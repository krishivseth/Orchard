from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import httpx
from loguru import logger
from shared.types import (
    DeviceInfo, DeviceStatus, LLMModel, ChatMessage,
    InferenceRequest, InferenceResponse, DeviceHealthMetrics,
    ModelDeploymentRequest, ModelDeploymentStatus, DeviceType,
    ModelShardingConfig, ModelShard, ShardingStrategy,
    DistributedInferenceRequest, ShardedInferenceResponse
)
from llama_sharding import LlamaShardingEngine

app = FastAPI(title="Orchard LLM Distributor", version="1.0.0")

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage (in production, use Redis/PostgreSQL)
devices: Dict[str, DeviceInfo] = {}
models: Dict[str, LLMModel] = {}
chat_history: List[ChatMessage] = []
active_connections: List[WebSocket] = []
device_metrics: Dict[str, List[DeviceHealthMetrics]] = {}

# Initialize Llama sharding engine
llama_sharding_engine = LlamaShardingEngine()

# Initialize with some sample LLM models
def initialize_models():
    sample_models = [
        LLMModel(
            id="llama2-7b",
            name="Llama 2 7B",
            size_gb=13.0,
            min_memory_gb=16.0,
            description="Meta's Llama 2 7B parameter model",
            supported_devices=[DeviceType.MAC, DeviceType.IPAD]
        ),
        LLMModel(
            id="mistral-7b",
            name="Mistral 7B",
            size_gb=14.0,
            min_memory_gb=16.0,
            description="Mistral AI's 7B parameter model",
            supported_devices=[DeviceType.MAC, DeviceType.IPAD]
        ),
        LLMModel(
            id="phi-3-mini",
            name="Phi-3 Mini",
            size_gb=2.5,
            min_memory_gb=4.0,
            description="Microsoft's small but capable 3.8B model",
            supported_devices=[DeviceType.MAC, DeviceType.IPAD, DeviceType.IPHONE]
        ),
        LLMModel(
            id="llama-3.2-1b",
            name="Llama 3.2 1B",
            size_gb=1.5,
            min_memory_gb=2.0,
            description="Meta's Llama 3.2 1B parameter model - perfect for distributed inference",
            supported_devices=[DeviceType.MAC]
        ),
    ]
    for model in sample_models:
        models[model.id] = model

@app.on_event("startup")
async def startup_event():
    initialize_models()
    # Start background task for device health monitoring
    asyncio.create_task(monitor_device_health())
    logger.info("Orchard LLM Distributor started")

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                self.disconnect(connection)

manager = ConnectionManager()

# Device Management Endpoints
@app.get("/api/devices", response_model=List[DeviceInfo])
async def get_devices():
    """Get all registered devices"""
    return list(devices.values())

@app.post("/api/devices/register")
async def register_device(device: DeviceInfo):
    """Register a new device"""
    devices[device.id] = device
    logger.info(f"Device registered: {device.name} ({device.id})")
    return {"status": "registered", "device_id": device.id}

@app.post("/api/devices/{device_id}/heartbeat")
async def device_heartbeat(device_id: str, metrics: DeviceHealthMetrics):
    """Update device heartbeat and metrics"""
    if device_id in devices:
        devices[device_id].last_heartbeat = datetime.now()
        devices[device_id].cpu_usage_percent = metrics.cpu_usage_percent
        devices[device_id].available_memory_gb = metrics.memory_usage_gb
        
        # Store metrics history
        if device_id not in device_metrics:
            device_metrics[device_id] = []
        device_metrics[device_id].append(metrics)
        
        # Keep only last 100 metrics
        if len(device_metrics[device_id]) > 100:
            device_metrics[device_id] = device_metrics[device_id][-100:]
    
    return {"status": "ok"}

# LLM Model Endpoints
@app.get("/api/models", response_model=List[LLMModel])
async def get_models():
    """Get all available LLM models"""
    return list(models.values())

@app.get("/api/models/sharded-configs")
async def get_sharded_model_configs():
    """Get all sharded model configurations"""
    return {
        "configs": {model_id: config.model_dump(mode='json') for model_id, config in llama_sharding_engine.sharding_configs.items()}
    }

@app.post("/api/models/deploy")
async def deploy_model(deployment: ModelDeploymentRequest):
    """Deploy a model to selected devices"""
    if deployment.model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = models[deployment.model_id]
    deployment_results = []
    
    for device_id in deployment.device_ids:
        if device_id not in devices:
            continue
            
        device = devices[device_id]
        
        # Check if device has enough memory
        if device.available_memory_gb < model.min_memory_gb:
            deployment_results.append(ModelDeploymentStatus(
                model_id=deployment.model_id,
                device_id=device_id,
                status="failed",
                progress_percent=0,
                error_message="Insufficient memory"
            ))
            continue
        
        # Send deployment request to device
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"http://{device.ip_address}:{device.port}/deploy",
                    json={"model_id": deployment.model_id}
                )
                if response.status_code == 200:
                    device.current_model = deployment.model_id
                    device.status = DeviceStatus.ONLINE  # Device is ready for inference after successful deployment
                    deployment_results.append(ModelDeploymentStatus(
                        model_id=deployment.model_id,
                        device_id=device_id,
                        status="ready",  # Changed from "loading" to "ready" since deployment completed
                        progress_percent=100
                    ))
                else:
                    deployment_results.append(ModelDeploymentStatus(
                        model_id=deployment.model_id,
                        device_id=device_id,
                        status="failed",
                        progress_percent=0,
                        error_message="Device deployment failed"
                    ))
        except Exception as e:
            deployment_results.append(ModelDeploymentStatus(
                model_id=deployment.model_id,
                device_id=device_id,
                status="failed",
                progress_percent=0,
                error_message=str(e)
            ))
    
    return {"deployments": deployment_results}

@app.post("/api/models/deploy-llama-sharded")
async def deploy_llama_sharded_model(deployment: dict):
    """Deploy Llama 3.2 with sharding across multiple devices"""
    model_id = deployment.get("model_id")
    device_ids = deployment.get("device_ids", [])
    strategy = ShardingStrategy(deployment.get("strategy", "layer_split"))
    
    if model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Get device info
    target_devices = [devices[did] for did in device_ids if did in devices]
    if not target_devices:
        raise HTTPException(status_code=404, detail="No valid devices found")
    
    try:
        # Create Llama sharding configuration
        config = await llama_sharding_engine.create_llama_sharding_config(
            model_id, target_devices, strategy
        )
        
        # Store device connections
        for device in target_devices:
            llama_sharding_engine.device_connections[device.id] = f"{device.ip_address}:{device.port}"
        
        # Deploy shards to devices
        deployment_results = []
        for shard in config.shards:
            device = next(d for d in target_devices if d.id == shard.device_id)
            
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:  # Longer timeout for model loading
                    response = await client.post(
                        f"http://{device.ip_address}:{device.port}/llama/shard/deploy",
                        json={"shard": shard.model_dump()}
                    )
                    
                    if response.status_code == 200:
                        # Mark device as running this sharded model
                        device.current_model = model_id
                        device.status = DeviceStatus.ONLINE
                        deployment_results.append(ModelDeploymentStatus(
                            model_id=model_id,
                            device_id=device.id,
                            status="ready",
                            progress_percent=100
                        ))
                    else:
                        deployment_results.append(ModelDeploymentStatus(
                            model_id=model_id,
                            device_id=device.id,
                            status="failed",
                            progress_percent=0,
                            error_message="Llama shard deployment failed"
                        ))
                        
            except Exception as e:
                deployment_results.append(ModelDeploymentStatus(
                    model_id=model_id,
                    device_id=device.id,
                    status="failed",
                    progress_percent=0,
                    error_message=str(e)
                ))
        
        # Store sharding configuration
        llama_sharding_engine.sharding_configs[model_id] = config
        
        return {
            "status": "success",
            "config": config.model_dump(mode='json'),
            "deployments": deployment_results
        }
        
    except Exception as e:
        logger.error(f"Llama sharded deployment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Chat and Inference Endpoints
@app.post("/api/chat", response_model=InferenceResponse)
async def chat(request: InferenceRequest):
    """Send a chat message to the distributed LLM"""
    # Find available devices running the requested model
    available_devices = [
        device for device in devices.values()
        if device.current_model == request.model_id and device.status == DeviceStatus.ONLINE
    ]
    
    if not available_devices:
        raise HTTPException(status_code=404, detail="No devices available for this model")
    
    # Simple load balancing - choose device with lowest CPU usage
    selected_device = min(available_devices, key=lambda d: d.cpu_usage_percent)
    
    try:
        start_time = datetime.now()
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"http://{selected_device.ip_address}:{selected_device.port}/inference",
                json=request.model_dump(mode='json')
            )
            
            if response.status_code == 200:
                result = response.json()
                processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
                
                # Store chat message
                user_message = ChatMessage(
                    id=str(uuid.uuid4()),
                    content=request.message,
                    role="user",
                    timestamp=start_time
                )
                assistant_message = ChatMessage(
                    id=str(uuid.uuid4()),
                    content=result["response"],
                    role="assistant",
                    timestamp=datetime.now(),
                    device_id=selected_device.id
                )
                
                chat_history.extend([user_message, assistant_message])
                
                # Broadcast to WebSocket clients
                await manager.broadcast({
                    "type": "new_message",
                    "message": assistant_message.model_dump(mode='json')
                })
                
                return InferenceResponse(
                    response=result["response"],
                    device_id=selected_device.id,
                    processing_time_ms=processing_time
                )
            else:
                raise HTTPException(status_code=500, detail="Inference failed on device")
                
    except Exception as e:
        logger.error(f"Chat inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/llama-sharded", response_model=ShardedInferenceResponse)
async def llama_sharded_chat(request: DistributedInferenceRequest):
    """Send a chat message to the sharded Llama 3.2 model"""
    
    if request.model_id not in llama_sharding_engine.sharding_configs:
        raise HTTPException(status_code=404, detail="No Llama sharding config found for model")
    
    config = llama_sharding_engine.sharding_configs[request.model_id]
    
    try:
        start_time = datetime.now()
        
        # Execute sharded Llama inference
        response = await llama_sharding_engine.execute_llama_sharded_inference(request, config)
        
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Store chat messages
        user_message = ChatMessage(
            id=str(uuid.uuid4()),
            content=request.message,
            role="user",
            timestamp=start_time
        )
        assistant_message = ChatMessage(
            id=str(uuid.uuid4()),
            content=response,
            role="assistant",
            timestamp=datetime.now(),
            device_id=",".join(config.devices_used)  # Multiple devices for sharded
        )
        
        chat_history.extend([user_message, assistant_message])
        
        # Broadcast to WebSocket clients
        await manager.broadcast({
            "type": "new_message",
            "message": assistant_message.model_dump(mode='json')
        })
        
        # Calculate shard contributions
        shard_contributions = {}
        for shard in config.shards:
            shard_contributions[shard.device_id] = 100.0 / len(config.shards)
        
        return ShardedInferenceResponse(
            response=response,
            device_ids=config.devices_used,
            processing_time_ms=processing_time,
            shard_contributions=shard_contributions
        )
        
    except Exception as e:
        logger.error(f"Llama sharded chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chat/history", response_model=List[ChatMessage])
async def get_chat_history():
    """Get chat history"""
    return chat_history[-50:]  # Return last 50 messages

@app.get("/api/devices/{device_id}/metrics", response_model=List[DeviceHealthMetrics])
async def get_device_metrics(device_id: str):
    """Get device metrics history"""
    if device_id not in device_metrics:
        return []
    return device_metrics[device_id]

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages if needed
    except WebSocketDisconnect:
        manager.disconnect(websocket)

async def monitor_device_health():
    """Monitor device health and mark offline devices"""
    while True:
        current_time = datetime.now()
        for device in devices.values():
            if (current_time - device.last_heartbeat) > timedelta(minutes=2):
                device.status = DeviceStatus.OFFLINE
                logger.warning(f"Device {device.name} marked as offline")
        
        await asyncio.sleep(30)  # Check every 30 seconds

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 