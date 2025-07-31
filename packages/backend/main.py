from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import httpx
from loguru import logger
import sys
sys.path.append('../shared')

from shared.types import (
    DeviceInfo, DeviceStatus, LLMModel, ChatMessage,
    InferenceRequest, InferenceResponse, DeviceHealthMetrics,
    ModelDeploymentRequest, ModelDeploymentStatus, DeviceType
)

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
    device.last_heartbeat = datetime.now()
    devices[device.id] = device
    await manager.broadcast({
        "type": "device_update",
        "device": device.dict()
    })
    logger.info(f"Device registered: {device.name} ({device.id})")
    return {"status": "registered", "device_id": device.id}

@app.post("/api/devices/{device_id}/heartbeat")
async def device_heartbeat(device_id: str, metrics: DeviceHealthMetrics):
    """Update device heartbeat and health metrics"""
    if device_id not in devices:
        raise HTTPException(status_code=404, detail="Device not found")
    
    device = devices[device_id]
    device.last_heartbeat = datetime.now()
    device.available_memory_gb = metrics.memory_usage_gb
    device.cpu_usage_percent = metrics.cpu_usage_percent
    device.temperature_celsius = metrics.temperature_celsius
    
    # Store metrics history
    if device_id not in device_metrics:
        device_metrics[device_id] = []
    device_metrics[device_id].append(metrics)
    
    # Keep only last 100 metrics per device
    if len(device_metrics[device_id]) > 100:
        device_metrics[device_id] = device_metrics[device_id][-100:]
    
    await manager.broadcast({
        "type": "device_metrics",
        "device_id": device_id,
        "metrics": metrics.dict()
    })
    
    return {"status": "ok"}

# LLM Model Endpoints
@app.get("/api/models", response_model=List[LLMModel])
async def get_models():
    """Get all available LLM models"""
    return list(models.values())

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
                    device.status = DeviceStatus.BUSY
                    deployment_results.append(ModelDeploymentStatus(
                        model_id=deployment.model_id,
                        device_id=device_id,
                        status="loading",
                        progress_percent=0
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
                json=request.dict()
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
                    "message": assistant_message.dict()
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

@app.get("/api/chat/history", response_model=List[ChatMessage])
async def get_chat_history():
    """Get chat history"""
    return chat_history[-50:]  # Return last 50 messages

@app.get("/api/devices/{device_id}/metrics", response_model=List[DeviceHealthMetrics])
async def get_device_metrics(device_id: str):
    """Get device health metrics history"""
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
            # Echo back for ping/pong
            await websocket.send_text(f"pong: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Background task for device health monitoring
async def monitor_device_health():
    """Monitor device health and update status"""
    while True:
        current_time = datetime.now()
        for device_id, device in devices.items():
            # Mark devices as offline if no heartbeat for 30 seconds
            if current_time - device.last_heartbeat > timedelta(seconds=30):
                if device.status != DeviceStatus.OFFLINE:
                    device.status = DeviceStatus.OFFLINE
                    await manager.broadcast({
                        "type": "device_update",
                        "device": device.dict()
                    })
                    logger.warning(f"Device {device.name} marked as offline")
            elif device.status == DeviceStatus.OFFLINE:
                device.status = DeviceStatus.ONLINE
                await manager.broadcast({
                    "type": "device_update", 
                    "device": device.dict()
                })
                logger.info(f"Device {device.name} back online")
        
        await asyncio.sleep(10)  # Check every 10 seconds

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 