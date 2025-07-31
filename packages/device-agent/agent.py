import asyncio
import json
import platform
import socket
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
import psutil
import httpx
from fastapi import FastAPI, HTTPException
import uvicorn
from loguru import logger
import sys
import os
sys.path.append('../shared')

from shared.types import (
    DeviceInfo, DeviceStatus, DeviceType, DeviceHealthMetrics,
    InferenceRequest, InferenceResponse
)

class LLMInferenceEngine:
    """Mock LLM inference engine - replace with actual model loading"""
    
    def __init__(self):
        self.loaded_model: Optional[str] = None
        self.model_responses = {
            "llama2-7b": "This is a response from Llama 2 7B running on {}",
            "mistral-7b": "This is a response from Mistral 7B running on {}",
            "phi-3-mini": "This is a response from Phi-3 Mini running on {}"
        }
    
    async def load_model(self, model_id: str) -> bool:
        """Load a model (simulated)"""
        logger.info(f"Loading model {model_id}")
        # Simulate loading time
        await asyncio.sleep(2)
        
        if model_id in self.model_responses:
            self.loaded_model = model_id
            logger.info(f"Model {model_id} loaded successfully")
            return True
        return False
    
    async def unload_model(self):
        """Unload current model"""
        if self.loaded_model:
            logger.info(f"Unloading model {self.loaded_model}")
            self.loaded_model = None
    
    async def inference(self, request: InferenceRequest) -> str:
        """Run inference on the loaded model"""
        if not self.loaded_model:
            raise ValueError("No model loaded")
        
        if self.loaded_model != request.model_id:
            raise ValueError(f"Requested model {request.model_id} not loaded")
        
        # Simulate inference time
        await asyncio.sleep(0.5)
        
        # Mock response based on the prompt
        device_name = platform.node()
        base_response = self.model_responses[self.loaded_model].format(device_name)
        
        # Simple echo with model identifier
        return f"{base_response}. You asked: '{request.message}'"

class DeviceAgent:
    def __init__(self, backend_url: str = "http://localhost:8000", port: int = 8001):
        self.device_id = str(uuid.uuid4())
        self.backend_url = backend_url
        self.port = port
        self.device_info = self._create_device_info()
        self.inference_engine = LLMInferenceEngine()
        self.app = FastAPI(title=f"Device Agent - {self.device_info.name}")
        self._setup_routes()
        
    def _create_device_info(self) -> DeviceInfo:
        """Create device info based on system properties"""
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        # Determine device type based on system info
        if system == "darwin":
            if "iphone" in platform.platform().lower():
                device_type = DeviceType.IPHONE
            elif "ipad" in platform.platform().lower():
                device_type = DeviceType.IPAD
            else:
                device_type = DeviceType.MAC
        else:
            device_type = DeviceType.MAC  # Default fallback
        
        # Get network info
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        
        # Get memory info
        memory = psutil.virtual_memory()
        total_memory_gb = memory.total / (1024**3)
        available_memory_gb = memory.available / (1024**3)
        
        return DeviceInfo(
            id=self.device_id,
            name=platform.node(),
            type=device_type,
            status=DeviceStatus.ONLINE,
            ip_address=local_ip,
            port=self.port,
            total_memory_gb=round(total_memory_gb, 1),
            available_memory_gb=round(available_memory_gb, 1),
            cpu_usage_percent=psutil.cpu_percent(),
            temperature_celsius=self._get_temperature(),
            last_heartbeat=datetime.now()
        )
    
    def _get_temperature(self) -> Optional[float]:
        """Get device temperature (mock implementation)"""
        try:
            # On macOS, you could use powermetrics or thermal state
            # For now, return a simulated temperature
            import random
            return round(random.uniform(45.0, 75.0), 1)
        except:
            return None
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/health")
        async def health():
            return {"status": "healthy", "device_id": self.device_id}
        
        @self.app.post("/deploy")
        async def deploy_model(request: dict):
            """Deploy a model to this device"""
            model_id = request.get("model_id")
            if not model_id:
                raise HTTPException(status_code=400, detail="model_id required")
            
            try:
                success = await self.inference_engine.load_model(model_id)
                if success:
                    self.device_info.current_model = model_id
                    self.device_info.status = DeviceStatus.ONLINE
                    return {"status": "deployed", "model_id": model_id}
                else:
                    raise HTTPException(status_code=400, detail="Model deployment failed")
            except Exception as e:
                logger.error(f"Model deployment error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/inference")
        async def inference(request: InferenceRequest):
            """Run inference on the loaded model"""
            try:
                start_time = datetime.now()
                response = await self.inference_engine.inference(request)
                processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
                
                return {
                    "response": response,
                    "processing_time_ms": processing_time,
                    "device_id": self.device_id
                }
            except Exception as e:
                logger.error(f"Inference error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get current device metrics"""
            return self._get_current_metrics().dict()
    
    def _get_current_metrics(self) -> DeviceHealthMetrics:
        """Get current device health metrics"""
        memory = psutil.virtual_memory()
        cpu_usage = psutil.cpu_percent()
        
        return DeviceHealthMetrics(
            device_id=self.device_id,
            memory_usage_gb=round((memory.total - memory.available) / (1024**3), 1),
            cpu_usage_percent=cpu_usage,
            temperature_celsius=self._get_temperature(),
            inference_count=0,  # Could track this
            average_response_time_ms=100,  # Could track this
            timestamp=datetime.now()
        )
    
    async def register_with_backend(self):
        """Register this device with the backend"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.backend_url}/api/devices/register",
                    json=self.device_info.dict()
                )
                if response.status_code == 200:
                    logger.info(f"Device registered successfully: {self.device_id}")
                    return True
                else:
                    logger.error(f"Registration failed: {response.text}")
                    return False
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return False
    
    async def send_heartbeat(self):
        """Send periodic heartbeat to backend"""
        while True:
            try:
                metrics = self._get_current_metrics()
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.backend_url}/api/devices/{self.device_id}/heartbeat",
                        json=metrics.dict()
                    )
                    if response.status_code != 200:
                        logger.warning(f"Heartbeat failed: {response.text}")
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
            
            await asyncio.sleep(10)  # Send heartbeat every 10 seconds
    
    async def start(self):
        """Start the device agent"""
        # Register with backend
        await self.register_with_backend()
        
        # Start heartbeat task
        asyncio.create_task(self.send_heartbeat())
        
        # Start FastAPI server
        config = uvicorn.Config(
            app=self.app,
            host="0.0.0.0",
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()

async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Orchard Device Agent")
    parser.add_argument("--backend", default="http://localhost:8000", help="Backend URL")
    parser.add_argument("--port", type=int, default=8001, help="Agent port")
    
    args = parser.parse_args()
    
    agent = DeviceAgent(backend_url=args.backend, port=args.port)
    logger.info(f"Starting device agent on port {args.port}")
    await agent.start()

if __name__ == "__main__":
    asyncio.run(main()) 