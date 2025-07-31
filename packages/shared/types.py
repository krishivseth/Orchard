from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from datetime import datetime

class DeviceType(str, Enum):
    MAC = "mac"
    IPHONE = "iphone"
    IPAD = "ipad"

class DeviceStatus(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    ERROR = "error"

class LLMModel(BaseModel):
    id: str
    name: str
    size_gb: float
    min_memory_gb: float
    description: str
    supported_devices: List[DeviceType]

class DeviceInfo(BaseModel):
    id: str
    name: str
    type: DeviceType
    status: DeviceStatus
    ip_address: str
    port: int
    total_memory_gb: float
    available_memory_gb: float
    cpu_usage_percent: float
    temperature_celsius: Optional[float] = None
    last_heartbeat: datetime
    current_model: Optional[str] = None

class ChatMessage(BaseModel):
    id: str
    content: str
    role: str  # "user" or "assistant"
    timestamp: datetime
    device_id: Optional[str] = None

class InferenceRequest(BaseModel):
    message: str
    model_id: str
    max_tokens: int = 150
    temperature: float = 0.7

class InferenceResponse(BaseModel):
    response: str
    device_id: str
    processing_time_ms: int

class DeviceHealthMetrics(BaseModel):
    device_id: str
    memory_usage_gb: float
    cpu_usage_percent: float
    temperature_celsius: Optional[float]
    inference_count: int
    average_response_time_ms: float
    timestamp: datetime

class ModelDeploymentRequest(BaseModel):
    model_id: str
    device_ids: List[str]

class ModelDeploymentStatus(BaseModel):
    model_id: str
    device_id: str
    status: str  # "loading", "ready", "failed"
    progress_percent: int
    error_message: Optional[str] = None 