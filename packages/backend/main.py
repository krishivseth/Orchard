import asyncio
import collections
import ipaddress
import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Deque, Dict, List, Optional

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

from llama_sharding import (
    HF_MODEL_NAME,
    UNSUPPORTED_STRATEGY_MESSAGE,
    LlamaShardingEngine,
    ShardingError,
    ShardInferenceError,
)
from shared.types import (
    ChatMessage,
    DeviceHealthMetrics,
    DeviceInfo,
    DeviceStatus,
    DeviceType,
    DistributedInferenceRequest,
    InferenceRequest,
    InferenceResponse,
    LLMModel,
    ModelDeploymentRequest,
    ModelDeploymentStatus,
    ModelShardingConfig,
    ShardedInferenceResponse,
    ShardingStrategy,
)


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        logger.warning(f"Invalid value for {name}; using default {default}")
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        logger.warning(f"Invalid value for {name}; using default {default}")
        return default


class Settings:
    """Runtime configuration read from environment variables."""

    def __init__(self) -> None:
        self.token: Optional[str] = os.getenv("ORCHARD_TOKEN") or None
        # Seconds without a heartbeat before a device is marked OFFLINE
        self.heartbeat_offline_seconds: float = _env_float("ORCHARD_HEARTBEAT_OFFLINE_SECONDS", 120)
        # Seconds without a heartbeat before a device is removed entirely
        self.device_removal_seconds: float = _env_float("ORCHARD_DEVICE_REMOVAL_SECONDS", 300)
        # Window in which a live device with the same id blocks re-registration from another address
        self.register_conflict_seconds: float = _env_float("ORCHARD_REGISTER_CONFLICT_SECONDS", 120)
        # How often the health monitor runs
        self.health_poll_seconds: float = _env_float("ORCHARD_HEALTH_POLL_SECONDS", 30)
        self.chat_history_limit: int = _env_int("ORCHARD_CHAT_HISTORY_LIMIT", 1000)
        self.chat_history_page: int = _env_int("ORCHARD_CHAT_HISTORY_PAGE", 50)
        self.metrics_history_limit: int = _env_int("ORCHARD_METRICS_HISTORY_LIMIT", 100)
        # Device agent HTTP timeouts
        self.device_timeout_seconds: float = _env_float("ORCHARD_DEVICE_TIMEOUT_SECONDS", 30)
        self.device_connect_timeout_seconds: float = _env_float("ORCHARD_DEVICE_CONNECT_TIMEOUT_SECONDS", 5)
        self.deploy_timeout_seconds: float = _env_float("ORCHARD_DEPLOY_TIMEOUT_SECONDS", 60)
        # Hugging Face model loaded by device agents for layer-split sharding (env ORCHARD_HF_MODEL)
        self.hf_model: str = HF_MODEL_NAME

    @property
    def auth_headers(self) -> Dict[str, str]:
        return {"X-Orchard-Token": self.token} if self.token else {}

    @property
    def deploy_timeout(self) -> httpx.Timeout:
        return httpx.Timeout(self.deploy_timeout_seconds, connect=self.device_connect_timeout_seconds)


settings = Settings()


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
devices: Dict[str, DeviceInfo] = {}
models: Dict[str, LLMModel] = {}
chat_history: Deque[ChatMessage] = collections.deque(maxlen=settings.chat_history_limit)
device_metrics: Dict[str, List[DeviceHealthMetrics]] = {}

http_client: Optional[httpx.AsyncClient] = None
health_monitor_task: Optional[asyncio.Task] = None

llama_sharding_engine = LlamaShardingEngine()
llama_sharding_engine.auth_headers = settings.auth_headers


def initialize_models() -> None:
    sample_models = [
        LLMModel(
            id="llama2-7b",
            name="Llama 2 7B",
            size_gb=13.0,
            min_memory_gb=16.0,
            description="Meta's Llama 2 7B parameter model",
            supported_devices=[DeviceType.MAC, DeviceType.IPAD],
        ),
        LLMModel(
            id="mistral-7b",
            name="Mistral 7B",
            size_gb=14.0,
            min_memory_gb=16.0,
            description="Mistral AI's 7B parameter model",
            supported_devices=[DeviceType.MAC, DeviceType.IPAD],
        ),
        LLMModel(
            id="phi-3-mini",
            name="Phi-3 Mini",
            size_gb=2.5,
            min_memory_gb=4.0,
            description="Microsoft's small but capable 3.8B model",
            supported_devices=[DeviceType.MAC, DeviceType.IPAD, DeviceType.IPHONE],
        ),
        LLMModel(
            id="llama-3.2-1b",
            name="Llama 3.2 1B",
            size_gb=1.5,
            min_memory_gb=2.0,
            description="Meta's Llama 3.2 1B parameter model - perfect for distributed inference",
            supported_devices=[DeviceType.MAC],
        ),
    ]
    for model in sample_models:
        models[model.id] = model


# --------------------------------------------------------------------------- #
# WebSocket manager
# --------------------------------------------------------------------------- #
class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict) -> None:
        payload = json.dumps(jsonable_encoder(message))
        for connection in list(self.active_connections):
            try:
                await connection.send_text(payload)
            except Exception as e:
                logger.debug(f"Dropping websocket client after send failure: {e!r}")
                self.disconnect(connection)


manager = ConnectionManager()


async def broadcast_device_update(device: DeviceInfo) -> None:
    await manager.broadcast({"type": "device_update", "device": device.model_dump(mode="json")})


async def broadcast_device_removed(device_id: str) -> None:
    await manager.broadcast({"type": "device_removed", "device_id": device_id})


async def forget_device(device_id: str) -> Optional[DeviceInfo]:
    """Remove a device from all in-memory state and notify clients."""
    device = devices.pop(device_id, None)
    device_metrics.pop(device_id, None)
    llama_sharding_engine.remove_device(device_id)
    await broadcast_device_removed(device_id)
    return device


# --------------------------------------------------------------------------- #
# Lifespan / background tasks
# --------------------------------------------------------------------------- #
async def monitor_device_health() -> None:
    """Mark stale devices offline and remove long-dead ones."""
    while True:
        try:
            now = datetime.now()
            to_remove = []
            for device_id, device in list(devices.items()):
                age = now - device.last_heartbeat
                if age > timedelta(seconds=settings.device_removal_seconds):
                    to_remove.append(device_id)
                    logger.info(f"Removing offline device: {device.name} ({device_id})")
                elif age > timedelta(seconds=settings.heartbeat_offline_seconds):
                    if device.status != DeviceStatus.OFFLINE:
                        device.status = DeviceStatus.OFFLINE
                        logger.warning(f"Device {device.name} marked as offline")
                        await broadcast_device_update(device)
            for device_id in to_remove:
                await forget_device(device_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Device health monitor iteration failed")
        await asyncio.sleep(settings.health_poll_seconds)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global http_client, health_monitor_task
    initialize_models()
    if settings.token is None:
        logger.warning("ORCHARD_TOKEN is not set: device agent authentication is DISABLED")
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(settings.device_timeout_seconds, connect=settings.device_connect_timeout_seconds)
    )
    llama_sharding_engine.http_client = http_client
    health_monitor_task = asyncio.create_task(monitor_device_health())
    logger.info("Orchard LLM Distributor started")
    try:
        yield
    finally:
        if health_monitor_task is not None:
            health_monitor_task.cancel()
            try:
                await health_monitor_task
            except asyncio.CancelledError:
                pass
            health_monitor_task = None
        llama_sharding_engine.http_client = None
        await http_client.aclose()
        http_client = None
        logger.info("Orchard LLM Distributor stopped")


app = FastAPI(title="Orchard LLM Distributor", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^(http://localhost(:\d+)?|http://127\.0\.0\.1(:\d+)?|file://.*|null)$",
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def get_http_client() -> httpx.AsyncClient:
    if http_client is None:
        raise HTTPException(status_code=503, detail="Server is starting up")
    return http_client


async def require_agent_token(x_orchard_token: Optional[str] = Header(default=None)) -> None:
    """Agent→backend auth. No-op when ORCHARD_TOKEN is unset."""
    if settings.token is None:
        return
    if x_orchard_token != settings.token:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Orchard-Token")


def _is_local_address(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return addr.is_private or addr.is_loopback or addr.is_link_local


def _device_url(device: DeviceInfo, path: str) -> str:
    return f"http://{device.ip_address}:{device.port}{path}"


def _record_metrics(device_id: str, metrics: DeviceHealthMetrics) -> None:
    history = device_metrics.setdefault(device_id, [])
    history.append(metrics)
    if len(history) > settings.metrics_history_limit:
        del history[: len(history) - settings.metrics_history_limit]


class ShardedDeploymentRequest(BaseModel):
    model_id: str
    device_ids: Optional[List[str]] = None
    strategy: ShardingStrategy = ShardingStrategy.LAYER_SPLIT


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
@app.get("/health")
async def health():
    return {"status": "ok", "service": "orchard-backend"}


# --------------------------------------------------------------------------- #
# Device endpoints
# --------------------------------------------------------------------------- #
@app.get("/api/devices", response_model=List[DeviceInfo])
async def get_devices():
    """Get all registered devices"""
    return list(devices.values())


@app.post("/api/devices/register", dependencies=[Depends(require_agent_token)])
async def register_device(device: DeviceInfo):
    """Register a new device"""
    try:
        if not _is_local_address(device.ip_address):
            raise HTTPException(
                status_code=400,
                detail="ip_address must be a private, loopback, or link-local IP address",
            )

        existing = devices.get(device.id)
        if existing is not None and existing.status == DeviceStatus.ONLINE:
            same_endpoint = existing.ip_address == device.ip_address and existing.port == device.port
            recent = (datetime.now() - existing.last_heartbeat) <= timedelta(seconds=settings.register_conflict_seconds)
            if not same_endpoint and recent:
                raise HTTPException(
                    status_code=409,
                    detail="A live device with this id is already registered from a different address",
                )

        # Evict OFFLINE devices sharing this name
        for other_id, other in list(devices.items()):
            if other_id != device.id and other.name == device.name and other.status == DeviceStatus.OFFLINE:
                logger.info(f"Removing offline duplicate device: {other.name} ({other_id})")
                await forget_device(other_id)

        devices[device.id] = device
        llama_sharding_engine.device_connections[device.id] = f"{device.ip_address}:{device.port}"
        logger.info(f"Device registered: {device.name} ({device.id}) at {device.ip_address}:{device.port}")
        await broadcast_device_update(device)
        return {"status": "registered", "device_id": device.id}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Device registration failed")
        raise HTTPException(status_code=500, detail="Device registration failed")


@app.post("/api/devices/{device_id}/heartbeat", dependencies=[Depends(require_agent_token)])
async def device_heartbeat(device_id: str, metrics: DeviceHealthMetrics):
    """Update device heartbeat and metrics"""
    try:
        device = devices.get(device_id)
        if device is None:
            raise HTTPException(status_code=404, detail="Device not found")

        device.last_heartbeat = datetime.now()
        device.status = DeviceStatus.ONLINE
        device.cpu_usage_percent = metrics.cpu_usage_percent
        if metrics.temperature_celsius is not None:
            device.temperature_celsius = metrics.temperature_celsius
        if metrics.total_memory_gb is not None:
            device.total_memory_gb = metrics.total_memory_gb
        if metrics.available_memory_gb is not None:
            device.available_memory_gb = metrics.available_memory_gb
        else:
            device.available_memory_gb = max(0.0, device.total_memory_gb - metrics.memory_usage_gb)

        _record_metrics(device_id, metrics)
        await broadcast_device_update(device)
        return {"status": "ok"}
    except HTTPException:
        raise
    except Exception:
        logger.exception(f"Heartbeat handling failed for device {device_id}")
        raise HTTPException(status_code=500, detail="Heartbeat handling failed")


@app.delete("/api/devices/{device_id}")
async def remove_device(device_id: str):
    """Manually remove a device"""
    try:
        if device_id not in devices:
            raise HTTPException(status_code=404, detail="Device not found")
        device = await forget_device(device_id)
        logger.info(f"Manually removed device: {device.name} ({device_id})")
        return {"status": "removed", "device_id": device_id}
    except HTTPException:
        raise
    except Exception:
        logger.exception(f"Failed to remove device {device_id}")
        raise HTTPException(status_code=500, detail="Failed to remove device")


@app.post("/api/devices/cleanup")
async def cleanup_offline_devices():
    """Manually cleanup all offline devices"""
    try:
        offline_ids = [d.id for d in devices.values() if d.status == DeviceStatus.OFFLINE]
        for device_id in offline_ids:
            device = await forget_device(device_id)
            if device is not None:
                logger.info(f"Cleaned up offline device: {device.name} ({device_id})")
        return {"status": "cleaned", "removed_count": len(offline_ids)}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Offline device cleanup failed")
        raise HTTPException(status_code=500, detail="Offline device cleanup failed")


@app.get("/api/devices/{device_id}/metrics", response_model=List[DeviceHealthMetrics])
async def get_device_metrics(device_id: str):
    """Get device metrics history"""
    return device_metrics.get(device_id, [])


# --------------------------------------------------------------------------- #
# Model endpoints
# --------------------------------------------------------------------------- #
@app.get("/api/models", response_model=List[LLMModel])
async def get_models():
    """Get all available LLM models"""
    return list(models.values())


@app.get("/api/models/sharded-configs")
async def get_sharded_model_configs():
    """Get all sharded model configurations"""
    return {
        "configs": {
            model_id: config.model_dump(mode="json")
            for model_id, config in llama_sharding_engine.sharding_configs.items()
        }
    }


def _failed(model_id: str, device_id: str, message: str) -> ModelDeploymentStatus:
    return ModelDeploymentStatus(
        model_id=model_id, device_id=device_id, status="failed", progress_percent=0, error_message=message
    )


def _ready(model_id: str, device_id: str) -> ModelDeploymentStatus:
    return ModelDeploymentStatus(model_id=model_id, device_id=device_id, status="ready", progress_percent=100)


async def _deploy_to_device(client: httpx.AsyncClient, model: LLMModel, device_id: str) -> ModelDeploymentStatus:
    device = devices.get(device_id)
    if device is None:
        return _failed(model.id, device_id, "Device not found")
    if device.available_memory_gb < model.min_memory_gb:
        return _failed(model.id, device_id, "Insufficient memory")

    try:
        response = await client.post(
            _device_url(device, "/deploy"),
            json={"model_id": model.id},
            headers=settings.auth_headers,
            timeout=settings.deploy_timeout,
        )
    except httpx.HTTPError as e:
        logger.error(f"Deploy request to {device.name} ({device_id}) failed: {e!r}")
        return _failed(model.id, device_id, "Device unreachable")

    if response.status_code != 200:
        logger.error(f"Deploy to {device.name} ({device_id}) returned HTTP {response.status_code}")
        message = f"Device deployment failed (HTTP {response.status_code})"
        if 400 <= response.status_code < 500:
            # Agent-side validation errors (e.g. model not pulled) are actionable for the user.
            try:
                detail = response.json().get("detail")
                if isinstance(detail, str) and detail:
                    message = detail[:300]
            except ValueError:
                pass
        return _failed(model.id, device_id, message)

    device.current_model = model.id
    device.status = DeviceStatus.ONLINE
    await broadcast_device_update(device)
    return _ready(model.id, device_id)


@app.post("/api/models/deploy")
async def deploy_model(deployment: ModelDeploymentRequest):
    """Deploy a model to selected devices"""
    try:
        model = models.get(deployment.model_id)
        if model is None:
            raise HTTPException(status_code=404, detail="Model not found")
        client = get_http_client()

        online = [d for d in devices.values() if d.status == DeviceStatus.ONLINE]
        if len(online) >= 2 and len(deployment.device_ids) == 1:
            logger.info(
                f"{len(online)} devices online; consider sharded deployment for better distribution"
            )

        results = await asyncio.gather(
            *[_deploy_to_device(client, model, device_id) for device_id in deployment.device_ids],
            return_exceptions=True,
        )
        deployments = []
        for device_id, result in zip(deployment.device_ids, results):
            if isinstance(result, BaseException):
                logger.error(f"Deploy to {device_id} raised: {result!r}")
                deployments.append(_failed(model.id, device_id, "Deployment failed"))
            else:
                deployments.append(result)
        return {"deployments": deployments}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Model deployment failed")
        raise HTTPException(status_code=500, detail="Model deployment failed")


async def _deploy_shard(client: httpx.AsyncClient, model_id: str, shard, device: DeviceInfo) -> ModelDeploymentStatus:
    try:
        response = await client.post(
            _device_url(device, "/llama/shard/deploy"),
            json={"shard": shard.model_dump(mode="json")},
            headers=settings.auth_headers,
            timeout=settings.deploy_timeout,
        )
    except httpx.HTTPError as e:
        logger.error(f"Shard {shard.shard_id} deploy to {device.name} ({device.id}) failed: {e!r}")
        return _failed(model_id, device.id, "Device unreachable")

    if response.status_code != 200:
        logger.error(f"Shard {shard.shard_id} deploy to {device.name} returned HTTP {response.status_code}")
        return _failed(model_id, device.id, f"Llama shard deployment failed (HTTP {response.status_code})")

    device.current_model = model_id
    device.status = DeviceStatus.ONLINE
    await broadcast_device_update(device)
    return _ready(model_id, device.id)


async def _deploy_sharded(deployment: ShardedDeploymentRequest, auto_selected: bool) -> dict:
    """Shared implementation for the sharded deployment endpoints."""
    if deployment.model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    if deployment.strategy != ShardingStrategy.LAYER_SPLIT:
        raise HTTPException(status_code=400, detail=UNSUPPORTED_STRATEGY_MESSAGE)
    client = get_http_client()

    if auto_selected or deployment.device_ids is None:
        target_devices = [d for d in devices.values() if d.status == DeviceStatus.ONLINE]
        if len(target_devices) < 2:
            raise HTTPException(
                status_code=400, detail="Auto-sharded deployment requires at least 2 online devices"
            )
    else:
        missing = [did for did in deployment.device_ids if did not in devices]
        if missing:
            raise HTTPException(status_code=404, detail=f"Unknown device ids: {', '.join(missing)}")
        target_devices = [devices[did] for did in deployment.device_ids]

    try:
        config: ModelShardingConfig = await llama_sharding_engine.create_llama_sharding_config(
            deployment.model_id, target_devices, deployment.strategy
        )
    except ShardingError as e:
        raise HTTPException(status_code=400, detail=str(e))

    logger.info(
        f"Sharding {deployment.model_id} ({deployment.strategy.value}, hf_model={config.model_name}) across "
        f"{len(target_devices)} devices: {[d.name for d in target_devices]}"
    )

    for device in target_devices:
        llama_sharding_engine.device_connections[device.id] = f"{device.ip_address}:{device.port}"

    by_id = {d.id: d for d in target_devices}
    results = await asyncio.gather(
        *[_deploy_shard(client, deployment.model_id, shard, by_id[shard.device_id]) for shard in config.shards],
        return_exceptions=True,
    )
    deployments: List[ModelDeploymentStatus] = []
    for shard, result in zip(config.shards, results):
        if isinstance(result, BaseException):
            logger.error(f"Shard {shard.shard_id} deploy raised: {result!r}")
            deployments.append(_failed(deployment.model_id, shard.device_id, "Shard deployment failed"))
        else:
            deployments.append(result)

    body = {
        "config": config.model_dump(mode="json"),
        "deployments": [d.model_dump(mode="json") for d in deployments],
        "auto_selected": auto_selected,
        "devices_used": [d.name for d in target_devices],
    }

    if all(d.status == "ready" for d in deployments):
        llama_sharding_engine.sharding_configs[deployment.model_id] = config
        return {"status": "success", **body}

    failed = [d.device_id for d in deployments if d.status != "ready"]
    logger.error(f"Sharded deployment of {deployment.model_id} failed on devices: {failed}")
    raise HTTPException(
        status_code=502,
        detail={"message": "Sharded deployment failed on one or more devices", "failed_devices": failed, **body},
    )


@app.post("/api/models/deploy-llama-sharded-auto")
async def deploy_llama_sharded_model_auto(deployment: ShardedDeploymentRequest):
    """Deploy a sharded model across ALL online devices"""
    try:
        return await _deploy_sharded(deployment, auto_selected=True)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Auto sharded deployment failed")
        raise HTTPException(status_code=500, detail="Sharded deployment failed")


@app.post("/api/models/deploy-llama-sharded")
async def deploy_llama_sharded_model(deployment: ShardedDeploymentRequest):
    """Deploy a sharded model across the given devices (or all online devices if none given)"""
    try:
        return await _deploy_sharded(deployment, auto_selected=deployment.device_ids is None)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Sharded deployment failed")
        raise HTTPException(status_code=500, detail="Sharded deployment failed")


# --------------------------------------------------------------------------- #
# Chat endpoints
# --------------------------------------------------------------------------- #
async def _store_and_broadcast(user_message: ChatMessage, assistant_message: ChatMessage) -> None:
    chat_history.extend([user_message, assistant_message])
    await manager.broadcast({"type": "new_message", "message": assistant_message.model_dump(mode="json")})


@app.post("/api/chat", response_model=InferenceResponse)
async def chat(request: InferenceRequest):
    """Send a chat message to the distributed LLM"""
    try:
        available = [
            d for d in devices.values()
            if d.current_model == request.model_id and d.status == DeviceStatus.ONLINE
        ]
        if not available:
            raise HTTPException(status_code=404, detail="No devices available for this model")
        client = get_http_client()

        selected = min(available, key=lambda d: d.cpu_usage_percent)
        start_time = datetime.now()
        try:
            response = await client.post(
                _device_url(selected, "/inference"),
                json=request.model_dump(mode="json"),
                headers=settings.auth_headers,
            )
        except httpx.HTTPError as e:
            logger.error(f"Inference request to {selected.name} ({selected.id}) failed: {e!r}")
            raise HTTPException(status_code=502, detail=f"Device {selected.id} is unreachable")

        if response.status_code != 200:
            logger.error(f"Inference on {selected.name} returned HTTP {response.status_code}")
            raise HTTPException(status_code=502, detail=f"Inference failed on device {selected.id}")

        try:
            text = response.json()["response"]
        except (ValueError, KeyError, TypeError):
            logger.error(f"Inference on {selected.name} returned a malformed response")
            raise HTTPException(status_code=502, detail=f"Malformed response from device {selected.id}")

        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        user_message = ChatMessage(id=str(uuid.uuid4()), content=request.message, role="user", timestamp=start_time)
        assistant_message = ChatMessage(
            id=str(uuid.uuid4()),
            content=text,
            role="assistant",
            timestamp=datetime.now(),
            device_id=selected.id,
            device_ids=[selected.id],
        )
        await _store_and_broadcast(user_message, assistant_message)

        return InferenceResponse(response=text, device_id=selected.id, processing_time_ms=processing_time)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Chat inference failed")
        raise HTTPException(status_code=500, detail="Chat inference failed")


@app.post("/api/chat/llama-sharded", response_model=ShardedInferenceResponse)
async def llama_sharded_chat(request: DistributedInferenceRequest):
    """Send a chat message to the sharded model"""
    try:
        config = llama_sharding_engine.sharding_configs.get(request.model_id)
        if config is None:
            raise HTTPException(status_code=404, detail="No Llama sharding config found for model")
        if config.strategy != ShardingStrategy.LAYER_SPLIT:
            raise HTTPException(status_code=400, detail=UNSUPPORTED_STRATEGY_MESSAGE)
        get_http_client()

        start_time = datetime.now()
        try:
            text = await llama_sharding_engine.execute_llama_sharded_inference(request, config)
        except ShardInferenceError as e:
            logger.error(f"Sharded inference failed: {e}")
            raise HTTPException(
                status_code=502,
                detail={"message": str(e), "shard_id": e.shard_id, "device_id": e.device_id},
            )
        except ShardingError as e:
            raise HTTPException(status_code=400, detail=str(e))
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)

        user_message = ChatMessage(id=str(uuid.uuid4()), content=request.message, role="user", timestamp=start_time)
        assistant_message = ChatMessage(
            id=str(uuid.uuid4()),
            content=text,
            role="assistant",
            timestamp=datetime.now(),
            device_id=config.devices_used[0] if config.devices_used else None,
            device_ids=list(config.devices_used),
        )
        await _store_and_broadcast(user_message, assistant_message)

        share = 100.0 / len(config.shards) if config.shards else 0.0
        shard_contributions = {shard.device_id: share for shard in config.shards}

        return ShardedInferenceResponse(
            response=text,
            device_ids=list(config.devices_used),
            processing_time_ms=processing_time,
            shard_contributions=shard_contributions,
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Sharded chat failed")
        raise HTTPException(status_code=500, detail="Sharded chat failed")


@app.get("/api/chat/history", response_model=List[ChatMessage])
async def get_chat_history():
    """Get recent chat history"""
    page = settings.chat_history_page
    if page <= 0 or len(chat_history) <= page:
        return list(chat_history)
    return list(chat_history)[-page:]


# --------------------------------------------------------------------------- #
# WebSocket
# --------------------------------------------------------------------------- #
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # Client messages are currently ignored
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug(f"WebSocket connection closed with error: {e!r}")
    finally:
        manager.disconnect(websocket)


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Orchard Backend Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    args = parser.parse_args()

    logger.info(f"Starting Orchard backend on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
