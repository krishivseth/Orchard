import argparse
import asyncio
import ipaddress
import os
import platform
import socket
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
import psutil
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from loguru import logger

# Add this directory to path for shared imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from shared.types import (  # noqa: E402
    DeviceInfo, DeviceStatus, DeviceType, DeviceHealthMetrics,
    InferenceRequest, ModelShard,
)
from ollama_inference import OllamaInferenceEngine, OllamaUnavailableError  # noqa: E402
from llama_sharded_inference import ShardNotLoadedError  # noqa: E402  (torch imported lazily)


ORCHARD_TOKEN = os.environ.get("ORCHARD_TOKEN") or None
DEVICE_ID_FILE = Path.home() / ".orchard" / "device_id"
HEARTBEAT_INTERVAL_S = 30
BACKOFF_MIN_S = 5
BACKOFF_MAX_S = 60


# --------------------------------------------------------------------------- auth

async def require_token(x_orchard_token: Optional[str] = Header(default=None)):
    """Require X-Orchard-Token on protected routes when ORCHARD_TOKEN is configured."""
    if ORCHARD_TOKEN is None:
        return
    if x_orchard_token != ORCHARD_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Orchard-Token")


def backend_headers() -> dict:
    return {"X-Orchard-Token": ORCHARD_TOKEN} if ORCHARD_TOKEN else {}


# --------------------------------------------------------------------------- network

def _is_private_routable(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return addr.version == 4 and addr.is_private and not addr.is_loopback and not addr.is_link_local


def get_network_ip(prefer_thunderbolt: bool = False, override: Optional[str] = None) -> str:
    """Pick the IP other devices should use to reach this agent.

    Priority: explicit override (--ip / ORCHARD_AGENT_IP) > Thunderbolt bridge (169.254.x.x,
    only with prefer_thunderbolt) > routable private IPv4 from the UDP-connect trick >
    any private IPv4 on an interface > 127.0.0.1.
    """
    override = override or os.environ.get("ORCHARD_AGENT_IP")
    if override:
        logger.info(f"Using configured agent IP: {override}")
        return override

    candidates = []
    try:
        for iface, addrs in psutil.net_if_addrs().items():
            for addr in addrs:
                if addr.family == socket.AF_INET and addr.address != "127.0.0.1":
                    candidates.append((iface, addr.address))
    except Exception as e:
        logger.warning(f"Could not enumerate network interfaces: {e}")

    if prefer_thunderbolt:
        for iface, ip in candidates:
            if ip.startswith("169.254.") or "bridge" in iface.lower() or "thunderbolt" in iface.lower():
                if ip.startswith("169.254."):
                    logger.info(f"Found Thunderbolt Bridge IP {ip} on {iface}")
                    return ip

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        if _is_private_routable(ip):
            logger.info(f"Found network IP via default route: {ip}")
            return ip
    except OSError:
        pass

    for iface, ip in candidates:
        if _is_private_routable(ip):
            logger.info(f"Found network IP {ip} on {iface}")
            return ip

    logger.warning("Using localhost as fallback IP")
    return "127.0.0.1"


# --------------------------------------------------------------------------- device id

def load_or_create_device_id(override: Optional[str] = None) -> str:
    if override:
        return override
    try:
        if DEVICE_ID_FILE.exists():
            existing = DEVICE_ID_FILE.read_text().strip()
            if existing:
                return existing
    except OSError as e:
        logger.warning(f"Could not read {DEVICE_ID_FILE}: {e}")

    device_id = str(uuid.uuid4())
    try:
        DEVICE_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
        DEVICE_ID_FILE.write_text(device_id + "\n")
        logger.info(f"Persisted new device id to {DEVICE_ID_FILE}")
    except OSError as e:
        logger.warning(f"Could not persist device id to {DEVICE_ID_FILE}: {e}")
    return device_id


# --------------------------------------------------------------------------- agent

class DeviceAgent:
    def __init__(
        self,
        backend_url: str = "http://localhost:8000",
        port: int = 8001,
        device_id: Optional[str] = None,
        ip_override: Optional[str] = None,
        prefer_thunderbolt: bool = False,
    ):
        self.device_id = load_or_create_device_id(device_id)
        self.backend_url = backend_url.rstrip("/")
        self.port = port
        self.ip_address = get_network_ip(prefer_thunderbolt=prefer_thunderbolt, override=ip_override)

        # Warm-up so subsequent cpu_percent(interval=None) calls return real values
        psutil.cpu_percent(interval=None)

        self.device_info = self._create_device_info()
        self.ollama_engine = OllamaInferenceEngine()

        self.llama_loader = None
        if os.environ.get("ORCHARD_USE_TORCH") == "1":
            from llama_sharded_inference import LlamaShardedLoader
            self.llama_loader = LlamaShardedLoader()
            logger.info("Torch-based LlamaShardedLoader enabled (ORCHARD_USE_TORCH=1)")

        self.inference_count = 0
        self.total_response_time_ms = 0.0
        self.registered = False

        self._client: Optional[httpx.AsyncClient] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

        self.app = FastAPI(title=f"Device Agent - {self.device_info.name}")
        self._setup_routes()

    # ------------------------------------------------------------------ device info

    def _create_device_info(self) -> DeviceInfo:
        system = platform.system().lower()
        if system == "darwin":
            plat = platform.platform().lower()
            if "iphone" in plat:
                device_type = DeviceType.IPHONE
            elif "ipad" in plat:
                device_type = DeviceType.IPAD
            else:
                device_type = DeviceType.MAC
        else:
            device_type = DeviceType.MAC

        memory = psutil.virtual_memory()
        return DeviceInfo(
            id=self.device_id,
            name=os.environ.get("ORCHARD_AGENT_NAME") or platform.node(),
            type=device_type,
            status=DeviceStatus.ONLINE,
            ip_address=self.ip_address,
            port=self.port,
            total_memory_gb=round(memory.total / (1024 ** 3), 1),
            available_memory_gb=round(memory.available / (1024 ** 3), 1),
            cpu_usage_percent=psutil.cpu_percent(interval=None),
            temperature_celsius=self._get_temperature(),
            last_heartbeat=datetime.now(),
        )

    def _get_temperature(self) -> Optional[float]:
        """Best-effort temperature from psutil sensors; None when unavailable (e.g. macOS)."""
        sensors = getattr(psutil, "sensors_temperatures", None)
        if sensors is None:
            return None
        try:
            readings = sensors()
        except Exception:
            return None
        for entries in readings.values():
            for entry in entries:
                if entry.current:
                    return float(entry.current)
        return None

    def _record_inference(self, elapsed_ms: float) -> None:
        self.inference_count += 1
        self.total_response_time_ms += elapsed_ms

    def _get_current_metrics(self) -> DeviceHealthMetrics:
        memory = psutil.virtual_memory()
        gb = 1024 ** 3
        avg = self.total_response_time_ms / self.inference_count if self.inference_count else 0.0
        return DeviceHealthMetrics(
            device_id=self.device_id,
            memory_usage_gb=round(memory.used / gb, 2),
            available_memory_gb=round(memory.available / gb, 2),
            total_memory_gb=round(memory.total / gb, 2),
            cpu_usage_percent=psutil.cpu_percent(interval=None),
            temperature_celsius=self._get_temperature(),
            inference_count=self.inference_count,
            average_response_time_ms=round(avg, 2),
            timestamp=datetime.now(),
        )

    # ------------------------------------------------------------------ routes

    def _setup_routes(self):
        protected = [Depends(require_token)]

        @self.app.get("/health")
        async def health():
            return {"status": "healthy", "device_id": self.device_id}

        @self.app.get("/metrics", dependencies=protected)
        async def get_metrics():
            return self._get_current_metrics().model_dump(mode="json")

        @self.app.post("/deploy", dependencies=protected)
        async def deploy_model(request: dict):
            model_id = request.get("model_id")
            if not model_id:
                raise HTTPException(status_code=400, detail="model_id required")
            try:
                await self.ollama_engine.load_model(model_id)
            except OllamaUnavailableError as e:
                logger.error(f"Model deployment error: {e}")
                raise HTTPException(status_code=502, detail=str(e))
            except ValueError as e:
                logger.error(f"Model deployment error: {e}")
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.exception("Model deployment error")
                raise HTTPException(status_code=500, detail=str(e))

            self.device_info.current_model = model_id
            self.device_info.status = DeviceStatus.ONLINE
            return {"status": "deployed", "model_id": model_id}

        @self.app.post("/inference", dependencies=protected)
        async def inference(request: InferenceRequest):
            loaded = self.ollama_engine.loaded_model
            if loaded is None:
                raise HTTPException(status_code=400, detail="No model deployed on this device")
            if request.model_id != loaded:
                raise HTTPException(
                    status_code=400,
                    detail=f"Requested model {request.model_id} not deployed (have {loaded})",
                )

            start_time = datetime.now()
            try:
                output = await self.ollama_engine.generate(
                    request.message,
                    model_id=request.model_id,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                )
            except HTTPException:
                raise
            except OllamaUnavailableError as e:
                logger.error(f"Inference error: {e}")
                raise HTTPException(status_code=502, detail=str(e))
            except TimeoutError as e:
                logger.error(f"Inference error: {e}")
                raise HTTPException(status_code=504, detail=str(e))
            except Exception as e:
                logger.exception("Inference error")
                raise HTTPException(status_code=500, detail=str(e))

            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._record_inference(processing_time)

            return {
                "response": output,
                "output": output,
                "processing_time_ms": int(processing_time),
                "device_id": self.device_id,
            }

        # ---------------------------------------------------------- layer-split sharding (torch)

        def _require_torch():
            if self.llama_loader is None:
                raise HTTPException(
                    status_code=400,
                    detail="Sharded inference requires this agent to run with ORCHARD_USE_TORCH=1 "
                    "and torch/transformers installed",
                )
            return self.llama_loader

        @self.app.post("/llama/shard/deploy", dependencies=protected)
        async def deploy_llama_shard(request: dict):
            loader = _require_torch()
            shard_data = request.get("shard")
            if not shard_data:
                raise HTTPException(status_code=400, detail="shard data required")
            try:
                shard = ModelShard(**shard_data)
                await loader.load_llama_shard(shard)
            except HTTPException:
                raise
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.exception("Llama shard deployment error")
                raise HTTPException(status_code=500, detail=f"Failed to load shard: {e}")
            self.device_info.current_model = shard.model_path
            return {"status": "success", "shard_id": shard.shard_id, **loader.loaded_shards[shard.shard_id]}

        @self.app.post("/llama/shard/unload", dependencies=protected)
        async def unload_llama_shard(request: dict):
            loader = _require_torch()
            await loader.unload_shard(request.get("shard_id"))
            self.device_info.current_model = None
            return {"status": "unloaded"}

        @self.app.post("/llama/session/end", dependencies=protected)
        async def llama_session_end(request: dict):
            loader = _require_torch()
            released = await loader.end_session(request.get("session_id"))
            return {"status": "ended" if released else "unknown"}

        @self.app.post("/llama/tokenize", dependencies=protected)
        async def llama_tokenize(request: dict):
            loader = _require_torch()
            text = request.get("text")
            if text is None:
                raise HTTPException(status_code=400, detail="text required")
            try:
                return {"input_ids": loader.tokenize(text)}
            except ShardNotLoadedError as e:
                raise HTTPException(status_code=409, detail=str(e))

        @self.app.post("/llama/detokenize", dependencies=protected)
        async def llama_detokenize(request: dict):
            loader = _require_torch()
            ids = request.get("input_ids")
            if not isinstance(ids, list):
                raise HTTPException(status_code=400, detail="input_ids list required")
            try:
                return {"text": loader.detokenize(ids)}
            except ShardNotLoadedError as e:
                raise HTTPException(status_code=409, detail=str(e))

        @self.app.post("/llama/forward", dependencies=protected)
        async def llama_forward(request: dict):
            loader = _require_torch()
            started = time.time()
            try:
                result = await loader.forward(
                    input_ids=request.get("input_ids"),
                    hidden_b64=request.get("hidden"),
                    shape=request.get("shape"),
                    dtype=request.get("dtype", "float16"),
                    temperature=request.get("temperature", 0.0) or 0.0,
                    session_id=request.get("session_id"),
                    reset=bool(request.get("reset", False)),
                )
            except ShardNotLoadedError as e:
                raise HTTPException(status_code=409, detail=str(e))
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.exception("Shard forward error")
                raise HTTPException(status_code=500, detail=f"Shard forward failed: {e}")
            self._record_inference((time.time() - started) * 1000)
            result["shard_id"] = request.get("shard_id")
            return result

    # ------------------------------------------------------------------ backend comms

    async def register_with_backend(self) -> bool:
        try:
            self.device_info.last_heartbeat = datetime.now()
            response = await self._client.post(
                f"{self.backend_url}/api/devices/register",
                json=self.device_info.model_dump(mode="json"),
                headers=backend_headers(),
            )
            if response.status_code == 200:
                logger.info("Successfully registered with backend")
                self.registered = True
                return True
            logger.error(f"Failed to register with backend: {response.status_code} {response.text[:200]}")
        except httpx.HTTPError as e:
            logger.error(f"Error registering with backend: {e}")
        self.registered = False
        return False

    async def send_heartbeat(self) -> bool:
        """Send a heartbeat. Returns True on success; re-registers on 404."""
        metrics = self._get_current_metrics()
        try:
            response = await self._client.post(
                f"{self.backend_url}/api/devices/{self.device_id}/heartbeat",
                json=metrics.model_dump(mode="json"),
                headers=backend_headers(),
            )
        except httpx.HTTPError as e:
            logger.error(f"Error sending heartbeat: {e}")
            return False

        if response.status_code == 200:
            return True
        if response.status_code == 404:
            logger.warning("Backend does not know this device; re-registering")
            self.registered = False
            return await self.register_with_backend()
        logger.warning(f"Heartbeat failed: {response.status_code} {response.text[:200]}")
        return False

    async def _heartbeat_loop(self):
        backoff = BACKOFF_MIN_S
        while True:
            if not self.registered:
                ok = await self.register_with_backend()
            else:
                ok = await self.send_heartbeat()

            if ok:
                backoff = BACKOFF_MIN_S
                delay = HEARTBEAT_INTERVAL_S
            else:
                delay = backoff
                backoff = min(backoff * 2, BACKOFF_MAX_S)
                logger.info(f"Retrying backend in {delay}s")
            await asyncio.sleep(delay)

    async def _deregister(self):
        if not self.registered or self._client is None:
            return
        try:
            await self._client.delete(
                f"{self.backend_url}/api/devices/{self.device_id}", headers=backend_headers()
            )
            logger.info("Deregistered from backend")
        except httpx.HTTPError as e:
            logger.warning(f"Could not deregister from backend: {e}")

    # ------------------------------------------------------------------ lifecycle

    async def start(self):
        if ORCHARD_TOKEN is None:
            logger.warning("ORCHARD_TOKEN not set: agent routes are unauthenticated")
        logger.info(
            f"Device {self.device_info.name} ({self.device_id}) at "
            f"{self.ip_address}:{self.port}, backend {self.backend_url}"
        )

        self._client = httpx.AsyncClient(timeout=10)
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        config = uvicorn.Config(self.app, host="0.0.0.0", port=self.port, log_level="info")
        server = uvicorn.Server(config)
        try:
            await server.serve()
        finally:
            if self._heartbeat_task is not None:
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except (asyncio.CancelledError, Exception):
                    pass
            await self._deregister()
            await self._client.aclose()
            await self.ollama_engine.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Orchard device agent")
    parser.add_argument("--backend", "--backend-url", dest="backend",
                        default=os.environ.get("ORCHARD_BACKEND_URL", "http://localhost:8000"),
                        help="Backend URL")
    parser.add_argument("--port", type=int, default=int(os.environ.get("ORCHARD_AGENT_PORT", "8001")),
                        help="Port to run on")
    parser.add_argument("--ip", default=None, help="Advertised IP address (overrides detection / ORCHARD_AGENT_IP)")
    parser.add_argument("--prefer-thunderbolt", action="store_true",
                        help="Prefer a 169.254.x.x Thunderbolt Bridge address when advertising")
    parser.add_argument("--device-id", default=None, help="Override the persisted device id")
    parser.add_argument("--name", default=None, help="Display name for this device (default: hostname, or ORCHARD_AGENT_NAME)")
    return parser.parse_args(argv)


async def main():
    args = parse_args()
    if args.name:
        os.environ["ORCHARD_AGENT_NAME"] = args.name
    agent = DeviceAgent(
        backend_url=args.backend,
        port=args.port,
        device_id=args.device_id,
        ip_override=args.ip,
        prefer_thunderbolt=args.prefer_thunderbolt,
    )
    await agent.start()


if __name__ == "__main__":
    asyncio.run(main())
