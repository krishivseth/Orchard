import os
from typing import Any, Dict, List, Optional, Tuple

import httpx
from loguru import logger

from shared.types import (
    DeviceInfo,
    DistributedInferenceRequest,
    ModelShard,
    ModelShardingConfig,
    ShardingStrategy,
)


# Hugging Face model the device agents load for layer-split sharding.
# This is the single source of truth for the model name (env ORCHARD_HF_MODEL).
HF_MODEL_NAME: str = os.getenv("ORCHARD_HF_MODEL") or "meta-llama/Llama-3.2-1B"

# Per-forward-call timeout: a single step over one shard (prefill can be slow on CPU).
FORWARD_TIMEOUT_SECONDS: float = 120.0

# Llama architecture configs keyed by model_id. Only models listed here can be sharded.
LLAMA_ARCHITECTURES: Dict[str, Dict[str, Any]] = {
    "llama-3.2-1b": {
        "num_hidden_layers": 16,
        "hidden_size": 2048,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 8192,
        "vocab_size": 128256,
        "max_position_embeddings": 131072,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "model_size_gb": 2.5,
    },
}

UNSUPPORTED_STRATEGY_MESSAGE = "Only layer_split sharding is implemented"


class ShardingError(Exception):
    """Raised when a sharding configuration cannot be built (caller error)."""


class ShardInferenceError(Exception):
    """Raised when a shard fails during inference or has no reachable device."""

    def __init__(self, message: str, shard_id: str, device_id: str):
        super().__init__(message)
        self.shard_id = shard_id
        self.device_id = device_id


class LlamaShardingEngine:
    """Layer-split pipeline over device agents.

    Each device holds a contiguous slice of transformer layers. The backend owns the
    token generation loop: it tokenizes on the first shard's device, forwards hidden
    states shard-to-shard, and the last shard samples the next token.
    """

    def __init__(self, http_client: Optional[httpx.AsyncClient] = None):
        self.sharding_configs: Dict[str, ModelShardingConfig] = {}
        self.device_connections: Dict[str, str] = {}  # device_id -> ip:port
        self.http_client: Optional[httpx.AsyncClient] = http_client
        self.auth_headers: Dict[str, str] = {}
        self.hf_model: str = HF_MODEL_NAME

    # ------------------------------------------------------------------ #
    # Configuration
    # ------------------------------------------------------------------ #
    def get_architecture(self, model_id: str) -> Dict[str, Any]:
        arch = LLAMA_ARCHITECTURES.get(model_id)
        if arch is None:
            raise ShardingError(f"No sharding architecture configured for model '{model_id}'")
        return arch

    def remove_device(self, device_id: str) -> List[str]:
        """Forget a device: drop its connection and any config that references it.

        Returns the model_ids whose sharding configs were removed.
        """
        self.device_connections.pop(device_id, None)
        removed = [
            model_id
            for model_id, config in self.sharding_configs.items()
            if device_id in config.devices_used
        ]
        for model_id in removed:
            del self.sharding_configs[model_id]
            logger.info(f"Removed sharding config for {model_id}: device {device_id} is gone")
        return removed

    def _validate_devices(self, arch: Dict[str, Any], devices: List[DeviceInfo]) -> None:
        max_devices = arch["num_hidden_layers"]
        if len(devices) < 2:
            raise ShardingError("Sharded deployment requires at least 2 devices")
        if len(devices) > max_devices:
            raise ShardingError(
                f"Sharded deployment supports at most {max_devices} devices for this model "
                f"(got {len(devices)})"
            )

    async def create_llama_sharding_config(
        self,
        model_id: str,
        devices: List[DeviceInfo],
        strategy: ShardingStrategy,
    ) -> ModelShardingConfig:
        """Create a sharding configuration for the given model and devices."""
        if strategy != ShardingStrategy.LAYER_SPLIT:
            raise ShardingError(UNSUPPORTED_STRATEGY_MESSAGE)
        arch = self.get_architecture(model_id)
        self._validate_devices(arch, devices)
        return self._create_layer_split_config(model_id, arch, devices)

    @staticmethod
    def _distribute_layers(total_layers: int, num_devices: int) -> List[Tuple[int, int]]:
        """Split [0, total_layers) into contiguous inclusive ranges, spreading the remainder over the first devices."""
        base, remainder = divmod(total_layers, num_devices)
        ranges = []
        current = 0
        for i in range(num_devices):
            count = base + (1 if i < remainder else 0)
            ranges.append((current, current + count - 1))
            current += count
        return ranges

    def _create_layer_split_config(
        self, model_id: str, arch: Dict[str, Any], devices: List[DeviceInfo]
    ) -> ModelShardingConfig:
        total_layers = arch["num_hidden_layers"]
        shards = []
        for i, (device, (start, end)) in enumerate(zip(devices, self._distribute_layers(total_layers, len(devices)))):
            device_layers = end - start + 1
            shards.append(ModelShard(
                shard_id=f"{model_id}-layer-shard-{i}",
                device_id=device.id,
                layer_start=start,
                layer_end=end,
                model_path=self.hf_model,
                shard_type="layers",
                memory_usage_gb=(device_layers / total_layers) * arch["model_size_gb"],
                llama_config={
                    "total_layers": total_layers,
                    "hf_model": self.hf_model,
                    "layer_start": start,
                    "layer_end": end,
                    "num_hidden_layers": device_layers,
                    "hidden_size": arch["hidden_size"],
                    "num_attention_heads": arch["num_attention_heads"],
                    "num_key_value_heads": arch["num_key_value_heads"],
                    "intermediate_size": arch["intermediate_size"],
                    "vocab_size": arch["vocab_size"],
                },
            ))

        return ModelShardingConfig(
            model_id=model_id,
            strategy=ShardingStrategy.LAYER_SPLIT,
            shards=shards,
            total_layers=total_layers,
            devices_used=[d.id for d in devices],
            model_name=self.hf_model,
        )

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    async def execute_llama_sharded_inference(
        self,
        request: DistributedInferenceRequest,
        config: ModelShardingConfig,
    ) -> str:
        """Run the token generation loop across the shards.

        Raises ShardInferenceError on any shard failure, ShardingError for unsupported strategies.
        """
        if config.strategy != ShardingStrategy.LAYER_SPLIT:
            raise ShardingError(UNSUPPORTED_STRATEGY_MESSAGE)
        if not config.shards:
            raise ShardingError("Sharding config has no shards")

        shards = sorted(config.shards, key=lambda s: s.layer_start)
        first, last = shards[0], shards[-1]
        first_address = self._device_address(first)

        tokenized = await self._call_shard(first_address, "/llama/tokenize", {"text": request.message}, first)
        input_ids = self._expect_int_list(tokenized, "input_ids", first, first_address)
        if not input_ids:
            raise ShardInferenceError(
                f"Tokenizer on device {first.device_id} returned no tokens",
                shard_id=first.shard_id,
                device_id=first.device_id,
            )

        generated: List[int] = []
        max_tokens = max(0, int(request.max_tokens))
        for step in range(max_tokens):
            next_token, eos = await self._forward_pipeline(shards, input_ids + generated, request.temperature)
            generated.append(next_token)
            if eos:
                logger.debug(f"EOS reached after {step + 1} tokens for {config.model_id}")
                break

        if not generated:
            return ""

        detokenized = await self._call_shard(first_address, "/llama/detokenize", {"input_ids": generated}, first)
        text = detokenized.get("text") if isinstance(detokenized, dict) else None
        if not isinstance(text, str):
            raise ShardInferenceError(
                f"Detokenizer on device {first.device_id} returned a malformed response",
                shard_id=first.shard_id,
                device_id=first.device_id,
            )
        return text

    async def _forward_pipeline(
        self, shards: List[ModelShard], input_ids: List[int], temperature: float
    ) -> Tuple[int, bool]:
        """One generation step: chain hidden states through every shard, return (next_token, eos)."""
        last = shards[-1]
        hidden: Optional[Dict[str, Any]] = None
        for shard in shards:
            address = self._device_address(shard)
            payload: Dict[str, Any] = {"shard_id": shard.shard_id, "temperature": temperature}
            if hidden is None:
                payload["input_ids"] = input_ids
            else:
                payload.update(hidden)
            out = await self._call_shard(address, "/llama/forward", payload, shard, timeout=FORWARD_TIMEOUT_SECONDS)

            if shard is last:
                next_token = out.get("next_token")
                if not isinstance(next_token, int) or isinstance(next_token, bool):
                    raise ShardInferenceError(
                        f"Last shard {shard.shard_id} on device {shard.device_id} returned no next_token",
                        shard_id=shard.shard_id,
                        device_id=shard.device_id,
                    )
                return next_token, bool(out.get("eos", False))

            # Intermediate shard: forward the opaque hidden tensor to the next shard.
            if not all(k in out for k in ("hidden", "shape", "dtype")):
                raise ShardInferenceError(
                    f"Shard {shard.shard_id} on device {shard.device_id} returned no hidden state",
                    shard_id=shard.shard_id,
                    device_id=shard.device_id,
                )
            hidden = {"hidden": out["hidden"], "shape": out["shape"], "dtype": out["dtype"]}

        # Unreachable: the loop always returns on the last shard.
        raise ShardInferenceError("Pipeline ended without a last shard", shard_id=last.shard_id, device_id=last.device_id)

    @staticmethod
    def _expect_int_list(body: Any, key: str, shard: ModelShard, address: str) -> List[int]:
        value = body.get(key) if isinstance(body, dict) else None
        if not isinstance(value, list) or not all(isinstance(v, int) and not isinstance(v, bool) for v in value):
            logger.error(f"Shard {shard.shard_id} on {address}: '{key}' missing or not a list of ints")
            raise ShardInferenceError(
                f"Device {shard.device_id} returned a malformed '{key}' for shard {shard.shard_id}",
                shard_id=shard.shard_id,
                device_id=shard.device_id,
            )
        return value

    def _device_address(self, shard: ModelShard) -> str:
        address = self.device_connections.get(shard.device_id)
        if not address:
            raise ShardInferenceError(
                f"No connection for device {shard.device_id} (shard {shard.shard_id})",
                shard_id=shard.shard_id,
                device_id=shard.device_id,
            )
        return address

    async def _call_shard(
        self,
        address: str,
        path: str,
        payload: dict,
        shard: ModelShard,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """POST to a device agent and return the parsed JSON body (must be an object)."""
        if self.http_client is None:
            raise ShardInferenceError(
                "HTTP client not initialised", shard_id=shard.shard_id, device_id=shard.device_id
            )
        kwargs: Dict[str, Any] = {"json": payload, "headers": self.auth_headers}
        if timeout is not None:
            kwargs["timeout"] = timeout
        try:
            response = await self.http_client.post(f"http://{address}{path}", **kwargs)
        except httpx.HTTPError as e:
            logger.error(f"Shard {shard.shard_id} on {address}{path}: request failed: {e!r}")
            raise ShardInferenceError(
                f"Device {shard.device_id} unreachable for shard {shard.shard_id}",
                shard_id=shard.shard_id,
                device_id=shard.device_id,
            ) from e

        if response.status_code != 200:
            logger.error(f"Shard {shard.shard_id} on {address}{path}: HTTP {response.status_code}")
            raise ShardInferenceError(
                f"Shard {shard.shard_id} on device {shard.device_id} returned HTTP {response.status_code}",
                shard_id=shard.shard_id,
                device_id=shard.device_id,
            )
        try:
            body = response.json()
        except ValueError as e:
            logger.error(f"Shard {shard.shard_id} on {address}{path}: invalid JSON: {e!r}")
            raise ShardInferenceError(
                f"Shard {shard.shard_id} on device {shard.device_id} returned a malformed response",
                shard_id=shard.shard_id,
                device_id=shard.device_id,
            ) from e
        if not isinstance(body, dict):
            raise ShardInferenceError(
                f"Shard {shard.shard_id} on device {shard.device_id} returned a malformed response",
                shard_id=shard.shard_id,
                device_id=shard.device_id,
            )
        return body
