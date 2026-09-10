"""
Real layer-split pipeline shard for Llama-family models.

Each device holds a contiguous slice of decoder layers. The first shard also owns the
token embeddings; the last shard owns the final norm and the LM head. Hidden states are
passed between devices as base64-encoded float16 tensors. The backend drives the
per-token generation loop.

torch / transformers are optional (ORCHARD_USE_TORCH=1) and imported lazily so the agent
can start without them.
"""
import asyncio
import base64
import gc
from typing import Any, Dict, List, Optional

from loguru import logger

from shared.types import ModelShard


def _import_torch():
    import torch  # noqa: WPS433 (lazy optional import)
    return torch


def _import_transformers():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    return AutoTokenizer, AutoModelForCausalLM


class ShardNotLoadedError(RuntimeError):
    pass


class LlamaShardedLoader:
    """Holds one contiguous layer slice of a Llama model and runs forward passes over it."""

    def __init__(self):
        self.shard: Optional[ModelShard] = None
        self.tokenizer = None
        self.embed_tokens = None
        self.layers = None
        self.rotary_emb = None
        self.norm = None
        self.lm_head = None
        self.total_layers: int = 0
        self.eos_token_ids: List[int] = []
        self._device = None
        self._dtype = None
        self._lock = asyncio.Lock()  # one forward at a time per shard

    # ------------------------------------------------------------------ device selection

    @property
    def device(self):
        if self._device is None:
            torch = _import_torch()
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")
        return self._device

    @property
    def dtype(self):
        if self._dtype is None:
            torch = _import_torch()
            # float16 on accelerators, float32 on CPU (fp16 matmuls are slow/unsupported on CPU)
            self._dtype = torch.float32 if self.device.type == "cpu" else torch.float16
        return self._dtype

    @property
    def is_first(self) -> bool:
        return self.shard is not None and self.shard.layer_start == 0

    @property
    def is_last(self) -> bool:
        return self.shard is not None and self.shard.layer_end == self.total_layers - 1

    @property
    def loaded_shards(self) -> Dict[str, Any]:
        if self.shard is None:
            return {}
        return {
            self.shard.shard_id: {
                "layer_start": self.shard.layer_start,
                "layer_end": self.shard.layer_end,
                "is_first": self.is_first,
                "is_last": self.is_last,
                "model": self.shard.model_path,
                "device": str(self.device),
            }
        }

    # ------------------------------------------------------------------ loading

    async def load_llama_shard(self, shard: ModelShard) -> bool:
        if shard.shard_type != "layers":
            raise ValueError(f"Only 'layers' shards are supported, got '{shard.shard_type}'")
        cfg = shard.llama_config or {}
        total_layers = cfg.get("total_layers")
        if total_layers is None:
            raise ValueError("shard.llama_config.total_layers is required")
        if shard.layer_start < 0 or shard.layer_end < shard.layer_start or shard.layer_end >= total_layers:
            raise ValueError(
                f"Invalid layer range {shard.layer_start}-{shard.layer_end} for {total_layers} layers"
            )
        async with self._lock:
            await self.unload_shard(shard.shard_id)
            await asyncio.to_thread(self._load_sync, shard, int(total_layers))
        logger.info(
            f"Loaded shard {shard.shard_id}: layers {shard.layer_start}-{shard.layer_end} "
            f"of {total_layers} from {shard.model_path} on {self.device} "
            f"(first={self.is_first}, last={self.is_last})"
        )
        return True

    def _load_sync(self, shard: ModelShard, total_layers: int) -> None:
        torch = _import_torch()
        AutoTokenizer, AutoModelForCausalLM = _import_transformers()

        model_name = shard.model_path
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Load the full checkpoint on CPU, then keep only what this shard needs.
        # (Selective weight loading would save peak RAM; full load keeps this simple and correct.)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=self.dtype, attn_implementation="sdpa", low_cpu_mem_usage=True
        ).eval()

        if len(model.model.layers) != total_layers:
            raise ValueError(
                f"Backend thinks {model_name} has {total_layers} layers but checkpoint has "
                f"{len(model.model.layers)}"
            )

        keep = list(model.model.layers[shard.layer_start : shard.layer_end + 1])
        self.layers = torch.nn.ModuleList(keep).to(self.device)
        self.rotary_emb = model.model.rotary_emb.to(self.device)
        self.embed_tokens = model.model.embed_tokens.to(self.device) if shard.layer_start == 0 else None
        if shard.layer_end == total_layers - 1:
            self.norm = model.model.norm.to(self.device)
            self.lm_head = model.lm_head.to(self.device)
        else:
            self.norm = None
            self.lm_head = None

        eos = model.generation_config.eos_token_id
        if eos is None:
            eos = tokenizer.eos_token_id
        self.eos_token_ids = list(eos) if isinstance(eos, (list, tuple)) else [eos]

        self.tokenizer = tokenizer
        self.shard = shard
        self.total_layers = total_layers

        # Drop the layers we don't own.
        del model
        gc.collect()

    async def unload_shard(self, shard_id: Optional[str] = None) -> None:
        if self.shard is None:
            return
        if shard_id is not None and shard_id != self.shard.shard_id:
            return
        logger.info(f"Unloading shard {self.shard.shard_id}")
        self.shard = None
        self.layers = self.embed_tokens = self.norm = self.lm_head = self.rotary_emb = None
        self.tokenizer = None
        gc.collect()
        try:
            torch = _import_torch()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif self.device.type == "mps":
                torch.mps.empty_cache()
        except Exception:  # pragma: no cover - best effort
            pass

    # ------------------------------------------------------------------ tokenizer

    def _require_loaded(self):
        if self.shard is None or self.layers is None:
            raise ShardNotLoadedError("No shard loaded on this device")

    def tokenize(self, text: str) -> List[int]:
        self._require_loaded()
        return self.tokenizer(text, return_tensors=None)["input_ids"]

    def detokenize(self, input_ids: List[int]) -> str:
        self._require_loaded()
        return self.tokenizer.decode(input_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------ forward

    async def forward(
        self,
        input_ids: Optional[List[int]] = None,
        hidden_b64: Optional[str] = None,
        shape: Optional[List[int]] = None,
        dtype: str = "float16",
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Run this shard's layers.

        First shard: expects input_ids. Others: expect hidden/shape/dtype.
        Returns {"hidden","shape","dtype"} unless this is the last shard, which returns
        {"next_token","eos"}.
        """
        self._require_loaded()
        async with self._lock:
            return await asyncio.to_thread(
                self._forward_sync, input_ids, hidden_b64, shape, dtype, float(temperature)
            )

    def _forward_sync(self, input_ids, hidden_b64, shape, dtype, temperature) -> Dict[str, Any]:
        torch = _import_torch()
        with torch.no_grad():
            if self.is_first:
                if not input_ids:
                    raise ValueError("First shard requires input_ids")
                ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
                h = self.embed_tokens(ids)
            else:
                if hidden_b64 is None or shape is None:
                    raise ValueError("Non-first shard requires hidden and shape")
                h = _decode_tensor(hidden_b64, shape, dtype).to(self.device, self.dtype)

            seq_len = h.shape[1]
            position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)
            cos, sin = self.rotary_emb(h, position_ids)
            for layer in self.layers:
                out = layer(
                    h,
                    attention_mask=None,  # sdpa applies a causal mask when mask is None
                    position_ids=position_ids,
                    position_embeddings=(cos, sin),
                )
                h = out[0] if isinstance(out, tuple) else out

            if not self.is_last:
                return _encode_tensor(h)

            logits = self.lm_head(self.norm(h[:, -1, :])).float()
            if temperature and temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = int(torch.multinomial(probs, num_samples=1).item())
            else:
                next_token = int(torch.argmax(logits, dim=-1).item())
            return {"next_token": next_token, "eos": next_token in self.eos_token_ids}

    def get_model_info(self) -> Dict[str, Any]:
        return {"loaded_shards": self.loaded_shards, "device": str(self.device) if self.shard else None}


# ---------------------------------------------------------------------- wire format


def _encode_tensor(t) -> Dict[str, Any]:
    torch = _import_torch()
    arr = t.detach().to("cpu", torch.float16).contiguous().numpy()
    return {
        "hidden": base64.b64encode(arr.tobytes()).decode("ascii"),
        "shape": list(arr.shape),
        "dtype": "float16",
    }


def _decode_tensor(b64: str, shape: List[int], dtype: str):
    import numpy as np

    torch = _import_torch()
    if dtype != "float16":
        raise ValueError(f"Unsupported hidden dtype {dtype}")
    arr = np.frombuffer(base64.b64decode(b64), dtype=np.float16).reshape(shape)
    return torch.from_numpy(arr.copy())
