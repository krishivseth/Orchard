"""
Real layer-split pipeline shard for Llama-family models.

Each device holds a contiguous slice of decoder layers. The first shard also owns the
token embeddings; the last shard owns the final norm and the LM head. Hidden states are
passed between devices as base64-encoded float16 tensors. The backend drives the
per-token generation loop and identifies each generation with a session_id; every shard
keeps a KV cache per session so each step only processes the newest token.

torch / transformers are optional (ORCHARD_USE_TORCH=1) and imported lazily so the agent
can start without them.
"""
import asyncio
import base64
import gc
import os
import time
from typing import Any, Dict, List, Optional

from loguru import logger

from shared.types import ModelShard


def _import_torch():
    import torch  # noqa: WPS433 (lazy optional import)
    return torch


def _import_transformers():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    return AutoTokenizer, AutoModelForCausalLM


def _new_cache():
    from transformers import DynamicCache
    return DynamicCache()


# Per-session KV caches: bounded in count and age so abandoned generations don't pin memory.
MAX_SESSIONS = int(os.environ.get("ORCHARD_KV_MAX_SESSIONS", "8"))
SESSION_TTL_SECONDS = float(os.environ.get("ORCHARD_KV_SESSION_TTL", "300"))


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
        self._sessions: Dict[str, Dict[str, Any]] = {}  # session_id -> {"cache", "last_used"}

    # ------------------------------------------------------------------ device selection

    @property
    def device(self):
        if self._device is None:
            torch = _import_torch()
            forced = os.environ.get("ORCHARD_TORCH_DEVICE")
            if forced:
                self._device = torch.device(forced)
            elif torch.cuda.is_available():
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
        # DynamicCache indexes by the attention module's layer_idx; make it local to this shard.
        for local_idx, layer in enumerate(keep):
            layer.self_attn.layer_idx = local_idx
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
        self._sessions.clear()
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
        session_id: Optional[str] = None,
        reset: bool = False,
    ) -> Dict[str, Any]:
        """Run this shard's layers over the NEW positions only.

        First shard: expects input_ids (the new tokens). Others: expect hidden/shape/dtype for
        the new positions. A session_id selects the KV cache; reset=True starts it fresh.
        Without a session_id the call is stateless (full sequence each time).
        Returns {"hidden","shape","dtype"} unless this is the last shard, which returns
        {"next_token","eos"}; both include "past_length".
        """
        self._require_loaded()
        async with self._lock:
            return await asyncio.to_thread(
                self._forward_sync, input_ids, hidden_b64, shape, dtype, float(temperature),
                session_id, reset,
            )

    async def end_session(self, session_id: Optional[str]) -> bool:
        async with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def _get_cache(self, session_id: Optional[str], reset: bool):
        if session_id is None:
            return _new_cache()
        now = time.time()
        # Evict stale sessions, then the oldest if we're over the cap.
        for sid in [sid for sid, sess in self._sessions.items() if now - sess["last_used"] > SESSION_TTL_SECONDS]:
            self._sessions.pop(sid, None)
        if reset or session_id not in self._sessions:
            while len(self._sessions) >= MAX_SESSIONS:
                oldest = min(self._sessions, key=lambda k: self._sessions[k]["last_used"])
                self._sessions.pop(oldest, None)
            self._sessions[session_id] = {"cache": _new_cache(), "last_used": now}
        sess = self._sessions[session_id]
        sess["last_used"] = now
        return sess["cache"]

    def _forward_sync(self, input_ids, hidden_b64, shape, dtype, temperature, session_id, reset) -> Dict[str, Any]:
        torch = _import_torch()
        cache = self._get_cache(session_id, reset)
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

            past_len = cache.get_seq_length()
            new_len = h.shape[1]
            position_ids = torch.arange(past_len, past_len + new_len, device=self.device).unsqueeze(0)
            cos, sin = self.rotary_emb(h, position_ids)
            for layer in self.layers:
                out = layer(
                    h,
                    # sdpa: causal mask for multi-token prefill, full attention over the
                    # cache for a single new token. Both are what we want.
                    attention_mask=None,
                    position_ids=position_ids,
                    past_key_value=cache,
                    use_cache=True,
                    cache_position=position_ids[0],
                    position_embeddings=(cos, sin),
                )
                h = out[0] if isinstance(out, tuple) else out

            if not self.is_last:
                result = _encode_tensor(h)
                result["past_length"] = past_len + new_len
                return result

            logits = self.lm_head(self.norm(h[:, -1, :])).float()
            if temperature and temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = int(torch.multinomial(probs, num_samples=1).item())
            else:
                next_token = int(torch.argmax(logits, dim=-1).item())
            return {
                "next_token": next_token,
                "eos": next_token in self.eos_token_ids,
                "past_length": past_len + new_len,
            }

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "loaded_shards": self.loaded_shards,
            "device": str(self.device) if self.shard else None,
            "active_sessions": len(self._sessions),
        }


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
