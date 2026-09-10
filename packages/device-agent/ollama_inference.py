import os
from typing import Dict, List, Optional, Any

import httpx
from loguru import logger


DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2:1b"

# Orchard catalog model id -> Ollama tag. Unknown ids are rejected rather than silently
# mapped to a default model.
MODEL_TAGS: Dict[str, str] = {
    "llama-3.2-1b": "llama3.2:1b",
    "llama2-7b": "llama2:7b",
    "mistral-7b": "mistral:7b",
    "phi-3-mini": "phi3:mini",
}


class OllamaUnavailableError(RuntimeError):
    """Raised when the Ollama server cannot be reached."""


class OllamaInferenceEngine:
    """Ollama-backed inference engine (HTTP API) for whole-model inference on one device."""

    def __init__(self, base_url: Optional[str] = None, timeout: float = 120.0):
        self.base_url = (base_url or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).rstrip("/")
        if not self.base_url.startswith("http"):
            self.base_url = f"http://{self.base_url}"
        self.default_model = os.environ.get("ORCHARD_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        self.loaded_model: Optional[str] = None   # model id as deployed by the backend
        self.ollama_models: List[str] = []
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def close(self):
        await self._client.aclose()

    # ------------------------------------------------------------------ models

    async def list_models(self) -> List[str]:
        """Return the model tags available in Ollama. Raises OllamaUnavailableError if unreachable."""
        try:
            response = await self._client.get("/api/tags", timeout=10.0)
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise OllamaUnavailableError(f"Cannot reach Ollama at {self.base_url}: {e}") from e
        models = [m.get("name") for m in response.json().get("models", []) if m.get("name")]
        self.ollama_models = models
        return models

    def resolve_model_tag(self, model_id: Optional[str]) -> str:
        """Map an Orchard model id to an Ollama tag. Raises ValueError for unknown ids."""
        if not model_id:
            return self.default_model
        if ":" in model_id:  # already an Ollama tag
            return model_id
        override = os.environ.get(f"ORCHARD_OLLAMA_TAG_{model_id.upper().replace('-', '_')}")
        if override:
            return override
        tag = MODEL_TAGS.get(model_id)
        if tag is None:
            raise ValueError(
                f"Unknown model id '{model_id}'. Known: {sorted(MODEL_TAGS)}. "
                f"Set ORCHARD_OLLAMA_TAG_{model_id.upper().replace('-', '_')} to map it."
            )
        return tag

    async def load_model(self, model_id: str) -> None:
        """Record a deployed model after verifying its Ollama tag exists."""
        tag = self.resolve_model_tag(model_id)
        available = await self.list_models()
        if tag not in available:
            raise ValueError(
                f"Model '{tag}' is not available in Ollama (have: {available or 'none'}). "
                f"Run `ollama pull {tag}` on this device."
            )
        self.loaded_model = model_id
        logger.info(f"Model {model_id} ready via Ollama tag {tag}")

    async def unload_model(self):
        if self.loaded_model:
            logger.info(f"Unloading model {self.loaded_model}")
            self.loaded_model = None

    async def generate(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate a completion for a deployed model."""
        tag = self.resolve_model_tag(model_id or self.loaded_model)
        return await self._run_ollama_inference(tag, prompt, max_tokens=max_tokens, temperature=temperature)

    # ------------------------------------------------------------------ internals

    async def _run_ollama_inference(
        self,
        model_name: str,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Call POST /api/generate (non-streaming). Raises on failure."""
        options: Dict[str, Any] = {}
        if max_tokens is not None:
            options["num_predict"] = int(max_tokens)
        if temperature is not None:
            options["temperature"] = float(temperature)

        payload: Dict[str, Any] = {"model": model_name, "prompt": prompt, "stream": False}
        if options:
            payload["options"] = options

        try:
            response = await self._client.post("/api/generate", json=payload)
        except httpx.TimeoutException as e:
            raise TimeoutError(f"Ollama inference timed out after {self._client.timeout}") from e
        except httpx.HTTPError as e:
            raise OllamaUnavailableError(f"Cannot reach Ollama at {self.base_url}: {e}") from e

        if response.status_code != 200:
            detail = response.text
            try:
                detail = response.json().get("error", detail)
            except ValueError:
                pass
            raise RuntimeError(f"Ollama inference failed ({response.status_code}): {detail}")

        return response.json().get("response", "").strip()

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            "loaded_model": self.loaded_model,
            "available_models": self.ollama_models,
        }
