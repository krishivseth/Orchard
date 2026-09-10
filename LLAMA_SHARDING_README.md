# Layer-Split Sharding

Orchard can split a Llama-family model across several devices so that no single device
has to hold the whole model. This document describes exactly what is implemented.

## What it does

- The backend divides the model's decoder layers into contiguous ranges, one per device
  (16 layers over 2 devices gives layers 0–7 and 8–15).
- Each device agent loads the checkpoint, keeps only its layer range, and frees the rest.
  The first shard also keeps the token embeddings; the last keeps the final norm and LM head.
- The backend runs the token loop. For every generated token the prompt-so-far goes to the
  first shard, hidden states flow shard to shard as base64 float16 tensors over HTTP, and the
  last shard returns the next token. Greedy decoding when `temperature` is 0, sampling otherwise.
- Each shard keeps a KV cache per generation session, so after the prompt is prefilled every
  step only processes the newest token. Sessions are released when generation ends and are
  bounded by `ORCHARD_KV_MAX_SESSIONS` (default 8) and `ORCHARD_KV_SESSION_TTL` (default
  300 s) on each agent.
- Output is verified identical to running the unsharded model with `generate()`.

## What it does not do

- Only one generation runs at a time per shard; concurrent chats queue.
- Only `layer_split` is implemented. `tensor_parallel` and `pipeline_parallel` are rejected
  with HTTP 400.
- Only `llama-3.2-1b` has a sharding architecture entry (`LLAMA_ARCHITECTURES` in
  `packages/backend/llama_sharding.py`). Add an entry to shard another Llama-family model.

## Requirements

- Device agents must run with `ORCHARD_USE_TORCH=1` and have `torch` and `transformers`
  installed in their environment. Without this, shard deploys return HTTP 400.
- Each agent needs access to the Hugging Face checkpoint named by `ORCHARD_HF_MODEL`
  (default `meta-llama/Llama-3.2-1B`, a gated model: run `huggingface-cli login` after
  accepting the license, or point the variable at an ungated mirror).
- Single-device chat does not use torch at all; it goes through Ollama.

## Usage

```bash
# On each device
ORCHARD_USE_TORCH=1 ORCHARD_TOKEN=<secret> python agent.py --backend-url http://<backend>:8000

# Deploy across all online devices
curl -X POST http://localhost:8000/api/models/deploy-llama-sharded-auto \
  -H 'Content-Type: application/json' -d '{"model_id":"llama-3.2-1b"}'

# Or pick devices explicitly
curl -X POST http://localhost:8000/api/models/deploy-llama-sharded \
  -H 'Content-Type: application/json' \
  -d '{"model_id":"llama-3.2-1b","device_ids":["dev-a","dev-b"]}'

# Chat
curl -X POST http://localhost:8000/api/chat/llama-sharded \
  -H 'Content-Type: application/json' \
  -d '{"message":"The capital of France is","model_id":"llama-3.2-1b","max_tokens":32,"temperature":0}'
```

The frontend's Model Management page exposes the same flow with a Shard button.

## Agent protocol

All routes require `X-Orchard-Token` when `ORCHARD_TOKEN` is set.

| Route | Body | Response |
|-------|------|----------|
| `POST /llama/shard/deploy` | `{"shard": ModelShard}` with `llama_config.total_layers` | `{"status":"success", ...}` |
| `POST /llama/shard/unload` | `{"shard_id"}` | `{"status":"unloaded"}` |
| `POST /llama/tokenize` | `{"text"}` | `{"input_ids":[...]}` |
| `POST /llama/detokenize` | `{"input_ids":[...]}` | `{"text"}` |
| `POST /llama/forward` | first shard: `{"input_ids":[new tokens]}`; others: `{"hidden","shape","dtype"}` for the new positions; always `temperature`, `session_id`, `reset` (true on the prefill step) | non-last: `{"hidden","shape","dtype","past_length"}`; last: `{"next_token","eos","past_length"}` |
| `POST /llama/session/end` | `{"session_id"}` | `{"status":"ended"|"unknown"}` |

## Failure behaviour

- A shard deploy that fails on any device returns HTTP 502 with the failing devices, and the
  config is not saved.
- A forward that fails or a device that goes away mid-generation returns HTTP 502 naming the
  shard and device. Nothing is ever echoed back as a fake answer.

## Files

- `packages/backend/llama_sharding.py`: config creation and the generation loop.
- `packages/device-agent/llama_sharded_inference.py`: layer slicing and forward pass.
- `packages/device-agent/agent.py`: the `/llama/*` routes.
