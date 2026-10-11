# Whisper API — lazy-loading fork

Windows-first [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
transcription server with an OpenAI-compatible API, forked from
[dniasoff/whisper-api](https://github.com/dniasoff/whisper-api) (MIT).
Runs as a Windows service (`whisper-api`); idle VRAM/RAM stays free for
other apps (e.g. LM Studio on the same 6 GB card).

## What the fork changes vs upstream

- **Lazy model loading** — nothing loads at startup. The first
  transcription request loads the model; after `MODEL_IDLE_UNLOAD_SECONDS`
  (default 900) of idleness it unloads again (`gc` + `cuda.empty_cache()`).
- **Per-request `model`** from a curated allowlist, one resident at a time
  (switching unloads first). Unknown id → `400`, before any load.
- **Per-request `device`** (`cuda`/`cpu`, else `WHISPER_DEVICE`, else auto).
  CUDA load *and* inference failures fall back to CPU with a clear log —
  never a bare 500 (total failure → `503` with the cause). The fallback is
  sticky; explicit `device=cuda` retries GPU.
- **`GET /v1/models`** documents the allowlist (params/VRAM/speed/accuracy)
  plus what is currently loaded.
- **`GET /v1/health`** works unloaded and reports `model_loaded`,
  `loaded_model`, `loaded_device`, `idle_for_seconds`.
- No startup warmup transcription (it would defeat lazy loading).
- Pinned `av==18.0.0` — av 19 removed `metadata_errors` used by
  faster-whisper decode and breaks every transcription.

## Model table

| id | params | VRAM | speed | accuracy | notes |
|---|---|---|---|---|---|
| tiny | 39M | ~1 GB | ~10x | lowest | |
| base | 74M | ~1 GB | ~7x | low | |
| small | 244M | ~2 GB | ~4x | good | |
| medium | 769M | ~5 GB | ~2x | high | |
| turbo | 809M | ~6 GB | ~8x | high | large-v3-turbo, best ratio (EN+SK) |
| large | 1550M | ~10 GB | ~1x | highest | does not fit 6 GB cards |

## Run

```powershell
# service (existing install points at this file)
net stop whisper-api
# copy server.py over the installed one, then:
net start whisper-api
# `net start` may report 3523 — slow start (~90 s), the service still comes up
curl.exe http://127.0.0.1:4444/v1/health
```

Env (service `Environment` registry key or process env):

| var | default | meaning |
|---|---|---|
| WHISPER_MODEL | turbo | default model id (`model=whisper-1` or omitted = this) |
| WHISPER_DEVICE | (empty) | empty/auto = GPU auto-detect; `cpu` forces CPU; unknown value is ignored with a warning |
| MODEL_IDLE_UNLOAD_SECONDS | 900 | unload model from RAM/VRAM after N s without requests |
| CUDA_DEVICE_ID | 0 | GPU index |
| bind | 127.0.0.1:4444 (hardcoded) | the Windows service wrapper rebinds via its own WHISPER_HOST/WHISPER_PORT |

## API

- `POST /v1/audio/transcriptions` — multipart `file`, optional `model`,
  `language`, `device`, `response_format` (json/text). Answers
  `{"text": ..., "language": ...}`. Also `POST /transcribe` (legacy alias,
  accepts `model`/`device` too).
- `model`: one of the allowlist ids, or `whisper-1` (= server default).
  Anything else → `400` with the allowed list. A different model than the
  loaded one unloads the current model first.
- `device`: `cuda` or `cpu` (else `400`). Omitted → `WHISPER_DEVICE`, else
  auto-detected GPU. Missing CUDA 12 DLLs (e.g. `cublas64_12.dll`) fail the
  GPU path with a logged fix checklist and a transparent CPU retry.
- `GET /v1/models`, `GET /v1/health`, `GET /docs` (Swagger).

Example:

```powershell
# explicit model + device
curl.exe -X POST http://127.0.0.1:4444/v1/audio/transcriptions `
  -F "file=@recording.wav" -F "model=turbo" -F "device=cuda" -F "language=sk"

# CPU (GPU busy elsewhere), default model
curl.exe -X POST http://127.0.0.1:4444/v1/audio/transcriptions `
  -F "file=@recording.wav" -F "device=cpu"
```

## Tests

```powershell
python -m pytest tests/ -q   # 15 tests, torch/faster-whisper stubbed (no GPU needed)
```

Covers: unloaded health, lazy load on first request, model switch with
unload (`empty_cache`), `whisper-1` default mapping, legacy `model_name`
alias, 400 on unknown model/device (before any load), 415/400 file errors,
CUDA load + CUDA inference (`cublas`) fallback to CPU, no retry on genuine
transcription errors, `/v1/models` + health loaded-state fields, manual
unload cycle.

## License

MIT — same as upstream. See [LICENSE](LICENSE).
