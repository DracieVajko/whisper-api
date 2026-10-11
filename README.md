# Whisper API — lazy-loading fork

Windows-first [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
transcription server with an OpenAI-compatible API, forked from
[dniasoff/whisper-api](https://github.com/dniasoff/whisper-api) (MIT).

## What the fork changes vs upstream

- **Lazy model loading** — nothing loads at startup. The first
  transcription request loads the model; after `MODEL_IDLE_UNLOAD_SECONDS`
  (default 900) of idleness it unloads again (`gc` + `cuda.empty_cache()`).
- **Per-request `model`** from a curated allowlist, one resident at a time
  (switching unloads first). Unknown id → `400`, before any load.
- **Per-request `device`** (`cuda`/`cpu`, else `WHISPER_DEVICE`, else auto).
  CUDA load *and* inference failures fall back to CPU with a clear log —
  never a bare 500. The fallback is sticky; explicit `device=cuda` retries GPU.
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
| WHISPER_MODEL | turbo | default model id |
| WHISPER_DEVICE | auto | cuda \| cpu \| auto |
| MODEL_IDLE_UNLOAD_SECONDS | 900 | unload after idleness (0 disables) |
| WHISPER_HOST / WHISPER_PORT | 127.0.0.1 / 4444 | bind; 0.0.0.0 only behind firewall, no auth |
| CUDA_DEVICE_ID | 0 | GPU index |

## API

- `POST /v1/audio/transcriptions` — multipart `file`, optional `model`,
  `language`, `device`, `response_format` (json/text). Answers
  `{"text": ..., "language": ...}`. Also `POST /transcribe` (legacy alias).
- `GET /v1/models`, `GET /v1/health`, `GET /docs` (Swagger).

Example:

```powershell
curl.exe -X POST http://127.0.0.1:4444/v1/audio/transcriptions `
  -F "file=@recording.wav" -F "model=turbo"
```

## Tests

```powershell
python -m pytest tests/ -q   # needs only fastapi + pytest (torch stubbed)
```

## License

MIT — same as upstream. See [LICENSE](LICENSE).
