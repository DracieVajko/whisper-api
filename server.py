"""Whisper API Server with lazy model loading.

Fork of dniasoff/whisper-api (MIT, Windows-first faster-whisper server).
Fork changes vs upstream:
  - Lazy model loading: nothing is loaded at startup; the first
    transcription request loads the model, idle timeout unloads it.
  - Per-request ``model`` selection from a curated allowlist (one model
    resident at a time; switching unloads first).
  - Per-request ``device`` selection (cuda/cpu) with CPU fallback that
    never surfaces a bare 500.
  - GET /v1/models documents the allowlist (params/VRAM/speed/accuracy).
  - GET /v1/health works without a loaded model and reports load state.
  - No startup warmup transcription (it would defeat lazy loading).

OpenAI-compatible: POST /v1/audio/transcriptions answers
``{"text": ..., "language": ...}``.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse
from faster_whisper import WhisperModel
from typing import Optional
import asyncio
import gc
import tempfile
import time
import torch
import os
import uvicorn
import logging

# -----------------------------
# Config
# -----------------------------
DEFAULT_MODEL = os.getenv("WHISPER_MODEL", "turbo")
# cuda | cpu | auto (auto = cuda when usable, else cpu)
CONFIGURED_DEVICE = os.getenv("WHISPER_DEVICE", "auto").strip().lower()
MODEL_IDLE_UNLOAD_SECONDS = int(os.getenv("MODEL_IDLE_UNLOAD_SECONDS", "900"))

# File upload limits
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
ALLOWED_AUDIO_EXTENSIONS = {
    ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a",
    ".wav", ".webm", ".ogg", ".opus", ".flac",
}

# Curated model allowlist: id -> (params, vram, speed, accuracy, notes).
# faster-whisper resolves these ids itself (turbo = large-v3-turbo).
MODEL_REGISTRY = {
    "tiny":   {"params": "39M",   "vram": "~1 GB",  "speed": "~10x", "accuracy": "lowest",  "notes": ""},
    "base":   {"params": "74M",   "vram": "~1 GB",  "speed": "~7x",  "accuracy": "low",     "notes": ""},
    "small":  {"params": "244M",  "vram": "~2 GB",  "speed": "~4x",  "accuracy": "good",    "notes": ""},
    "medium": {"params": "769M",  "vram": "~5 GB",  "speed": "~2x",  "accuracy": "high",    "notes": ""},
    "turbo":  {"params": "809M",  "vram": "~6 GB",  "speed": "~8x",  "accuracy": "high",    "notes": "large-v3-turbo, best speed/accuracy ratio"},
    "large":  {"params": "1550M", "vram": "~10 GB", "speed": "~1x",  "accuracy": "highest", "notes": "does not fit 6 GB cards"},
}

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

DEVICE_INDEX = int(os.getenv("CUDA_DEVICE_ID", "0"))
CHUNK_LENGTH_S = int(os.getenv("CHUNK_LENGTH", "15"))
PORT = int(os.getenv("WHISPER_PORT", "4444"))
HOST = os.getenv("WHISPER_HOST", "127.0.0.1")


def _cuda_usable() -> bool:
    """True when torch sees CUDA. Never raises."""
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _default_device() -> str:
    """Effective default device: explicit config wins, else auto-detect."""
    if CONFIGURED_DEVICE in ("cuda", "cpu"):
        return CONFIGURED_DEVICE
    return "cuda" if _cuda_usable() else "cpu"


# Sticky default: a failed CUDA load permanently falls back to CPU so every
# later request does not pay for the same doomed attempt. An explicit
# per-request device=cuda still retries the GPU.
_effective_device = _default_device()


def _compute_type_for(device: str) -> str:
    return "float16" if device == "cuda" else "int8"


# -----------------------------
# Lazy model state (one resident model at most)
# -----------------------------
model = None
_loaded_model_id: Optional[str] = None
_loaded_device: Optional[str] = None
_loaded_compute_type: Optional[str] = None
_last_used = 0.0
_model_lock = asyncio.Lock()
_gpu_lock = asyncio.Lock()


def _resolve_model_id(value: Optional[str]) -> str:
    """Map the OpenAI `model` field (or legacy `model_name`) to an id.

    whisper-1/empty means the configured default. Anything outside the
    allowlist is a 400 raised before any load or VRAM is touched.
    """
    v = (value or "").strip() or "whisper-1"
    if v == "whisper-1":
        return DEFAULT_MODEL
    if v not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{v}'. Allowed: {', '.join(sorted(MODEL_REGISTRY))}",
        )
    return v


def _resolve_device(value: Optional[str]) -> str:
    v = (value or "").strip().lower() or _effective_device
    if v not in ("cuda", "cpu"):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown device '{value}'. Allowed: cuda, cpu.",
        )
    return v


def _unload_locked(reason: str) -> None:
    """Drop the resident model. Caller must hold _model_lock."""
    global model, _loaded_model_id, _loaded_device, _loaded_compute_type
    if model is None:
        return
    logger.info(f"Unloading Whisper model '{_loaded_model_id}' ({reason}) to free RAM/VRAM...")
    del model
    model = None
    _loaded_model_id = None
    _loaded_device = None
    _loaded_compute_type = None
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    logger.info("[OK] Model unloaded, VRAM cache cleared")


def _load_locked(model_id: str, device: str):
    """Instantiate and return the model. Caller must hold _model_lock."""
    global _effective_device
    compute_type = _compute_type_for(device)
    logger.info(f"Loading Whisper model: {model_id} on {device} ({compute_type})")
    try:
        return WhisperModel(
            model_id,
            device=device,
            device_index=DEVICE_INDEX,
            compute_type=compute_type,
            num_workers=1,
        )
    except Exception as e:
        if device == "cuda":
            logger.warning(f"CUDA load failed ({e}); falling back to CPU")
            _effective_device = "cpu"  # sticky: stop retrying a broken GPU by default
            return WhisperModel(model_id, device="cpu", compute_type="int8", num_workers=1)
        raise


async def get_model(model_id: str, device: str):
    """Return the resident model, loading/switching as needed (thread-safe)."""
    global model, _loaded_model_id, _loaded_device, _loaded_compute_type, _last_used
    async with _model_lock:
        if model is not None and (_loaded_model_id, _loaded_device) != (model_id, device):
            logger.info(
                f"Switching model '{_loaded_model_id}' ({_loaded_device}) -> "
                f"'{model_id}' ({device}): unloading current model first (one model in memory)"
            )
            _unload_locked(f"switch to '{model_id}'")
        if model is None:
            logger.info(f"Lazy loading Whisper model '{model_id}' on {device}...")
            model = _load_locked(model_id, device)
            # _load_locked may have fallen back to CPU; reflect reality.
            actual_device = device
            if device == "cuda" and _effective_device == "cpu":
                actual_device = "cpu"
            _loaded_model_id = model_id
            _loaded_device = actual_device
            _loaded_compute_type = _compute_type_for(actual_device)
            logger.info(
                f"[OK] Model loaded successfully: {model_id} "
                f"on {_loaded_device} with {_loaded_compute_type}"
            )
        _last_used = time.monotonic()
        return model


async def _idle_unloader():
    """Background task: unload the model after sustained idleness."""
    while True:
        await asyncio.sleep(60)
        async with _model_lock:
            if model is not None and (time.monotonic() - _last_used) > MODEL_IDLE_UNLOAD_SECONDS:
                _unload_locked(f"idle {MODEL_IDLE_UNLOAD_SECONDS}s")


def _load_state() -> dict:
    return {
        "loaded_model": _loaded_model_id,
        "loaded_device": _loaded_device,
        "loaded_compute_type": _loaded_compute_type,
        "model_loaded": model is not None,
        "idle_unload_seconds": MODEL_IDLE_UNLOAD_SECONDS,
        "idle_for_seconds": round(time.monotonic() - _last_used, 1) if model is not None else None,
    }


# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(title="Whisper Assistant API (lazy fork)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return RedirectResponse(url="/docs")


@app.get("/v1/health")
def health():
    state = _load_state()
    return {
        "status": "ok",
        "device": _loaded_device or _effective_device,
        "model": _loaded_model_id or DEFAULT_MODEL,
        "compute_type": _loaded_compute_type or _compute_type_for(_effective_device),
        "cuda_available": _cuda_usable(),
        "cuda_version": getattr(torch.version, "cuda", None),
        **state,
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "default_model": DEFAULT_MODEL,
        **_load_state(),
        "data": [
            {"id": mid, "object": "model", **meta}
            for mid, meta in sorted(MODEL_REGISTRY.items())
        ],
    }


@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = Form(default="whisper-1"),
    model_name: Optional[str] = Form(default=None),  # legacy alias
    language: Optional[str] = Form(default=None),
    device: Optional[str] = Form(default=None),
    temperature: float = Form(default=0.0),
    response_format: str = Form(default="json"),
):
    # OpenAI field wins; legacy alias is honored for older clients.
    model_id = _resolve_model_id(model if (model or "").strip() else (model_name or ""))
    target_device = _resolve_device(device)

    suffix = os.path.splitext(file.filename or "")[1].lower() or ".wav"
    if suffix not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio format: {suffix}. Supported: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}",
        )

    file_content = await file.read()
    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(file_content) / (1024 * 1024):.1f} MB. Maximum: {MAX_FILE_SIZE / (1024 * 1024):.0f} MB",
        )
    if not file_content:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    logger.info(f"Processing transcription: {file.filename} ({len(file_content) / 1024:.1f} KB, model={model_id}, device={target_device})")

    m = await get_model(model_id, target_device)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            tmp.write(file_content)
            tmp.flush()

        try:
            async with _gpu_lock:
                segments, info = m.transcribe(
                    tmp_path,
                    language=language,
                    condition_on_previous_text=True,
                    temperature=0.0,
                    beam_size=1,
                    best_of=1,
                    word_timestamps=False,
                    chunk_length=15,
                    vad_filter=True,
                    vad_parameters={
                        "min_speech_duration_ms": 200,
                        "min_silence_duration_ms": 250,
                        "speech_pad_ms": 120,
                    },
                )
                text = " ".join(s.text for s in segments).strip()
                detected = getattr(info, "language", language)
        except Exception as e:
            # GPU inference blew up after a successful load (e.g. missing
            # cuBLAS): fall back to CPU for this request instead of a bare 500.
            if target_device == "cuda":
                logger.warning(f"CUDA inference failed ({e}); retrying on CPU")
                async with _model_lock:
                    _unload_locked("cuda inference failed")
                m_cpu = await get_model(model_id, "cpu")
                async with _gpu_lock:
                    segments, info = m_cpu.transcribe(
                        tmp_path,
                        language=language,
                        condition_on_previous_text=True,
                        temperature=0.0,
                        beam_size=1,
                        best_of=1,
                        word_timestamps=False,
                        chunk_length=15,
                        vad_filter=True,
                        vad_parameters={
                            "min_speech_duration_ms": 200,
                            "min_silence_duration_ms": 250,
                            "speech_pad_ms": 120,
                        },
                    )
                    text = " ".join(s.text for s in segments).strip()
                    detected = getattr(info, "language", language)
            else:
                raise

        if response_format == "text":
            return PlainTextResponse(text)
        return JSONResponse({"text": text, "language": detected})
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


@app.post("/transcribe")
async def transcribe_alias(
    file: UploadFile = File(...),
    language: Optional[str] = Form(default=None),
):
    return await transcribe_audio(file=file, language=language)


@app.on_event("startup")
async def _start_unloader():
    logger.info("=" * 60)
    logger.info("Whisper API Server - lazy mode (no model loaded at startup)")
    logger.info(f"Default model: {DEFAULT_MODEL}, default device: {_effective_device}")
    logger.info(f"Idle unload after: {MODEL_IDLE_UNLOAD_SECONDS}s")
    logger.info("=" * 60)
    asyncio.create_task(_idle_unloader())


if __name__ == "__main__":
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    uvicorn.run("server:app", host=HOST, port=PORT, reload=False, workers=1)
