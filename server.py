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
MODEL_SIZE = os.getenv("WHISPER_MODEL", "turbo")

# Idle timeout (seconds) after which the model is unloaded from RAM/VRAM.
# Configurable via env MODEL_IDLE_UNLOAD_SECONDS (default 900 = 15 minutes).
try:
    MODEL_IDLE_UNLOAD_SECONDS = int(os.getenv("MODEL_IDLE_UNLOAD_SECONDS", "900"))
except ValueError:
    MODEL_IDLE_UNLOAD_SECONDS = 900

# Allowed per-request models (all multilingual, EN+SK). Only ONE model is
# ever loaded at a time to protect the shared 6GB VRAM.
ALLOWED_MODELS = ("tiny", "base", "small", "medium", "turbo", "large")

# Documented resource requirements / speed estimates (see README + config).
MODEL_INFO = {
    "tiny":   {"params": "39M",   "vram": "~1 GB",  "speed": "~10x", "accuracy": "lowest"},
    "base":   {"params": "74M",   "vram": "~1 GB",  "speed": "~7x",  "accuracy": "low"},
    "small":  {"params": "244M",  "vram": "~2 GB",  "speed": "~4x",  "accuracy": "good"},
    "medium": {"params": "769M",  "vram": "~5 GB",  "speed": "~2x",  "accuracy": "high"},
    "turbo":  {"params": "809M",  "vram": "~6 GB",  "speed": "~8x",  "accuracy": "high",
               "notes": "large-v3-turbo, best speed/accuracy ratio"},
    "large":  {"params": "1550M", "vram": "~10 GB", "speed": "~1x",  "accuracy": "highest"},
}

# File upload limits
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
ALLOWED_AUDIO_EXTENSIONS = {
    ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a",
    ".wav", ".webm", ".ogg", ".flac", ".opus"
}

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Auto-detect CUDA availability and check for modern GPUs
# NOTE: this only queries device properties, it does NOT allocate VRAM
# and does NOT load the Whisper model (that happens lazily on first request).
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

# Use print to ensure logs appear during module import (before uvicorn captures logging)
print("="*60, flush=True)
print("Whisper API Server - GPU Detection", flush=True)
print("="*60, flush=True)

if torch.cuda.is_available():
    print("[OK] CUDA is available on this system", flush=True)
    try:
        capability = torch.cuda.get_device_capability(0)
        compute_cap = float(f"{capability[0]}.{capability[1]}")
        gpu_name = torch.cuda.get_device_name(0)

        print(f"GPU Name: {gpu_name}", flush=True)
        print(f"GPU Compute Capability: sm_{capability[0]}{capability[1]} ({compute_cap})", flush=True)
        print(f"CUDA Toolkit Version: {torch.version.cuda}", flush=True)
        print(f"PyTorch Version: {torch.__version__}", flush=True)

        # Check PyTorch compatibility
        # PyTorch 2.9+ requires compute capability >= 7.5 (sm_75)
        # Older versions support down to 5.0 or 6.0
        pytorch_min_capability = 7.5  # PyTorch 2.9.x minimum requirement

        print(f"PyTorch Minimum Required: sm_75 (7.5+)")

        if compute_cap < pytorch_min_capability:
            print(f"[X] GPU INCOMPATIBLE: Your GPU (sm_{capability[0]}{capability[1]}) is below PyTorch minimum (sm_75)")
            print(f"   Explanation: PyTorch {torch.__version__} dropped support for older GPUs")
            print(f"   - Your GPU: Compute Capability {compute_cap} (Pascal/Maxwell architecture)")
            print(f"   - Required: Compute Capability 7.5+ (Turing/Volta/Ampere/Ada/Hopper)")
            print(f"   - Supported GPUs: RTX 20/30/40 series, Tesla V100+, A100, H100")
            print(f"")
            print(f"   To use this GPU, you would need to downgrade PyTorch to version 2.4 or earlier.")
            print(f"   Current configuration will use CPU mode (int8 precision).")
            print(f"Final Device: CPU (GPU incompatible with PyTorch version)")
        elif compute_cap >= 5.0:
            # GPU is compatible, enable CUDA
            DEVICE = "cuda"
            COMPUTE_TYPE = "float16"
            print(f"[OK] GPU COMPATIBLE: Compute capability {compute_cap} meets PyTorch requirement (7.5+)")
            print(f"[OK] CUDA ACCELERATION ENABLED")
            print(f"Final Device: CUDA (GPU acceleration active with float16 precision)")
        else:
            print(f"[X] GPU TOO OLD: Compute capability {compute_cap} < 5.0")
            print(f"   This GPU is from pre-2014 era and is not supported by any modern CUDA toolkit.")
            print(f"Final Device: CPU (GPU too old)")

    except Exception as e:
        print(f"[X] Could not get GPU information: {e}")
        print(f"Final Device: CPU (GPU detection failed)")
else:
    print("[X] CUDA is NOT available on this system")
    print("   Possible reasons:")
    print("   - No NVIDIA GPU detected")
    print("   - NVIDIA drivers not installed")
    print("   - PyTorch CPU-only version installed")
    print(f"Final Device: CPU (no CUDA support)")

# Optional device override via env: WHISPER_DEVICE=cpu forces CPU mode,
# WHISPER_DEVICE=cuda (or unset) keeps auto-detection. Per-request `device`
# field takes precedence over both.
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "").strip().lower()
if WHISPER_DEVICE == "cpu" and DEVICE == "cuda":
    DEVICE = "cpu"
    COMPUTE_TYPE = "int8"
    print("WHISPER_DEVICE=cpu: forcing CPU mode (int8)")
elif WHISPER_DEVICE not in ("", "auto", "cuda"):
    print(f"Ignoring unknown WHISPER_DEVICE={WHISPER_DEVICE!r} (use 'cuda' or 'cpu')")
    WHISPER_DEVICE = ""

print("="*60)
print(f"Active Configuration: DEVICE={DEVICE}, COMPUTE_TYPE={COMPUTE_TYPE}, MODEL={MODEL_SIZE}")
print(f"Allowed models: {', '.join(ALLOWED_MODELS)} (one loaded at a time)")
print(f"Lazy loading: ENABLED (model loads on first request, unloads after {MODEL_IDLE_UNLOAD_SECONDS}s idle)")
print("="*60)

DEVICE_INDEX = int(os.getenv("CUDA_DEVICE_ID", "0"))
CHUNK_LENGTH_S = int(os.getenv("CHUNK_LENGTH", "15"))
VAD_FILTER = os.getenv("VAD_FILTER", "false").lower() == "true"
PORT = 4444
HOST = "127.0.0.1"

# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(title="Whisper Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Disabled for security since we're localhost-only
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Lazy model management
# -----------------------------
# The model is intentionally None at startup: it is loaded on the first
# transcription request and unloaded after MODEL_IDLE_UNLOAD_SECONDS of
# inactivity, so idle VRAM/RAM stays free for other apps (e.g. LM Studio).
model: Optional[WhisperModel] = None
_loaded_model_name: Optional[str] = None
_last_used_ts: float = 0.0
_gpu_lock = asyncio.Lock()
_idle_task: Optional[asyncio.Task] = None


def _resolve_target_device(device_arg: Optional[str]) -> str:
    """Resolve which device to use: explicit per-request value wins,
    then WHISPER_DEVICE env, then the auto-detected default
    (which may have been muted to CPU by an earlier fallback)."""
    if device_arg:
        return device_arg
    if WHISPER_DEVICE in ("cuda", "cpu"):
        return WHISPER_DEVICE
    return DEVICE


def _ensure_model_locked(wanted_model: str, target_device: str) -> WhisperModel:
    """Load `wanted_model` on `target_device` if needed and return it.

    Only ONE model is ever kept in memory: requesting a different model or
    device unloads the current one first (VRAM is shared with other apps).
    MUST only be called with `_gpu_lock` held, so concurrent requests can
    never trigger a double load. CUDA failures fall back to CPU (int8)
    instead of crashing.
    """
    global model, DEVICE, COMPUTE_TYPE, _last_used_ts, _loaded_model_name
    want_compute = "float16" if target_device == "cuda" else "int8"

    if (model is not None and _loaded_model_name == wanted_model
            and DEVICE == target_device and COMPUTE_TYPE == want_compute):
        _last_used_ts = time.time()
        return model

    if model is not None:
        logger.info(
            f"Switching model '{_loaded_model_name}' ({DEVICE}/{COMPUTE_TYPE}) -> "
            f"'{wanted_model}' ({target_device}/{want_compute}): "
            f"unloading current model first (one model in memory)"
        )
        _unload_model(reason=f"switch to '{wanted_model}'")
    else:
        logger.info(f"Lazy loading Whisper model '{wanted_model}' on {target_device} ({want_compute})...")

    try:
        logger.info(f"Loading Whisper model: {wanted_model}")
        model = WhisperModel(
            wanted_model,
            device=target_device,
            device_index=DEVICE_INDEX,
            compute_type=want_compute,
            num_workers=1,
        )
        DEVICE, COMPUTE_TYPE = target_device, want_compute
        logger.info(f"[OK] Model loaded successfully: {wanted_model} on {DEVICE} with {COMPUTE_TYPE}")
    except Exception as e:
        # Fall back to CPU if CUDA fails (e.g. missing CUDA 12 DLLs such as
        # cublas64_12.dll required by CTranslate2) instead of crashing.
        if target_device == "cuda":
            logger.error(f"[X] Failed to load model '{wanted_model}' on CUDA: {e}")
            logger.error(
                "Possible cause: CTranslate2 is built against the CUDA 12 runtime and needs "
                "CUDA 12 DLLs (cublas64_12.dll, cudart64_12.dll, cudnn DLLs) visible in PATH. "
                "A CUDA 13-only toolkit (cublas64_13.dll) is NOT enough."
            )
            logger.error(
                "Fix options: (1) install the CUDA 12.x runtime and add its bin dir to the "
                "service PATH, (2) pip-install matching nvidia-*-cu12 wheels into the service "
                "Python env, or (3) copy the missing *_12.dll files next to the service binaries. "
                "Continuing with CPU fallback for now."
            )
            logger.info("Falling back to CPU mode...")
            try:
                model = WhisperModel(
                    wanted_model,
                    device="cpu",
                    device_index=DEVICE_INDEX,
                    compute_type="int8",
                    num_workers=1,
                )
                DEVICE = "cpu"
                COMPUTE_TYPE = "int8"
                logger.info(f"[OK] Model loaded successfully: {wanted_model} on CPU with int8 [fallback]")
            except Exception as e2:
                logger.error(f"[X] CPU fallback also failed: {e2}")
                raise HTTPException(status_code=503, detail=f"Failed to load Whisper model '{wanted_model}': {e2}")
        else:
            logger.error(f"Failed to load model '{wanted_model}': {e}")
            raise HTTPException(status_code=503, detail=f"Failed to load Whisper model '{wanted_model}': {e}")

    _loaded_model_name = wanted_model
    _last_used_ts = time.time()
    return model


def _unload_model(reason: str = "idle") -> bool:
    """Unload the model from RAM/VRAM. Returns True if a model was unloaded."""
    global model, _loaded_model_name
    if model is None:
        return False
    logger.info(f"Unloading Whisper model '{_loaded_model_name or MODEL_SIZE}' ({reason}) to free RAM/VRAM...")
    try:
        del model
    finally:
        model = None
        _loaded_model_name = None
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"torch.cuda.empty_cache() failed: {e}")
    logger.info("[OK] Model unloaded, VRAM cache cleared")
    return True


# Markers identifying a missing/broken GPU library (CTranslate2/CUDA),
# as opposed to a genuine transcription error (bad audio, etc.).
_GPU_LIBRARY_ERROR_MARKERS = (
    "cublas", "cudnn", "cudart", "nvrtc", "cutensor", "ctranslate",
    "cublas64_12.dll",
)


def _is_gpu_library_error(e: Exception) -> bool:
    msg = f"{type(e).__name__}: {e}".lower()
    return any(m in msg for m in _GPU_LIBRARY_ERROR_MARKERS)


def _transcribe_file(active_model: WhisperModel, tmp_path: str, language: Optional[str]):
    """Run transcription with the service's fixed VAD/beam settings.

    Returns (text, info). Must be called with `_gpu_lock` held.
    """
    segments, info = active_model.transcribe(
        tmp_path,
        language=language,
        # Less aggressive duplicate detection:
        condition_on_previous_text=True,
        temperature=0.0,                          # Single temperature, no fallback retries
        beam_size=1,
        best_of=1,
        word_timestamps=False,
        chunk_length=15,
        vad_filter=True,
        vad_parameters={
            "min_speech_duration_ms": 200,
            "min_silence_duration_ms": 250,
            "speech_pad_ms": 120
        }
    )
    text = " ".join(s.text for s in segments).strip()
    return text, info


async def _idle_unloader() -> None:
    """Background task: unload the model after MODEL_IDLE_UNLOAD_SECONDS idle."""
    logger.info(f"Idle unloader started (unload after {MODEL_IDLE_UNLOAD_SECONDS}s without requests)")
    try:
        while True:
            await asyncio.sleep(60)
            if model is None:
                continue
            idle_for = time.time() - _last_used_ts
            if idle_for >= MODEL_IDLE_UNLOAD_SECONDS:
                async with _gpu_lock:
                    # Re-check under the lock (a request may have arrived meanwhile).
                    if model is not None and (time.time() - _last_used_ts) >= MODEL_IDLE_UNLOAD_SECONDS:
                        _unload_model(reason=f"idle for {idle_for:.0f}s")
    except asyncio.CancelledError:
        logger.info("Idle unloader stopped")

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.get("/v1/health")
def health():
    # Works with or without a loaded model (no model access here).
    return {
        "status": "ok",
        "device": DEVICE,
        "model": MODEL_SIZE,
        "compute_type": COMPUTE_TYPE,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": getattr(torch.version, "cuda", None),
        "model_loaded": model is not None,
        "loaded_model": _loaded_model_name,
        "loaded_device": DEVICE if model is not None else None,
        "idle_unload_seconds": MODEL_IDLE_UNLOAD_SECONDS,
        "idle_for_seconds": round(time.time() - _last_used_ts, 1) if model is not None else None,
    }

@app.get("/v1/models")
def list_models():
    # Works with or without a loaded model (static metadata + load state).
    return {
        "object": "list",
        "default_model": MODEL_SIZE,
        "loaded_model": _loaded_model_name,
        "loaded_device": DEVICE if model is not None else None,
        "loaded_compute_type": COMPUTE_TYPE if model is not None else None,
        "data": [
            {"id": name, "object": "model", **MODEL_INFO[name]}
            for name in ALLOWED_MODELS
        ],
    }

@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(...),
    model_name: str = Form(default="whisper-1"),
    model: Optional[str] = Form(default=None),
    device: Optional[str] = Form(default=None),
    language: Optional[str] = Form(default=None),
    temperature: float = Form(default=0.0),
    response_format: str = Form(default="json"),
):
    global DEVICE, COMPUTE_TYPE
    # Per-request model selection: the OpenAI-standard `model` field wins,
    # the legacy `model_name` field is still honored. 'whisper-1' (or an
    # omitted field) means the server default from WHISPER_MODEL.
    raw_model = (model or "").strip() or (model_name or "").strip() or "whisper-1"
    if raw_model == "whisper-1":
        wanted_model = MODEL_SIZE
    elif raw_model in ALLOWED_MODELS:
        wanted_model = raw_model
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{raw_model}'. Allowed models: {', '.join(ALLOWED_MODELS)} "
                   f"(or 'whisper-1' = server default '{MODEL_SIZE}').",
        )

    # Per-request device selection (validated here, resolved inside the lock
    # path via _resolve_target_device).
    raw_device = (device or "").strip().lower()
    if raw_device and raw_device not in ("cuda", "cpu"):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown device '{device}'. Allowed: 'cuda', 'cpu'.",
        )
    target_device = _resolve_target_device(raw_device or None)

    # Validate file extension
    suffix = os.path.splitext(file.filename or "")[1].lower() or ".wav"
    if suffix not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio format: {suffix}. Supported formats: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}"
        )

    # Read file and validate size
    file_content = await file.read()
    file_size = len(file_content)

    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {file_size / (1024*1024):.1f} MB. Maximum allowed: {MAX_FILE_SIZE / (1024*1024):.0f} MB"
        )

    if file_size == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    logger.info(f"Processing transcription: {file.filename} ({file_size / 1024:.1f} KB, model={wanted_model}, device={target_device})")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            tmp.write(file_content)
            tmp.flush()

        async with _gpu_lock:
            # Lazy-load inside the lock: concurrent requests serialize here,
            # so a double load is impossible. A different model/device unloads
            # the current one first (one model in memory). Segments are
            # consumed inside the lock as well, so the idle unloader can never
            # free the model mid-transcription.
            active_model = _ensure_model_locked(wanted_model, target_device)
            try:
                text, info = _transcribe_file(active_model, tmp_path, language)
            except (RuntimeError, OSError) as e:
                # The model can load on CUDA fine but still fail at inference
                # (e.g. missing cublas64_12.dll required by CTranslate2).
                # Fall back to CPU and retry once instead of failing.
                if DEVICE != "cuda" or not _is_gpu_library_error(e):
                    raise
                logger.error(f"[X] GPU inference failed: {e}")
                logger.error(
                    "CTranslate2 needs CUDA 12 runtime DLLs (cublas64_12.dll, cudart64_12.dll, "
                    "cudnn DLLs) visible in PATH; a CUDA 13-only toolkit is NOT enough. "
                    "Fix options: (1) install the CUDA 12.x runtime, (2) pip-install matching "
                    "nvidia-*-cu12 wheels, or (3) copy the missing *_12.dll files next to the "
                    "service binaries. Falling back to CPU for now."
                )
                logger.info("Falling back to CPU (int8) and retrying transcription...")
                _unload_model(reason="gpu inference failed")
                DEVICE = "cpu"
                COMPUTE_TYPE = "int8"
                active_model = _ensure_model_locked(wanted_model, "cpu")
                text, info = _transcribe_file(active_model, tmp_path, language)
            _last_used_ts = time.time()

        if response_format == "text":
            return PlainTextResponse(text)
        return JSONResponse({"text": text, "language": getattr(info, "language", language)})

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
    model: Optional[str] = Form(default=None),
    device: Optional[str] = Form(default=None),
):
    return await transcribe_audio(file=file, model_name="whisper-1", model=model, device=device, language=language, response_format="json")

@app.on_event("startup")
async def startup_log():
    """Log configuration at startup. Deliberately does NOT load the model
    (no warmup transcription) so idle RAM/VRAM stays free."""
    global _idle_task
    # Log GPU/CUDA configuration
    logger.info("="*60)
    logger.info("Whisper API Server - GPU/CUDA Configuration")
    logger.info("="*60)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Compute Type: {COMPUTE_TYPE}")
    logger.info(f"Model: {MODEL_SIZE}")
    logger.info(f"PyTorch Version: {torch.__version__}")

    if torch.cuda.is_available():
        logger.info("CUDA Status: Available")
        try:
            capability = torch.cuda.get_device_capability(0)
            compute_cap = float(f"{capability[0]}.{capability[1]}")
            gpu_name = torch.cuda.get_device_name(0)

            logger.info(f"GPU Name: {gpu_name}")
            logger.info(f"GPU Compute Capability: sm_{capability[0]}{capability[1]} ({compute_cap})")
            logger.info(f"CUDA Toolkit Version: {torch.version.cuda}")

            # Explain why CUDA might be disabled
            pytorch_min_capability = 7.5  # PyTorch 2.9.x requirement
            logger.info(f"PyTorch Minimum Required: sm_75 (7.5+)")

            if DEVICE == "cuda":
                logger.info("[OK] CUDA ACCELERATION ENABLED")
                logger.info(f"   GPU is compatible and will be used for acceleration")
            else:
                # CUDA is available but not being used - explain why
                logger.warning("[X] CUDA DISABLED - GPU Not Compatible")
                logger.warning(f"   Your GPU: {gpu_name} with compute capability {compute_cap} (sm_{capability[0]}{capability[1]})")
                logger.warning(f"   PyTorch {torch.__version__} requires: Compute capability 7.5+ (sm_75+)")
                logger.warning(f"")
                # Determine GPU architecture based on compute capability
                if compute_cap >= 9.0:
                    arch = "Hopper (2022+)"
                elif compute_cap >= 8.9:
                    arch = "Ada Lovelace (2022)"
                elif compute_cap >= 8.0:
                    arch = "Ampere (2020)"
                elif compute_cap >= 7.5:
                    arch = "Turing (2018)"
                elif compute_cap >= 7.0:
                    arch = "Volta (2017)"
                elif compute_cap >= 6.0:
                    arch = "Pascal (2016)"
                elif compute_cap >= 5.0:
                    arch = "Maxwell (2014)"
                else:
                    arch = "Kepler or older (2012-)"

                logger.warning(f"   Explanation:")
                logger.warning(f"   - Your GPU is based on {arch} architecture")
                logger.warning(f"   - PyTorch 2.9+ dropped support for GPUs older than Turing/Volta (2018)")
                logger.warning(f"   - Supported GPUs: RTX 20/30/40 series, Tesla V100+, A100, H100")
                logger.warning(f"")
                logger.warning(f"   Options:")
                logger.warning(f"   1. Continue with CPU mode (current - works fine, just slower)")
                logger.warning(f"   2. Downgrade PyTorch to version 2.4 to use this GPU")
                logger.warning(f"   3. Upgrade GPU to RTX 2060 or newer for CUDA support")

        except Exception as e:
            logger.error(f"Error getting GPU details: {e}")
    else:
        logger.info("CUDA Status: Not Available")
        logger.info("   Possible reasons:")
        logger.info("   - No NVIDIA GPU detected")
        logger.info("   - NVIDIA drivers not installed")
        logger.info("   - PyTorch CPU-only version installed")

    logger.info("="*60)

    # No warmup transcription: the model stays unloaded until the first request.
    if MODEL_SIZE not in ALLOWED_MODELS and MODEL_SIZE != "whisper-1":
        logger.warning(f"WHISPER_MODEL='{MODEL_SIZE}' is not in the per-request allowlist ({', '.join(ALLOWED_MODELS)})")
    logger.info(f"Lazy loading enabled: model '{MODEL_SIZE}' will load on first transcription request")
    logger.info(f"Idle unload: model will be released after {MODEL_IDLE_UNLOAD_SECONDS}s without requests")

    _idle_task = asyncio.create_task(_idle_unloader())

@app.on_event("shutdown")
async def shutdown_cleanup():
    global _idle_task
    if _idle_task is not None:
        _idle_task.cancel()
        try:
            await _idle_task
        except asyncio.CancelledError:
            pass
        _idle_task = None

if __name__ == "__main__":
    # Add script directory to Python path for module import
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    uvicorn.run("server:app", host=HOST, port=PORT, reload=False, workers=1)
