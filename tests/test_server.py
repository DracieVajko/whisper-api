"""Offline tests for the lazy-loading server (no torch/GPU needed).

Run: python -m pytest tests/ -q   (needs fastapi + pytest only;
torch / faster-whisper / uvicorn are stubbed, nothing loads VRAM).
"""

import asyncio
import sys
import types

import pytest
from fastapi import HTTPException


def _install_stubs(monkeypatch, calls, cuda_available=True,
                   fail_cuda_load=False, fail_cuda_infer=False):
    torch_mod = types.ModuleType("torch")

    class _Cuda:
        @staticmethod
        def is_available():
            return cuda_available

        @staticmethod
        def get_device_capability(index=0):
            return (8, 6)  # RTX 3060 (sm_86), mirrors production

        @staticmethod
        def get_device_name(index=0):
            return "NVIDIA GeForce RTX 3060 Laptop GPU"

        @staticmethod
        def empty_cache():
            calls.append("empty_cache")

    torch_mod.cuda = _Cuda
    torch_mod.version = types.SimpleNamespace(cuda="13.0")
    torch_mod.__version__ = "2.14.1+cu130"  # mirrors production
    monkeypatch.setitem(sys.modules, "torch", torch_mod)

    fw = types.ModuleType("faster_whisper")

    class FakeModel:
        def __init__(self, size, device="cpu", device_index=0, compute_type="int8", num_workers=1):
            if device == "cuda" and fail_cuda_load:
                raise RuntimeError("cublas64_12.dll missing")
            calls.append(("load", size, device, compute_type))
            self._device = device

        def transcribe(self, path, **kwargs):
            if self._device == "cuda" and fail_cuda_infer:
                raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")

            class _Seg:
                text = "hello"

            class _Info:
                language = "sk"

            return iter([_Seg()]), _Info()

    fw.WhisperModel = FakeModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fw)
    monkeypatch.setitem(sys.modules, "uvicorn", types.ModuleType("uvicorn"))


def _fresh_server(monkeypatch, calls, **kw):
    for mod in [m for m in list(sys.modules) if m == "server" or m.startswith("server.")]:
        del sys.modules[mod]
    _install_stubs(monkeypatch, calls, **kw)
    monkeypatch.setenv("WHISPER_MODEL", "turbo")
    monkeypatch.setenv("MODEL_IDLE_UNLOAD_SECONDS", "900")
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    import server
    return server


class _Upload:
    def __init__(self, name="a.webm", data=b"0123456789"):
        self.filename = name
        self._data = data

    async def read(self):
        return self._data


def _run(coro):
    return asyncio.run(coro)


def test_health_works_unloaded(monkeypatch):
    calls = []
    server = _fresh_server(monkeypatch, calls)
    h = server.health()
    assert h["status"] == "ok"
    assert h["model_loaded"] is False
    assert h["loaded_model"] is None
    assert not [c for c in calls if isinstance(c, tuple)]  # no loads


def test_first_transcribe_lazy_loads(monkeypatch):
    calls = []
    server = _fresh_server(monkeypatch, calls)
    res = _run(server.transcribe_audio(file=_Upload(), model="turbo", model_name=None, language=None, device=None))
    body = res.body.decode()
    assert '"hello"' in body
    assert ("load", "turbo", "cuda", "float16") in calls


def test_model_switch_unloads_first(monkeypatch):
    calls = []
    server = _fresh_server(monkeypatch, calls)
    _run(server.transcribe_audio(file=_Upload(), model="tiny", model_name=None, language=None, device=None))
    _run(server.transcribe_audio(file=_Upload(), model="small", model_name=None, language=None, device=None))
    loads = [c for c in calls if isinstance(c, tuple)]
    assert loads == [("load", "tiny", "cuda", "float16"), ("load", "small", "cuda", "float16")]
    assert "empty_cache" in calls  # switch unloaded first (gc + cuda.empty_cache)


def test_unknown_model_400_without_load(monkeypatch):
    calls = []
    server = _fresh_server(monkeypatch, calls)
    with pytest.raises(HTTPException) as e:
        _run(server.transcribe_audio(file=_Upload(), model="xxl", model_name=None, language=None, device=None))
    assert e.value.status_code == 400
    assert not [c for c in calls if isinstance(c, tuple)]


def test_unknown_device_400(monkeypatch):
    calls = []
    server = _fresh_server(monkeypatch, calls)
    with pytest.raises(HTTPException) as e:
        _run(server.transcribe_audio(file=_Upload(), model="tiny", device="tpu"))
    assert e.value.status_code == 400


def test_legacy_model_name_alias(monkeypatch):
    calls = []
    server = _fresh_server(monkeypatch, calls, cuda_available=False)
    _run(server.transcribe_audio(file=_Upload(), model="", model_name="base", language=None, device=None))
    assert ("load", "base", "cpu", "int8") in calls


def test_cuda_failure_falls_back_to_cpu(monkeypatch):
    calls = []
    server = _fresh_server(monkeypatch, calls, fail_cuda_load=True)
    res = _run(server.transcribe_audio(file=_Upload(), model="tiny", model_name=None, language=None, device="cuda"))
    assert '"hello"' in res.body.decode()
    assert ("load", "tiny", "cpu", "int8") in calls


def test_models_endpoint(monkeypatch):
    calls = []
    server = _fresh_server(monkeypatch, calls)
    out = server.list_models()
    assert out["object"] == "list"
    assert out["default_model"] == "turbo"
    ids = [m["id"] for m in out["data"]]
    assert {"tiny", "turbo", "large"} <= set(ids)
    assert all("vram" in m and "speed" in m for m in out["data"])


def test_bad_extension_415_and_empty_400(monkeypatch):
    calls = []
    server = _fresh_server(monkeypatch, calls)
    with pytest.raises(HTTPException) as e:
        _run(server.transcribe_audio(file=_Upload(name="a.exe"), model="tiny", model_name=None, language=None, device=None))
    assert e.value.status_code == 415
    with pytest.raises(HTTPException) as e2:
        _run(server.transcribe_audio(file=_Upload(data=b""), model="tiny", model_name=None, language=None, device=None))
    assert e2.value.status_code == 400


def test_whisper1_maps_to_default(monkeypatch):
    calls = []
    server = _fresh_server(monkeypatch, calls)
    _run(server.transcribe_audio(file=_Upload(), model="whisper-1", model_name=None, language=None, device=None))
    loads = [c for c in calls if isinstance(c, tuple)]
    assert loads == [("load", "turbo", "cuda", "float16")]


def test_inference_cublas_falls_back_to_cpu(monkeypatch):
    """CUDA load succeeds but inference raises cublas error -> unload,
    reload on CPU, retry once (production PC behavior)."""
    calls = []
    server = _fresh_server(monkeypatch, calls, fail_cuda_infer=True)
    res = _run(server.transcribe_audio(file=_Upload(), model="tiny", model_name=None, language=None, device="cuda"))
    assert '"hello"' in res.body.decode()
    loads = [c for c in calls if isinstance(c, tuple)]
    assert loads == [("load", "tiny", "cuda", "float16"), ("load", "tiny", "cpu", "int8")]
    assert "empty_cache" in calls  # failed GPU model was unloaded first
    assert server.health()["loaded_device"] == "cpu"


def test_non_gpu_error_propagates_without_retry(monkeypatch):
    """A genuine transcription error (not a CUDA library error) must not
    trigger a pointless CPU reload."""
    calls = []
    server = _fresh_server(monkeypatch, calls)

    def boom(model, path, language):
        raise ValueError("bad audio")

    monkeypatch.setattr(server, "_transcribe_file", boom)
    with pytest.raises(ValueError):
        _run(server.transcribe_audio(file=_Upload(), model="tiny", model_name=None, language=None, device="cuda"))
    loads = [c for c in calls if isinstance(c, tuple)]
    assert loads == [("load", "tiny", "cuda", "float16")]  # exactly one load, no retry


def test_models_endpoint_loaded_state(monkeypatch):
    calls = []
    server = _fresh_server(monkeypatch, calls, cuda_available=False)
    _run(server.transcribe_audio(file=_Upload(), model="small", model_name=None, language=None, device=None))
    out = server.list_models()
    assert out["loaded_model"] == "small"
    assert out["loaded_device"] == "cpu"
    assert out["loaded_compute_type"] == "int8"


def test_health_loaded_fields(monkeypatch):
    calls = []
    server = _fresh_server(monkeypatch, calls, cuda_available=False)
    _run(server.transcribe_audio(file=_Upload(), model="small", model_name=None, language=None, device=None))
    h = server.health()
    assert h["model_loaded"] is True
    assert h["loaded_model"] == "small"
    assert h["loaded_device"] == "cpu"
    assert isinstance(h["idle_for_seconds"], float)


def test_unload_model_cycle(monkeypatch):
    calls = []
    server = _fresh_server(monkeypatch, calls)
    _run(server.transcribe_audio(file=_Upload(), model="tiny", model_name=None, language=None, device=None))
    assert server.health()["model_loaded"] is True
    assert server._unload_model("test") is True
    assert "empty_cache" in calls
    h = server.health()
    assert h["model_loaded"] is False
    assert h["loaded_model"] is None
    assert server._unload_model("test") is False  # nothing to unload
