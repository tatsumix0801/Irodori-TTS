import importlib


def test_gradio_safe_device_uses_cpu_for_rocm_cuda_by_default(monkeypatch):
    gradio_runtime = importlib.import_module("irodori_tts.gradio_runtime")

    monkeypatch.setattr(gradio_runtime, "is_rocm_runtime", lambda: True)
    monkeypatch.delenv(gradio_runtime.ROCM_CUDA_OPT_IN_ENV, raising=False)

    assert gradio_runtime.gradio_safe_device("cuda") == "cpu"
    assert gradio_runtime.gradio_safe_device("cuda:0") == "cpu"


def test_gradio_safe_device_keeps_non_cuda_devices_on_rocm(monkeypatch):
    gradio_runtime = importlib.import_module("irodori_tts.gradio_runtime")

    monkeypatch.setattr(gradio_runtime, "is_rocm_runtime", lambda: True)
    monkeypatch.delenv(gradio_runtime.ROCM_CUDA_OPT_IN_ENV, raising=False)

    assert gradio_runtime.gradio_safe_device("cpu") == "cpu"
    assert gradio_runtime.gradio_safe_device("mps") == "mps"


def test_gradio_safe_device_keeps_cuda_when_rocm_opt_in_is_set(monkeypatch):
    gradio_runtime = importlib.import_module("irodori_tts.gradio_runtime")

    monkeypatch.setattr(gradio_runtime, "is_rocm_runtime", lambda: True)
    monkeypatch.setenv(gradio_runtime.ROCM_CUDA_OPT_IN_ENV, "1")

    assert gradio_runtime.gradio_safe_device("cuda") == "cuda"


def test_gradio_safe_device_keeps_cuda_on_non_rocm(monkeypatch):
    gradio_runtime = importlib.import_module("irodori_tts.gradio_runtime")

    monkeypatch.setattr(gradio_runtime, "is_rocm_runtime", lambda: False)
    monkeypatch.delenv(gradio_runtime.ROCM_CUDA_OPT_IN_ENV, raising=False)

    assert gradio_runtime.gradio_safe_device("cuda") == "cuda"
