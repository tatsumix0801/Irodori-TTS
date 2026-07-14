import importlib

import pytest

MODULE_NAMES = ["gradio_app", "gradio_app_voicedesign"]


def test_gradio_safe_precision_downgrades_bf16_on_cpu():
    gradio_runtime = importlib.import_module("irodori_tts.gradio_runtime")

    assert gradio_runtime.gradio_safe_precision("cpu", "bf16") == "fp32"


def test_gradio_safe_precision_keeps_bf16_on_cuda():
    gradio_runtime = importlib.import_module("irodori_tts.gradio_runtime")

    assert gradio_runtime.gradio_safe_precision("cuda", "bf16") == "bf16"


def test_gradio_safe_precision_keeps_fp32_on_cpu():
    gradio_runtime = importlib.import_module("irodori_tts.gradio_runtime")

    assert gradio_runtime.gradio_safe_precision("cpu", "fp32") == "fp32"


def test_gradio_safe_precision_keeps_bf16_on_cuda_with_index():
    gradio_runtime = importlib.import_module("irodori_tts.gradio_runtime")

    assert gradio_runtime.gradio_safe_precision("CUDA:0", "bf16") == "bf16"


def test_gradio_safe_precision_keeps_bf16_on_xpu():
    gradio_runtime = importlib.import_module("irodori_tts.gradio_runtime")

    assert gradio_runtime.gradio_safe_precision("xpu", "bf16") == "bf16"


def test_gradio_safe_precision_is_case_and_whitespace_insensitive_for_bf16():
    gradio_runtime = importlib.import_module("irodori_tts.gradio_runtime")

    assert gradio_runtime.gradio_safe_precision("cpu", "  BF16  ") == "fp32"


def test_gradio_precision_note_none_when_unchanged():
    gradio_runtime = importlib.import_module("irodori_tts.gradio_runtime")

    assert gradio_runtime.gradio_precision_note("model", "fp32", "fp32") is None


def test_gradio_precision_note_message_when_downgraded():
    gradio_runtime = importlib.import_module("irodori_tts.gradio_runtime")

    note = gradio_runtime.gradio_precision_note("codec", "bf16", "fp32")

    assert note is not None
    assert "bf16" in note
    assert "fp32" in note
    assert "codec" in note


@pytest.fixture(params=MODULE_NAMES)
def gradio_module(request):
    return importlib.import_module(request.param)


def test_build_runtime_key_downgrades_bf16_precision_on_cpu_device(gradio_module):
    runtime_key = gradio_module._build_runtime_key(
        checkpoint="dummy.pt",
        model_device="cpu",
        model_precision="bf16",
        codec_device="cpu",
        codec_precision="bf16",
    )

    assert runtime_key.model_precision == "fp32"
    assert runtime_key.codec_precision == "fp32"


def test_build_runtime_key_keeps_bf16_precision_on_cuda_device(gradio_module, monkeypatch):
    gradio_runtime = importlib.import_module("irodori_tts.gradio_runtime")
    monkeypatch.setattr(gradio_runtime, "is_rocm_runtime", lambda: False)

    runtime_key = gradio_module._build_runtime_key(
        checkpoint="dummy.pt",
        model_device="cuda",
        model_precision="bf16",
        codec_device="cuda",
        codec_precision="bf16",
    )

    assert runtime_key.model_precision == "bf16"
    assert runtime_key.codec_precision == "bf16"


def test_build_runtime_key_downgrades_bf16_precision_on_rocm_coercion(
    gradio_module, monkeypatch
):
    gradio_runtime = importlib.import_module("irodori_tts.gradio_runtime")
    monkeypatch.setattr(gradio_runtime, "is_rocm_runtime", lambda: True)
    monkeypatch.delenv(gradio_runtime.ROCM_CUDA_OPT_IN_ENV, raising=False)

    runtime_key = gradio_module._build_runtime_key(
        checkpoint="dummy.pt",
        model_device="cuda",
        model_precision="bf16",
        codec_device="cuda",
        codec_precision="bf16",
    )

    assert runtime_key.model_device == "cpu"
    assert runtime_key.codec_device == "cpu"
    assert runtime_key.model_precision == "fp32"
    assert runtime_key.codec_precision == "fp32"


def test_runtime_device_messages_includes_precision_downgrade_note(gradio_module):
    runtime_key = gradio_module._build_runtime_key(
        checkpoint="dummy.pt",
        model_device="cpu",
        model_precision="bf16",
        codec_device="cpu",
        codec_precision="fp32",
    )

    messages = gradio_module._runtime_device_messages(
        runtime_key=runtime_key,
        requested_model_device="cpu",
        requested_codec_device="cpu",
        requested_model_precision="bf16",
        requested_codec_precision="fp32",
    )

    assert any("bf16" in message and "fp32" in message for message in messages)
