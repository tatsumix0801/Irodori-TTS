import importlib

import pytest

MODULE_NAMES = ["gradio_app", "gradio_app_voicedesign"]


@pytest.fixture(params=MODULE_NAMES)
def gradio_module(request):
    return importlib.import_module(request.param)


def test_build_arg_parser_defaults_match_todays_behavior(gradio_module):
    parser = gradio_module.build_arg_parser()
    args = parser.parse_args([])

    assert args.decode_mode == "sequential"
    assert args.compile_model is False
    assert args.compile_dynamic is False


def test_build_arg_parser_accepts_perf_flags(gradio_module):
    parser = gradio_module.build_arg_parser()
    args = parser.parse_args(
        ["--decode-mode", "batch", "--compile-model", "--compile-dynamic"]
    )

    assert args.decode_mode == "batch"
    assert args.compile_model is True
    assert args.compile_dynamic is True


def test_build_arg_parser_rejects_invalid_decode_mode(gradio_module):
    parser = gradio_module.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--decode-mode", "bogus"])


def test_build_runtime_key_honors_perf_options_compile_flags(gradio_module, monkeypatch):
    monkeypatch.setattr(
        gradio_module,
        "_PERF",
        gradio_module._PerfOptions(
            decode_mode="batch", compile_model=True, compile_dynamic=True
        ),
    )

    runtime_key = gradio_module._build_runtime_key(
        checkpoint="dummy.pt",
        model_device="cpu",
        model_precision="fp32",
        codec_device="cpu",
        codec_precision="fp32",
    )

    assert runtime_key.compile_model is True
    assert runtime_key.compile_dynamic is True


def test_build_runtime_key_defaults_keep_compile_disabled(gradio_module, monkeypatch):
    monkeypatch.setattr(gradio_module, "_PERF", gradio_module._PerfOptions())

    runtime_key = gradio_module._build_runtime_key(
        checkpoint="dummy.pt",
        model_device="cpu",
        model_precision="fp32",
        codec_device="cpu",
        codec_precision="fp32",
    )

    assert runtime_key.compile_model is False
    assert runtime_key.compile_dynamic is False
