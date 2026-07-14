import re
from pathlib import Path

import pytest
import torch

import perf_bootstrap

REPO_ROOT = Path(__file__).resolve().parent.parent
ENTRYPOINT_FILES = [
    "train.py",
    "infer.py",
    "gradio_app.py",
    "gradio_app_voicedesign.py",
]
_GUARD_PATTERN = re.compile(r"^(import torch|from torch|import gradio|from irodori_tts)")


@pytest.fixture(autouse=True)
def _restore_torch_thread_count():
    original = torch.get_num_threads()
    try:
        yield
    finally:
        torch.set_num_threads(original)


def test_apply_env_defaults_sets_both_keys_on_empty_env():
    env: dict[str, str] = {}

    applied = perf_bootstrap.apply_env_defaults(env)

    assert env == perf_bootstrap.ROCM_ENV_DEFAULTS
    assert applied == perf_bootstrap.ROCM_ENV_DEFAULTS


def test_apply_env_defaults_respects_pre_existing_values():
    env = {"MIOPEN_FIND_MODE": "0"}

    applied = perf_bootstrap.apply_env_defaults(env)

    assert env["MIOPEN_FIND_MODE"] == "0"
    assert env["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] == "1"
    assert "MIOPEN_FIND_MODE" not in applied
    assert applied == {"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1"}


def test_configure_cpu_threads_uses_omp_num_threads_when_set():
    n = perf_bootstrap.configure_cpu_threads(env={"OMP_NUM_THREADS": "4"})

    assert n == 4
    assert torch.get_num_threads() == 4


def test_configure_cpu_threads_defaults_to_half_cpu_count():
    n = perf_bootstrap.configure_cpu_threads(env={}, cpu_count=32)

    assert n == 16
    assert torch.get_num_threads() == 16


def test_configure_cpu_threads_falls_back_on_invalid_omp_num_threads():
    n = perf_bootstrap.configure_cpu_threads(
        env={"OMP_NUM_THREADS": "garbage"}, cpu_count=32
    )

    assert n == 16
    assert torch.get_num_threads() == 16


def test_configure_cpu_threads_falls_back_on_non_positive_omp_num_threads():
    n = perf_bootstrap.configure_cpu_threads(env={"OMP_NUM_THREADS": "0"}, cpu_count=32)

    assert n == 16
    assert torch.get_num_threads() == 16


@pytest.mark.parametrize("filename", ENTRYPOINT_FILES)
def test_entrypoint_imports_perf_bootstrap_before_torch_or_gradio(filename):
    source_lines = (REPO_ROOT / filename).read_text().splitlines()

    bootstrap_line_idx = next(
        (i for i, line in enumerate(source_lines) if line.strip() == "import perf_bootstrap"
         or line.strip().startswith("import perf_bootstrap  #")),
        None,
    )
    guard_line_idx = next(
        (i for i, line in enumerate(source_lines) if _GUARD_PATTERN.match(line)), None
    )

    assert bootstrap_line_idx is not None, f"{filename} does not import perf_bootstrap"
    assert guard_line_idx is not None, f"{filename} has no torch/gradio/irodori_tts import to guard"
    assert bootstrap_line_idx < guard_line_idx, (
        f"{filename}: import perf_bootstrap must precede the first "
        f"torch/gradio/irodori_tts import"
    )
