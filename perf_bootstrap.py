"""Performance bootstrap for Irodori-TTS entrypoints.

This module MUST be imported before torch (and therefore before
irodori_tts, which imports torch transitively via .model) so that the
ROCm environment variables below are already in the process environment
when torch's ROCm/HIP backend initializes. Once torch has loaded, setting
these variables has no effect.

That ordering requirement is also why this module lives at the repo
root instead of inside the irodori_tts package: irodori_tts/__init__.py
itself imports torch, so anything living inside the package would
already be too late.

Usage: entrypoints (train.py, infer.py, gradio_app.py,
gradio_app_voicedesign.py) should do `import perf_bootstrap` as their
first import after `from __future__ import annotations`, before any
torch/gradio/irodori_tts import. Importing this module has the side
effect of calling apply_env_defaults() immediately.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping

ROCM_ENV_DEFAULTS: dict[str, str] = {
    "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1",
    "MIOPEN_FIND_MODE": "2",
}


def apply_env_defaults(env: MutableMapping[str, str] | None = None) -> dict[str, str]:
    """Set ROCM_ENV_DEFAULTS on env using setdefault semantics.

    Never overrides a value the caller/user already set. Returns a dict
    of only the keys this call actually set. Does not import torch.
    """
    if env is None:
        env = os.environ

    applied: dict[str, str] = {}
    for key, value in ROCM_ENV_DEFAULTS.items():
        if key not in env:
            env[key] = value
            applied[key] = value
    return applied


def configure_cpu_threads(
    env: Mapping[str, str] | None = None, cpu_count: int | None = None
) -> int:
    """Tune torch's intra-op CPU thread count and return the chosen value.

    Honors OMP_NUM_THREADS from env if it is a valid positive integer.
    Otherwise falls back to max(1, cpu_count // 2) (physical cores),
    where cpu_count defaults to os.cpu_count().
    """
    import torch

    if env is None:
        env = os.environ

    n: int | None = None
    raw = env.get("OMP_NUM_THREADS")
    if raw is not None:
        try:
            candidate = int(raw)
        except ValueError:
            candidate = None
        if candidate is not None and candidate > 0:
            n = candidate

    if n is None:
        if cpu_count is None:
            cpu_count = os.cpu_count() or 1
        n = max(1, cpu_count // 2)

    torch.set_num_threads(n)
    return n


apply_env_defaults()
