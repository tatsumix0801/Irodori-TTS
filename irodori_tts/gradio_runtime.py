from __future__ import annotations

import os

import torch

ROCM_CUDA_OPT_IN_ENV = "IRODORI_TTS_GRADIO_ALLOW_ROCM_CUDA"
_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}


def is_rocm_runtime() -> bool:
    return getattr(torch.version, "hip", None) is not None


def _is_cuda_device(device: str) -> bool:
    return str(device).strip().lower().split(":", 1)[0] == "cuda"


def rocm_cuda_opted_in() -> bool:
    value = os.environ.get(ROCM_CUDA_OPT_IN_ENV, "")
    return value.strip().lower() in _TRUTHY_ENV_VALUES


def gradio_safe_device(device: str) -> str:
    device_text = str(device).strip()
    if _is_cuda_device(device_text) and is_rocm_runtime() and not rocm_cuda_opted_in():
        return "cpu"
    return device_text


def gradio_device_note(kind: str, requested: str, effective: str) -> str | None:
    requested_text = str(requested).strip()
    effective_text = str(effective).strip()
    if requested_text == effective_text:
        return None
    return (
        f"warning: ROCm {kind} device {requested_text!r} was requested; using "
        f"{effective_text!r} for Gradio inference because ROCm CUDA output can produce "
        f"invalid speech on this environment. Set {ROCM_CUDA_OPT_IN_ENV}=1 to opt in."
    )


def gradio_safe_precision(effective_device: str, precision: str) -> str:
    device_kind = str(effective_device).strip().lower().split(":", 1)[0]
    precision_text = str(precision).strip()
    if precision_text.lower() == "bf16" and device_kind not in ("cuda", "xpu"):
        return "fp32"
    return precision_text


def gradio_precision_note(kind: str, requested: str, effective: str) -> str | None:
    requested_text = str(requested).strip()
    effective_text = str(effective).strip()
    if requested_text == effective_text:
        return None
    return (
        f"warning: {kind} precision {requested_text!r} was requested; using "
        f"{effective_text!r} because bf16 requires a CUDA or XPU device and the "
        f"{kind} device is not CUDA/XPU for this Gradio run."
    )
