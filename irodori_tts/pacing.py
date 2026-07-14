from __future__ import annotations

import re

import torch

INTERNAL_PERIOD_PATTERN = re.compile(r"。\s*(?=\S)")


def smooth_internal_periods(text: str) -> str:
    return INTERNAL_PERIOD_PATTERN.sub("、", text)


def compress_long_internal_silences(
    audio: torch.Tensor,
    *,
    sample_rate: int,
    silence_threshold: float = 0.01,
    min_silence_seconds: float = 0.45,
    target_silence_seconds: float = 0.28,
) -> torch.Tensor:
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be > 0, got {sample_rate}")
    if min_silence_seconds <= 0:
        raise ValueError(f"min_silence_seconds must be > 0, got {min_silence_seconds}")
    if target_silence_seconds < 0:
        raise ValueError(f"target_silence_seconds must be >= 0, got {target_silence_seconds}")
    if silence_threshold < 0:
        raise ValueError(f"silence_threshold must be >= 0, got {silence_threshold}")

    squeeze = False
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
        squeeze = True
    if audio.ndim != 2:
        raise ValueError(f"audio must have shape (channels, samples), got {tuple(audio.shape)}")

    total_samples = int(audio.shape[1])
    min_silence_samples = max(1, int(round(float(min_silence_seconds) * int(sample_rate))))
    target_silence_samples = int(round(float(target_silence_seconds) * int(sample_rate)))
    if total_samples <= 0 or target_silence_samples >= min_silence_samples:
        return audio.squeeze(0) if squeeze else audio

    silent = audio.detach().abs().amax(dim=0) <= float(silence_threshold)
    segments: list[torch.Tensor] = []
    cursor = 0
    idx = 0
    while idx < total_samples:
        if not bool(silent[idx]):
            idx += 1
            continue

        start = idx
        while idx < total_samples and bool(silent[idx]):
            idx += 1
        end = idx
        run_len = end - start
        is_internal = start > 0 and end < total_samples
        should_compress = is_internal and run_len >= min_silence_samples
        if should_compress:
            if cursor < start:
                segments.append(audio[:, cursor:start])
            if target_silence_samples > 0:
                segments.append(audio[:, start : start + target_silence_samples])
            cursor = end

    if cursor == 0:
        return audio.squeeze(0) if squeeze else audio
    if cursor < total_samples:
        segments.append(audio[:, cursor:])
    compressed = torch.cat(segments, dim=1) if segments else audio[:, :0]
    return compressed.squeeze(0) if squeeze else compressed
