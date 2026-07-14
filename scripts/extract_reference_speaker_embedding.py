#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
import torchaudio

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from irodori_tts.codec import patchify_latent  # noqa: E402
from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey  # noqa: E402
from irodori_tts.model import patch_sequence_with_mask  # noqa: E402
from irodori_tts.speaker_inversion import (  # noqa: E402
    SPEAKER_INVERSION_PACING_PROFILE_KEY,
    save_speaker_inversion_safetensors,
)


def _load_wav(path: Path) -> tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(str(path))
    if wav.ndim != 2 or wav.numel() == 0:
        raise ValueError(f"Unexpected waveform shape for {path}: {tuple(wav.shape)}")
    if wav.shape[0] != 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.float(), int(sr)


def _time_pool_tokens(tokens: torch.Tensor, target_tokens: int) -> torch.Tensor:
    if tokens.ndim != 2:
        raise ValueError(f"Expected tokens shape (T,D), got {tuple(tokens.shape)}")
    if int(target_tokens) <= 0:
        raise ValueError(f"target_tokens must be > 0, got {target_tokens}")
    if tokens.shape[0] == target_tokens:
        return tokens

    pooled: list[torch.Tensor] = []
    total = int(tokens.shape[0])
    for i in range(int(target_tokens)):
        start = math.floor(i * total / target_tokens)
        end = math.floor((i + 1) * total / target_tokens)
        if end <= start:
            end = min(total, start + 1)
        pooled.append(tokens[start:end].mean(dim=0))
    return torch.stack(pooled, dim=0)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_extraction_metadata(
    *,
    source_wav: str,
    checkpoint: str,
    tokens: int,
    pacing_profile: str | None,
) -> dict[str, str]:
    metadata = {
        "irodori_tts.source_wav": str(source_wav),
        "irodori_tts.source_method": "reference-speaker-encoder extraction",
        "irodori_tts.checkpoint": str(checkpoint),
        "irodori_tts.tokens": str(int(tokens)),
    }
    if pacing_profile is not None and pacing_profile.strip():
        metadata[SPEAKER_INVERSION_PACING_PROFILE_KEY] = pacing_profile.strip()
    return metadata


def extract_embedding(args: argparse.Namespace) -> dict[str, object]:
    checkpoint = Path(args.checkpoint).expanduser()
    source_wav = Path(args.source_wav).expanduser().resolve()
    output = Path(args.output).expanduser()

    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=str(checkpoint),
            model_device=args.model_device,
            model_precision=args.model_precision,
            codec_repo=args.codec_repo,
            codec_device=args.codec_device,
            codec_precision=args.codec_precision,
            codec_deterministic_encode=True,
            codec_deterministic_decode=True,
            compile_model=False,
            compile_dynamic=False,
        )
    )
    runtime.model.eval()

    wav, sr = _load_wav(source_wav)
    runtime_dtype = next(runtime.model.parameters()).dtype

    with torch.inference_mode():
        ref_latent = runtime.codec.encode_waveform(
            wav.unsqueeze(0),
            sample_rate=sr,
            normalize_db=None if args.no_normalize else float(args.ref_normalize_db),
            ensure_max=True,
        ).cpu()
        ref_latent_patched = patchify_latent(ref_latent, runtime.model_cfg.latent_patch_size).to(
            device=runtime.model_device,
            dtype=runtime_dtype,
        )
        ref_mask = torch.ones(
            (1, ref_latent_patched.shape[1]),
            dtype=torch.bool,
            device=runtime.model_device,
        )
        ref_latent_patched, ref_mask = patch_sequence_with_mask(
            seq=ref_latent_patched,
            mask=ref_mask,
            patch_size=runtime.model_cfg.speaker_patch_size,
        )
        ref_state = runtime.model.speaker_encoder(ref_latent_patched, ref_mask)
        ref_state = runtime.model.speaker_norm(ref_state)
        ref_state, ref_mask = runtime.model._prepend_masked_mean_token(ref_state, ref_mask)
        tokens = ref_state[0][ref_mask[0]].detach().cpu().float()
        embedding = _time_pool_tokens(tokens, int(args.tokens)).contiguous()

    metadata = build_extraction_metadata(
        source_wav=str(source_wav),
        checkpoint=str(checkpoint),
        tokens=int(args.tokens),
        pacing_profile=args.pacing_profile,
    )
    save_speaker_inversion_safetensors(
        output,
        {"speaker_embedding": embedding},
        metadata=metadata,
    )

    stats = {
        "source_wav": str(source_wav),
        "source_sample_rate": sr,
        "source_samples": int(wav.shape[-1]),
        "source_seconds": float(wav.shape[-1]) / float(sr),
        "checkpoint": str(checkpoint),
        "method": "reference-speaker-encoder extraction, time-pooled to target tokens",
        "pacing_profile": args.pacing_profile,
        "speaker_patch_size": int(runtime.model_cfg.speaker_patch_size),
        "latent_patch_size": int(runtime.model_cfg.latent_patch_size),
        "latent_steps": int(ref_latent.shape[1]),
        "full_reference_tokens": int(tokens.shape[0]),
        "shape": list(embedding.shape),
        "dtype": str(embedding.dtype).replace("torch.", ""),
        "finite": bool(torch.isfinite(embedding).all().item()),
        "rms": float(torch.sqrt((embedding * embedding).mean()).item()),
        "absmean": float(embedding.abs().mean().item()),
        "sha256": _sha256(output),
    }
    stats["sha256_prefix"] = str(stats["sha256"])[:16]

    if args.stats_json:
        stats_path = Path(args.stats_json).expanduser()
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a Speaker Inversion-compatible embedding from reference wav."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-wav", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats-json", default=None)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument(
        "--pacing-profile",
        default=None,
        help="Speaker embeddingのmetadataにirodori_tts.pacing_profileとして書き込む値(未指定時は書き込まない)",
    )
    parser.add_argument("--model-device", default="cpu")
    parser.add_argument("--model-precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--codec-repo", default="Aratako/Semantic-DACVAE-Japanese-32dim")
    parser.add_argument("--codec-device", default="cpu")
    parser.add_argument("--codec-precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--ref-normalize-db", type=float, default=-16.0)
    parser.add_argument("--no-normalize", action="store_true")
    args = parser.parse_args()

    stats = extract_embedding(args)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
