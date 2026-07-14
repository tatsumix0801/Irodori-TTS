#!/usr/bin/env python3
"""Re-embed the IRODORI SilentCipher watermark into a wav (e.g. after mastering).

Mastering with a noise gate strips the watermark carriers in silent gaps and can
break payload decoding (observed 2026-07-13: gated master lost decode on one of
two samples). Re-encoding after mastering restores a full-confidence watermark.

Usage: rewatermark.py <in.wav> <out.wav>   (in == out is allowed)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torchaudio

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from irodori_tts.watermark import SilentCipherWatermarker  # noqa: E402


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    in_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])

    wav, sr = torchaudio.load(str(in_path))
    if wav.shape[0] != 1:
        wav = wav.mean(dim=0, keepdim=True)

    encoder = SilentCipherWatermarker(device="cpu")
    encoded = encoder.encode_one(wav.squeeze(0), sample_rate=sr)
    if encoded.dim() == 1:
        encoded = encoded.unsqueeze(0)
    peak = float(encoded.abs().max())
    if peak > 1.0:
        encoded = encoded / peak * 0.999

    torchaudio.save(str(out_path), encoded.to(torch.float32), sr)
    print(f"[rewatermarked] {out_path}")


if __name__ == "__main__":
    main()
