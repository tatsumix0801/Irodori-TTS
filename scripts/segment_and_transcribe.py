#!/usr/bin/env python3
"""Segment a long single-speaker WAV into clips, transcribe with Whisper, and emit a JSONL manifest."""

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Input WAV file")
    p.add_argument("--clips-dir", required=True, help="Output directory for clips")
    p.add_argument("--out-manifest", required=True, help="Output JSONL manifest path")
    p.add_argument("--prefer-expressive", action="store_true", help="Score and filter for expressiveness")
    p.add_argument("--min-silence-len", type=int, default=700, help="Minimum silence length in ms (default: 700)")
    p.add_argument("--silence-thresh", type=int, default=-55, help="Silence threshold in dBFS (default: -55)")
    p.add_argument("--min-clip-sec", type=float, default=0.5, help="Minimum clip duration in seconds (default: 0.5)")
    p.add_argument("--max-clip-sec", type=float, default=15.0, help="Maximum clip duration in seconds (default: 15.0)")
    p.add_argument("--whisper-model", default="openai/whisper-large-v3", help="HF Whisper model ID")
    p.add_argument("--keep-floor-sec", type=float, default=180.0, help="Minimum total kept duration in seconds (default: 180)")
    return p.parse_args()


def log(msg: str):
    print(msg, file=sys.stderr)


def compute_expressiveness(audio_path: str) -> tuple[float, float, float]:
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    if len(y) < sr * 0.1:
        return (0.0, 0.0, len(y) / sr)

    f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr)
    voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
    f0_var = float(np.var(voiced_f0)) if len(voiced_f0) > 1 else 0.0

    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rms_var = float(np.var(rms)) if len(rms) > 1 else 0.0

    return (f0_var, rms_var, len(y) / sr)


def main():
    args = parse_args()

    input_path = Path(args.input).resolve()
    clips_dir = Path(args.clips_dir).resolve()
    manifest_path = Path(args.out_manifest).resolve()

    clips_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    log(f"Loading audio: {input_path}")
    audio = AudioSegment.from_wav(str(input_path))
    sample_rate = audio.frame_rate
    log(f"  Duration: {len(audio)/1000:.1f}s  Sample rate: {sample_rate}Hz  Channels: {audio.channels}  dBFS: {audio.dBFS:.1f}")

    log(f"Detecting non-silent segments (thresh={args.silence_thresh} dBFS, min_silence={args.min_silence_len}ms)...")
    ranges = detect_nonsilent(audio, min_silence_len=args.min_silence_len, silence_thresh=args.silence_thresh)
    log(f"  Found {len(ranges)} non-silent segments")

    min_ms = int(args.min_clip_sec * 1000)
    max_ms = int(args.max_clip_sec * 1000)

    clip_ranges = []
    for start_ms, end_ms in ranges:
        if end_ms - start_ms < min_ms:
            continue
        offset = start_ms
        while offset < end_ms:
            chunk_end = min(offset + max_ms, end_ms)
            if chunk_end - offset < min_ms:
                break
            clip_ranges.append((offset, chunk_end))
            offset = chunk_end

    log(f"  {len(clip_ranges)} clips after duration filtering")

    log("Exporting clips...")
    exported = []
    for i, (start_ms, end_ms) in enumerate(clip_ranges):
        clip_path = clips_dir / f"clip_{i:05d}.wav"
        segment = audio[start_ms:end_ms].set_channels(1).set_frame_rate(sample_rate)
        segment.export(str(clip_path), format="wav")
        exported.append((str(clip_path), (end_ms - start_ms) / 1000.0))
    log(f"  Exported {len(exported)} clips to {clips_dir}")

    log(f"Loading Whisper model: {args.whisper_model}")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.whisper_model, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to("cuda")
    processor = AutoProcessor.from_pretrained(args.whisper_model)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.float16,
        device="cuda",
    )

    log("Transcribing clips...")
    records = []
    for i, (clip_path, duration_sec) in enumerate(exported):
        if i % 50 == 0:
            log(f"  Transcribing {i}/{len(exported)}...")

        result = pipe(clip_path, generate_kwargs={"language": "japanese", "task": "transcribe"})
        text = result["text"].strip()

        if len(text) < 2:
            continue

        records.append({"audio": clip_path, "text": text, "duration": duration_sec})

    log(f"  Kept {len(records)} clips after quality filtering")

    if args.prefer_expressive and records:
        log("Computing expressiveness scores...")
        raw_scores = []
        for j, rec in enumerate(records):
            if j % 50 == 0:
                log(f"  Scoring {j}/{len(records)}...")
            raw_scores.append(compute_expressiveness(rec["audio"]))

        arr_f0 = np.array([s[0] for s in raw_scores], dtype=float)
        arr_rms = np.array([s[1] for s in raw_scores], dtype=float)
        arr_dur = np.array([s[2] for s in raw_scores], dtype=float)

        def normalize(a):
            mn, mx = a.min(), a.max()
            return (a - mn) / (mx - mn + 1e-9)

        norm_f0 = normalize(arr_f0)
        norm_rms = normalize(arr_rms)
        dur_score = 1.0 - normalize(np.abs(arr_dur - 5.0))

        scores = 0.5 * norm_f0 + 0.3 * norm_rms + 0.2 * dur_score

        for rec, score in zip(records, scores, strict=False):
            rec["expr_score"] = float(score)

        threshold = 0.5
        kept = [r for r in records if r.get("expr_score", 0.0) >= threshold]
        total_kept_dur = sum(r["duration"] for r in kept)

        if total_kept_dur < args.keep_floor_sec:
            log(f"  Below keep-floor ({total_kept_dur:.1f}s < {args.keep_floor_sec}s), lowering threshold...")
            sorted_recs = sorted(records, key=lambda r: r.get("expr_score", 0.0), reverse=True)
            kept = []
            cumulative = 0.0
            for rec in sorted_recs:
                kept.append(rec)
                cumulative += rec["duration"]
                if cumulative >= args.keep_floor_sec:
                    break

        records = kept
        log(f"  {len(records)} clips after expressiveness filtering")

    total_dur = sum(dur for _, dur in exported)
    kept_dur = sum(r["duration"] for r in records)

    log("Writing manifest...")
    with open(str(manifest_path), "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps({"audio": rec["audio"], "text": rec["text"]}, ensure_ascii=False) + "\n")

    log("")
    log("=== Summary ===")
    log(f"  Total clips extracted : {len(exported)}")
    log(f"  Total duration        : {total_dur:.1f}s")
    log(f"  Kept clips            : {len(records)}")
    log(f"  Kept duration         : {kept_dur:.1f}s")
    log(f"  Manifest              : {manifest_path}")


if __name__ == "__main__":
    main()
