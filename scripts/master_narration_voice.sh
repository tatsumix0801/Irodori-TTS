#!/usr/bin/env bash
# Mastering chain for Irodori-TTS narration renders (narration-serious01 tuned, 2026-07-13, v2).
# Fixes the codec's dark timbre: presence shelf + exciter + isolated air-band (>10kHz)
# synthesis + compression + EBU R128 loudness normalization -- WITHOUT amplifying the
# noise floor in silent gaps (v1 raised the silence floor +5.4..+6.8dB and the >10kHz
# air band by +26..35dB; see docs/harness noise diagnosis for the full writeup).
#
# v2 changes vs v1:
#   1. Front-loaded downward expander (agate) right after the rumble highpass, BEFORE any
#      boost/exciter stage. Fast attack (3ms) + fast release (15ms) so it clamps shut well
#      before intra-gap breath noise (a slow ~250-300ms release leaves the gate mostly open
#      during the first ~40ms after speech ends, which is exactly where breath energy lives --
#      measured empirically, not a guess). Threshold/ratio tuned so real word onsets are only
#      shaved by a couple of ms of pre-voice buildup (verified sample-accurate against raw).
#   2. Two-pass LINEAR loudnorm (measure pass with print_format=json, then a second pass with
#      measured_I/TP/LRA/thresh + linear=true) instead of single-pass dynamic loudnorm. Dynamic
#      mode's frame-adaptive gain automation was the single biggest broadband lift source
#      (+4-7dB) because it over-boosts quiet passages non-uniformly; linear mode applies one
#      constant gain for the whole file, so gaps get exactly the same dB as everything else
#      (no pumping) -- this constant gain still lifts the floor a little (~3dB, unavoidable
#      given the -16 LUFS target vs natural loudness), which is why front-gate margin matters.
#   3. Speech-band shaping (mud cut, treble shelf +2@6k, serial exciter, presence bump, parallel
#      air path >10kHz at vol 0.6, compressor) is unchanged from v1 -- the air path's own
#      exciter still produces a noise-independent near-zero-input floor (an exciter driven this
#      hard behaves close to a fixed ~drive gain for tiny inputs), but the front gate now
#      starves it enough that after the air mix + compressor + loudnorm the >10kHz band lands
#      well under -70dBFS in gaps.
# Calibrated against the narration-serious01 reference profile:
#   target 6-10kHz energy ratio ~3-5% (was 0.7% raw), 10-16kHz air ~0.2-0.5% (was 0.0),
#   spectral tilt (1-8kHz) ~ -3.4 dB/oct (was -6.8). Watermark verified to survive
#   (SilentCipher payload decode conf ~0.9 post-mastering).
# Measured (fixed/char samples, dBFS): silence floor raw -48.6/-54.3 -> mastered -51.8/-86.0
#   (was -41.8/-48.9 in v1); 10-16kHz band -88.5/-83.9 -> -80.9/-98.4 (was -61.4/n.m. in v1,
#   both now under the -70dBFS gap target); SNR 36.4/41.4 -> 39.6/73.3 (raw was the floor).
#
# Usage: master_narration_voice.sh <in.wav> <out.wav> [treble_gain_db=2] [air_mix=0.6]
set -euo pipefail

IN="${1:?usage: master_narration_voice.sh <in.wav> <out.wav> [treble_db] [air_mix]}"
OUT="${2:?usage: master_narration_voice.sh <in.wav> <out.wav> [treble_db] [air_mix]}"
TREBLE="${3:-2}"
AIRMIX="${4:-0.6}"

GATE="highpass=f=55, agate=threshold=-30dB:ratio=10:attack=3:release=15:range=-40dB:knee=1"

FILTER_PRE="[0:a]${GATE}, equalizer=f=320:t=q:w=1.3:g=-1.5, treble=g=${TREBLE}:f=6000:t=s, \
aexciter=amount=1.2:drive=9:blend=0:freq=7500:ceil=16500, equalizer=f=3200:t=q:w=1.6:g=1[pre]; \
[pre]asplit=2[main][exc]; \
[exc]aexciter=amount=8:drive=9.5:blend=0:freq=5200:ceil=18000, \
highpass=f=10000, highpass=f=10000, highpass=f=10000, highpass=f=10000, volume=${AIRMIX}[air]; \
[main][air]amix=inputs=2:normalize=0, \
acompressor=threshold=-20dB:ratio=2.2:attack=8:release=140:makeup=1.5[precomp]"

TMPLOG="$(mktemp)"
trap 'rm -f "$TMPLOG"' EXIT

# Pass 1: measure loudnorm stats on the fully pre-processed (gated/EQ'd/exciter'd/compressed)
# signal -- must use the identical filter chain as pass 2 so the measurement matches.
ffmpeg -y -loglevel info -i "$IN" -filter_complex \
"${FILTER_PRE}; [precomp]loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json[out]" \
-map "[out]" -f null - 2>"$TMPLOG"

STATS="$(grep -A20 "Parsed_loudnorm" "$TMPLOG" | tail -12)"
MEASURED_I="$(echo "$STATS" | grep '"input_i"' | grep -oE '\-?[0-9]+\.[0-9]+' | head -1)"
MEASURED_TP="$(echo "$STATS" | grep '"input_tp"' | grep -oE '\-?[0-9]+\.[0-9]+' | head -1)"
MEASURED_LRA="$(echo "$STATS" | grep '"input_lra"' | grep -oE '\-?[0-9]+\.[0-9]+' | head -1)"
MEASURED_THRESH="$(echo "$STATS" | grep '"input_thresh"' | grep -oE '\-?[0-9]+\.[0-9]+' | head -1)"
TARGET_OFFSET="$(echo "$STATS" | grep '"target_offset"' | grep -oE '\-?[0-9]+\.[0-9]+' | head -1)"

if [[ -z "$MEASURED_I" || -z "$MEASURED_TP" || -z "$MEASURED_LRA" || -z "$MEASURED_THRESH" ]]; then
  echo "[master_narration_voice] loudnorm pass 1 failed to produce measured stats" >&2
  exit 1
fi

# Pass 2: apply the same chain with linear=true and the measured stats -- a single static
# gain for the whole file, so silence gaps get the same dB as speech (no dynamic pumping).
ffmpeg -y -loglevel error -i "$IN" -filter_complex \
"${FILTER_PRE}; [precomp]loudnorm=I=-16:TP=-1.5:LRA=11:measured_I=${MEASURED_I}:measured_TP=${MEASURED_TP}:measured_LRA=${MEASURED_LRA}:measured_thresh=${MEASURED_THRESH}:offset=${TARGET_OFFSET}:linear=true[out]" \
-map "[out]" -ar 48000 "$OUT"

# Pass 3: re-embed the SilentCipher watermark. The front gate silences the gaps and with
# them the original watermark carriers, which breaks payload decoding (measured: char
# sample lost decode entirely, fixed dropped to conf 0.64). Re-encoding on the mastered
# signal restores full-confidence decoding without undoing the gap cleanup.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
(cd "${SCRIPT_DIR}/.." && TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 MIOPEN_FIND_MODE=2 \
  uv run --no-sync python scripts/rewatermark.py "$OUT" "$OUT")

echo "[mastered] $OUT"
