import importlib

import torch

from irodori_tts.speaker_inversion import (
    SPEAKER_INVERSION_PACING_PROFILE_KEY,
    load_speaker_inversion_metadata,
    save_speaker_inversion_safetensors,
)

extract_reference_speaker_embedding = importlib.import_module(
    "scripts.extract_reference_speaker_embedding"
)
build_extraction_metadata = extract_reference_speaker_embedding.build_extraction_metadata


def test_build_extraction_metadata_omits_pacing_profile_when_none():
    metadata = build_extraction_metadata(
        source_wav="/tmp/ref.wav",
        checkpoint="/tmp/ckpt.pt",
        tokens=32,
        pacing_profile=None,
    )

    assert metadata == {
        "irodori_tts.source_wav": "/tmp/ref.wav",
        "irodori_tts.source_method": "reference-speaker-encoder extraction",
        "irodori_tts.checkpoint": "/tmp/ckpt.pt",
        "irodori_tts.tokens": "32",
    }
    assert SPEAKER_INVERSION_PACING_PROFILE_KEY not in metadata


def test_build_extraction_metadata_includes_pacing_profile_when_set():
    metadata = build_extraction_metadata(
        source_wav="/tmp/ref.wav",
        checkpoint="/tmp/ckpt.pt",
        tokens=32,
        pacing_profile="shuten_fluent_v1",
    )

    assert metadata[SPEAKER_INVERSION_PACING_PROFILE_KEY] == "shuten_fluent_v1"


def test_build_extraction_metadata_omits_pacing_profile_when_blank():
    metadata = build_extraction_metadata(
        source_wav="/tmp/ref.wav",
        checkpoint="/tmp/ckpt.pt",
        tokens=32,
        pacing_profile="   ",
    )

    assert SPEAKER_INVERSION_PACING_PROFILE_KEY not in metadata


def test_pacing_profile_roundtrips_through_safetensors_metadata(tmp_path):
    output = tmp_path / "voice.speaker.safetensors"
    metadata = build_extraction_metadata(
        source_wav="/tmp/ref.wav",
        checkpoint="/tmp/ckpt.pt",
        tokens=32,
        pacing_profile="shuten_fluent_v1",
    )

    save_speaker_inversion_safetensors(
        output,
        {"speaker_embedding": torch.randn(32, 768, dtype=torch.float32)},
        metadata=metadata,
    )

    loaded = load_speaker_inversion_metadata(output)
    assert loaded[SPEAKER_INVERSION_PACING_PROFILE_KEY] == "shuten_fluent_v1"
