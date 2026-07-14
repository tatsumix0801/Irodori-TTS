import torch
from safetensors.torch import save_file

from irodori_tts.pacing import compress_long_internal_silences, smooth_internal_periods
from irodori_tts.speaker_inversion import (
    SPEAKER_INVERSION_PACING_PROFILE_KEY,
    load_speaker_inversion_metadata,
)


def test_smooth_internal_periods_keeps_final_period():
    text = "今日は短いテストです。次の文章を読みます。よろしくお願いします。"

    assert smooth_internal_periods(text) == (
        "今日は短いテストです、次の文章を読みます、よろしくお願いします。"
    )


def test_compress_long_internal_silences_preserves_edges():
    sample_rate = 1000
    tone = torch.ones(1, 100)
    long_silence = torch.zeros(1, 1000)
    edge_silence = torch.zeros(1, 300)
    audio = torch.cat([edge_silence, tone, long_silence, tone, edge_silence], dim=1)

    compressed = compress_long_internal_silences(
        audio,
        sample_rate=sample_rate,
        silence_threshold=0.01,
        min_silence_seconds=0.45,
        target_silence_seconds=0.28,
    )

    assert compressed.shape == (1, 300 + 100 + 280 + 100 + 300)


def test_load_speaker_inversion_metadata_reads_pacing_profile(tmp_path):
    path = tmp_path / "voice.speaker.safetensors"
    save_file(
        {"speaker_embedding": torch.zeros(2, 3)},
        str(path),
        metadata={SPEAKER_INVERSION_PACING_PROFILE_KEY: "shuten_fluent_v1"},
    )

    assert load_speaker_inversion_metadata(path) == {
        SPEAKER_INVERSION_PACING_PROFILE_KEY: "shuten_fluent_v1"
    }
