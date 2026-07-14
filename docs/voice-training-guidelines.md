# Irodori-TTS Voice Training Guidelines

Last updated: 2026-06-20

このドキュメントは、新規ボイスを Speaker Inversion などで学習するときの運用メモ。
特に ROCm 環境で Gradio / GPU 推論が入力テキストと無関係な音声を出す問題を踏まえ、
学習失敗と推論経路の不具合を混同しないための確認手順をまとめる。

## 結論

- 学習は GPU を使ってよい。
- 品質判定と最終 QA は CPU 推論を基準にする。
- ROCm 環境の `cuda` 推論で、入力 `text` と無関係な日本語や英語断片が出る場合がある。
- その症状は、まず学習データや学習済みボイスではなく、推論デバイス経路を疑う。
- Gradio 版は ROCm では既定で CPU に退避する。`IRODORI_TTS_GRADIO_ALLOW_ROCM_CUDA=1` は検証用だけにする。

## 事前ゲート

新規ボイス作業を始める前に、以下を確認する。

- 音声素材の権利、同意、利用範囲が明確である。
- 個人利用、非公開、非配布、非商用、なりすまし不可などの制約を README や build note に残す。
- 生成音声の SilentCipher 透かしを無効化しない。
- 出力先の名前が既存ボイスと衝突しない。
- `voice_build/`、`outputs/speaker_inversion/<name>/`、`gradio_outputs/` などの生成物を誤ってコミットしない。

## データ準備

入力音声は、学習品質の上限を決める。ここが荒いと後工程では直しにくい。

- 48 kHz mono を基本にする。
- 対象話者だけの音声にする。
- BGM、強いノイズ、エコー、別話者の被り、長い無音を除く。
- セリフと transcript が対応していることを確認する。
- 感情強めボイスでも、叫び、笑い、息、相づちだけに偏らせない。
- 普通の読み上げ、短い会話、感情表現を混ぜる。
- 採用尺は最低 180 秒を目安にする。
- 4 分前後でも動くが薄い。安定させるなら 8-10 分以上を目標にする。

Speaker Inversion でも manifest の `text` は学習で使われる。
「声だけ良ければ transcript は雑でよい」とは扱わない。

## Curation

長い単一話者 WAV からクリップと仮 manifest を作る例。

```bash
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 MIOPEN_FIND_MODE=2 \
uv run --no-sync python scripts/segment_and_transcribe.py \
  --input voice_build/<name>_source_48k_mono.wav \
  --clips-dir voice_build/<name>_clips \
  --out-manifest voice_build/<name>_pre_manifest.jsonl \
  --prefer-expressive
```

確認すること:

- kept duration が 180 秒以上ある。
- クリップ数が極端に少なくない。
- transcript の一部を目視し、対象話者の内容として読める。
- 別話者、音楽、環境音だけ、Whisper の幻覚、同じ文言の反復を取り除く。
- 低音量素材では silence threshold を調整した場合、その理由を build note に残す。

## Latent Manifest

`train.py` が読む manifest は、各行に `text` と `latent_path` が必要。
`segment_and_transcribe.py` の出力は `audio` と `text` の仮 manifest なので、
`prepare_manifest.py` で DACVAE latent 付き manifest に変換する。

```bash
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 MIOPEN_FIND_MODE=2 \
uv run --no-sync python prepare_manifest.py \
  --dataset json \
  --data-files voice_build/<name>_pre_manifest.jsonl \
  --audio-column audio \
  --text-column text \
  --output-manifest voice_build/<name>_manifest.jsonl \
  --latent-dir voice_build/<name>_latents \
  --device cuda
```

確認すること:

- skip が高すぎない。目安として 5% 未満。
- 出力 manifest の行数が期待通り。
- `latent_path` の実体が存在する。
- `text` が空、文字化け、別言語だらけになっていない。

## Speaker Inversion Training

ROCm 環境では数値不安定が出ることがあるため、既存の成功レシピを基準にする。

推奨初期値:

- `speaker_inversion_tokens`: 32
- `max_steps`: 1500
- `save_every`: 250
- `learning_rate`: 0.0001
- `precision`: fp32
- `init-checkpoint`: `Irodori-TTS-500M-v3.safetensors`

例:

```bash
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 MIOPEN_FIND_MODE=2 \
uv run --no-sync python train.py \
  --config configs/train_500m_v3_speaker_inversion.yaml \
  --manifest voice_build/<name>_manifest.jsonl \
  --init-checkpoint path/to/Irodori-TTS-500M-v3.safetensors \
  --output-dir outputs/speaker_inversion/<name>_t32_s1500 \
  --speaker-inversion-tokens 32 \
  --max-steps 1500 \
  --save-every 250 \
  --precision fp32 \
  --lr 0.0001
```

学習中に見ること:

- loss がノイジーでも全体として下がる。
- `checkpoint_0000250.speaker.safetensors` などが定期保存される。
- `checkpoint_final.speaker.safetensors` が出る。
- NaN / Inf が出たら、bf16 や高すぎる learning rate を疑う。

## Checkpoint QA

最終 `.speaker.safetensors` は最低限ロード検査する。

確認すること:

- `speaker_embedding` が存在する。
- shape が `(32, 768)` である。
- dtype が `float32` である。
- NaN / Inf がない。
- RMS が極端に 0 へ潰れていない。

## Inference QA

今回の事故を避けるため、最終判定は CPU 推論で行う。

固定文:

```text
こんにちは、私はAIです。これは音声合成のテストです。
```

CPU 推論例:

```bash
uv run --no-sync python infer.py \
  --checkpoint path/to/Irodori-TTS-500M-v3.safetensors \
  --ref-embed outputs/speaker_inversion/<name>_final/<name>.speaker.safetensors \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --output-wav voice_build/<name>_sample_cpu.wav \
  --model-device cpu \
  --codec-device cpu
```

ASR 確認例:

```bash
/home/motoki/tools/whisper/bin/whisper \
  voice_build/<name>_sample_cpu.wav \
  --model faster-whisper-large-v3-turbo \
  --language ja
```

合格ライン:

- 音声が無音ではない。
- 秒数が入力文に対して極端に短くない。
- Whisper 結果が入力文と概ね一致する。
- SilentCipher 透かしが有効である。
- 生成文が別セリフ、英語、謎言語になっていない。

GPU 推論も参考として試してよいが、ROCm では合否判定に使わない。
GPU だけで別セリフになる場合、まず推論経路の問題として扱う。

## Gradio QA

Gradio 版は ROCm 環境で `cuda` を要求されても、既定では `cpu` に退避する。

確認すること:

- Gradio 起動後、Model Device と Codec Device が `cpu` になっている。
- Run Log に ROCm `cuda` から `cpu` へ退避した警告が出ても、それは想定内。
- 古いサーバープロセスが残っている場合、修正前コードのまま動き続けるので再起動する。
- `IRODORI_TTS_GRADIO_ALLOW_ROCM_CUDA=1` を設定した状態で品質判定しない。

症状別の見方:

- 入力文と無関係な日本語を喋る: 推論デバイス経路を確認する。
- 一部だけ日本語で一部が謎言語: 推論デバイス経路を確認する。
- `Thank you.` など英語断片になる: 推論デバイス経路を確認する。
- predicted duration が極端に短い: duration predictor か GPU 経路の異常を疑う。
- CPU 推論でも同じ症状: データ、manifest、checkpoint、学習設定を疑う。

## Build Note に残す項目

各ボイスごとに、最低限これを記録する。

- source wav の由来、利用制約、サンプルレート、チャンネル、尺。
- curation コマンドと kept clips / kept duration。
- transcript spot-check の結果。
- latent manifest の行数と skip 数。
- training recipe: tokens, steps, lr, precision, seed, config, init checkpoint。
- checkpoint 統計: shape, dtype, NaN/Inf, RMS。
- CPU 推論サンプルのパス、秒数、RMS、ASR 結果。
- SilentCipher 透かし確認結果。
- caveat: データが薄い、音量が低い、感情表現が偏る、別話者残りの可能性など。

## 判断ルール

- CPU 推論で固定文が読めないものは不合格。
- GPU 推論だけ壊れるものは、ボイス不合格ではなく ROCm 推論経路の既知リスクとして扱う。
- データ尺が 180 秒未満なら、学習前に停止して素材追加を検討する。
- 別話者混入が明確なら、学習前に除去する。
- 学習が NaN 化したら、bf16 をやめて fp32、learning rate を下げる、checkpoint から再試行する。
- 品質改善の優先順位は、データ追加、curation 改善、steps/tokens 調整、LoRA 併用の順で考える。
