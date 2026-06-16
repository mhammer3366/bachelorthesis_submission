# Vorarlberg Dialect TTS — Bachelor Thesis Code Appendix

**Scripts-only** reproduction code for the bachelor thesis *A Data-Driven Pipeline for Alemannic Dialect TTS with a Focus on Vorarlberg*. The pipeline collects Swiss and Vorarlberg dialect speech, builds merged classification datasets, trains dialect classifiers, benchmarks ASR, evaluates Vorarlberg and SRF corpora, and fine-tunes Chatterbox (Kartoffelbox) TTS on Vorarlberg speech.

> This repository contains **no training datasets or model checkpoints**. It **does** include **50 demo synthesis WAVs** (25 sentences × base vs fine-tuned) under [`tts/samples/`](tts/samples/) so reviewers can listen without re-running GPU inference. All other scripts expect corpora and trained weights on disk via environment variables.

See also [REPO_OVERVIEW.md](REPO_OVERVIEW.md) for a folder-by-folder map.

---

## What this repo contains

| Included | Not included |
|----------|--------------|
| Python / shell / YAML scripts for every thesis chapter | Merged TSVs, feature caches (~164 GB), `.npy` / `.pt` weights |
| Chatterbox fine-tuning code under `tts/src/` + `tts/thesis_eval/` | TTS checkpoints, TensorBoard logs |
| **Demo synthesis WAVs** (`tts/samples/`, 50 files, ~11 MB) | Full Vorarlberg / Swiss corpora |
| `thesis_eval/scripts/` (loss curves, val loss, synthesis eval) | `clip_labels.csv` (~3 GB), Label Studio exports |
| Verification scripts (`scripts/verify_thesis_numbers.py`) | Thesis PDF (`main-thesis.pdf` — author machine only) |

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python 3.11+** | 3.10 may work; tested with 3.11 |
| **[uv](https://github.com/astral-sh/uv)** | Dependency management (`uv sync`) |
| **ffmpeg / ffprobe** | Audio probing, resampling, MP3→WAV |
| **CUDA** (optional) | Classification training, ASR, TTS fine-tuning |
| **HuggingFace token** | Required for pyannote diarization (`HF_TOKEN` env var) |
| **Playwright** | Only if running SRF scrapers: `uv run playwright install chromium` |

For TTS fine-tuning, also install the Chatterbox sub-project:

```bash
cd tts
uv sync
```

---

## Environment setup

Clone the repo and install dependencies:

```bash
git clone <repo-url> bachelorthesis_submission
cd bachelorthesis_submission
uv sync
```

Set path variables before running any pipeline step:

```bash
export THESIS_ROOT="$(pwd)"                    # repo root (default when unset)
export DATA_ROOT=/path/to/your/datasets          # audio + source TSVs
export MODELS_ROOT=/path/to/your/models          # checkpoints (optional)
export CHECKPOINTS_DIR=$MODELS_ROOT/TTS/chatterbox   # Chatterbox fine-tunes
export CHATTERBOX_ROOT=$THESIS_ROOT/tts/chatterbox-finetuning
export HF_TOKEN=hf_...                           # pyannote + gated HF models
```

Derived paths (override if needed):

```bash
export MERGED_DATA_DIR=$THESIS_ROOT/data_preparation/merged_datasets
export MERGED_DE_DATA_DIR=$THESIS_ROOT/data_preparation/merged_datasets_plus_de
export CLASSIFIER_MODELS_DIR=$THESIS_ROOT/models
```

**How paths work:** Most scripts default `THESIS_ROOT` to the repository root via `Path(__file__).resolve().parents[N]`. Dataset locations default to `$DATA_ROOT/...` but fall back to the author's layout (`/home/ai/AI-DataPool/Datasets`) if unset — **always set `DATA_ROOT` on a fresh machine**. See [docs/PATH_AUDIT.md](docs/PATH_AUDIT.md) for per-script details.

---

## Step-by-step reproduction by chapter

### Chapter 3 — Data collection

**Swiss SRF podcast metadata**

1. Obtain SRF API credentials and store the access token:
   ```bash
   uv run python data_collection/get_access_token.py
   ```
2. Scrape podcast metadata and download URLs:
   ```bash
   uv run python data_collection/srf_webscraper_2.py   # or srf_webscraper.py
   ```
3. Convert downloaded MP3s to WAV (after placing files under `$DATA_ROOT`):
   ```bash
   uv run python data_collection/mp3_to_wav.py
   ```

**Vorarlberg dialect audio**

- Curate podcast/audio folders under `$DATA_ROOT/audio/Vorarlberg/<show>/original/`.
- Print per-show episode counts and hours:
  ```bash
  uv run python data_preparation/merged_datasets/vorarlberg_distribution_print.py
  ```
- Build Vorarlberg phoneme TSV (requires eSpeak phoneme model):
  ```bash
  uv run python dialect_classification/feature_extraction/phoneme_vorarlberg.py
  uv run python dialect_classification/feature_extraction/vorarlberg/add_speaker_and_duration.py
  ```

### Chapter 3 — Data preparation (Swiss merge)

1. Merge SDS-200 + STT4SG-350 (+ optional Common Voice DE) into train/valid/test TSVs:
   ```bash
   uv run python data_preparation/merged_datasets/merge_datasets.py
   uv run python data_preparation/merged_datasets/merge_german.py   # adds German rows
   ```
2. Fix audio paths after moving corpora:
   ```bash
   uv run python data_preparation/merged_datasets/update_path.py
   ```
3. Distribution statistics for thesis tables:
   ```bash
   uv run python data_preparation/merged_datasets/distribution_prints.py
   uv run python data_preparation/merged_datasets/enhanced_distribution_analysis_for_presentation.py
   ```

Outputs land in `$THESIS_ROOT/data_preparation/merged_datasets/` and `merged_datasets_plus_de/` (created on first run).

### Chapter 4 — Feature extraction & classifiers

**Feature extraction** (run after merged TSVs exist):

```bash
uv run python dialect_classification/feature_extraction/1_mel_spectogram.py
uv run python dialect_classification/feature_extraction/2_phoneme.py
uv run python dialect_classification/feature_extraction/3_wav2vec_base_layer6.py
uv run python dialect_classification/feature_extraction/4_xlsr_300m_layer8.py
```

**Training** (7-way Swiss + binary German/dialect):

```bash
uv run python dialect_classification/train/1_mel_cnn.py
uv run python dialect_classification/train/2_nb_phoneme.py
uv run python dialect_classification/train/3_linear_phoneme.py
uv run python dialect_classification/train/4_wav2vec_base_layer6.py
uv run python dialect_classification/train/5_xlsr_300m_layer8.py
uv run python dialect_classification/train/6_nb_phoneme_binary.py
uv run python dialect_classification/train/7_linear_phoneme_binary.py
```

**Evaluation**

- Vorarlberg dialect (7-way phoneme NB):
  ```bash
  uv run python eval/dialect_evaluation/vorarlberg_dialect_test.py
  ```
- Vorarlberg / Tirol / Wien binary evaluation:
  ```bash
  uv run python eval/dialect_evaluation/vorarlberg_results/binary_classification/binary_vorarlberg_evaluation.py
  ```
- Swiss SRF two-stage labeling (binary → 7-way):
  ```bash
  uv run python eval/swiss_dialect/two_stage_swiss_srf_espeak_classification.py
  ```
- Neighbor-confusion figures:
  ```bash
  uv run python scripts/plot_neighbor_confusion.py
  uv run python scripts/speaker_cm.py
  ```

### Chapter 5 — ASR benchmark & pyannote pipeline

1. Resample Swiss podcasts to 16 kHz mono:
   ```bash
   uv run python asr/pyannote_pipeline/resample_swiss.py
   ```
2. Diarization + faster-whisper transcription (set `HF_TOKEN` first):
   ```bash
   uv run python asr/pyannote_pipeline/pyannote_swiss.py
   # or orchestrated:
   uv run python asr/pyannote_pipeline/run_audio_processing.py
   ```
3. ASR model comparison on merged test set:
   ```bash
   uv run python asr/models/bench_whisper_turbo.py
   uv run python asr/models/bench_differend_asr_models.py
   ```

### Chapter 6 — Chatterbox fine-tuning & thesis_eval

Code lives in `tts/src/` and `tts/thesis_eval/`.

**Listen first (no GPU):** pre-generated outputs from the 25-sentence benchmark are in [`tts/samples/`](tts/samples/) — compare `kartoffelbox_base/sentence_XX.wav` with `vorarlberg_finetuned_10_epochs/sentence_XX.wav` (see `tts/samples/README.md` for suggested pairs).

**Fine-tuning** (requires Vorarlberg metadata TSV + GPU):

```bash
cd tts
uv sync
# Example — edit metadata path inside script first:
bash src/run_finetune_local_dataset.sh
# or:
bash src/run_finetune.sh
```

**Chapter 6 evaluation tasks** (inference / plotting; checkpoints must exist under `$CHECKPOINTS_DIR`):

```bash
cd tts
export CHECKPOINTS_DIR=$MODELS_ROOT/TTS/chatterbox

uv run python thesis_eval/scripts/task1_loss_curves.py    # training loss curves
uv run python thesis_eval/scripts/task2_val_loss.py       # recovered validation loss
uv run python thesis_eval/scripts/task3_synth_eval.py     # 25-sentence synthesis benchmark
uv run python thesis_eval/scripts/task4_dataset_size.py   # dataset size reconciliation
```

Key thesis numbers (10-epoch Vorarlberg run): train_loss **0.936**, ~52 h wall-clock, recovered val loss **3.393**, synthesis WER 0.550 → **0.287**.

---

## Verification

Recompute thesis tables from local data (requires merged TSVs + datasets on disk):

```bash
uv run python scripts/verify_thesis_numbers.py
uv run python scripts/run_extra_verify.py
uv run python scripts/comparison_prints_of_all_models.py
```

---

## External data required

Not shipped with this appendix. Obtain separately:

| Corpus | Typical path under `$DATA_ROOT` |
|--------|----------------------------------|
| SDS-200 | `audio/Schweiz/SDS-200/` |
| STT4SG-350 | `audio/Schweiz/STT4SG-350/` |
| Common Voice DE (optional) | `audio/Deutschland/cv22-de/` |
| Self-collected SRF audio | `audio/Schweiz/srf_audio_downloads/` or `16000_mono_wav/` |
| Vorarlberg podcasts | `audio/Vorarlberg/` |
| Austria eval (Tirol/Wien) | `audio/Österreich/sliced_16000_mono/` |
| Chatterbox base + fine-tuned checkpoints | `$MODELS_ROOT/TTS/chatterbox/` |
| Reference voice clip (TTS eval) | `tts/chatterbox-finetuning/voice_samples/daniel_ganahl_unfall_montafonerisch.wav` (not in git) |

---

## Path requirements

Full per-script audit: [docs/PATH_AUDIT.md](docs/PATH_AUDIT.md).

**Supervisor checklist after clone:**

1. Set `DATA_ROOT`, `MODELS_ROOT`, `HF_TOKEN`.
2. Run pipeline steps in chapter order; each step creates output dirs under `$THESIS_ROOT`.
3. Scripts with hardcoded `$DATA_ROOT/...` defaults work once `DATA_ROOT` is exported.
4. TTS eval tasks need checkpoints at `$CHECKPOINTS_DIR/<run_name>/`.

---

## License

MIT — see [LICENSE](LICENSE).
