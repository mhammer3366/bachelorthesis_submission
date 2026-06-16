# Vorarlberg Dialect TTS — Bachelor Thesis Code Appendix

**Scripts-only** reproduction code for the bachelor thesis *A Data-Driven Pipeline for Alemannic Dialect TTS with a Focus on Vorarlberg*.

> This repository contains **no datasets, audio, checkpoints, result CSVs/JSON, or the thesis PDF**. All scripts expect data on disk via `DATA_ROOT` and generated artifacts via `THESIS_ROOT`.

## Thesis document

The full thesis PDF is kept locally (not in git): `main-thesis.pdf` (author machine only).

Public code appendix: [docs/APPENDIX_LINK.md](docs/APPENDIX_LINK.md) → https://github.com/mhammer3366/bachelorthesis_submission

## Prerequisites

- **Python 3.11+** (3.10+ may work)
- **[uv](https://github.com/astral-sh/uv)** package manager
- **ffmpeg** (audio resampling / probing)
- **CUDA** optional but recommended (classification, ASR, TTS)
- **Playwright** for SRF scrapers: `uv sync && uv run playwright install chromium`
- **Hugging Face account** + `HF_TOKEN` for pyannote diarization (gated models)

## Setup

```bash
git clone https://github.com/mhammer3366/bachelorthesis_submission.git
cd bachelorthesis_submission
uv sync
```

## Environment variables

Set these before running pipeline steps (also see [docs/PATH_AUDIT.md](docs/PATH_AUDIT.md)):

```bash
export THESIS_ROOT="$(pwd)"   # default in scripts: repo root
export DATA_ROOT=/path/to/AI-DataPool/Datasets
export MODELS_ROOT=/path/to/AI-DataPool/Models          # optional
export CLASSIFIER_MODELS_DIR=$THESIS_ROOT/models        # Ch. 6 NB classifier
export HF_TOKEN=hf_...                                  # pyannote only
export MERGED_DATA_DIR=$THESIS_ROOT/data_preparation/merged_datasets
export MERGED_DE_DATA_DIR=$THESIS_ROOT/data_preparation/merged_datasets_plus_de
```

| Variable | When needed |
|----------|-------------|
| `THESIS_ROOT` | Merged TSVs, saved features, `models/`, `eval/` outputs (defaults to repository root) |
| `DATA_ROOT` | All on-disk audio corpora; Swiss pyannote pipeline I/O |
| `MODELS_ROOT` | TTS / Chatterbox checkpoints (not in git) |
| `HF_TOKEN` | `asr/pyannote_pipeline/pyannote_swiss.py` |
| `CLASSIFIER_MODELS_DIR` | `tts/thesis_eval/scripts/task3_synth_eval.py` |

## Step-by-step reproduction

### Chapter 3 — Data collection & preparation

1. **SRF metadata (optional self-collected Swiss audio)**  
   Configure API access, then:
   ```bash
   uv run python data_collection/get_access_token.py   # local credentials
   uv run python data_collection/srf_webscraper.py
   uv run python data_collection/srf_webscraper_2.py
   ```
2. **Convert downloads** (if needed): `uv run python data_collection/mp3_to_wav.py`
3. **Merge SDS-200 + STT4SG-350 (+ optional CV DE)** into train/valid/test TSVs:
   ```bash
   uv run python data_preparation/merged_datasets/merge_datasets.py
   uv run python data_preparation/merged_datasets/merge_german.py   # + Common Voice DE
   ```
4. **Distribution checks**:
   ```bash
   uv run python data_preparation/merged_datasets/vorarlberg_distribution_print.py
   uv run python data_preparation/merged_datasets/self_collected_srf_distribution_print.py
   ```
5. **Verify thesis tables** (requires merged TSVs + `DATA_ROOT` audio):
   ```bash
   uv run python scripts/verify_thesis_numbers.py
   ```

### Chapter 4 — Dialect classification

1. **Features** (long-running; writes under `$THESIS_ROOT/data_preparation/feature_extraction/`):
   ```bash
   uv run python dialect_classification/feature_extraction/1_mel_spectogram.py
   uv run python dialect_classification/feature_extraction/2_phoneme.py
   uv run python dialect_classification/feature_extraction/3_wav2vec_base_layer6.py
   uv run python dialect_classification/feature_extraction/4_xlsr_300m_layer8.py
   uv run python dialect_classification/feature_extraction/phoneme_vorarlberg.py
   ```
2. **Train classifiers** (`dialect_classification/train/1_mel_cnn.py` … `7_linear_phoneme_binary.py`):
   ```bash
   uv run python dialect_classification/train/3_linear_phoneme.py
   uv run python dialect_classification/train/6_nb_phoneme_binary.py
   ```
3. **Figures / verification**:
   ```bash
   uv run python scripts/plot_neighbor_confusion.py
   uv run python scripts/speaker_cm.py
   uv run python scripts/comparison_prints_of_all_models.py
   ```

### Chapter 5 — ASR benchmark & evaluation

1. **Resample Swiss podcasts** (paths under `$DATA_ROOT/audio/Schweiz/`):
   ```bash
   uv run python asr/pyannote_pipeline/resample_swiss.py
   ```
2. **Diarization + transcription** (set `HF_TOKEN`):
   ```bash
   export HF_TOKEN=hf_...
   uv run python asr/pyannote_pipeline/pyannote_swiss.py
   uv run python asr/pyannote_pipeline/pynnote_postprocessing.py
   uv run python asr/pyannote_pipeline/run_audio_processing.py
   ```
3. **Whisper benchmarks**:
   ```bash
   uv run python asr/models/bench_differend_asr_models.py
   uv run python asr/models/bench_whisper_turbo.py
   ```
4. **Swiss SRF two-stage labeling**:
   ```bash
   uv run python eval/swiss_dialect/two_stage_swiss_srf_espeak_classification.py
   ```
5. **Vorarlberg binary / neighbor eval** — scripts under `eval/dialect_evaluation/`.

Manual listening labels: see `labeling/README.md` (artifacts not in git).

### Chapter 6 — TTS (Chatterbox / Kartoffelbox fine-tuning)

In-repo copy of the fine-tuning fork (scripts only):

```
tts/
├── src/                    # Chatterbox package + run_finetune*.sh + finetune_config.yaml
└── thesis_eval/scripts/    # task1–task4 + task3 synthesis benchmark
```

1. **Install TTS deps** (CUDA torch as on your machine; see `tts/src/pyproject.toml` if present or author env).
2. **Prepare Vorarlberg metadata TSV** on disk: `$DATA_ROOT/audio/Vorarlberg/vorarlberger_daten_16000.tsv` (and binary-filtered variant for some runs).
3. **Fine-tune** (edit shell script paths for `DATA_ROOT` / `MODELS_ROOT`):
   ```bash
   cd tts/src
   bash run_finetune_local_dataset.sh    # example entry; see run_finetune*.sh
   ```
4. **Point checkpoints**:
   ```bash
   export MODELS_ROOT=/path/to/AI-DataPool/Models
   # expect: $MODELS_ROOT/TTS/chatterbox/vorarlberg_finetuned_10_epochs/
   ```
5. **Chapter 6 evaluation** (inference-only; writes under `tts/thesis_eval/` at runtime):
   ```bash
   uv run python tts/thesis_eval/scripts/task1_loss_curves.py
   uv run python tts/thesis_eval/scripts/task2_val_loss.py
   uv run python tts/thesis_eval/scripts/task3_synth_eval.py
   uv run python tts/thesis_eval/scripts/task4_dataset_size.py
   ```
   See `tts/thesis_eval/REPORT.md` and [docs/VERIFICATION_SUMMARY.md](docs/VERIFICATION_SUMMARY.md).

## Repository map

| Thesis chapter | Folder |
|----------------|--------|
| Ch. 3 — Data collection | `data_collection/` |
| Ch. 3 — Merging & stats | `data_preparation/` |
| Ch. 4 — Classification | `dialect_classification/` |
| Ch. 5 — ASR & pyannote | `asr/` |
| Ch. 4–5 — Evaluation | `eval/` |
| Ch. 5 — Manual labeling | `labeling/` |
| Ch. 6 — TTS | `tts/` |
| Cross-cutting | `scripts/`, `docs/` |

Details: [REPO_OVERVIEW.md](REPO_OVERVIEW.md). Supervisor checklist: [docs/SUPERVISOR_CHECKLIST.md](docs/SUPERVISOR_CHECKLIST.md).

## License

Released under the **MIT License** — see [LICENSE](LICENSE).
