# Repository Overview

Scripts-only appendix for the Vorarlberg dialect TTS bachelor thesis. **No data files, model weights, or evaluation dumps** are included.

---

## Top-level layout

```
bachelorthesis_submission/
├── README.md
├── REPO_OVERVIEW.md
├── pyproject.toml
├── LICENSE
├── .gitignore
├── data_collection/          # SRF scrapers (Ch. 3)
├── data_preparation/         # merge scripts, distribution analysis (Ch. 3)
├── dialect_classification/   # feature extraction + training (Ch. 4)
├── asr/                      # pyannote + whisper benchmarks (Ch. 5)
├── eval/                     # Vorarlberg + Swiss SRF evaluation scripts (Ch. 4–5)
├── labeling/                 # README only (manual eval artifacts not in git)
├── scripts/                  # verify_thesis_numbers, plotting, utilities
├── tts/                      # Chatterbox src + thesis_eval (Ch. 6)
└── docs/
    ├── APPENDIX_LINK.md
    └── VERIFICATION_SUMMARY.md
```

---

## Folder details

### `data_collection/` *(Ch. 3 — self-collected Swiss German)*

| File | Purpose |
|------|---------|
| `srf_webscraper.py`, `srf_webscraper_2.py` | Scrape SRF podcast metadata & audio URLs |
| `srf_api.py`, `get_access_token.py` | SRF API helpers |
| `mp3_to_wav.py` | Convert downloaded MP3 to WAV |

**Not included:** `srf_audio_metadata.csv`, downloaded audio — regenerate via scraper.

### `data_preparation/` *(Ch. 3 — merged classification dataset)*

| Subfolder / file | Purpose |
|------------------|---------|
| `merged_datasets/` | `merge_datasets.py`, `merge_german.py`, distribution analysis |
| `augment_tsv_with_client_and_duration.py` | TSV augmentation helper |
| `test.py` | Development smoke test |

**Not included:** merged TSV files, `saved_features_*` (~164 GB), plot PNGs.

### `dialect_classification/` *(Ch. 4)*

| Subfolder | Purpose |
|-----------|---------|
| `feature_extraction/` | Mel spectrogram, eSpeak phoneme, wav2vec/XLSR embedding extractors |
| `train/` | Classifier training (`1_mel_cnn.py` … `7_linear_phoneme_binary.py`) |

**Not included:** extracted feature caches, `.pt` / `.joblib` weights, `*_preds.csv`.

### `asr/` *(Ch. 5)*

| Subfolder | Purpose |
|-----------|---------|
| `models/` | Benchmark faster-whisper vs HF whisper |
| `pyannote_pipeline/` | Swiss podcast diarization + transcription pipeline |

**Not included:** `bench_outputs/` prediction CSVs.

### `eval/` *(Ch. 4–5)*

| Subfolder | Purpose |
|-----------|---------|
| `swiss_dialect/` | Two-stage SRF eSpeak classification |
| `dialect_evaluation/` | Vorarlberg/Tirol/Wien binary evaluation, Label Studio batch prep |

**Not included:** `clip_labels.csv` (~3 GB), distribution CSVs, batch TSVs.

### `scripts/`

| Script | Purpose |
|--------|---------|
| `verify_thesis_numbers.py` | Recompute thesis consistency checks |
| `run_extra_verify.py` | Supplementary checks |
| `speaker_cm.py` | Speaker-level binary confusion matrix |
| `plot_neighbor_confusion.py` | Neighbor vs non-neighbor confusion figures |
| `comparison_prints_of_all_models.py` | Tabular model comparison |

### `tts/` *(Ch. 6)*

Points to external `chatterbox-finetuning` and `thesis_eval/`. See [`tts/README.md`](tts/README.md).

---

## Copy provenance

| Source | What was copied |
|--------|-----------------|
| `bachelorthesis_2/` (primary) | `data_preparation/`, `dialect_classification/`, `asr/`, `eval/`, `scripts/` |
| `bachelorthesis/` (supplement) | `data_collection/` SRF scrapers only |

**Deliberately omitted:** third-party podcast-corpus reference tooling, all CSV/TSV/audio/JSON results, `models/` metrics dumps, `verification_output.json`, duplicate prototype classifiers superseded by `dialect_classification/train/`.

---

## External dependencies (not in git)

| Asset | Typical path | Used by |
|-------|--------------|---------|
| SDS-200 splits | `$DATA_ROOT/audio/Schweiz/SDS-200/...` | merge, verify |
| STT4SG-350 | `$DATA_ROOT/audio/Schweiz/STT4SG-350` | merge, verify |
| Vorarlberg podcasts | `$DATA_ROOT/audio/Vorarlberg` | Ch. 3 table, TTS training |
| Common Voice 22 DE | `$DATA_ROOT/audio/Deutschland/cv22-de/...` | merge_german |
| Chatterbox checkpoints | `$MODELS_ROOT/TTS/chatterbox/` | Ch. 6 |
| Full thesis_eval | `chatterbox-finetuning/thesis_eval/` | Ch. 6 scripts, figures |

---

## Appendix citation

Full LaTeX block: [`docs/APPENDIX_LINK.md`](docs/APPENDIX_LINK.md).

Suggested GitHub name: **`vorarlberg-dialect-tts-thesis`**
