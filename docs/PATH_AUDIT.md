# Path audit (supervisor)

Scripts resolve paths from environment variables with defaults suitable for the author machine. Set these once per session:

| Variable | Default (if unset) | Purpose |
|----------|------------------|---------|
| `THESIS_ROOT` | Repository root (`Path(__file__).parents[N]`) | Merged TSVs, `models/`, `eval/` outputs on the legacy tree |
| `DATA_ROOT` | `/home/ai/AI-DataPool/Datasets` | Audio corpora (Vorarlberg, Schweiz, SDS, CV, …) |
| `MODELS_ROOT` | `$DATA_ROOT/../Models` | TTS checkpoints, optional classifier weights |
| `CLASSIFIER_MODELS_DIR` | `$THESIS_ROOT/models` | Chapter 6 dialect NB model for synthesis eval |
| `HF_TOKEN` | *(empty)* | Hugging Face token for pyannote diarization |
| `TTS_REF_VOICE` | `tts/thesis_eval/voice_samples/daniel_ganahl_unfall_montafonerisch.wav` | Reference clip for Ch. 6 synthesis benchmark |

## Layout note

Historical code used the folder name `data_preperation` (typo). This submission repo uses `data_preparation/`. Runtime paths in Python use `data_preparation/` under `REPO_ROOT`. If you symlink a full legacy checkout, map `data_preperation` → `data_preparation` or set `THESIS_ROOT` to that legacy tree.

## By area

| Area | Path pattern | Env |
|------|----------------|-----|
| Merged Swiss TSVs | `REPO_ROOT/data_preparation/merged_datasets/*.tsv` | `THESIS_ROOT` |
| Phoneme / mel features | `REPO_ROOT/data_preparation/feature_extraction/...` | `THESIS_ROOT` |
| Classifier checkpoints | `REPO_ROOT/models/*` | `THESIS_ROOT` or `CLASSIFIER_MODELS_DIR` |
| Vorarlberg audio | `DATA_ROOT/audio/Vorarlberg/**` | `DATA_ROOT` |
| Swiss pyannote I/O | `DATA_ROOT/audio/Schweiz/16000_mono_wav*` | `DATA_ROOT` |
| Chatterbox checkpoints | `MODELS_ROOT/TTS/chatterbox/<run_name>/` | `MODELS_ROOT` |
| Ch. 6 eval scripts | `tts/thesis_eval/scripts/` | in-repo |

## Docstring-only paths

Some evaluation scripts still mention `/home/ai/bachelorthesis_2/...` in module docstrings (documentation of the author run). Executable assignments use `REPO_ROOT` / `DATA_ROOT`.

## Manual edits still possible

- `asr/pyannote_pipeline/pyannote_faster_whisper_large.py` — local `INPUT_ROOT` / `OUTPUT_ROOT` if not using `DATA_ROOT` layout
- Optional datasets referenced in comments (swissdials, Austria eval TSVs)
