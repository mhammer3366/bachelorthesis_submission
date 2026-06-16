# Path Audit — `bachelorthesis_submission`

Last updated: 2026-06-16. Audited after supervisor-ready path fixes.

## Environment variables

| Variable | Default when unset | Purpose |
|----------|-------------------|---------|
| `THESIS_ROOT` | Repository root (`Path(__file__).parents[N]`) | Merged TSVs, `models/`, `eval/` outputs |
| `DATA_ROOT` | `/home/ai/AI-DataPool/Datasets` | All audio corpora and source TSVs |
| `MODELS_ROOT` | `$DATA_ROOT/../Models` | Classifier + TTS checkpoint storage |
| `CHECKPOINTS_DIR` | `$MODELS_ROOT/TTS/chatterbox` | Chatterbox fine-tuned runs |
| `CHATTERBOX_ROOT` | `$THESIS_ROOT/tts/chatterbox-finetuning` | TTS source + thesis_eval |
| `MERGED_DATA_DIR` | `$THESIS_ROOT/data_preparation/merged_datasets` | Swiss merge outputs |
| `MERGED_DE_DATA_DIR` | `$THESIS_ROOT/data_preparation/merged_datasets_plus_de` | Merge + German CV |
| `CLASSIFIER_MODELS_DIR` | `$THESIS_ROOT/models` | Dialect classifier weights |
| `HF_TOKEN` | *(empty)* | pyannote / gated HuggingFace models |
| `REF_VOICE_WAV` | `$CHATTERBOX_ROOT/voice_samples/daniel_ganahl_unfall_montafonerisch.wav` | TTS synthesis reference |
| `SPEAKER_CM_OUT` | `$THESIS_ROOT/speaker_cm_out.json` | Speaker CM JSON output |

**Supervisor action:** Always export `DATA_ROOT` (and `MODELS_ROOT` for TTS/eval). Defaults point to the author's machine and will fail elsewhere if unset.

## Path resolution pattern

Most Python scripts use:

```python
REPO_ROOT = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parents[N]))
```

where `N` depends on script depth (1 for `scripts/`, 2 for `dialect_classification/train/`, 4 for deep eval scripts).

TTS `thesis_eval/scripts/task*.py` use:

```python
CHATTERBOX_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(os.environ.get("THESIS_ROOT", CHATTERBOX_ROOT.parents[1]))
```

Shared constants also exist in [`thesis_paths.py`](../thesis_paths.py) at repo root.

## Fixed in this audit

| Issue | Resolution |
|-------|------------|
| `/home/ai/bachelorthesis_2/...` hardcodes | Replaced with `REPO_ROOT / ...` or `THESIS_ROOT` env |
| `/home/student/AI-DataPool/...` | Replaced with `$DATA_ROOT/...` |
| `/home/max/AI-DataPool/...` | Replaced with `$DATA_ROOT/...` |
| `data_preperation/` typo | Normalized to `data_preparation/` in submission tree |
| pyannote HF token in source | Moved to `HF_TOKEN` env var |
| TTS thesis_eval absolute backup paths | Use `CHATTERBOX_ROOT` + `CHECKPOINTS_DIR` |

## Scripts by category

### Ready after `export DATA_ROOT=...` (no manual edits)

| Script | Notes |
|--------|-------|
| `dialect_classification/train/*.py` | Outputs to `$THESIS_ROOT/models/` |
| `dialect_classification/feature_extraction/*.py` | Reads merged TSVs from `$THESIS_ROOT/data_preparation/` |
| `data_preparation/merged_datasets/merge_*.py` | Source TSVs under `$DATA_ROOT/audio/Schweiz/` |
| `asr/pyannote_pipeline/resample_swiss.py` | `$DATA_ROOT/audio/Schweiz/srf_audio_downloads` |
| `asr/pyannote_pipeline/pyannote_swiss.py` | Requires `HF_TOKEN` |
| `eval/swiss_dialect/two_stage_*.py` | SRF master TSV under `$DATA_ROOT` |
| `scripts/verify_thesis_numbers.py` | Needs merged TSVs + datasets |
| `tts/chatterbox-finetuning/thesis_eval/scripts/task*.py` | Needs `$CHECKPOINTS_DIR` checkpoints |

### Require `$MODELS_ROOT` or trained weights

| Script | Expected weights |
|--------|------------------|
| `eval/swiss_dialect/*.py` | `$THESIS_ROOT/models/{4_linear_phoneme,5_nb_phoneme_binary}` |
| `eval/dialect_evaluation/vorarlberg_*.py` | `$THESIS_ROOT/models/nb_phoneme_binary` or `5_nb_phoneme_binary` |
| `tts/.../task3_synth_eval.py` | Base + fine-tuned Chatterbox under `$CHECKPOINTS_DIR` |
| `asr/models/bench_*.py` | Merged test TSV + audio paths valid |

### Optional / dev utilities (hardcoded `$DATA_ROOT` paths in body)

These use `$DATA_ROOT`-relative paths with author fallback; set `DATA_ROOT` before use:

- `scripts/clean_tsv.py`, `check_phonemes.py`, `check_where_it_comes_from.py`, `debug_tsv_lines.py`
- `data_collection/mp3_to_wav.py` (example swissdials path in `__main__`)
- `data_preparation/augment_tsv_with_client_and_duration.py` (docstring example only)

### Austria eval scripts (external TSV paths)

Under `$DATA_ROOT/audio/Österreich/sliced_16000_mono/`:

- `binary_tirol_evaluation.py`, `binary_wien_evaluation.py`
- `merge_tsv_tirol.py`, `merge_tsv_wien.py`

Obtain Austria slices separately or skip these eval steps.

## Remaining manual steps for supervisor

1. **Obtain all corpora** listed in README § External data.
2. **Set environment variables** (`DATA_ROOT`, `MODELS_ROOT`, `HF_TOKEN`).
3. **Run pipeline in chapter order** — intermediate TSVs/features are not in git.
4. **Place TTS checkpoints** under `$CHECKPOINTS_DIR` for Chapter 6 eval tasks.
5. **Reference voice WAV** for task3: copy to `tts/chatterbox-finetuning/voice_samples/` or set `REF_VOICE_WAV`.
6. **Optional:** `data_collection/mp3_to_wav.py` — edit `input_dir`/`output_dir` in `__main__` for your layout.

## py_compile

All `.py` files under the repo (excluding `.venv`) are syntax-checked via:

```bash
find . -name '*.py' ! -path './.venv/*' ! -path '*/__pycache__/*' -print0 | xargs -0 python3 -m py_compile
```

Run after any edit to confirm import/syntax validity (does not verify runtime dependencies).
