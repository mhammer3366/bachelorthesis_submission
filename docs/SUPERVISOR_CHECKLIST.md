# Supervisor checklist

## Automated (this repo)

- [x] No `.csv`, `.tsv`, or `.wav` tracked in the tree (excluding `.venv`)
- [x] Python sources compile: `find . -path ./.venv -prune -o -name '*.py' -print | xargs python3 -m py_compile`
- [x] MIT [LICENSE](../LICENSE); mentioned in [README](../README.md)
- [x] GitHub URL in [APPENDIX_LINK.md](APPENDIX_LINK.md)

## You must provide locally

1. **Datasets** under `DATA_ROOT` (Vorarlberg podcasts, Swiss SRF downloads, SDS-200, STT4SG-350, Common Voice 22 DE, etc.).
2. **Legacy artifacts** on disk: merged TSVs, extracted features (~100+ GB), trained `.joblib` / `.pt` models, eval CSVs — produced by running Ch. 3–5 scripts or copied from the author environment.
3. **`HF_TOKEN`** for pyannote (`export HF_TOKEN=...`) before `asr/pyannote_pipeline/pyannote_swiss.py`.
4. **TTS checkpoints** under `MODELS_ROOT/TTS/chatterbox/` (not in git) for Chapter 6 fine-tuned inference.
5. **Reference voice** for synthesis eval: set `TTS_REF_VOICE` or place clip at `tts/thesis_eval/voice_samples/daniel_ganahl_unfall_montafonerisch.wav`.
6. **GPU / CUDA** recommended for training, ASR benchmark, and TTS.
7. **Playwright**: `uv run playwright install chromium` before SRF scrapers.

## Scripts needing manual attention

| Script | Why |
|--------|-----|
| `data_collection/get_access_token.py` | SRF API credentials (not stored in git) |
| `asr/pyannote_pipeline/pyannote_swiss.py` | Requires `HF_TOKEN`; long GPU job |
| `tts/src/run_finetune*.sh` | Edit dataset/checkpoint paths for your machine before training |
| `tts/thesis_eval/scripts/task3_synth_eval.py` | Needs checkpoints + `CLASSIFIER_MODELS_DIR` NB model |
| `eval/swiss_dialect/two_stage_swiss_srf_espeak_classification.py` | Large SRF metadata + GPU for wav2vec |

## Suggested review order

1. Read [README](../README.md) prerequisites and env vars.
2. `uv sync` — confirm lockfile resolves.
3. Spot-check `scripts/verify_thesis_numbers.py` against thesis tables (with author data paths).
4. Skim `docs/PATH_AUDIT.md` for path conventions.
