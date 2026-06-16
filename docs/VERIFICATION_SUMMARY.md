# Verification summary (condensed)

Audited thesis numbers and Chapter 6 metrics (author machine). Re-run locally with datasets and legacy outputs present:

```bash
export THESIS_ROOT=/path/to/bachelorthesis_submission   # or legacy bachelorthesis_2 tree with TSVs/models
export DATA_ROOT=/path/to/AI-DataPool/Datasets
uv run python scripts/verify_thesis_numbers.py
```

## Chapter 6 (TTS) — key confirmed values

| Metric | Value |
|--------|-------|
| 10-epoch fine-tune steps | 68,760 |
| Epoch-averaged train loss | 0.936 |
| Recovered validation loss (total) | 3.393 |
| 25-sentence WER (base → fine-tuned) | 0.550 → 0.287 |
| Dialect proxy (25-sentence set) | 20% → 40% |

Scripts: `tts/thesis_eval/scripts/task1_loss_curves.py` … `task4_dataset_size.py`, `task3_synth_eval.py`. See `tts/thesis_eval/REPORT.md` for methodology.

Full audit notes: author `cursor_verification_results.md` (not in git).
