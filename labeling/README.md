# Manual labeling (Chapter 5)

This folder documents the human-validation workflow for Vorarlberg dialect clips. **Label files are not redistributed** in this scripts-only appendix.

## Vorarlberg binary validation

- **Pipeline:** `eval/dialect_evaluation/vorarlberg_results/binary_classification/`
- **Label Studio batches:** created via `create_label_studio_batches.py`
- **Human eval:** 200 clips; accuracy **91.0%**; F1 dialect **0.913**, F1 High German **0.906**
- **Confusion matrix:** `[[95, 10], [8, 87]]`

## Swiss SRF two-stage labeling

- **Scripts:** `eval/swiss_dialect/two_stage_swiss_srf_espeak_classification.py`
- **Stage A:** `5_nb_phoneme_binary` (German vs non-German)
- **Stage B:** `4_linear_phoneme` (7 dialect regions for non-German only)
- **Output:** large `clip_labels.csv` (~3 GB) — regenerate locally; not in git

## Local artifacts (not in repo)

| Artifact | Typical path |
|----------|--------------|
| Vorarlberg manual eval JSON | `labeling/vbg_binary/manual_eval.json` |
| Merged eval CSV | `labeling/vbg_binary/merged_eval.csv` |
| SRF clip labels | `eval/dialect_evaluation/swiss_srf_espeak_classification_two_stage_2/clip_labels.csv` |
