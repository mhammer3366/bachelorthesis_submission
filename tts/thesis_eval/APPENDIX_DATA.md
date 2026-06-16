# Thesis Appendix — Extracted Data Tables

Read-only extraction from existing files. No training or synthesis was run.

---

## Task 1: Per-dialect ASR breakdown

**Source file:** `/home/ai/JKU/bachelorthesis_2/asr/bench_outputs/20250903_180349__metrics_summary.json`

**Structure:** The JSON is organised as one top-level object per ASR model. Each model entry contains `overall` (macro metrics), `per_label` (string keys `"0"`–`"7"` with WER/CER/sacreBLEU/chrF/TER), `preds_csv`, and `successful_predictions` (2,400 total). This matches the expected per-label layout; no structural surprises.

**Benchmark setup:** 2,400 utterances from `merged_datasets_plus_de` (Swiss dialect labels 0–6 plus Standard German label 7), up to 300 samples per label. Metrics below are taken from `per_label`.

| Region | Label | WER (Whisper) | CER (Whisper) | WER (Wav2Vec2) | CER (Wav2Vec2) |
|--------|-------|---------------|---------------|----------------|----------------|
| Basel | 0 | 0.2357 | 0.1182 | 0.8822 | 0.4254 |
| Bern | 1 | 0.2795 | 0.1598 | 0.9091 | 0.4656 |
| Innerschweiz | 2 | 0.2180 | 0.1118 | 0.8691 | 0.4091 |
| Ostschweiz | 3 | 0.2678 | 0.1379 | 0.8829 | 0.4126 |
| Wallis | 4 | 0.3598 | 0.1891 | 0.8919 | 0.4509 |
| Zürich | 5 | 0.2421 | 0.1181 | 0.8798 | 0.4069 |
| Graubünden | 6 | 0.2547 | 0.1399 | 0.8545 | 0.3952 |
| Standard German | 7 | 0.0894 | 0.0222 | 0.4891 | 0.1004 |

**Model keys in JSON:** `faster-whisper:large-v3` (Whisper) and `hf:wav2vec2:jonatasgrosman/wav2vec2-large-xlsr-53-german` (Wav2Vec2).

**Overall WER/CER (all labels pooled):** Whisper WER 0.2395 / CER 0.1206; Wav2Vec2 WER 0.8244 / CER 0.3723.

Whisper is best on every dialect label; Standard German (label 7) is easiest for both models. Wallis (label 4) is the hardest Swiss dialect for Whisper (WER 0.360).

---

## Task 2: Full training run summary

**Sources:** `training_args.bin` (loaded via `CustomTrainingArguments` from `src/finetune_t3.py`), `trainer_state.json`, and `train_results.json` under each checkpoint directory. Dataset paths come from matching `src/run_finetune*.sh` scripts where they exist; otherwise the path is inferred from the run name and throughput (see notes column).

**Effective batch size** = `per_device_train_batch_size` × (`train_batch_size` / `per_device_train_batch_size`) × `gradient_accumulation_steps` from `training_args.bin` and `trainer_state.json`. `train_batch_size` in HuggingFace logs is micro-batch × number of GPUs (not including gradient accumulation).

| Run name | Dataset | Epochs | Effective batch | Total steps | Train loss |
|----------|---------|--------|-----------------|-------------|------------|
| vorarlberg_finetuned_10_epochs | `vorarlberger_daten_16000.tsv` (inferred; no dedicated `.sh`; same hyperparams as 1-epoch run) | 10 | 24 | 68,760 | 0.936 |
| vorarlberg_finetuned_1_epoch | `vorarlberger_daten_16000.tsv` (inferred) | 1 | 24 | 6,876 | 2.028 |
| chatterbox_finetuned_vorarlberg_binary | `vorarlberg_daten_16000_binary_classified.tsv` (inferred; 146,819 rows ≈ 7,326 steps × 20 batch) | 8 | 20 | 58,608 | 0.022 |
| chatterbox_finetuned_überleaba_more_epochs | `überleaba_podcast/.../arrow_dataset_16000_1` (inferred; same dataset as 1-epoch überleaba run) | 10 | 24 | 32,120 | 0.090 |
| chatterbox_finetuned_überleaba | `überleaba_podcast/sliced_16000/arrow_dataset_16000_1` (`run_finetune_überleaba.sh`) | 1 | 24 | 3,211 | 4553.793 |
| chatterbox_finetuned_stt_all | `STT4SG-350/.../updated_train_all_split.tsv` (`run_finetune_local_dataset_2.sh`) | 1 | 24 | 8,319 | 4.411 |
| chatterbox_finetuned_only_east_more_epochs | `STT4SG-350/.../arrow_dataset_3` (`run_finetune_only_east_more_epochs.sh`) | 3 | 24 | 3,900 | 3.757 |
| chatterbox_finetuned_only_east | `STT4SG-350/.../arrow_dataset_3` (`run_finetune_only_east.sh`) | 1 | 24 | 1,307 | 4.600 |
| chatterbox_finetuned_wien_1 | `Österreich/.../Wien/master_with_phonemes_binary.tsv` (inferred; ~7,260 samples/epoch from throughput) | 8 | 20 | 2,904 | 0.142 |
| chatterbox_finetuned_wien_2 | same as `wien_1` (identical `trainer_state.json` / `train_results.json`) | 8 | 20 | 2,904 | 0.142 |
| chatterbox_finetuned_überleaba_5_epochs_multilingual | `vorarlberger_daten_16000.tsv` (`run_finetune_überleaba_5_epochs_multilingual.sh`) | 5 (planned) | 24 | 4,000 | **aborted** (no `train_results.json`; only `checkpoint-4000/trainer_state.json`, epoch ≈ 0.49) |

### Notes on unexpected or missing data

- **`training_args.bin` does not store dataset paths.** `DataArguments` (`metadata_file` / `dataset_dir`) are passed separately at launch and are not serialized into the checkpoint. Dataset column above uses shell scripts or inference from run name + steps/epoch × effective batch.
- **`chatterbox_finetuned_überleaba` (1 epoch)** completed but has epoch-averaged train loss **4553.79**, consistent with the broken 1-epoch überleaba run cited in the thesis audit (loss ~4500). The 10-epoch überleaba follow-up (`überleaba_more_epochs`) trained normally (loss 0.090).
- **`chatterbox_finetuned_überleaba_5_epochs_multilingual`** stopped at step 4,000 of a planned 5-epoch run (~49% of epoch 1 per `checkpoint-4000/trainer_state.json`). No root-level `training_args.bin` or `train_results.json` exists.
- **`chatterbox_finetuned_wien_1` and `wien_2`** report identical step counts, batch settings, and train loss; they may be duplicate exports of the same training job.
- **`chatterbox_finetuned_stt_all`:** the saved checkpoint uses `per_device_train_batch_size=6` (effective batch 24), while `run_finetune_local_dataset_2.sh` specifies `per_device_train_batch_size=4`; the launch command may have differed from the checked-in script.
- **Train loss** values are epoch-averaged figures from each run's `train_results.json` (`train_loss` field), except for the aborted multilingual run.
- **All runs report `eval_loss = NaN`** during training (SpeechDataCollator label-masking bug); that is not reflected in this table.

### Extraction commands (reproducible)

```bash
# Task 1: values read directly from metrics_summary.json (printed above)

# Task 2: training_args.bin requires CustomTrainingArguments class
source thesis_eval/.venv/bin/activate
cd /home/ai/AI-DataPool/Other/Backup/home_dirs/max_150/experiments/TTS/chatterbox/chatterbox-finetuning
python3 -c "
import json, sys, torch
sys.path.insert(0,'src')
from finetune_t3 import CustomTrainingArguments
# ... load per checkpoint (see thesis_eval/scripts for pattern)
"
```
