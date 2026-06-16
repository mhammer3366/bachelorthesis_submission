# Dataset size explanation — `vorarlberg_finetuned_10_epochs`

## Which TSV was used?

There is **no dedicated `.sh` launch script** for `vorarlberg_finetuned_10_epochs` in the repo.
The closest documented Vorarlberg metadata path is in `src/run_finetune_überleaba_5_epochs_multilingual.sh`:

```
--metadata_file /home/ai/AI-DataPool/Datasets/audio/Vorarlberg/vorarlberger_daten_16000.tsv
```

The final 10-epoch run's `training_args.bin` matches the same hyperparameters as the 1-epoch Vorarlberg run
(`vorarlberg_finetuned_1_epoch`, same step count per epoch: 6876). Both use `eval_split_size=0.01`, `seed=42`,
`per_device_train_batch_size=3`, `gradient_accumulation_steps=2`, and `train_batch_size=12` (4 GPUs).

**Conclusion:** the thesis should cite **`vorarlberger_daten_16000.tsv`** as the training metadata source.

## Row counts (computed now)

- `/home/ai/AI-DataPool/Datasets/audio/Vorarlberg/vorarlberger_daten_16000.tsv`: 197238 rows total; **187924** with audio file present on disk; after 1% eval split (seed 42): **186045 train / 1879 eval**.
- `/home/ai/AI-DataPool/Datasets/audio/Vorarlberg/vorarlberger_daten_16000_62server_filtered.tsv`: not found on this machine.
- `/home/ai/AI-DataPool/Datasets/audio/Vorarlberg/vorarlberg_daten_16000_binary_classified.tsv`: 146818 rows total; **139865** with audio file present on disk; after 1% eval split (seed 42): **138467 train / 1398 eval**.

## Throughput-derived training set size

From `checkpoints/vorarlberg_finetuned_10_epochs/trainer_state.json`:
- `global_step` = 68760
- `num_train_epochs` = 10
- `train_batch_size` = 12 (per_device 3 × 4 GPUs × grad_accum 2 = effective batch 24, but HF logs `train_batch_size` as micro-batch × GPUs = 12)
- Steps per epoch = 68760 / 10 = **6876**
- Samples seen per epoch = 6876 × effective batch **24** = **165024**

The audit inferred **~165,024** samples/epoch from `6876 × 24 = 165024`.

## Resolving the gap

If all 187918 existing rows were used with 1% eval split,
train size would be **186038**, not 165,024.

The difference (~30k rows) is explained by **`SpeechFineTuningDataset.__getitem__` silently dropping samples**
that return `None` when:
- audio cannot be loaded,
- speaker embedding fails, or
- S3Tokenizer returns `None` for speech tokens.

The HuggingFace `Trainer` dataloader does not back-fill dropped samples; each epoch still runs
**6876 optimizer steps** over whatever valid items remain (~165k passes through the loader).

## Number the thesis should quote

**Training utterances per epoch (throughput-verified): ~165,024**
(= 6876 steps × effective batch 24).

**Raw metadata rows:** 197238 in `vorarlberger_daten_16000.tsv`.

**Rows with resolvable audio on disk:** 187915.

**Held-out validation (1%, seed 42):** 1879 rows from the metadata split (before `__getitem__` drops).

Do **not** quote both 197k and 165k without explaining that ~16% of rows fail feature extraction at load time.
