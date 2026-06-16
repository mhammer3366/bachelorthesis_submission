# Chapter 6 Evaluation Report — Vorarlberg TTS (inference-only)

**Base directory:** `/home/ai/AI-DataPool/Other/Backup/home_dirs/max_150/experiments/TTS/chatterbox/chatterbox-finetuning/thesis_eval/`

All artifacts were produced **without retraining**. No `run_finetune_*.sh` scripts were executed.

---

## Files created

### Figures (Task 1)
| File | Description |
|------|-------------|
| `figures/loss_vs_step_vorarlberg_10epochs.png` | Training loss vs step (y clipped 0–5; early spike ~20 at step 110) |
| `figures/loss_vs_step_vorarlberg_10epochs_log.png` | Same curve, log y-axis |
| `figures/learning_rate_vs_step_vorarlberg_10epochs.png` | Cosine-with-restarts LR schedule |
| `figures/loss_comparison_vorarlberg_runs.png` | Overlay: 10-epoch, binary, überleaba_more_epochs |
| `train_scalars_all_runs.csv` | Auditable scalars (step, loss, lr, grad_norm) for 3 runs — 15,948 rows |

### Validation loss (Task 2)
| File | Description |
|------|-------------|
| `final_val_loss.json` | Recovered validation loss with fixed eval collator |
| `scripts/eval_collator.py` | Eval-only collator (drops all-masked samples) |
| `scripts/task2_val_loss.py` | Standalone eval script |
| `cache/valid_metadata_rows.json` | 187,922 rows with existing audio (cached) |

### Synthesis evaluation (Task 3)
| File | Description |
|------|-------------|
| `test_sentences.txt` | 25 fixed test sentences (Chapter 6 benchmark set) |
| `synth_eval_results.csv` | Per-sentence metrics (50 rows = 2 models × 25 sentences) |
| `samples/kartoffelbox_base/sentence_*.wav` | Base Kartoffelbox syntheses |
| `samples/vorarlberg_finetuned_10_epochs/sentence_*.wav` | Fine-tuned syntheses |
| `scripts/task3_synth_eval.py` | Standalone harness |

### Dataset size (Task 4)
| File | Description |
|------|-------------|
| `dataset_size_explanation.md` | Reconciled training-set size |

### Scripts & utilities
| File | Description |
|------|-------------|
| `scripts/task1_loss_curves.py` | TensorBoard / trainer_state → PNGs + CSV |
| `scripts/task4_dataset_size.py` | Dataset arithmetic |
| `scripts/eval_bootstrap.py` | Perth watermarker stub for headless eval |

### Dialect & ASR evaluation (Chapter 4)
| File | Description |
|------|-------------|
| `/home/ai/JKU/bachelorthesis_2/plot_neighbor_confusion.py` | Neighbor vs non-neighbor confusion plots from model test CMs |
| `figures/neighbor_confusion_overview.png` | All 7-way phoneme/feature models — error breakdown |
| `figures/neighbor_confusion_3_nb_phoneme.png` | Per-class neighbor confusion, NB Phoneme |
| `figures/neighbor_confusion_4_linear_phoneme.png` | Per-class neighbor confusion, Linear Phoneme |
| `figures/neighbor_confusion_3_wav2vec_base_layer6.png` | Per-class neighbor confusion, Wav2Vec2 Base L6 |
| `figures/neighbor_confusion_7_xlsr_300m_layer8.png` | Per-class neighbor confusion, XLSR 300M L8 |

---

## Key numbers

### Task 1 — Training curves (`vorarlberg_finetuned_10_epochs`)

Computed from `train_scalars_all_runs.csv` via `task1_loss_curves.py` (source: `trainer_state.json` log_history, 6,876 points):

| Metric | Value |
|--------|-------|
| Steps per epoch | 6,876 |
| Total steps (10 epochs) | 68,760 |
| Early loss spike | **20.46** at step **230** |
| Epoch-averaged train_loss (`train_results.json`) | **0.936** |
| Last logged step loss (step 68,750) | 1.542 |

Learning-rate schedule confirmed visually in `figures/learning_rate_vs_step_vorarlberg_10epochs.png` (cosine_with_restarts, peak 3×10⁻⁵).

### Task 2 — Final checkpoint validation loss

**Command:** `CUDA_VISIBLE_DEVICES=6 python scripts/task2_val_loss.py`

| Metric | Value |
|--------|-------|
| mean_loss_text | **0.684** |
| mean_loss_speech | **2.709** |
| mean_total_loss | **3.393** |
| Eval rows (1% split, seed 42) | 1,880 |
| Samples with valid speech labels | 282 |
| Skipped (None or all-masked labels) | 1,598 |
| Empty batches during eval | 238 |

> **Note:** This is the validation loss of the **final checkpoint only**. Per-step `eval_loss` during training was NaN (SpeechDataCollator bug) and is **not** recovered without retraining.

### Task 3 — Synthesis metrics (shared reference: `voice_samples/daniel_ganahl_unfall_montafonerisch.wav`)

**Command:** `CUDA_VISIBLE_DEVICES=6 python scripts/task3_synth_eval.py` (re-run 2025-06-15 on updated `test_sentences.txt`; first attempt on GPU 7 aborted with a CUDA device-side assert on fine-tuned sentence_03)

ASR: faster-whisper **large-v3** (German). Speaker similarity: Chatterbox **VoiceEncoder** cosine (Resemblyzer unavailable). Dialect proxy: Chapter-4 **5_nb_phoneme_binary** Naive Bayes on wav2vec2-espeak phonemes.

| Model | Mean WER ↓ | Mean CER ↓ | Mean RTF | Mean speaker cosine | % classified dialect |
|-------|------------|------------|----------|---------------------|----------------------|
| **kartoffelbox_base** | 0.550 | 0.327 | 0.669 | 0.916 | 20.0% |
| **vorarlberg_finetuned_10_epochs** | **0.287** | **0.214** | 0.990* | 0.874 | 40.0% |

\*Fine-tuned mean RTF inflated by two failed/near-failed generations (sentence_03 RTF = 4.45, sentence_19 RTF = 4.13); median RTF ≈ 0.69 for both models.

**Largest WER improvements (fine-tuned vs base):**
- sentence_01 (greeting): 2.286 → **0.286**
- sentence_07 (long prefix): 2.000 → **0.100**
- sentence_23 (Standard German control): 1.444 → **0.000**
- sentence_10 (voicemail): 1.750 → **0.375**
- sentence_09 (long clause): 1.700 → **0.400**
- sentence_04 (team meeting): 1.250 → **0.125**

**Failures / outliers (fine-tuned struggles):**
- sentence_03 (thanks): WER 1.00, speaker cosine 0.59 — near-silent / collapsed output
- sentence_19 (loanwords): WER 1.00, speaker cosine 0.39 — ASR mismatch on technical terms
- sentence_24 (Gemeindehaus): WER 0.667 — partial intelligibility loss on control sentence

### Task 4 — Dataset size (what to quote)

| Quantity | Value |
|----------|-------|
| Metadata TSV | `vorarlberger_daten_16000.tsv` |
| Raw rows (`wc -l` − 1) | 197,238 |
| Rows with audio on disk | 187,922 |
| **Training samples per epoch (throughput-verified)** | **165,024** (= 6,876 steps × effective batch 24) |
| Gap (load-time drops) | ~21,000 rows/epoch via `__getitem__` → `None` |

---

## Chapter 4 — Swiss SRF dialect labeling (`swiss_srf_espeak_classification_two_stage_2`)

**Source:** `/home/ai/JKU/bachelorthesis_2/eval/dialect_evaluation/swiss_srf_espeak_classification_two_stage_2/`  
**Pipeline:** `two_stage_swiss_srf_espeak_classification.py` — Stage A: `5_nb_phoneme_binary` (German vs non-German); Stage B: `4_linear_phoneme` (7 dialect regions, only for non-German speakers).

| Artifact | Role |
|----------|------|
| `clip_labels.csv` | Per-clip final label (11,075,360 rows) |
| `speaker_assignments.csv` | Speaker-level two-stage decisions (245,520 speakers) |
| `distribution_overall.csv` | Clip/hour/speaker counts by region |
| `distribution_by_podcast.csv`, `distribution_by_episode.csv` | Program-level breakdown |

### Label distribution (`distribution_overall.csv`)

| Region | Clips | % all | % excl. German | Hours | Speakers |
|--------|-------|-------|----------------|-------|----------|
| German | 5,874,248 | 47.4% | — | 8,659.7 | 99,755 |
| Zürich | 3,147,115 | 25.4% | 48.2% | 3,765.2 | 47,890 |
| Bern | 1,425,122 | 11.5% | 21.8% | 1,790.7 | 48,467 |
| Ostschweiz | 809,174 | 6.5% | 12.4% | 965.2 | 20,125 |
| Basel | 537,801 | 4.3% | 8.2% | 628.5 | 10,228 |
| Graubünden | 259,686 | 2.1% | 4.0% | 330.6 | 5,608 |
| Innerschweiz | 194,616 | 1.6% | 3.0% | 265.0 | 3,918 |
| Wallis | 149,250 | 1.2% | 2.3% | 171.7 | 9,529 |

**Headline numbers:** 11.08M clips, 245,520 speakers, 16,577 h total (7,917 h dialect-only). **59.4%** of speakers classified as non-German (`speaker_assignments.csv`). Held-out classifier accuracy on the merged Swiss+DE test set: binary **97.9%** (`models/5_nb_phoneme_binary/test_metrics.json`), 7-way **84.2%** (`models/4_linear_phoneme/test_metrics.json`).

---

## Chapter 4 — Vorarlberg dialect evaluation (`vorarlberg_results`)

**Source:** `/home/ai/JKU/bachelorthesis_2/eval/dialect_evaluation/vorarlberg_results/`

### Multiclass podcast classification (`vorarlberg_classification_results.csv`)

30 s phoneme chunks from Vorarlberg podcasts, classified with the Swiss 7-way `nb_phoneme` model (`vorarlberg_dialect_test.py`):

| Metric | Value |
|--------|-------|
| Chunks | **33,359** |
| Episode–speaker pairs | **1,143** |
| Mean confidence | **0.978** |

**Predicted region (top classes):** Ostschweiz **61.9%** (20,640), Basel **14.3%** (4,776), Zürich **8.7%** (2,887), Wallis **7.7%** (2,566). Geographic neighbor Ostschweiz dominates — consistent with Vorarlberg's eastern-Alpine position.

### Binary German vs dialect (`binary_classification/concat_30/speaker_preds.csv`)

90 s phoneme chunks per speaker, `5_nb_phoneme_binary`:

| Class | Speakers | Share |
|-------|----------|-------|
| Dialect | 771 | **67.0%** |
| High German | 380 | 33.0% |

### Human validation (Label Studio, `batches/self_labeled_vorarlberg.tsv` + `pack.csv`)

200 manually labeled speaker clips vs system predictions:

| Metric | Value |
|--------|-------|
| Accuracy | **91.0%** |
| F1 dialect | **0.913** |
| F1 high German | **0.906** |
| Confusion matrix | [[95, 10], [8, 87]] (rows=true dialect/high German) |

---

## ASR benchmark (`asr/bench_outputs`)

**Source:** `/home/ai/JKU/bachelorthesis_2/asr/bench_outputs/`  
Balanced sample: **300 utterances × 8 labels** = **2,400** records (`20250903_180349__run_manifest.json`, labels 0–7 = Basel…Graubünden + German).

### Main benchmark (`20250903_180349__metrics_table.csv`)

| Rank | Model | WER ↓ | CER ↓ | sacreBLEU ↑ | chrF ↑ | TER ↓ |
|------|-------|-------|-------|-------------|--------|-------|
| 1 | faster-whisper:large-v3 | **0.240** | **0.121** | **64.9** | **84.6** | **23.1** |
| 2 | hf:whisper:openai/whisper-large-v3 | 0.243 | 0.126 | 64.4 | 84.4 | 23.5 |
| 3 | hf:wav2vec2:jonatasgrosman/wav2vec2-large-xlsr-53-german | 0.824 | 0.372 | 5.9 | 45.4 | 70.1 |
| 4 | hf:whisper:Flurin17/whisper-large-v3-turbo-swiss-german | 1.168 | 2.718 | 6.3 | 54.2 | 115.6 |

Per-label WER (faster-whisper): German **0.089**, Innerschweiz **0.218**, Basel **0.236**, Zürich **0.242**; worst: Wallis **0.360** (`20250903_180349__metrics_summary.json`).

### Turbo comparison (`20250903_203225__metrics_table.csv`, also 2,400 utt.)

| Model | WER | CER | sacreBLEU | chrF | TER |
|-------|-----|-----|-----------|------|-----|
| faster-whisper:large-v3-turbo | 0.287 | 0.144 | 59.4 | 81.2 | 27.8 |
| hf:whisper:openai/whisper-large-v3-turbo | 0.319 | 0.164 | 57.3 | 79.0 | 31.1 |

### Subsample run (`20251031_001450__metrics_table.csv`, 1,000 utt.)

| Model | WER | CER | RTF |
|-------|-----|-----|-----|
| faster-whisper:large-v3 | 0.249 | 0.140 | 0.071 |
| nemo:parakeet-tdt-0.6b-v3 | 0.368 | 0.200 | 0.005 |

---

## Neighbor confusion analysis (Swiss 7-way classifiers)

Geographic neighbors (Basel↔Bern, Zürich↔Ostschweiz/Innerschweiz, etc.) are defined in `plot_neighbor_confusion.py` (same schema as `comparison_prints_of_all_models.py`).

**Test-set error decomposition** (from `models/*/test_metrics.json`):

| Model | Accuracy | Neighbor errors | Non-neighbor errors |
|-------|----------|-----------------|---------------------|
| Linear Phoneme | **84.2%** | 11.8% | 4.0% |
| NB Phoneme | 83.5% | 11.8% | 4.7% |
| XLSR 300M L8 | 32.3% | 29.4% | 38.3% |
| Wav2Vec2 Base L6 | 29.3% | 29.0% | 41.7% |
| MelSpec CNN | 19.1% | 36.3% | 44.7% |

Phoneme models misclassify **~3× more often to geographic neighbors than to distant regions**; most errors are neighbor confusions, not random off-diagonal noise.

**Figures** (generated by `plot_neighbor_confusion.py`):

| File | Description |
|------|-------------|
| `figures/neighbor_confusion_overview.png` | Stacked correct / neighbor / non-neighbor counts for all 7-way models |
| `figures/neighbor_confusion_3_nb_phoneme.png` | Per-class breakdown, NB Phoneme |
| `figures/neighbor_confusion_4_linear_phoneme.png` | Per-class breakdown, Linear Phoneme |

```bash
cd /home/ai/JKU/bachelorthesis_2
/home/ai/JKU/.plot_venv/bin/python plot_neighbor_confusion.py \
  --models 3_nb_phoneme 4_linear_phoneme 3_wav2vec_base_layer6 7_xlsr_300m_layer8
```

---

## What these results show

Fine-tuning T3 on ~165k Vorarlberg utterances per epoch for 10 epochs produced a model that **synthesizes more intelligible speech** on the updated 25-sentence benchmark (mean WER 0.55 → 0.29, mean CER 0.33 → 0.21) while **preserving strong voice cloning** (mean speaker cosine 0.92 → 0.87). Gains are largest on greetings, long prefix-heavy sentences, and several Standard German control lines at the end of the list (e.g. sentence_23: WER 1.44 → 0.00).

The Chapter-4 dialect classifier rates **40%** of fine-tuned outputs as dialect vs **20%** for the base Kartoffelbox model on this **predominantly Standard German** test set (short greetings, administrative phrasing, compounds, numbers, loanwords, and four control sentences). Lower absolute percentages vs the earlier 17-sentence dialect-heavy list are expected; the relative fine-tuned > base shift still indicates phoneme patterns closer to the non-Standard-German class — interpret as a proxy only.

Training converged from an early loss spike (~20.5 at step 230) to an epoch-averaged train loss of **0.936** (`train_results.json`). The recovered validation loss is **3.39** (text 0.68 + speech 2.71), giving the thesis a citeable number where `eval_loss = NaN` previously blocked any claim.

Remaining failures are **isolated collapsed generations** (sentence_03, sentence_19: WER 1.0, low speaker cosine) rather than systematic regression across sentence types; most compound, numeric, and multi-clause items show clear fine-tuned WER reductions.

---

## Gaps and limitations

| Item | Status |
|------|--------|
| Per-step validation loss curve during training | **Not recovered** — would require retraining with fixed collator |
| `chatterbox_finetuned_vorarlberg_binary` in Task 3 | **Skipped** — base + final 10-epoch comparison is the core result |
| MCD (mel-cepstral distortion) | **Skipped** — optional stretch; not computed |
| Resemblyzer speaker similarity | **Unavailable** (`pkg_resources` / webrtcvad); used Chatterbox VoiceEncoder instead |
| Dialect classifier sklearn version | Warning: model trained with sklearn 1.7.1, evaluated with 1.9.0 — predictions still ran |
| `resemble-perth` watermarking | Stubbed in eval (`eval_bootstrap.py`) — synthesis WAVs are unwatermarked |
| Test sentence count | **25 sentences** — greetings, prefix variations, long clauses, compounds/phoneme traps, numbers/dates, loanwords, Standard German controls |
| Listening tests / MOS | **Not on disk** — subjective evaluation still needed for Chapter 6 |
| Vorarlberg binary eval script path | `evaluation_of_vorarlberg_self_labeled_and_system.py` expects `/home/ai/bachelorthesis_2/…` (works when repo is symlinked there; use `pack.csv` + TSV merge otherwise) |
| `labels.csv` (Label Studio export) | **Empty** — human labels live in `self_labeled_vorarlberg.tsv` |
| Vorarlberg multiclass ground truth | **No reference labels** — podcast chunks scored by Swiss-trained 7-way model only |

---

## How to reproduce

```bash
EVAL="/home/ai/AI-DataPool/Other/Backup/home_dirs/max_150/experiments/TTS/chatterbox/chatterbox-finetuning/thesis_eval"
source "$EVAL/.venv/bin/activate"
cd "$EVAL/scripts"

python task1_loss_curves.py
python task4_dataset_size.py
CUDA_VISIBLE_DEVICES=6 python task2_val_loss.py
CUDA_VISIBLE_DEVICES=6 python task3_synth_eval.py
```

GPU 6–7 were used because GPUs 0–5 were occupied (~24 GB used each).
