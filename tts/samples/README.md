# TTS synthesis samples (Chapter 6)

**25 test sentences** × **2 models** = **50 WAV files** from the inference-only evaluation (`thesis_eval`, June 2025).

| Folder | Model |
|--------|--------|
| `kartoffelbox_base/` | Base Kartoffelbox v0.1 (Standard German TTS) |
| `vorarlberg_finetuned_10_epochs/` | Fine-tuned on ~165k Vorarlberg utterances / epoch (10 epochs) |

- **Reference voice:** same speaker clip for all utterances (`daniel_ganahl_unfall_montafonerisch.wav` on the author machine).
- **Input text:** `test_sentences.txt` (one sentence per line, `sentence_01` … `sentence_25`).
- **Metrics:** mean WER 0.550 → 0.287 (base → fine-tuned); see thesis Chapter 6.

**Suggested listening pairs for a quick A/B check:**

| File | Content (short) |
|------|-----------------|
| `sentence_01.wav` | Greeting — largest WER improvement |
| `sentence_07.wav` | Long prefix-heavy sentence |
| `sentence_23.wav` | Standard German control (fine-tuned WER 0.00) |

Compare the same filename in both folders.
