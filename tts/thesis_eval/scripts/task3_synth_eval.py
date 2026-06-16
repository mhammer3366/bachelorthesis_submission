#!/usr/bin/env python3
"""Task 3: Synthesis evaluation harness — WER/CER, RTF, speaker cosine, dialect classifier."""
from __future__ import annotations

import os
from pathlib import Path

CHATTERBOX_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(os.environ.get("THESIS_ROOT", CHATTERBOX_ROOT.parents[1]))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/home/ai/AI-DataPool/Datasets"))
MODELS_ROOT = Path(os.environ.get("MODELS_ROOT", DATA_ROOT.parent / "Models"))
CHECKPOINTS_DIR = Path(os.environ.get("CHECKPOINTS_DIR", MODELS_ROOT / "TTS" / "chatterbox"))
BASE = CHATTERBOX_ROOT


import csv
import re
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from joblib import load

BASE = CHATTERBOX_ROOT
SRC = BASE / "src"
SCRIPTS = BASE / "thesis_eval" / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SRC))

import eval_bootstrap  # noqa: F401, E402

from chatterbox.models.s3gen import S3GEN_SR
from chatterbox.models.s3tokenizer import S3_SR
from chatterbox.tts import ChatterboxTTS

OUT = BASE / "thesis_eval"
SAMPLES = OUT / "samples"
SENTENCES = OUT / "test_sentences.txt"
REF_VOICE = BASE / "voice_samples" / "daniel_ganahl_unfall_montafonerisch.wav"
NB_DIR = Path(os.environ.get("CLASSIFIER_MODELS_DIR", str(REPO_ROOT / "models"))) / "5_nb_phoneme_binary"
PHONEME_MODEL = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"

MODELS = {
    "kartoffelbox_base": BASE.parent / "kartoffelbox_model_v0.1",
    "vorarlberg_finetuned_10_epochs": BASE / "checkpoints" / "vorarlberg_finetuned_10_epochs",
}

CLASS_NAMES = ["dialect", "german"]  # non_german=0, german=1 in NB model


def load_sentences(path: Path) -> list[str]:
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    return " ".join(s.split())


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            cost = 0 if ca == cb else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[lb]


def wer_cer(ref: str, hyp: str) -> tuple[float, float]:
    ref_n, hyp_n = normalize_text(ref), normalize_text(hyp)
    if not ref_n:
        return (0.0 if not hyp_n else 1.0), (0.0 if not hyp else 1.0)
    ref_w, hyp_w = ref_n.split(), hyp_n.split()
    w_dist = edit_distance(" ".join(ref_w), " ".join(hyp_w))
    # word-level WER via token edit distance
    def tok_dist(r, h):
        n, m = len(r), len(h)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if r[i - 1] == h[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        return dp[n][m]

    w_err = tok_dist(ref_w, hyp_w) / max(len(ref_w), 1)
    c_err = edit_distance(ref_n.replace(" ", ""), hyp_n.replace(" ", "")) / max(len(ref_n.replace(" ", "")), 1)
    return w_err, c_err


class PhonemeExtractor:
    def __init__(self, device: str):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(PHONEME_MODEL)
        self.model = Wav2Vec2ForCTC.from_pretrained(PHONEME_MODEL).to(device).eval()

    @torch.no_grad()
    def extract(self, wav_path: Path) -> str:
        y, _ = librosa.load(str(wav_path), sr=16000, mono=True)
        inputs = self.processor(y, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)
        text = self.processor.batch_decode(pred_ids)[0].strip()
        return text or "NO_PHONEME"


class DialectClassifier:
    def __init__(self, model_dir: Path):
        self.model = load(model_dir / "nb_model.joblib")
        self.vectorizer = load(model_dir / "vectorizer.joblib")

    def predict(self, phoneme: str) -> tuple[str, float]:
        if not phoneme.strip():
            return "unknown", 0.0
        X = self.vectorizer.transform([phoneme])
        proba = self.model.predict_proba(X)[0]
        pred = int(proba.argmax())
        return CLASS_NAMES[pred], float(proba[pred])


def speaker_embedding(ve, wav_path: Path) -> np.ndarray:
    y, _ = librosa.load(str(wav_path), sr=S3_SR, mono=True)
    emb = ve.embeds_from_wavs([y], sample_rate=S3_SR)[0]
    return emb


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def synthesize_and_eval(
    model_name: str,
    ckpt_dir: Path,
    sentences: list[str],
    ref_voice: Path,
    asr_model,
    phoneme_ext: PhonemeExtractor,
    dialect_clf: DialectClassifier,
    device: str,
) -> list[dict]:
    print(f"\n=== Model: {model_name} ({ckpt_dir}) ===")
    tts = ChatterboxTTS.from_local(str(ckpt_dir), device=device)
    ref_emb = speaker_embedding(tts.ve, ref_voice)

    out_dir = SAMPLES / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, text in enumerate(sentences, 1):
        sid = f"sentence_{i:02d}"
        wav_path = out_dir / f"{sid}.wav"
        t0 = time.perf_counter()
        wav = tts.generate(text, audio_prompt_path=str(ref_voice))
        dt = time.perf_counter() - t0
        wav_np = wav.squeeze().numpy()
        sf.write(str(wav_path), wav_np, S3GEN_SR)
        duration = len(wav_np) / S3GEN_SR
        rtf = dt / max(duration, 1e-6)

        segments, _ = asr_model.transcribe(str(wav_path), language="de")
        hyp = " ".join(s.text for s in segments).strip()
        w, c = wer_cer(text, hyp)

        synth_emb = speaker_embedding(tts.ve, wav_path)
        spk_cos = cosine_sim(ref_emb, synth_emb)

        phon = phoneme_ext.extract(wav_path)
        d_pred, d_prob = dialect_clf.predict(phon)

        row = {
            "model": model_name,
            "sentence_id": sid,
            "text": text,
            "wav_path": str(wav_path),
            "wer": round(w, 4),
            "cer": round(c, 4),
            "rtf": round(rtf, 4),
            "speaker_cosine": round(spk_cos, 4),
            "dialect_pred": d_pred,
            "dialect_prob": round(d_prob, 4),
            "asr_hypothesis": hyp,
        }
        rows.append(row)
        print(f"  {sid}: WER={w:.3f} CER={c:.3f} RTF={rtf:.3f} spk={spk_cos:.3f} dialect={d_pred}({d_prob:.2f})")

    del tts
    torch.cuda.empty_cache()
    return rows


def print_summary(rows: list[dict]) -> None:
    from collections import defaultdict

    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    print("\n=== Per-model summary ===")
    print(f"{'model':<35} {'WER':>6} {'CER':>6} {'RTF':>6} {'spk_cos':>8} {'%dialect':>9}")
    for model, rs in sorted(by_model.items()):
        mean_wer = np.mean([x["wer"] for x in rs])
        mean_cer = np.mean([x["cer"] for x in rs])
        mean_rtf = np.mean([x["rtf"] for x in rs])
        mean_spk = np.mean([x["speaker_cosine"] for x in rs])
        pct_dialect = 100 * sum(1 for x in rs if x["dialect_pred"] == "dialect") / len(rs)
        print(f"{model:<35} {mean_wer:6.3f} {mean_cer:6.3f} {mean_rtf:6.3f} {mean_spk:8.3f} {pct_dialect:8.1f}%")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if not REF_VOICE.is_file():
        raise SystemExit(f"Reference voice not found: {REF_VOICE}")
    if not NB_DIR.is_dir():
        raise SystemExit(f"Dialect model dir not found: {NB_DIR}")

    sentences = load_sentences(SENTENCES)
    print(f"Loaded {len(sentences)} test sentences from {SENTENCES}")

    from faster_whisper import WhisperModel

    asr = WhisperModel("large-v3", device=device, compute_type="float16" if device == "cuda" else "int8")
    phoneme_ext = PhonemeExtractor(device)
    dialect_clf = DialectClassifier(NB_DIR)

    all_rows = []
    for name, ckpt in MODELS.items():
        if not ckpt.is_dir():
            print(f"SKIP missing checkpoint: {ckpt}")
            continue
        all_rows.extend(
            synthesize_and_eval(
                name, ckpt, sentences, REF_VOICE, asr, phoneme_ext, dialect_clf, device
            )
        )

    csv_path = OUT / "synth_eval_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model", "sentence_id", "text", "wav_path", "wer", "cer", "rtf",
                "speaker_cosine", "dialect_pred", "dialect_prob", "asr_hypothesis",
            ],
        )
        w.writeheader()
        w.writerows(all_rows)

    print_summary(all_rows)
    print(f"\nWrote {csv_path} ({len(all_rows)} rows)")
    print("Task 3 complete.")


if __name__ == "__main__":
    main()
