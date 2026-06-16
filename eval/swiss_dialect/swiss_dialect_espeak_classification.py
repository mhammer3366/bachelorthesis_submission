#!/usr/bin/env python3

import os
from pathlib import Path
REPO_ROOT = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parents[2]))
# classify_from_phonemes_by_podcast.py
# Stream-friendly dialect classification with podcast/episode awareness and prints.

import os, csv
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
from joblib import load
from tqdm import tqdm

# ---------- CONFIG ----------
TSV_PATH   = "/home/ai/AI-DataPool/Datasets/audio/Schweiz/16000_mono_wav_splitted_1/master_index_with_phonemes.tsv"
MODEL_DIR  = str(REPO_ROOT / "models/nb_phoneme")
OUTPUT_DIR = str(REPO_ROOT / "eval/dialect_evaluation/all_swiss_srf_results")

TARGET_CHUNK_SECS = 90.0
READ_CHUNK_ROWS   = 250_000
CLS_BATCH_SIZE    = 8_000
DEFAULT_ROW_SECS  = 5.0
MISSING_PHONEME   = "NO_PHONEME"

# If True, we flush state when podcast changes in the stream (expects TSV roughly grouped by podcast).
GROUP_BY_PODCAST = True

# Optional: top-up short speakers using their own utterances (simple loop)
PAD_SHORT_SPEAKERS = False

# Columns auto-detect: looks for path/clip_path, phoneme, optional text, duration/duration_sec
# ---------------------------

def detect_cols(tsv: str) -> Dict[str, Optional[str]]:
    head = pd.read_csv(tsv, sep="\t", nrows=0)
    cols = set(head.columns)
    path = "path" if "path" in cols else ("clip_path" if "clip_path" in cols else None)
    if not path: raise ValueError("Need 'path' or 'clip_path'.")
    phon = "phoneme" if "phoneme" in cols else None
    if not phon: raise ValueError("Need 'phoneme'.")
    text = "text" if "text" in cols else None
    dur  = "duration" if "duration" in cols else ("duration_sec" if "duration_sec" in cols else None)
    return {"path": path, "phoneme": phon, "text": text, "duration": dur}

def parse_pod_ep_sp(path_str: str) -> Tuple[str,str,str]:
    """
    Extract (podcast, episode, speaker) from the path.
    Default layout: .../<podcast>/<episode>/<speaker>/<file>
    Adjust indices if your hierarchy differs.
    """
    parts = Path(path_str).parts
    podcast = parts[-4] if len(parts) >= 4 else "unknown_podcast"
    episode = parts[-3] if len(parts) >= 3 else "unknown_episode"
    speaker = parts[-2] if len(parts) >= 2 else "unknown_speaker"
    return podcast, episode, speaker

class NBClassifier:
    def __init__(self, model_dir: str):
        self.model = load(Path(model_dir, "nb_model.joblib"))
        self.vec   = load(Path(model_dir, "vectorizer.joblib"))
    def predict(self, phonemes: List[str]):
        X = self.vec.transform(phonemes)
        y = self.model.predict(X)
        conf = self.model.predict_proba(X).max(axis=1)
        return y, conf

class ConcatState:
    """
    Rolling state for ONE podcast at a time (podcast,episode,speaker).
    """
    def __init__(self, target_secs: float, pad_short: bool):
        self.target = target_secs
        self.pad    = pad_short
        self.buf: Dict[Tuple[str,str], Dict[str,object]] = {}  # key=(episode,speaker)
        self.pool: Dict[Tuple[str,str], List[Tuple[str,str,str,float]]] = {}
        self.current_episode_printed: Optional[str] = None
        self.emitted_chunks = 0

    def _ensure_episode_print(self, podcast: str, episode: str):
        key = f"{podcast}/{episode}"
        if self.current_episode_printed != key:
            print(f"→ Processing episode: {podcast}/{episode}")
            self.current_episode_printed = key

    def add(self, podcast: str, episode: str, speaker: str,
            path: str, phon: str, text: str, dur: float) -> Optional[dict]:
        self._ensure_episode_print(podcast, episode)
        es_key = (episode, speaker)
        self.pool.setdefault(es_key, []).append((path, phon, text, dur))
        st = self.buf.setdefault(es_key, {"paths":[], "phs":[], "txts":[], "dur":0.0, "idx":0, "episode":episode, "speaker":speaker, "podcast":podcast})
        st["paths"].append(path)
        if phon and phon != MISSING_PHONEME: st["phs"].append(phon)
        if text: st["txts"].append(text)
        st["dur"] += float(dur)

        if st["dur"] >= self.target:
            return self._emit(es_key, pad_if_short=False)
        return None

    def _emit(self, es_key: Tuple[str,str], pad_if_short: bool):
        st = self.buf[es_key]
        if pad_if_short and self.pad and st["dur"] < self.target:
            pool = self.pool.get(es_key, [])
            i = 0
            while st["dur"] < self.target and pool and i < len(pool)*3:
                p, ph, tx, du = pool[i % len(pool)]
                if ph and ph != MISSING_PHONEME:
                    st["paths"].append(p); st["phs"].append(ph)
                    if tx: st["txts"].append(tx)
                    st["dur"] += float(du)
                i += 1

        out = {
            "podcast": st["podcast"],
            "episode": st["episode"],
            "speaker": st["speaker"],
            "chunk_id": f"{st['podcast']}_{st['episode']}_{st['speaker']}_{st['idx']}",
            "audio_paths": "|".join(st["paths"]),
            "phonemes": " ".join(st["phs"]).strip(),
            "text": " ".join(st["txts"]).strip(),
            "duration": float(st["dur"]),
            "num_files": int(len(st["paths"])),
        }
        st["paths"].clear(); st["phs"].clear(); st["txts"].clear()
        st["dur"] = 0.0; st["idx"] += 1
        self.emitted_chunks += 1
        return out

    def flush_all(self) -> List[dict]:
        outs = []
        for es_key, st in list(self.buf.items()):
            if st["paths"]:
                o = self._emit(es_key, pad_if_short=True)
                if o and o["phonemes"]:
                    outs.append(o)
        return outs

def build_chunks_grouped(tsv: str, out_csv: str) -> int:
    cols = detect_cols(tsv)
    use = [c for c in [cols["path"], cols["phoneme"], cols["text"], cols["duration"]] if c]

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    n_chunks_total = 0

    # File writer
    writer = None
    def ensure_writer():
        nonlocal writer
        if writer is None:
            f = open(out_csv, "w", newline="", encoding="utf-8")
            writer = csv.DictWriter(f, fieldnames=[
                "podcast","episode","speaker","chunk_id","audio_paths","phonemes","text","duration","num_files"
            ])
            writer.writeheader()

    # State for the current podcast only
    current_podcast: Optional[str] = None
    state: Optional[ConcatState]   = None
    chunks_this_podcast = 0

    reader = pd.read_csv(tsv, sep="\t", usecols=use, dtype=str,
                         chunksize=READ_CHUNK_ROWS, low_memory=True)

    for df in tqdm(reader, desc="Concat", unit="rows"):
        df = df.rename(columns={
            cols["path"]:"path", cols["phoneme"]:"phoneme",
            **({cols["text"]:"text"} if cols["text"] else {}),
            **({cols["duration"]:"duration"} if cols["duration"] else {})
        }).dropna(subset=["path","phoneme"])

        # duration vector
        if "duration" in df.columns:
            durs = pd.to_numeric(df["duration"], errors="coerce").fillna(DEFAULT_ROW_SECS).astype(float).to_numpy()
        else:
            durs = np.full(len(df), DEFAULT_ROW_SECS, dtype=float)

        paths = df["path"].tolist()
        phons = df["phoneme"].tolist()
        texts = df["text"].tolist() if "text" in df.columns else [""] * len(df)

        for p, ph, tx, du in zip(paths, phons, texts, durs):
            pod, ep, sp = parse_pod_ep_sp(str(p))

            # Handle podcast boundary (if requested)
            if GROUP_BY_PODCAST and current_podcast is not None and pod != current_podcast:
                # flush previous podcast completely
                outs = state.flush_all()
                if outs:
                    ensure_writer()
                    for o in outs:
                        writer.writerow(o)
                    n_chunks_total += len(outs)
                    chunks_this_podcast += len(outs)
                print(f"✔ Finished podcast: {current_podcast} (chunks: {chunks_this_podcast})")
                # reset for new podcast
                state = ConcatState(TARGET_CHUNK_SECS, PAD_SHORT_SPEAKERS)
                chunks_this_podcast = 0

            # Initialize state for very first row or podcast switch
            if state is None or current_podcast != pod:
                current_podcast = pod
                if state is None:
                    state = ConcatState(TARGET_CHUNK_SECS, PAD_SHORT_SPEAKERS)
                print(f"\n=== Podcast: {current_podcast} ===")

            out = state.add(pod, ep, sp, str(p), str(ph), str(tx or ""), float(du))
            if out and out["phonemes"]:
                ensure_writer()
                writer.writerow(out)
                n_chunks_total += 1
                chunks_this_podcast += 1

    # End of file: flush remaining (last podcast)
    if state is not None:
        outs = state.flush_all()
        if outs:
            ensure_writer()
            for o in outs:
                writer.writerow(o)
            n_chunks_total += len(outs)
            chunks_this_podcast += len(outs)
        if current_podcast is not None:
            print(f"✔ Finished podcast: {current_podcast} (chunks: {chunks_this_podcast})")

    return n_chunks_total

def classify_chunks(model_dir: str, in_csv: str, out_csv: str, summary_csv: str):
    clf = NBClassifier(model_dir)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    writer = None
    summary: Dict[Tuple[str,str,str], Dict[str,object]] = {}  # key=(pod,ep,sp)

    reader = pd.read_csv(in_csv, dtype=str, chunksize=CLS_BATCH_SIZE, low_memory=True)
    for df in tqdm(reader, desc="Classify", unit="chunks"):
        df["phonemes"] = df["phonemes"].fillna("").astype(str).str.strip()
        df = df[df["phonemes"] != ""]
        if len(df) == 0: continue

        y, conf = clf.predict(df["phonemes"].tolist())
        df["predicted_dialect"] = y
        df["confidence"] = conf

        if writer is None:
            writer = csv.DictWriter(open(out_csv, "w", newline="", encoding="utf-8"),
                                    fieldnames=list(df.columns))
            writer.writeheader()
        for r in df.itertuples(index=False):
            writer.writerow(r._asdict())

        for (pod, ep, sp), g in df.groupby(["podcast","episode","speaker"]):
            entry = summary.setdefault((pod,ep,sp), {
                "num_chunks":0,"total_files":0,"total_duration":0.0,"preds":[], "confs":[]
            })
            entry["num_chunks"]     += len(g)
            entry["total_files"]    += g["num_files"].astype(int).sum()
            entry["total_duration"] += g["duration"].astype(float).sum()
            entry["preds"]  += g["predicted_dialect"].tolist()
            entry["confs"]  += g["confidence"].astype(float).tolist()

    # write summary
    rows = []
    for (pod,ep,sp), s in summary.items():
        preds = pd.Series(s["preds"])
        rows.append({
            "podcast": pod, "episode": ep, "speaker": sp,
            "num_chunks": s["num_chunks"],
            "total_files": s["total_files"],
            "total_duration": s["total_duration"],
            "predicted_dialect": preds.mode().iloc[0] if len(preds) else -1,
            "avg_confidence": float(np.mean(s["confs"])) if s["confs"] else 0.0,
            "confidence_std": float(np.std(s["confs"])) if len(s["confs"])>1 else 0.0,
        })
    pd.DataFrame(rows).sort_values(["podcast","episode","speaker"]).to_csv(summary_csv, index=False)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    chunks_csv = str(Path(OUTPUT_DIR, f"chunks_from_phonemes_by_podcast{'_padded' if PAD_SHORT_SPEAKERS else ''}.csv"))

    print(f"➤ Building ~{TARGET_CHUNK_SECS}s chunks from {TSV_PATH}")
    n = build_chunks_grouped(TSV_PATH, chunks_csv)
    print(f"✓ Total chunks: {n} → {chunks_csv}")

    per_chunk_csv = str(Path(OUTPUT_DIR, "classification_per_chunk.csv"))
    summary_csv   = str(Path(OUTPUT_DIR, "episode_speaker_summary.csv"))
    print(f"➤ Classifying with model at {MODEL_DIR}")
    classify_chunks(MODEL_DIR, chunks_csv, per_chunk_csv, summary_csv)
    print(f"✓ Per-chunk → {per_chunk_csv}")
    print(f"✓ Summary   → {summary_csv}")

if __name__ == "__main__":
    main()
