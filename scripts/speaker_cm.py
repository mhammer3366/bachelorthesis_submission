
import os
from pathlib import Path
REPO_ROOT = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parents[1]))
import csv
from collections import Counter, defaultdict
from pathlib import Path

BASE = REPO_ROOT
speaker_map = {}
with open(BASE / "data_preparation/merged_datasets_plus_de/merged_test.tsv", encoding="utf-8") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        speaker_map[row["audio_path"]] = row["client_id"]

sp = defaultdict(lambda: {"preds": [], "true": None})
with open(BASE / "models/7_linear_phoneme_binary/test_preds.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        ap = row["audio_path"]
        cid = None
        for p, c in speaker_map.items():
            if p in ap or ap in p:
                cid = c
                break
        if not cid and ap.startswith("concat_"):
            parts = ap.split("_")
            pid = "_".join(parts[1:-2]) if len(parts) > 3 else parts[1]
            if pid in speaker_map.values():
                cid = pid
        if not cid:
            continue
        sp[cid]["true"] = int(row["binary_label"])
        sp[cid]["preds"].append(int(row["pred_binary"]))

cm = [[0, 0], [0, 0]]
for d in sp.values():
    cm[d["true"]][Counter(d["preds"]).most_common(1)[0][0]] += 1

tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
out = {
    "speaker_cm": cm,
    "n_speakers": len(sp),
    "accuracy": round((tn + tp) / (tn + fp + fn + tp), 3),
    "german_recall": round(tp / (fn + tp), 3),
    "non_german_recall": round(tn / (tn + fp), 3),
    "balanced_accuracy": round((tp / (fn + tp) + tn / (tn + fp)) / 2, 3),
    "german_total": fn + tp,
}
Path(os.environ.get("SPEAKER_CM_OUT", str(BASE / "speaker_cm_out.json"))).write_text(str(out))
