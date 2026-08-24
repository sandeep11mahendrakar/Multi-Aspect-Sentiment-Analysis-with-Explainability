"""E7B: evaluate models on the human-labeled 199-review gold subset (non-circular).

Uses results/e7a/e7a_labeling_round1.csv (Sandeep's blind labels, score 0-9,
collapsed <=3 neg | 4-6 neu | >=7 pos) as ground truth and runs inference with
both DistilBERT variants on the raw texts.

Run: venv\\Scripts\\python.exe -u src\\experiments\\eval_gold_subset.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GOLD_CSV = os.path.join(REPO_ROOT, "results", "e7a", "e7a_labeling_round1.csv")
OUT_JSON = os.path.join(REPO_ROOT, "results", "e7a", "gold_subset_eval.json")
MODELS = [
    "distilbert_3class_baseline",
    "distilbert_3class_baseline_textuallabels",
]
MAX_LEN = 128


def collapse(score):
    return 0 if score <= 3 else (2 if score >= 7 else 1)


def main():
    df = pd.read_csv(GOLD_CSV).dropna(subset=["score"]).reset_index(drop=True)
    y = df["score"].astype(int).apply(collapse).to_numpy()
    texts = df["Review Text"].tolist()
    print(f"gold subset: {len(df)} reviews | dist "
          f"{[int((y == i).sum()) for i in range(3)]} (neg/neu/pos)")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"n": len(df),
           "human_dist": {["negative", "neutral", "positive"][i]: int((y == i).sum())
                          for i in range(3)},
           "models": {}}
    for name in MODELS:
        mdir = os.path.join(REPO_ROOT, "models", name)
        tok = AutoTokenizer.from_pretrained(mdir)
        model = AutoModelForSequenceClassification.from_pretrained(mdir).to(dev).eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(texts), 64):
                enc = tok(texts[i:i + 64], truncation=True, max_length=MAX_LEN,
                          padding=True, return_tensors="pt").to(dev)
                preds.append(model(**enc).logits.argmax(dim=-1).cpu().numpy())
        pred = np.concatenate(preds)
        res = {
            "accuracy": round(float(accuracy_score(y, pred)), 4),
            "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
            "confusion_rows_human_cols_pred":
                confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
        }
        out["models"][name] = res
        print(f"\n{name}: acc={res['accuracy']} macro_f1={res['macro_f1']}")
        print(pd.DataFrame(
            res["confusion_rows_human_cols_pred"],
            index=[f"human_{l}" for l in ("neg", "neu", "pos")],
            columns=[f"pred_{l}" for l in ("neg", "neu", "pos")]).to_string())

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n-> {os.path.relpath(OUT_JSON, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
