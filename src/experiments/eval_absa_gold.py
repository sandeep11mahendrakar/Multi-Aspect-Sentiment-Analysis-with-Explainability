"""Path B step 6: PRIMARY eval of the ABSA model on the human gold set.

Reads Sandeep's labeled absa_gold_labeling.xlsx (score column filled, 0-9 scale,
<=3 negative | 4-6 neutral | >=7 positive) and scores the trained
distilbert_absa_aspectconditioned model against those human labels.

Run after labeling:
  venv\\Scripts\\python.exe -u src\\experiments\\eval_absa_gold.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GOLD_XLSX = os.path.join(REPO_ROOT, "results", "e7b_path_b", "absa_gold_labeling.xlsx")
OUT_JSON = os.path.join(REPO_ROOT, "results", "e7b_path_b", "absa_gold_eval.json")
MODEL_DIR = os.path.join(REPO_ROOT, "models", "distilbert_absa_aspectconditioned_consensus")
LABELS = ["negative", "neutral", "positive"]
MAX_LEN = 96


def collapse(score):
    return 0 if score <= 3 else (2 if score >= 7 else 1)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    df = pd.read_excel(GOLD_XLSX, sheet_name="gold")
    df = df.dropna(subset=["score"]).reset_index(drop=True)
    if len(df) == 0:
        print("No labeled rows found in the 'score' column - label the xlsx first.")
        return 1
    y = df["score"].astype(int).apply(collapse).to_numpy()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(dev).eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(df), 64):
            enc = tok(df["aspect"].iloc[i:i + 64].tolist(),
                      df["sentence"].iloc[i:i + 64].tolist(),
                      truncation=True, max_length=MAX_LEN, padding=True,
                      return_tensors="pt").to(dev)
            preds.extend(model(**enc).logits.argmax(dim=-1).cpu().numpy().tolist())
    pred = np.array(preds)

    acc = accuracy_score(y, pred)
    mf1 = f1_score(y, pred, average="macro")
    rec = recall_score(y, pred, average=None, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])

    print(f"===== HUMAN GOLD SET (PRIMARY) {args.tag} =====")
    print(f"model: {args.model_dir}")
    print(f"n={len(df)} | acc={acc:.4f} macro_f1={mf1:.4f} "
          f"| recall neg={rec[0]:.3f} neu={rec[1]:.3f} pos={rec[2]:.3f}")
    print(pd.DataFrame(cm, index=[f"human_{l}" for l in LABELS],
                       columns=[f"pred_{l}" for l in LABELS]).to_string())

    out = {"model_dir": os.path.relpath(args.model_dir, REPO_ROOT),
           "n_labeled": len(df), "accuracy": round(float(acc), 4),
           "macro_f1": round(float(mf1), 4),
           "precision": None,
           "recall": {l: round(float(r), 4) for l, r in zip(LABELS, rec)},
           "confusion_rows_human_cols_pred": cm.tolist()}
    tp = np.diag(cm).astype(float)
    ppc = [tp[i] / cm[:, i].sum() if cm[:, i].sum() else 0.0 for i in range(3)]
    out["precision"] = {l: round(float(p), 4) for l, p in zip(LABELS, ppc)}
    out_path = OUT_JSON.replace(".json", f"{args.tag}.json") if args.tag else OUT_JSON
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n-> {os.path.relpath(out_path, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
