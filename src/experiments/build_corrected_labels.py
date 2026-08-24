"""E7B step 1: build textually-corrected labels for rating-3 reviews.

E7A verdict (results/e7a/e7a_analysis.json):
  - only ~28% of rating-3 reviews are truly neutral by text sentiment
  - human review validated the baseline DistilBERT as near-correct on the
    disagreement set, so its predictions are used as the relabeling source

Keeps rating>=4 -> positive and rating<=2 -> negative UNCHANGED.
Relabels rating==3 rows to the model's predicted textual sentiment.

Output: data/processed/corrected_rating3_labels.csv
    row_index      positional index into the 22,641-row usable frame
                   (identical ordering to train_transformer.load_split)
    textual_label  negative | neutral | positive
    pred_conf      model confidence

Run: venv\\Scripts\\python.exe -u src\\experiments\\build_corrected_labels.py
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_CSV = os.path.join(REPO_ROOT, "data", "raw", "reviews.csv")
MODEL_DIR = os.path.join(REPO_ROOT, "models", "distilbert_3class_baseline")
OUT_CSV = os.path.join(REPO_ROOT, "data", "processed", "corrected_rating3_labels.csv")
MAX_LEN = 128
LABELS = ["negative", "neutral", "positive"]


def main():
    df = pd.read_csv(RAW_CSV).dropna(subset=["Review Text"]).reset_index(drop=True)
    neu_idx = df.index[df["Rating"] == 3].to_numpy()
    print(f"usable rows: {len(df)} | rating-3 rows to relabel: {len(neu_idx)}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(dev).eval()

    texts = df["Review Text"].iloc[neu_idx].tolist()
    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            enc = tok(texts[i:i + 64], truncation=True, max_length=MAX_LEN,
                      padding=True, return_tensors="pt").to(dev)
            probs.append(torch.softmax(model(**enc).logits, dim=-1).cpu().numpy())
    P = np.vstack(probs)

    out = pd.DataFrame({
        "row_index": neu_idx,
        "textual_label": [LABELS[i] for i in P.argmax(axis=1)],
        "pred_conf": P.max(axis=1),
    })
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    dist = out["textual_label"].value_counts(normalize=True)
    print("\nformer 'neutral' bucket, textually relabeled:")
    for l in LABELS:
        print(f"  {l:<10} {int((out['textual_label'] == l).sum()):5d}  ({dist.get(l, 0)*100:.1f}%)")
    print(f"\nfull corrected distribution:")
    n_pos = int((df['Rating'] >= 4).sum())
    n_neg = int((df['Rating'] <= 2).sum())
    print(f"  positive {(df['Rating'] >= 4).sum() + int((out['textual_label'] == 'positive').sum())}")
    print(f"  negative {(df['Rating'] <= 2).sum() + int((out['textual_label'] == 'negative').sum())}")
    print(f"  neutral  {int((out['textual_label'] == 'neutral').sum())}")
    print(f"-> {os.path.relpath(OUT_CSV, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
