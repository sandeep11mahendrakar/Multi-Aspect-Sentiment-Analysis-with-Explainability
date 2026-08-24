"""Evaluation harness: full metrics from a predictions CSV (y_true, y_pred).

Run from repo root:
    python src\\evaluate.py                       # baseline predictions
    python src\\evaluate.py --pred <path> --name <label>
"""

import argparse
import json
import os

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = ["negative", "neutral", "positive"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=os.path.join(REPO_ROOT, "results", "metrics", "baseline_predictions.csv"))
    ap.add_argument("--name", default="TF-IDF + LogisticRegression (baseline)")
    args = ap.parse_args()

    df = pd.read_csv(args.pred)
    y_true, y_pred = df["y_true"], df["y_pred"]

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    prec, rec, f1s, support = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    per_class = pd.DataFrame(
        {"precision": prec, "recall": rec, "f1": f1s, "support": support}, index=LABELS
    )
    dist_true = y_true.value_counts()
    dist_pct = (y_true.value_counts(normalize=True) * 100).round(2)

    out_dir = os.path.dirname(args.pred)
    stem = os.path.splitext(os.path.basename(args.pred))[0]
    per_class.to_csv(os.path.join(out_dir, f"{stem}_per_class.csv"))
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in LABELS], columns=[f"pred_{l}" for l in LABELS])
    cm_df.to_csv(os.path.join(out_dir, f"{stem}_confusion_matrix.csv"))

    summary = {
        "model": args.name,
        "n_test": int(len(df)),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class": {l: {"precision": round(float(prec[i]), 4),
                          "recall": round(float(rec[i]), 4),
                          "f1": round(float(f1s[i]), 4),
                          "support": int(support[i])} for i, l in enumerate(LABELS)},
        "class_distribution_test": {l: int(dist_true.get(l, 0)) for l in LABELS},
        "class_distribution_pct": {l: float(dist_pct.get(l, 0.0)) for l in LABELS},
        "confusion_matrix_rows_true_cols_pred": cm.tolist(),
        "labels_order": LABELS,
    }
    with open(os.path.join(out_dir, f"{stem}_evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 60)
    print(f"EVALUATION: {args.name}")
    print("=" * 60)
    print(f"n_test       : {len(df)}")
    print(f"Accuracy     : {acc:.4f}")
    print(f"Macro-F1     : {macro_f1:.4f}")
    print(f"Weighted-F1  : {weighted_f1:.4f}")
    print("\nPer-class metrics:")
    print(per_class.round(4).to_string())
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm_df.to_string())
    print("\nTest-set class distribution:")
    for l in LABELS:
        print(f"  {l:<10} {dist_true.get(l, 0):>5} ({dist_pct.get(l, 0):.2f}%)")
    print(f"\nSaved: {stem}_evaluation.json / _per_class.csv / _confusion_matrix.csv")


if __name__ == "__main__":
    main()
