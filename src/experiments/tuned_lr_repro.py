"""Reproduce and explain the tuned-LR regression (nb02: 0.8072 < untuned 0.8231).

Findings (verified):
  - CV (f1_weighted, cv=5) genuinely prefers C=10 over C=1 -> GridSearchCV worked
    correctly BY ITS OWN CRITERION; the metric it maximizes is not accuracy.
  - Test accuracy peaks at C=1 (default): C=10 mildly overfits -> -1.6 acc pts.
  - Solver is NOT the cause: saga(C=1,l2) == lbfgs(C=1) == 0.8223.
  - class_weight='balanced' collapses accuracy (~0.70): trades majority
    precision for minority recall.

Run from repo root after src/train.py:
    python src\\experiments\\tuned_lr_repro.py
"""

import json
import os
import sys
import warnings

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLEANED_CSV = os.path.join(REPO_ROOT, "data", "processed", "cleaned_reviews.csv")
METRICS_DIR = os.path.join(REPO_ROOT, "results", "metrics")

warnings.filterwarnings("ignore")


def eval_config(A, B, ytr, yte, with_cv=True, **kw):
    m = LogisticRegression(max_iter=2000, random_state=42, **kw).fit(A, ytr)
    out = {
        "test_accuracy": round(float(accuracy_score(yte, m.predict(B))), 4),
        "test_macro_f1": round(float(f1_score(yte, m.predict(B), average="macro")), 4),
        "n_iter": int(m.n_iter_[0]),
    }
    if with_cv:
        cv = cross_val_score(LogisticRegression(max_iter=2000, random_state=42, **kw),
                             A, ytr, cv=5, scoring="f1_weighted")
        out["cv_f1_weighted"] = round(float(cv.mean()), 4)
    return out


def main():
    df = pd.read_csv(CLEANED_CSV).dropna(subset=["final_text", "sentiment"])
    Xtr, Xte, ytr, yte = train_test_split(
        df["final_text"], df["sentiment"], test_size=0.2, random_state=42,
        stratify=df["sentiment"])
    tfidf = joblib.load(os.path.join(REPO_ROOT, "models", "tfidf_vectorizer.pkl"))
    A, B = tfidf.transform(Xtr), tfidf.transform(Xte)

    results = {
        "lbfgs_C=0.1": eval_config(A, B, ytr, yte, C=0.1),
        "lbfgs_C=1_default": eval_config(A, B, ytr, yte, C=1),
        "lbfgs_C=10": eval_config(A, B, ytr, yte, C=10),
        "saga_C=10_l2_nb02_best": eval_config(A, B, ytr, yte, solver="saga", penalty="l2", C=10),
        "saga_C=1_l2": eval_config(A, B, ytr, yte, solver="saga", penalty="l2", C=1),
        "saga_C=10_l2_balanced": eval_config(A, B, ytr, yte, with_cv=False,
                                             solver="saga", penalty="l2", C=10,
                                             class_weight="balanced"),
        "conclusion": (
            "GridSearchCV selected C=10 because it maximizes f1_weighted "
            "(CV 0.8073 > 0.7976 at C=1), but test ACCURACY is lower at C=10 "
            "(~0.806-0.807) than at C=1 (~0.822). No implementation bug: "
            "metric-objective mismatch + mild overfitting. saga==lbfgs at equal "
            "hyperparams; class_weight='balanced' costs ~12 accuracy points."
        ),
    }

    os.makedirs(METRICS_DIR, exist_ok=True)
    with open(os.path.join(METRICS_DIR, "tuned_lr_repro.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print("\nSaved -> results/metrics/tuned_lr_repro.json")


if __name__ == "__main__":
    sys.exit(main())
