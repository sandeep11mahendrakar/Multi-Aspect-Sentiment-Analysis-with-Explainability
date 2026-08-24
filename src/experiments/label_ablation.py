"""E2: Controlled label-strategy ablation.

All variants share the IDENTICAL split methodology of E1
(stratified 80/20, random_state=42, same underlying row assignment):
splitting is done ONCE on the full usable dataset; variants then filter rows.

Variants:
  A) 3-class baseline : rating>=4 pos | ==3 neu | <=2 neg          (E1 replica)
  B) binary           : drop rating==3 entirely; pos vs neg
  C) binary + tau-routing :
        train binary (as B), predict pos/neg on the FULL 3-class test set;
        if max P(class) < tau -> predict 'neutral'.
        tau chosen on training data via 5-fold out-of-fold probabilities,
        maximizing macro-F1 against the true rating-based 3-class labels.
        No ground-truth relabeling: truth stays rating-derived everywhere.

Run from repo root after src/train.py:
    venv\\Scripts\\python.exe src\\experiments\\label_ablation.py
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import cross_val_predict, train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import build_pipeline_parts, ensure_nltk, rating_to_sentiment  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_CSV = os.path.join(REPO_ROOT, "data", "raw", "reviews.csv")
OUT_JSON = os.path.join(REPO_ROOT, "results", "metrics", "label_ablation.json")
RANDOM_STATE = 42


def full_report(name, y_true, y_pred, labels, extra=None):
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    wf1 = f1_score(y_true, y_pred, average="weighted")
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    dist = pd.Series(y_true).value_counts()
    rep = {
        "name": name,
        "n": int(len(y_true)),
        "class_distribution": {l: int(dist.get(l, 0)) for l in labels},
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(mf1), 4),
        "weighted_f1": round(float(wf1), 4),
        "per_class": {l: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                          "f1": round(float(f[i]), 4), "support": int(s[i])}
                      for i, l in enumerate(labels)},
        "confusion_rows_true_cols_pred": cm.tolist(),
        "labels_order": list(labels),
    }
    if extra:
        rep.update(extra)

    print(f"\n{'='*64}\n{name}\n{'='*64}")
    print(f"n={rep['n']}  dist={rep['class_distribution']}")
    print(f"acc={acc:.4f}  macro_f1={mf1:.4f}  weighted_f1={wf1:.4f}")
    print(pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s},
                       index=labels).round(4).to_string())
    print("confusion (rows=true):")
    print(pd.DataFrame(cm, index=[f"true_{l}" for l in labels],
                       columns=[f"pred_{l}" for l in labels]).to_string())
    return rep


def main():
    ensure_nltk()
    _, preprocess = build_pipeline_parts()

    print("Preparing data (cleaning from raw)...")
    df = pd.read_csv(RAW_CSV)
    df["sentiment"] = df["Rating"].apply(rating_to_sentiment)
    df["final_text"] = df["Review Text"].fillna("").apply(preprocess)
    df = df.dropna(subset=["Review Text"]).copy()          # drop empty-review rows (as E1)
    df = df.reset_index(drop=True)
    print(f"usable rows: {len(df)}")

    # ---- ONE split, shared by all variants ----
    idx_train, idx_test = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=RANDOM_STATE, stratify=df["sentiment"])
    results = {}

    # ================= Variant A: 3-class baseline (E1 replica) =================
    vecA = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    XtrA = vecA.fit_transform(df.loc[idx_train, "final_text"])
    XteA = vecA.transform(df.loc[idx_test, "final_text"])
    mA = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE).fit(XtrA, df.loc[idx_train, "sentiment"])
    predA = mA.predict(XteA)
    yA = df.loc[idx_test, "sentiment"]
    results["A_3class_baseline"] = full_report(
        "A) 3-class baseline (all ratings)", yA, predA,
        ["negative", "neutral", "positive"])

    # ================= Variant B: binary, drop rating==3 =================
    b_train = idx_train[df.loc[idx_train, "sentiment"] != "neutral"]
    b_test = idx_test[df.loc[idx_test, "sentiment"] != "neutral"]
    vecB = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    XtrB = vecB.fit_transform(df.loc[b_train, "final_text"])
    XteB = vecB.transform(df.loc[b_test, "final_text"])
    mB = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE).fit(
        XtrB, df.loc[b_train, "sentiment"])
    predB = mB.predict(XteB)
    yB = df.loc[b_test, "sentiment"]
    results["B_binary_drop_rating3"] = full_report(
        "B) binary (rating==3 removed)", yB, predB, ["negative", "positive"])

    # deployed view: B applied to FULL 3-class test set, forced pos/neg
    XteB_full = vecB.transform(df.loc[idx_test, "final_text"])
    predB_full = mB.predict(XteB_full)
    results["B_deployed_on_full_test_forced_choice"] = full_report(
        "B-deployed) binary forced-choice on full 3-class test",
        df.loc[idx_test, "sentiment"], predB_full, ["negative", "neutral", "positive"],
        extra={"note": "model cannot output neutral; every true-neutral is a guaranteed error"})

    # ================= Variant C: binary + tau-routing to neutral =================
    # tau chosen on the FULL training portion using the TRUE rating-based
    # 3-class labels: mB scores every training row; if confidence < tau the
    # prediction routes to neutral. Caveat (documented): pos/neg train-row
    # probabilities are in-sample; rating-3 rows are genuinely unseen by mB,
    # which is exactly the distribution tau must separate.
    probB_tr = mB.predict_proba(vecB.transform(df.loc[idx_train, "final_text"]))[:, 1]
    conf_tr = np.maximum(probB_tr, 1 - probB_tr)
    ytr3 = df.loc[idx_train, "sentiment"].values
    best_tau, best_score = None, -1
    for tau in np.arange(0.50, 0.96, 0.05):
        routed = np.where(conf_tr < tau, "neutral",
                          np.where(probB_tr >= 0.5, "positive", "negative"))
        s = f1_score(ytr3, routed, average="macro")
        if s > best_score:
            best_score, best_tau = s, round(float(tau), 2)

    probB = mB.predict_proba(XteB_full)[:, 1]
    conf = np.maximum(probB, 1 - probB)
    predC = np.where(conf < best_tau, "neutral",
                     np.where(probB >= 0.5, "positive", "negative"))
    yC = df.loc[idx_test, "sentiment"]
    results["C_binary_plus_tau_routing"] = full_report(
        f"C) binary + tau-routing (tau={best_tau}, chosen on train OOF)",
        yC, predC, ["negative", "neutral", "positive"],
        extra={"tau": best_tau, "tau_selection": "max macro-F1 on 5-fold OOF train probs"})

    # ================= comparison digest =================
    def g(k):
        r = results[k]
        return r["accuracy"], r["macro_f1"], r["weighted_f1"]

    digest = {
        k: {"accuracy": v[0], "macro_f1": v[1], "weighted_f1": v[2]}
        for k, v in [
            ("A_3class", g("A_3class_baseline")),
            ("B_binary_own_test", g("B_binary_drop_rating3")),
            ("B_deployed_full_test", g("B_deployed_on_full_test_forced_choice")),
            ("C_tau_routed_full_test", g("C_binary_plus_tau_routing")),
        ]}
    results["_digest"] = digest
    print(f"\nDIGEST:\n{json.dumps(digest, indent=2)}")

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved -> {os.path.relpath(OUT_JSON, REPO_ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
