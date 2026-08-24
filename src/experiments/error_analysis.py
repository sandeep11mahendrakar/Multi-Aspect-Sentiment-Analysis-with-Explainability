"""E6: Quantitative error analysis of the DistilBERT 3-class baseline.

Uses results/metrics/distilbert_3class_baseline_predictions.csv plus the
reconstructed E1/E2 test split, the raw review texts, and model confidences
(recomputed from models/distilbert_3class_baseline).

Run from repo root:
    venv\\Scripts\\python.exe -u src\\experiments\\error_analysis.py
"""

import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_CSV = os.path.join(REPO_ROOT, "data", "raw", "reviews.csv")
PREDS_CSV = os.path.join(REPO_ROOT, "results", "metrics", "distilbert_3class_baseline_predictions.csv")
MODEL_DIR = os.path.join(REPO_ROOT, "models", "distilbert_3class_baseline")
OUT_JSON = os.path.join(REPO_ROOT, "results", "metrics", "error_analysis.json")
MAX_LEN = 128
LABELS = ["negative", "neutral", "positive"]
SEED = 42

NEGATION_RE = re.compile(r"\b(not|no|never|nothing|without|hardly|barely|n't)\b", re.I)
BUT_RE = re.compile(r"\bbut\b|\bhowever\b|\bthough\b", re.I)
ASPECT_RES = {
    "fit_size": re.compile(r"\b(fit|fits|fitted|fitting|size|sized|sizing|small|large|tight|loose|run)s?\b", re.I),
    "quality": re.compile(r"\b(quality|material|fabric|cheap|well.?made|flimsy)\b", re.I),
    "shipping": re.compile(r"\b(shipping|delivery|delivered|arrived|package)\b", re.I),
}


def main():
    # ---- reconstruct identical test split ----
    df = pd.read_csv(RAW_CSV)
    df = df.dropna(subset=["Review Text"]).reset_index(drop=True)
    df["label"] = df["Rating"].apply(lambda r: 0 if r <= 2 else (2 if r >= 4 else 1))
    _, idx_test = train_test_split(np.arange(len(df)), test_size=0.2,
                                   random_state=SEED, stratify=df["label"])
    test = df.loc[idx_test].reset_index(drop=True)

    preds = pd.read_csv(PREDS_CSV)
    assert len(preds) == len(test) and (preds["y_true"] == test["label"]).all(), \
        "prediction file does not align with reconstructed test split"
    test["pred"] = preds["y_pred"].values

    # ---- confidences from saved model ----
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(test), 64):
            enc = tok(list(test["Review Text"].iloc[i:i+64]), truncation=True,
                      max_length=MAX_LEN, padding=True, return_tensors="pt").to(dev)
            probs.append(torch.softmax(model(**enc).logits, dim=-1).cpu().numpy())
    P = np.vstack(probs)
    test["prob"] = P.max(axis=1)

    test["words"] = test["Review Text"].str.split().str.len()
    test["has_negation"] = test["Review Text"].str.contains(NEGATION_RE)
    test["has_contrast"] = test["Review Text"].str.contains(BUT_RE)
    for k, rx in ASPECT_RES.items():
        test[f"has_{k}"] = test["Review Text"].str.contains(rx)
    test["correct"] = test["label"] == test["pred"]

    out = {"total": int(len(test)), "errors": int((~test["correct"]).sum()),
           "error_rate": round(float((~test["correct"]).mean()), 4)}

    # ---- 1. error taxonomy ----
    taxo = (test[~test["correct"]]
            .groupby(["true_label", "wrong_pred"])
            if False else None)
    et = test[~test["correct"]].copy()
    et["true_label"] = et["label"].map(dict(enumerate(LABELS)))
    et["wrong_pred"] = et["pred"].map(dict(enumerate(LABELS)))
    tax = et.groupby(["true_label", "wrong_pred"]).agg(
        n=("correct", "size"), mean_conf=("prob", "mean"),
        mean_words=("words", "mean"), pct_negation=("has_negation", "mean"),
        pct_contrast=("has_contrast", "mean")).round(3)
    out["taxonomy"] = tax.reset_index().to_dict("records")
    print("=== ERROR TAXONOMY ===")
    print(tax.to_string())

    # ---- 2. length effect ----
    ln = test.groupby(["label", "correct"])["words"].agg(["count", "mean", "median"]).round(1)
    out["length_by_class_correctness"] = ln.reset_index().to_dict("records")
    print("\n=== WORD COUNT: correct vs misclassified (per true class) ===")
    print(ln.to_string())
    q = test["words"].quantize if False else test["words"].quantile([0.25, 0.5, 0.75]).to_dict()
    bins = [0, q[0.25], q[0.5], q[0.75], 10**9]
    names = ["short25", "mid-low", "mid-high", "long25"]
    test["len_bin"] = pd.cut(test["words"], bins=bins, labels=names)
    acc_by_len = test.groupby("len_bin", observed=True)["correct"].mean().round(4)
    out["accuracy_by_length_bin"] = acc_by_len.to_dict()
    print("\naccuracy by length bin:", dict(acc_by_len))

    # ---- 3. negation & contrast ----
    ng = test.groupby("correct")[["has_negation", "has_contrast"]].mean().round(4)
    out["negation_contrast_rates"] = {str(k): v for k, v in ng.to_dict("index").items()}
    print("\n=== negation/contrast rates (correct vs errors) ===")
    print(ng.to_string())

    # ---- 4. aspects in specific confusion flows ----
    flows = {
        "neg->neu": (test["label"] == 0) & (test["pred"] == 1),
        "neg->pos": (test["label"] == 0) & (test["pred"] == 2),
        "neg_correct": (test["label"] == 0) & (test["pred"] == 0),
        "neu->pos": (test["label"] == 1) & (test["pred"] == 2),
        "neu_correct": (test["label"] == 1) & (test["pred"] == 1),
        "pos->neu": (test["label"] == 2) & (test["pred"] == 1),
    }
    ar = {}
    for name, mask in flows.items():
        sub = test[mask]
        ar[name] = {"n": int(mask.sum()),
                    **{c: round(float(sub[c].mean()), 3) for c in ["has_fit_size", "has_quality", "has_shipping", "has_negation", "has_contrast"]},
                    "mean_rating": round(float(sub["Rating"].mean()), 2)}
    out["flow_profiles"] = ar
    print("\n=== CONFUSION-FLOW PROFILES ===")
    print(pd.DataFrame(ar).T.to_string())

    # ---- 5. confidence ----
    cf = test.groupby("correct")["prob"].agg(["mean", "median"])
    conf_wrong = float((test[~test["correct"]]["prob"] > 0.8).mean())
    out["confidence"] = {**{str(k): round(float(v), 4) for k, v in cf["mean"].items()},
                         "pct_confident_wrong_gt08": round(conf_wrong, 4)}
    print("\nmean confidence:", out["confidence"])

    # ---- 6. label-noise probe: rating-3 reviews predicted positive with high conf ----
    noisy = test[(test["label"] == 1) & (test["pred"] == 2)].nlargest(5, "prob")
    out["examples_neu_to_pos_high_conf"] = [
        {"text": t[:220], "rating": int(r), "prob_pos": round(float(p), 3)}
        for t, r, p in zip(noisy["Review Text"], noisy["Rating"], noisy["prob"])]
    print("\n=== SAMPLE: true-neutral (rating 3) predicted positive, high confidence ===")
    for e in out["examples_neu_to_pos_high_conf"]:
        print(f"[r{e['rating']} p={e['prob_pos']}] {e['text']}")

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved -> {os.path.relpath(OUT_JSON, REPO_ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
