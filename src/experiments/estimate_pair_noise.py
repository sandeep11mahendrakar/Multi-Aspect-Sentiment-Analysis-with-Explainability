"""Path B step 2: estimate weak-supervision noise rate of aspect pairs.

For every unique aspect-sentence, the validated baseline DistilBERT predicts
sentence-level sentiment independently. Disagreement between that prediction
and the INHERITED document-level proxy label estimates the weak-label noise
rate (dilution: mixed reviews averaging opposing aspect signals).

Caveat: the baseline model is not ground truth (37% doc-level agreement with
the human on rating-3 reviews), so treat outputs as a relative noise estimate,
not an absolute error rate.

Run: venv\\Scripts\\python.exe -u src\\experiments\\estimate_pair_noise.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PAIRS_CSV = os.path.join(REPO_ROOT, "data", "processed", "aspect_sentiment_pairs.csv")
MODEL_DIR = os.path.join(REPO_ROOT, "models", "distilbert_3class_baseline")
OUT_JSON = os.path.join(REPO_ROOT, "results", "e7b_path_b", "pair_noise_estimate.json")
LABELS = ["negative", "neutral", "positive"]
MAX_LEN = 96


def disagree_rate(sub):
    return float((sub["pred"] != sub["proxy_id"]).mean()) if len(sub) else float("nan")


def main():
    pairs = pd.read_csv(PAIRS_CSV)
    uniq = pairs.drop_duplicates("sentence").reset_index(drop=True)
    print(f"pairs: {len(pairs):,} | unique sentences to score: {len(uniq):,}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(dev).eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(uniq), 128):
            enc = tok(uniq["sentence"].iloc[i:i + 128].tolist(), truncation=True,
                      max_length=MAX_LEN, padding=True, return_tensors="pt").to(dev)
            preds.extend(model(**enc).logits.argmax(dim=-1).cpu().numpy().tolist())
    sent_pred = pd.Series(preds)

    pairs["pred"] = pairs["sentence"].map(dict(zip(uniq["sentence"], sent_pred)))
    pairs["proxy_id"] = pairs["proxy_label"].map({l: i for i, l in enumerate(LABELS)})
    pairs = pairs.dropna(subset=["pred"])
    pairs["pred"] = pairs["pred"].astype(int)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    out = {"n_pairs_scored": len(pairs),
           "overall_disagreement": round(disagree_rate(pairs), 4)}

    print(f"\noverall noise estimate (proxy != sentence-level model): "
          f"{out['overall_disagreement']:.1%}")

    print("\nby aspect:")
    out["by_aspect"] = {}
    for a, sub in pairs.groupby("aspect"):
        r = round(disagree_rate(sub), 4)
        out["by_aspect"][a] = {"n": len(sub), "disagreement": r}
        print(f"  {a:<9} n={len(sub):6,d}  disagreement={r:.1%}")

    print("\nby label source:")
    out["by_source"] = {}
    for src, sub in pairs.groupby("label_source"):
        r = round(disagree_rate(sub), 4)
        out["by_source"][src] = {"n": len(sub), "disagreement": r}
        print(f"  {src:<14} n={len(sub):6,d}  disagreement={r:.1%}")

    print("\nconfusion rows=proxy(cols) inherited -> cols=sentence model pred:")
    cm = pd.crosstab(pairs["proxy_label"], pairs["pred"].map(dict(enumerate(LABELS))))
    print(cm.to_string())
    out["proxy_vs_sentence_confusion"] = cm.to_dict()

    # where does the dilution go? per proxy class, distribution of sentence preds
    print("\nper inherited class, sentence-model distribution:")
    out["per_proxy_class"] = {}
    for lbl in LABELS:
        sub = pairs[pairs["proxy_label"] == lbl]
        dist = sub["pred"].value_counts(normalize=True)
        out["per_proxy_class"][lbl] = {l: round(float(dist.get(i, 0)), 4)
                                       for i, l in enumerate(LABELS)}
        print(f"  inherited {lbl:<8} -> neg {dist.get(0,0):5.1%} "
              f"neu {dist.get(1,0):5.1%} pos {dist.get(2,0):5.1%}")

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n-> {os.path.relpath(OUT_JSON, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
