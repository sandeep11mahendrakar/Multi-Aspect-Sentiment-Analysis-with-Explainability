"""Path B: consensus-filtered aspect pair rebuild (E8 Phase 2).

Two-teacher consensus: keep a pair ONLY when the inherited document-level
proxy label agrees with the validated baseline DistilBERT's independent
sentence-level prediction (bare sentence, max_len 96, same as
estimate_pair_noise.py). Drops diluted/misleading weak labels while keeping
the aspect tags untouched (audit found ~4% clear mis-tags -> tagger repair
not the dominant lever).

Run: venv\\Scripts\\python.exe -u src\\experiments\\build_pairs_consensus.py
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PAIRS_CSV = os.path.join(REPO_ROOT, "data", "processed", "aspect_sentiment_pairs.csv")
MODEL_DIR = os.path.join(REPO_ROOT, "models", "distilbert_3class_baseline")
OUT_CSV = os.path.join(REPO_ROOT, "data", "processed", "aspect_sentiment_pairs_consensus.csv")
LABELS = ["negative", "neutral", "positive"]
L2I = {l: i for i, l in enumerate(LABELS)}
MAX_LEN = 96


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pairs = pd.read_csv(PAIRS_CSV)
    uniq = pairs.drop_duplicates("sentence").reset_index(drop=True)
    print(f"pairs: {len(pairs):,} | unique sentences to score: {len(uniq):,}")

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(dev := device).eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(uniq), 128):
            enc = tok(uniq["sentence"].iloc[i:i + 128].tolist(), truncation=True,
                      max_length=MAX_LEN, padding=True, return_tensors="pt").to(device)
            preds.extend(model(**enc).logits.argmax(dim=-1).cpu().numpy().tolist())
    uniq["sent_pred"] = preds

    pairs = pairs.merge(uniq[["sentence", "sent_pred"]], on="sentence", how="left")
    assert pairs["sent_pred"].notna().all()
    pairs["proxy_id"] = pairs["proxy_label"].map(L2I)

    kept = pairs[pairs["sent_pred"] == pairs["proxy_id"]].copy()
    kept["teacher"] = "doc+sentence_consensus"
    kept = kept.drop(columns=["proxy_id"]).reset_index(drop=True)
    print(f"\nconsensus kept {len(kept):,}/{len(pairs):,} "
          f"({len(kept)/len(pairs):.1%}) | dropped {len(pairs)-len(kept):,}")
    print("\nby aspect:")
    for a, sub in pairs.groupby("aspect"):
        k = int((sub["sent_pred"] == sub["proxy_id"]).sum())
        print(f"  {a:<9} {k:6,d}/{len(sub):6,d} ({k/len(sub):5.1%})")
    print("by source:")
    for src, sub in pairs.groupby("label_source"):
        k = int((sub["sent_pred"] == sub["proxy_id"]).sum())
        print(f"  {src:<14} {k:6,d}/{len(sub):6,d} ({k/len(sub):5.1%})")
    print("\nkept class balance per aspect:")
    table = kept.groupby(["aspect", "proxy_label"]).size().unstack(fill_value=0)
    for a in ["quality", "price", "fit"]:
        if a in table.index:
            t = table.loc[a]
            tot = t.sum()
            print(f"  {a:<9} n={tot:6,d} | neg {t.get('negative',0)/tot:5.1%} "
                  f"neu {t.get('neutral',0)/tot:5.1%} pos {t.get('positive',0)/tot:5.1%}")

    cols = ["row_index", "sentence_id", "aspect", "sentence", "proxy_label",
            "rating", "label_source", "sent_pred", "teacher"]
    kept[cols].to_csv(OUT_CSV, index=False)
    print(f"\n-> {os.path.relpath(OUT_CSV, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
