"""Path B step 3: scaffold the aspect-conditioned ABSA input format.

Input format: [aspect] [SEP] [sentence], built via the DistilBERT tokenizer's
text/text-pair interface (aspect as first segment, sentence as second) so
special-token handling stays correct.

This script verifies the format end-to-end on real pairs and measures token
lengths to pick a training max_len.

Run: venv\\Scripts\\python.exe -u src\\experiments\\scaffold_aspect_input.py
"""

import os
import sys

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PAIRS_CSV = os.path.join(REPO_ROOT, "data", "processed", "aspect_sentiment_pairs.csv")
TOKENIZER_DIR = os.path.join(REPO_ROOT, "models", "distilbert_3class_baseline")


def main():
    pairs = pd.read_csv(PAIRS_CSV)
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    print("=== format check: one sample pair per aspect ===")
    encodings = []
    for aspect, sub in pairs.groupby("aspect"):
        row = sub.iloc[0]
        enc = tok(row["aspect"], row["sentence"], truncation=True, max_length=128)
        encodings.append(enc)
        print(f"\n[{aspect}] proxy={row['proxy_label']}")
        print(f"  decoded: {tok.decode(enc['input_ids'])}")

    print("\n=== round-trip integrity ===")
    # BERT pair format is [CLS] aspect [SEP] sentence [SEP]
    ok = all(e["input_ids"][0] == tok.cls_token_id
             and e["input_ids"].count(tok.sep_token_id) == 2
             and e["input_ids"][-1] == tok.sep_token_id
             for e in encodings)
    print(f"all samples are well-formed [CLS] aspect [SEP] sentence [SEP]: {ok}")
    if not ok:
        return 1

    print("\n=== token length distribution over all pairs (max_len planning) ===")
    lens = []
    step = 2000
    for i in range(0, len(pairs), step):
        chunk = pairs.iloc[i:i + step]
        enc = tok(chunk["aspect"].tolist(), chunk["sentence"].tolist(),
                  truncation=False)
        lens.extend(len(ids) for ids in enc["input_ids"])
    lens = np.array(lens)
    for q in (50, 90, 95, 99):
        print(f"  p{q}: {np.percentile(lens, q):.0f} tokens")
    print(f"  max: {lens.max()} | share > 96: {(lens > 96).mean():.2%} "
          f"| share > 128: {(lens > 128).mean():.2%}")

    print("\nSCAFFOLD OK - ready to wire into train_transformer.py as an ABSA variant")
    return 0


if __name__ == "__main__":
    sys.exit(main())
