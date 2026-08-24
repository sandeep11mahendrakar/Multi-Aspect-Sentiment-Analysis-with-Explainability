"""Path B step 1: build weak-supervision aspect-sentiment training pairs.

Per review (usable frame, 22,641 rows):
  - split into sentences via src/sentence_splitter.split_sentences
  - tag aspects via tag_aspects (quality/price/fit/shipping)
  - each aspect-sentence inherits the review's CORRECTED document label as its
    proxy sentiment (NOT the raw star rating):
      rating>=4 -> positive, rating<=2 -> negative (unchanged by E7B),
      rating==3 -> model textual label from data/processed/corrected_rating3_labels.csv
    Rationale: E7B showed textual labels track human judgment better on
    ambiguous cases (+2.8pp macro-F1 on gold-199); inheriting rating-based
    noise at sentence level would relocate the E7A problem instead of fixing it.

Output: data/processed/aspect_sentiment_pairs.csv
    row_index     positional index into the 22,641-row usable frame
    sentence_id   sentence position within the review
    aspect        quality | price | fit | shipping
    sentence      aspect-tagged sentence text
    proxy_label   negative | neutral | positive (inherited corrected doc label)
    rating        original star rating (for filtering/analysis)
    label_source  rating | model_relabel

Run: venv\\Scripts\\python.exe -u src\\experiments\\build_aspect_pairs.py
"""

import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from sentence_splitter import split_sentences, tag_aspects  # noqa: E402

RAW_CSV = os.path.join(REPO_ROOT, "data", "raw", "reviews.csv")
CORRECTED_CSV = os.path.join(REPO_ROOT, "data", "processed", "corrected_rating3_labels.csv")
OUT_CSV = os.path.join(REPO_ROOT, "data", "processed", "aspect_sentiment_pairs.csv")
ASPECTS = ["quality", "price", "fit", "shipping"]


def main():
    df = pd.read_csv(RAW_CSV).dropna(subset=["Review Text"]).reset_index(drop=True)

    doc_label = df["Rating"].apply(lambda r: "negative" if r <= 2 else
                                   ("positive" if r >= 4 else "neutral"))
    label_source = pd.Series("rating", index=df.index)
    corr = pd.read_csv(CORRECTED_CSV)
    df.loc[corr["row_index"], "_corr"] = corr["textual_label"].values
    relabeled = df.index.isin(corr["row_index"])
    doc_label[relabeled] = df.loc[relabeled, "_corr"]
    label_source[relabeled] = "model_relabel"
    df.drop(columns="_corr", inplace=True)
    print(f"usable rows: {len(df)} | doc labels from model_relabel: {int(relabeled.sum())}")

    rows = []
    for idx, text in zip(df.index, df["Review Text"]):
        for sid, sent in enumerate(split_sentences(text)):
            for aspect in sorted(tag_aspects(sent)):
                rows.append((idx, sid, aspect, sent, doc_label[idx],
                             int(df.at[idx, "Rating"]), label_source[idx]))
    pairs = pd.DataFrame(rows, columns=["row_index", "sentence_id", "aspect",
                                        "sentence", "proxy_label", "rating",
                                        "label_source"])
    pairs.to_csv(OUT_CSV, index=False)

    print(f"\naspect-sentiment pairs: {len(pairs):,} "
          f"({pairs['sentence'].nunique():,} unique sentences)")
    print(f"\nclass balance per aspect:")
    table = pairs.groupby(["aspect", "proxy_label"]).size().unstack(fill_value=0)
    for a in ASPECTS:
        if a in table.index:
            t = table.loc[a]
            tot = t.sum()
            print(f"  {a:<9} n={tot:6,d} | neg {t.get('negative',0)/tot:5.1%} "
                  f"neu {t.get('neutral',0)/tot:5.1%} pos {t.get('positive',0)/tot:5.1%}")
    print(f"\nby label_source:")
    for src, n in pairs["label_source"].value_counts().items():
        print(f"  {src:<14} {n:,}")
    print(f"\n-> {os.path.relpath(OUT_CSV, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
