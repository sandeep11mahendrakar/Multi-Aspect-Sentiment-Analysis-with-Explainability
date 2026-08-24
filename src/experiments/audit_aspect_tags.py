"""Path B step: audit keyword-based aspect tagging on a stratified sample.

Samples 100 pairs from aspect_sentiment_pairs.csv (seed42, proportional by
aspect) for human/agent judgment of whether the tagged aspect is actually
discussed in the sentence. Output CSV drives the audit verdict recorded in
PROJECT_MEMORY.md.

Run: venv\\Scripts\\python.exe -u src\\experiments\\audit_aspect_tags.py
"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PAIRS_CSV = os.path.join(REPO_ROOT, "data", "processed", "aspect_sentiment_pairs.csv")
OUT_CSV = os.path.join(REPO_ROOT, "results", "e7b_path_b", "aspect_tag_audit_sample.csv")

N = 100
SEED = 42


def main():
    pairs = pd.read_csv(PAIRS_CSV)
    # dedupe identical (sentence, aspect) rows for audit fairness; keep first
    uniq = pairs.drop_duplicates(subset=["sentence", "aspect"]).reset_index(drop=True)
    frac = (uniq["aspect"].value_counts(normalize=True)
            .reindex(["fit", "quality", "price", "shipping"]).fillna(0))
    counts = (frac * N).round().astype(int)
    counts.iloc[-1] += N - counts.sum()  # fix rounding drift on smallest stratum

    parts = []
    for aspect, k in counts.items():
        sub = uniq[uniq["aspect"] == aspect]
        take = min(k, len(sub))
        if take >= sub["proxy_label"].nunique() and (sub["proxy_label"].value_counts() >= 2).all():
            _, sample = train_test_split(sub, test_size=take, random_state=SEED,
                                         stratify=sub["proxy_label"])
        else:
            sample = sub.sample(n=take, random_state=SEED)
        parts.append(sample)
        print(f"{aspect:<9} target {k:>3} -> sampled {take}")
    audit = pd.concat(parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    audit["verdict"] = ""
    audit.to_csv(OUT_CSV, index=False)

    with pd.option_context("display.max_colwidth", 200):
        for i, r in audit.iterrows():
            print(f"[{i:03d}] ({r['aspect']}) {r['sentence']}")
    print(f"\n-> {os.path.relpath(OUT_CSV, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    main()
