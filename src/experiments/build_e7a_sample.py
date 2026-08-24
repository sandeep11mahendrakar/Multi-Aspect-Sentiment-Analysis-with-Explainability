"""E7A: build the blinded human-labeling sample for rating-3 reviews.

Outputs (results/e7a/):
   e7a_labeling_round1.csv  200 reviews: review_id, review_text, score, comment
                           score = human sentiment on 0..9 integer scale
                            (NO star ratings, NO model output - fully blinded)
                            Mapping to 3 classes (applied at analysis time):
                            score <= 3 -> negative | 4..6 -> neutral | >= 7 -> positive
  e7a_labeling_round2.csv  50 of those, re-shuffled independently (kappa subset)
  e7a_key.csv              the blinding key: id -> rating, split, model prediction,
                           model confidence, aspect tags per sentence
  e7a_sentences_view.csv   per-sentence view of the 200 sampled reviews

Sampling is stratified by the baseline DistilBERT's PREDICTED class on all
usable rating-3 reviews (full dataset, not just test split), proportional
allocation with a minimum quota per class. Seed 42.

Run from repo root:
    venv\\Scripts\\python.exe -u src\\experiments\\build_e7a_sample.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sentence_splitter import split_sentences, tag_aspects  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_CSV = os.path.join(REPO_ROOT, "data", "raw", "reviews.csv")
MODEL_DIR = os.path.join(REPO_ROOT, "models", "distilbert_3class_baseline")
OUT_DIR = os.path.join(REPO_ROOT, "results", "e7a")
MAX_LEN = 128
SEED = 42
N_SAMPLE = 200
N_DOUBLE = 50
MIN_PER_CLASS = 15
LABELS = ["negative", "neutral", "positive"]


def main():
    rng = np.random.RandomState(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(RAW_CSV).dropna(subset=["Review Text"]).reset_index(drop=True)
    neu = df[df["Rating"] == 3].reset_index(drop=True)
    print(f"rating-3 usable reviews: {len(neu)}")

    # ---- model predictions on ALL rating-3 reviews ----
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(dev).eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(neu), 64):
            enc = tok(list(neu["Review Text"].iloc[i:i+64]), truncation=True,
                      max_length=MAX_LEN, padding=True, return_tensors="pt").to(dev)
            probs.append(torch.softmax(model(**enc).logits, dim=-1).cpu().numpy())
    P = np.vstack(probs)
    neu["pred_label"] = [LABELS[i] for i in P.argmax(axis=1)]
    neu["pred_conf"] = P.max(axis=1)
    print("prediction distribution on rating-3:",
          neu["pred_label"].value_counts().to_dict())

    # ---- stratified sample by predicted class ----
    quotas = {}
    for cls in LABELS:
        n_cls = (neu["pred_label"] == cls).sum()
        quotas[cls] = max(MIN_PER_CLASS, int(round(N_SAMPLE * n_cls / len(neu))))
    # fix rounding drift against total
    while sum(quotas.values()) > N_SAMPLE:
        quotas[max(quotas, key=quotas.get)] -= 1
    while sum(quotas.values()) < N_SAMPLE:
        quotas[min(LABELS, key=lambda c: quotas[c])] += 1
    print("sampling quotas:", quotas)

    parts = []
    for cls in LABELS:
        pool = neu[neu["pred_label"] == cls]
        take = min(quotas[cls], len(pool))
        parts.append(pool.sample(n=take, random_state=SEED))
    sample = pd.concat(parts).sample(frac=1.0, random_state=rng.randint(10**6)).reset_index(drop=True)
    if len(sample) != N_SAMPLE:
        print(f"WARNING: sampled {len(sample)} (pool smaller than target in some class)")

    sample_ids = sample.index.to_numpy()  # positional ids into `neu`
    out = sample[["Review Text"]].copy()
    out.insert(0, "review_id", sample_ids)
    out["score"] = ""   # human sentiment, -5..+5 (blinded annotation)
    out["comment"] = ""

    r1_path = os.path.join(OUT_DIR, "e7a_labeling_round1.csv")
    out.to_csv(r1_path, index=False, encoding="utf-8-sig")

    # ---- round 2: independent shuffle of a blind subset ----
    double_idx = rng.choice(len(out), size=min(N_DOUBLE, len(out)), replace=False)
    round2 = out.iloc[np.sort(double_idx)].sample(frac=1.0, random_state=rng.randint(10**6)).reset_index(drop=True)
    r2_path = os.path.join(OUT_DIR, "e7a_labeling_round2.csv")
    round2.to_csv(r2_path, index=False, encoding="utf-8-sig")

    # ---- blinding key + sentence view ----
    key_cols = ["Rating", "pred_label", "pred_conf"]
    key = sample[["Review Text"] + key_cols].copy()
    key.insert(0, "review_id", sample_ids)
    key.insert(2, "in_round2", np.isin(np.arange(len(sample)), double_idx))
    key["split_hint"] = ""  # filled below without leaking into labeling files
    key.to_csv(os.path.join(OUT_DIR, "e7a_key.csv"), index=False)

    sent_rows = []
    for rid, text in zip(sample_ids, sample["Review Text"]):
        for sid, sent in enumerate(split_sentences(text)):
            sent_rows.append({"review_id": rid, "sentence_id": sid,
                              "sentence": sent,
                              "aspects": ",".join(sorted(tag_aspects(sent)))})
    pd.DataFrame(sent_rows).to_csv(
        os.path.join(OUT_DIR, "e7a_sentences_view.csv"), index=False, encoding="utf-8-sig")

    meta = {
        "seed": SEED, "n_rating3_usable": int(len(neu)), "n_sampled": int(len(sample)),
        "n_round2": int(len(round2)),
        "quotas_by_predicted_class": {k: int(v) for k, v in quotas.items()},
        "model": MODEL_DIR,
        "blinding": "labeling files contain only review_id + text; key file held separately",
    }
    with open(os.path.join(OUT_DIR, "e7a_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\nsampled: {len(sample)} | round2: {len(round2)} | sentences: {len(sent_rows)}")
    print(f"-> {os.path.relpath(r1_path, REPO_ROOT)}")
    print(f"-> {os.path.relpath(r2_path, REPO_ROOT)}")
    print("\nNext: streamlit run apps/labeling_app.py")


if __name__ == "__main__":
    sys.exit(main())
