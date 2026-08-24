"""Path B step 4: build a ~100-sentence human gold eval set for the ABSA model.

Sampled from results/e7a/e7a_sentences_view.csv (sentences of the 200 E7A
reviews), stratified across aspect x baseline-model confidence tercile.
Shipping is EXCLUDED (dropped from training; different noise profile).

Output:
  results/e7b_path_b/absa_gold_labeling.xlsx   Sandeep's blind labeling input
  results/e7b_path_b/absa_gold_key.csv         BLINDING KEY (pred/conf per gold_id)

Labeling scale (same as E7A): 0-9 overall sentiment of the SENTENCE for the
stated aspect: <=3 negative | 4-6 neutral | >=7 positive.

Run: venv\\Scripts\\python.exe -u src\\experiments\\build_absa_gold_sample.py
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from sentence_splitter import tag_aspects  # noqa: E402

VIEW_CSV = os.path.join(REPO_ROOT, "results", "e7a", "e7a_sentences_view.csv")
MODEL_DIR = os.path.join(REPO_ROOT, "models", "distilbert_3class_baseline")
OUT_DIR = os.path.join(REPO_ROOT, "results", "e7b_path_b")
N_TOTAL = 99  # 3 aspects x 3 confidence bands x 11
ASPECTS = ["quality", "price", "fit"]
LABELS = ["negative", "neutral", "positive"]
MAX_LEN = 96


def main():
    view = pd.read_csv(VIEW_CSV)
    # re-tag aspects from raw sentence text (authoritative taxonomy); drop shipping
    rows = []
    for _, r in view.iterrows():
        for a in sorted(tag_aspects(str(r["sentence"]))):
            if a in ASPECTS:
                rows.append((r["review_id"], r["sentence_id"], r["sentence"], a))
    cand = pd.DataFrame(rows, columns=["review_id", "sentence_id", "sentence", "aspect"])
    cand = cand.drop_duplicates(subset=["sentence", "aspect"]).reset_index(drop=True)
    print(f"candidates: {len(cand)} ({cand['aspect'].value_counts().to_dict()})")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(dev).eval()
    preds, confs = [], []
    with torch.no_grad():
        for i in range(0, len(cand), 128):
            enc = tok(cand["aspect"].iloc[i:i + 128].tolist(),
                      cand["sentence"].iloc[i:i + 128].tolist(),
                      truncation=True, max_length=MAX_LEN, padding=True,
                      return_tensors="pt").to(dev)
            probs = torch.softmax(model(**enc).logits, dim=-1)
            preds.extend(probs.argmax(dim=-1).cpu().numpy().tolist())
            confs.extend(probs.max(dim=-1).values.cpu().numpy().tolist())
    cand["model_pred"] = [LABELS[p] for p in preds]
    cand["model_conf"] = np.round(confs, 4)

    cand["conf_band"] = pd.qcut(cand["model_conf"], q=3,
                                labels=["low", "mid", "high"])
    sample = (cand.groupby(["aspect", "conf_band"], observed=True)
              .apply(lambda g: g.sample(min(len(g), N_TOTAL // 9), random_state=42))
              .reset_index(drop=True))
    sample.insert(0, "gold_id", range(len(sample)))

    os.makedirs(OUT_DIR, exist_ok=True)
    key = sample[["gold_id", "model_pred", "model_conf"]].copy()
    xlsx_rows = sample[["gold_id", "aspect", "sentence"]].copy()
    xlsx_rows["score"] = ""
    xlsx_rows["comment"] = ""

    key.to_csv(os.path.join(OUT_DIR, "absa_gold_key.csv"), index=False)
    with pd.ExcelWriter(os.path.join(OUT_DIR, "absa_gold_labeling.xlsx"),
                        engine="openpyxl") as xw:
        xlsx_rows.to_excel(xw, index=False, sheet_name="gold")
        ws = xw.sheets["gold"]
        ws.column_dimensions["C"].width = 100
    print(f"\ngold sample: {len(sample)} sentences | "
          f"dist:\n{sample.groupby(['aspect', 'conf_band'], observed=True).size()}")
    print(f"-> {os.path.relpath(os.path.join(OUT_DIR, 'absa_gold_labeling.xlsx'), REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
