"""E7A analysis: quantify label noise in rating-3 reviews using 0..9 scores.

Run AFTER completing both labeling rounds:
    venv\\Scripts\\python.exe -u src\\experiments\\analyze_e7a.py

Score -> 3-class mapping (documented in build_e7a_sample.py):
    score <= 3 -> negative | 4..6 -> neutral | >= 7 -> positive
Kappa is computed on the collapsed 3-class labels (primary) and reported
as raw score agreement as well. Decision thresholds:
  >= 70% of rating-3 reviews truly neutral -> labels usable (Path A)
  <  60% -> structurally noisy, go ABSA (Path B)
"""

import json
import os
import sys

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
E7A_DIR = os.path.join(REPO_ROOT, "results", "e7a")
LABELS = ["negative", "neutral", "positive"]


def to_label(score: int) -> str:
    return "negative" if score <= 3 else ("positive" if score >= 7 else "neutral")


def main():
    r1 = pd.read_csv(os.path.join(E7A_DIR, "e7a_labeling_round1.csv"), encoding="utf-8-sig",
                     keep_default_na=False)
    key = pd.read_csv(os.path.join(E7A_DIR, "e7a_key.csv"))
    scored = r1[r1["score"] != ""].copy()
    if len(scored) == 0:
        print("No scores found yet. Open apps/labeling_app.py in Streamlit and label first.")
        return 1
    scored["score"] = scored["score"].astype(int)
    scored["label"] = scored["score"].apply(to_label)

    merged = scored.merge(key, on="review_id", how="left")
    dist = scored["label"].value_counts(normalize=True).round(4)
    mean_score_by_rating = None  # all rating-3 here; kept for future extensions

    print(f"Labeled: {len(scored)} / {len(r1)}")
    print(f"Mean human score: {scored['score'].mean():+.2f} (std {scored['score'].std():.2f})")
    print("\nCollapsed label distribution of rating-3 reviews:")
    for l in LABELS:
        print(f"  {l:<10} {dist.get(l, 0)*100:5.1f}%")

    # score distribution granularity
    print("\nScore histogram:")
    print(scored["score"].value_counts().sort_index().to_string())

    cm = confusion_matrix(merged["label"], merged["pred_label"], labels=LABELS)
    cm_df = pd.DataFrame(cm, index=[f"human_{l}" for l in LABELS],
                         columns=[f"model_{l}" for l in LABELS])
    agree_model = float((merged["label"] == merged["pred_label"]).mean())
    print(f"\nHuman vs model prediction agreement (collapsed): {agree_model:.4f}")
    print(cm_df.to_string())

    results = {
        "n_labeled": int(len(scored)),
        "mean_human_score": round(float(scored["score"].mean()), 3),
        "pct_neutral": round(float(dist.get("neutral", 0)), 4),
        "human_label_distribution": {l: round(float(dist.get(l, 0)), 4) for l in LABELS},
        "score_histogram": {str(k): int(v) for k, v in
                            scored["score"].value_counts().sort_index().items()},
        "human_vs_model_agreement": round(agree_model, 4),
        "human_vs_model_confusion": cm.tolist(),
        "labels_order": LABELS,
    }

    # ---- Cohen's kappa on double-labeled subset ----
    r2_path = os.path.join(E7A_DIR, "e7a_labeling_round2.csv")
    if os.path.exists(r2_path):
        r2 = pd.read_csv(r2_path, encoding="utf-8-sig", keep_default_na=False)
        r2_done = r2[r2["score"] != ""].copy()
        if len(r2_done):
            r2_done["score"] = r2_done["score"].astype(int)
            r2_done["label"] = r2_done["score"].apply(to_label)
            pair = r1[r1["score"] != ""].copy()
            pair["score"] = pair["score"].astype(int)
            pair["label_r1"] = pair["score"].apply(to_label)
            pair = pair[["review_id", "label_r1", "score"]].merge(
                r2_done[["review_id", "label", "score"]].rename(
                    columns={"label": "label_r2", "score": "score_r2"}),
                on="review_id")
            kappa = cohen_kappa_score(pair["label_r1"], pair["label_r2"])
            same = float((pair["label_r1"] == pair["label_r2"]).mean())
            exact_scores = float((pair["score"] == pair["score_r2"]).mean())
            within1 = float(((pair["score"] - pair["score_r2"]).abs() <= 1).mean())
            results.update({
                "kappa_subset_n": int(len(pair)),
                "percent_raw_agreement_rounds": round(same, 4),
                "cohen_kappa_collapsed": round(float(kappa), 4),
                "percent_exact_score_agreement": round(exact_scores, 4),
                "percent_within_1_point": round(within1, 4),
            })
            print(f"\nDouble-labeled pairs: {len(pair)}")
            print(f"Raw agreement (collapsed): {same:.4f} | Cohen's kappa: {kappa:.4f}")
            print(f"Exact score agreement: {exact_scores:.4f} | within ±1 point: {within1:.4f}")

    verdict = ("labels mostly fine -> Path A (relabel + retrain)"
               if results["pct_neutral"] >= 0.70 else
               "structurally noisy -> Path B (ABSA)" if results["pct_neutral"] < 0.60 else
               "borderline (0.60-0.70): decide with model-agreement numbers above")
    results["decision_hint"] = verdict
    print(f"\nDecision hint: {verdict}")

    out = os.path.join(E7A_DIR, "e7a_analysis.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved -> {os.path.relpath(out, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
