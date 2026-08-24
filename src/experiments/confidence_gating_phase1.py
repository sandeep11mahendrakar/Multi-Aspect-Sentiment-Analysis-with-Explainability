"""Phase 1 confidence-gating analysis for models/distilbert_3class_baseline.

No training. Steps:
  1. Reproduce the seed42 80/20 split (identical row assignment to E5/E7B).
  2. Carve a stratified 10% calibration split OUT OF THE TRAIN PORTION
     (test set is never used for fitting anything).
  3. Inference: save per-row softmax probs on calibration rows and all 4,529 test rows.
  4. Fit temperature T on the calibration split by NLL minimization.
  5. ECE before/after + reliability diagram bins.
  6. Risk-coverage curve on calibrated test probs;
     accuracy + coverage at tau = .70 / .80 / .90 (raw-prob taus reported too).

Outputs -> results/e8_confidence_gating/
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_CSV = os.path.join(REPO_ROOT, "data", "raw", "reviews.csv")
MODEL_DIR = os.path.join(REPO_ROOT, "models", "distilbert_3class_baseline")
OUT_DIR = os.path.join(REPO_ROOT, "results", "e8_confidence_gating")
MAX_LEN = 128
SEED = 42
BATCH = 128


def load_split():
    df = pd.read_csv(RAW_CSV)
    df = df.dropna(subset=["Review Text"]).reset_index(drop=True)
    df["label"] = df["Rating"].apply(lambda r: 0 if r <= 2 else (2 if r >= 4 else 1))
    idx_train, idx_test = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=SEED, stratify=df["label"])
    return df, np.array(idx_train), np.array(idx_test)


@torch.no_grad()
def predict_logits(model, tok, texts, device):
    logits = []
    for i in range(0, len(texts), BATCH):
        enc = tok(list(texts[i:i + BATCH]), truncation=True, max_length=MAX_LEN,
                  padding="max_length", return_tensors="pt").to(device)
        with torch.autocast("cuda", enabled=(device == "cuda")):
            logits.append(model(**enc).logits.float().cpu())
    return torch.cat(logits).numpy()


def nll(logits, y, temp):
    z = torch.tensor(logits) / temp
    return torch.nn.functional.cross_entropy(z, torch.tensor(y, dtype=torch.long)).item()


def ece(probs, y, n_bins=15):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(float)
    bins = np.clip((conf * n_bins).astype(int), 0, n_bins - 1)
    e = 0.0
    rows = []
    for b in range(n_bins):
        m = bins == b
        if not m.any():
            continue
        c, a, n = conf[m].mean(), correct[m].mean(), int(m.sum())
        e += n / len(y) * abs(c - a)
        rows.append({"bin": b, "conf_mean": round(float(c), 4),
                     "acc_mean": round(float(a), 4), "count": n})
    return e, rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df, idx_train, idx_test = load_split()

    # stratified calibration split carved from TRAIN only
    idx_cal, idx_fit = train_test_split(
        idx_train, test_size=0.9, random_state=SEED, stratify=df.loc[idx_train, "label"])
    print(f"rows={len(df)} train={len(idx_train)} cal={len(idx_cal)} test={len(idx_test)}")

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()
    print("model loaded:", MODEL_DIR)

    logits_cal = predict_logits(model, tok, df.loc[idx_cal, "Review Text"].values, device)
    logits_te = predict_logits(model, tok, df.loc[idx_test, "Review Text"].values, device)
    y_cal = df.loc[idx_cal, "label"].values.astype(int)
    y_te = df.loc[idx_test, "label"].values.astype(int)

    probs_cal_raw = torch.softmax(torch.tensor(logits_cal), dim=1).numpy()
    probs_te_raw = torch.softmax(torch.tensor(logits_te), dim=1).numpy()
    assert len(y_te) == 4529, f"test size {len(y_te)} != 4529"
    acc_full = accuracy_score(y_te, probs_te_raw.argmax(1))
    mf1_full = f1_score(y_te, probs_te_raw.argmax(1), average="macro")
    print(f"sanity: test acc={acc_full:.4f} macro_f1={mf1_full:.4f} "
          f"(expected ~.8547/~.6952)")

    # ---- temperature scaling: minimize NLL on calibration split ----
    grid = np.linspace(0.5, 5.0, 901)
    losses = [nll(logits_cal, y_cal, t) for t in grid]
    T = float(grid[int(np.argmin(losses))])
    print(f"fitted temperature T={T:.4f} (cal NLL {min(losses):.4f} vs T=1: "
          f"{nll(logits_cal, y_cal, 1.0):.4f})")

    probs_cal_T = torch.softmax(torch.tensor(logits_cal) / T, dim=1).numpy()
    probs_te_T = torch.softmax(torch.tensor(logits_te) / T, dim=1).numpy()

    ece_raw_cal, bins_raw_cal = ece(probs_cal_raw, y_cal)
    ece_T_cal, bins_T_cal = ece(probs_cal_T, y_cal)
    ece_raw_te, bins_raw_te = ece(probs_te_raw, y_te)
    ece_T_te, bins_T_te = ece(probs_te_T, y_te)
    print(f"ECE(cal): raw={ece_raw_cal:.4f} -> T-scaled={ece_T_cal:.4f}")
    print(f"ECE(test, reference-only): raw={ece_raw_te:.4f} -> T-scaled={ece_T_te:.4f}")

    # ---- risk-coverage on test, calibrated probs ----
    def risk_coverage(probs, y):
        conf = probs.max(axis=1)
        order = np.argsort(-conf)
        preds = probs.argmax(1)
        correct = (preds == y)[order]
        n = len(y)
        cov_grid = np.arange(1, n + 1) / n
        cum_acc = np.cumsum(correct) / np.arange(1, n + 1)
        return conf, order, cov_grid, cum_acc, correct

    conf_T, _, cov_grid, cum_acc_T, _ = risk_coverage(probs_te_T, y_te)
    conf_raw, _, _, cum_acc_raw, _ = risk_coverage(probs_te_raw, y_te)

    thresholds = [0.70, 0.80, 0.90]
    gate_rows = []
    for tau in thresholds:
        for name, conf, cacc in [("T-scaled", conf_T, cum_acc_T), ("raw", conf_raw, cum_acc_raw)]:
            m = conf >= tau
            gate_rows.append({
                "tau": tau, "probs": name,
                "coverage": round(float(m.mean()), 4),
                "committed_n": int(m.sum()),
                "abstained_n": int((~m).sum()),
                "committed_acc": round(float((y_te[m] == probs_te_T.argmax(1)[m]).mean()), 4)
                if m.any() else None,
                "coverage_curve_acc_at_same_cov": None,
            })
    # committed_acc must be computed against matching prob set; redo cleanly:
    gate_rows = []
    for tau in thresholds:
        for name, probs in [("raw", probs_te_raw), ("T-scaled", probs_te_T)]:
            conf = probs.max(axis=1)
            m = conf >= tau
            gate_rows.append({
                "tau": tau, "probs": name,
                "coverage": round(float(m.mean()), 4),
                "committed_n": int(m.sum()),
                "abstained_n": int((~m).sum()),
                "committed_acc": round(float(accuracy_score(
                    y_te[m], probs.argmax(1)[m])), 4) if m.any() else None,
                "committed_macro_f1": round(float(f1_score(
                    y_te[m], probs.argmax(1)[m], average="macro")), 4) if m.any() else None,
            })

    # coverage -> accuracy lookup at fixed coverages for the writeup
    cov_points = [0.80, 0.85, 0.90, 0.95]
    curve_points = []
    for cp in cov_points:
        i = int(np.argmin(np.abs(cov_grid - cp)))
        curve_points.append({"coverage": round(float(cov_grid[i]), 4),
                             "acc": round(float(cum_acc_T[i]), 4)})

    pd.DataFrame({
        "row_index": idx_test, "y_true": y_te,
        "p_neg_raw": probs_te_raw[:, 0], "p_neu_raw": probs_te_raw[:, 1],
        "p_pos_raw": probs_te_raw[:, 2],
        "p_neg_T": probs_te_T[:, 0], "p_neu_T": probs_te_T[:, 1], "p_pos_T": probs_te_T[:, 2],
        "max_prob_raw": conf_raw, "max_prob_T": conf_T,
    }).to_csv(os.path.join(OUT_DIR, "test_probs.csv"), index=False)

    out = {
        "model": os.path.relpath(MODEL_DIR, REPO_ROOT),
        "split": {"train": len(idx_train), "cal_from_train": len(idx_cal),
                  "test": len(idx_test), "seed": SEED},
        "temperature": round(T, 4),
        "cal_nll_raw": round(nll(logits_cal, y_cal, 1.0), 4),
        "cal_nll_T": round(min(losses), 4),
        "ece": {"cal_raw": round(ece_raw_cal, 4), "cal_T": round(ece_T_cal, 4),
                "test_raw_reference_only": round(ece_raw_te, 4),
                "test_T_reference_only": round(ece_T_te, 4)},
        "reliability_bins_cal_raw": bins_raw_cal,
        "reliability_bins_cal_T": bins_T_cal,
        "full_test": {"accuracy": round(acc_full, 4), "macro_f1": round(mf1_full, 4)},
        "gates": gate_rows,
        "risk_coverage_points_T": curve_points,
    }
    path = os.path.join(OUT_DIR, "phase1_calibration.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print("\n===== GATES (test, n=4529) =====")
    print(pd.DataFrame(gate_rows).to_string(index=False))
    print("\nrisk-coverage (T-scaled):")
    for p in curve_points:
        print(f"  coverage {p['coverage']:.2f} -> acc {p['acc']:.4f}")
    print(f"\n-> {os.path.relpath(path, REPO_ROOT)}")
    print(f"-> {os.path.relpath(os.path.join(OUT_DIR, 'test_probs.csv'), REPO_ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
