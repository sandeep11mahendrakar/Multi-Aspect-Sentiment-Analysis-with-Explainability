"""Path B step 5: train aspect-conditioned DistilBERT ([aspect] [SEP] [sentence]).

Constraints (operator-mandated):
  1. Train/test split GROUPED BY REVIEW (row_index): all sentences of one review
     stay on the same side -> no sentence-level leakage across the split.
  2. Class weights ARE passed into the training call (WeightedTrainer);
     verified non-None before trainer.train() and logged to metrics.
  3. Shipping aspect EXCLUDED from training (544 pairs, different noise profile)
     - recorded as a limitation.

Run: venv\\Scripts\\python.exe -u src\\experiments\\train_absa.py [--max-steps 10]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PAIRS_CSV = os.path.join(REPO_ROOT, "data", "processed", "aspect_sentiment_pairs.csv")
METRICS_DIR = os.path.join(REPO_ROOT, "results", "e7b_path_b")
MODELS_DIR = os.path.join(REPO_ROOT, "models")

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 96
SEED = 42
LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}
ASPECTS = ["quality", "price", "fit"]  # shipping excluded (limitation)


class PairDataset(Dataset):
    def __init__(self, aspects, sentences, labels, tokenizer):
        self.enc = tokenizer(list(aspects), list(sentences), truncation=True,
                             max_length=MAX_LEN, padding="max_length",
                             return_tensors="pt")
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kw):
        super().__init__(*args, **kw)
        if class_weights is None:
            raise ValueError("class_weights must be provided (operator mandate #2)")
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = nn.functional.cross_entropy(outputs.logits, labels,
                                           weight=self.class_weights)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, y = eval_pred
    pred = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "weighted_f1": f1_score(y, pred, average="weighted"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--pairs-csv", default=PAIRS_CSV,
                    help="override pairs input (e.g. consensus-filtered rebuild)")
    ap.add_argument("--run-name", default="distilbert_absa_aspectconditioned")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    pairs = pd.read_csv(args.pairs_csv)
    pairs = pairs[pairs["aspect"].isin(ASPECTS)].reset_index(drop=True)
    y = pairs["proxy_label"].map(LABEL2ID).to_numpy()
    groups = pairs["row_index"].to_numpy()
    print(f"pairs used: {len(pairs):,} (shipping excluded) | "
          f"label dist {np.bincount(y).tolist()} | source: {os.path.basename(args.pairs_csv)}")

    out_dir = os.path.join(MODELS_DIR, args.run_name)
    metrics_path = os.path.join(METRICS_DIR, f"{args.run_name}_training.json")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_idx, te_idx = next(gss.split(pairs, y, groups))
    overlap = set(groups[tr_idx]).intersection(set(groups[te_idx]))
    assert not overlap, "leakage: reviews present in both sides"
    print(f"train={len(tr_idx):,} test={len(te_idx):,} | "
          f"shared reviews between sides: {len(overlap)}")
    print(f"train label dist {np.bincount(y[tr_idx]).tolist()} | "
          f"test label dist {np.bincount(y[te_idx]).tolist()}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = PairDataset(pairs.loc[tr_idx, "aspect"], pairs.loc[tr_idx, "sentence"],
                           y[tr_idx], tok)
    test_pairs = pairs.loc[te_idx].reset_index(drop=True)
    test_y = y[te_idx]
    test_ds = PairDataset(test_pairs["aspect"], test_pairs["sentence"], test_y, tok)

    w = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=y[tr_idx])
    w_tensor = torch.tensor(w, dtype=torch.float32).to(device)
    print(f"class weights (balanced, train): {w.tolist()}")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID)

    targs = TrainingArguments(
        output_dir=out_dir,
        seed=SEED,
        data_seed=SEED,
        fp16=(device == "cuda"),
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        num_train_epochs=4,
        warmup_steps=max(1, int(0.10 * np.ceil(len(tr_idx) / 32) * 4)),
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=300,
        save_strategy="steps",
        save_steps=300,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=100,
        report_to=[],
        max_steps=(args.max_steps if args.max_steps > 0 else -1),
    )

    trainer = WeightedTrainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=test_ds,
        compute_metrics=compute_metrics, class_weights=w_tensor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    assert isinstance(trainer, WeightedTrainer) and trainer.class_weights is not None

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    pred_out = trainer.predict(test_ds)
    pred = np.argmax(pred_out.predictions, axis=1)
    acc = accuracy_score(test_y, pred)
    mf1 = f1_score(test_y, pred, average="macro")
    rec = recall_score(test_y, pred, average=None, labels=[0, 1, 2], zero_division=0)

    print(f"\n===== WEAK HELD-OUT (secondary sanity number) =====")
    print(f"acc={acc:.4f} macro_f1={mf1:.4f} | recall neg={rec[0]:.3f} "
          f"neu={rec[1]:.3f} pos={rec[2]:.3f}")
    cm = confusion_matrix(test_y, pred, labels=[0, 1, 2])
    print(pd.DataFrame(cm, index=[f"true_{l}" for l in LABELS],
                       columns=[f"pred_{l}" for l in LABELS]).to_string())

    per_aspect = {}
    print("\nper-aspect recall / macro-F1:")
    for a in ASPECTS:
        mask = (test_pairs["aspect"] == a).to_numpy()
        ya, pa = test_y[mask], pred[mask]
        ra = recall_score(ya, pa, average=None, labels=[0, 1, 2], zero_division=0)
        fa = f1_score(ya, pa, average="macro", zero_division=0)
        per_aspect[a] = {
            "n": int(mask.sum()),
            "recall_neg": round(float(ra[0]), 4),
            "recall_neu": round(float(ra[1]), 4),
            "recall_pos": round(float(ra[2]), 4),
            "macro_f1": round(float(fa), 4),
        }
        print(f"  {a:<8} n={int(mask.sum()):6,d} | recall neg={ra[0]:.3f} "
              f"neu={ra[1]:.3f} pos={ra[2]:.3f} | macroF1={fa:.4f}")

    if args.max_steps > 0:
        print("SMOKE TEST OK")
        return 0

    results = {
        "run": args.run_name,
        "pairs_csv": os.path.relpath(args.pairs_csv, REPO_ROOT),
        "input_format": "[CLS] aspect [SEP] sentence [SEP]",
        "max_len": MAX_LEN, "seed": SEED,
        "split": "GroupShuffleSplit by row_index (review-level, no leakage)",
        "shipping_excluded": True,
        "class_weights_balanced": w.tolist(),
        "weak_heldout": {
            "n_test": int(len(test_y)),
            "accuracy": round(float(acc), 4),
            "macro_f1": round(float(mf1), 4),
            "recall_neg": round(float(rec[0]), 4),
            "recall_neu": round(float(rec[1]), 4),
            "recall_pos": round(float(rec[2]), 4),
            "confusion_rows_true_cols_pred": cm.tolist(),
            "per_aspect": per_aspect,
        },
        "train_time_seconds": round(elapsed, 1),
        "note": "weak held-out labels are the SAME proxy labels used for training "
                "-> secondary/sanity only. PRIMARY eval = human gold set "
                "(results/e7b_path_b/absa_gold_labeling.xlsx) once labeled.",
    }
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    print(f"\nSaved model -> {out_dir}")
    print(f"Saved metrics -> {metrics_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
