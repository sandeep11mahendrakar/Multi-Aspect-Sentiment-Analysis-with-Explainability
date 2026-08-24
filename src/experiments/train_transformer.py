"""E5: DistilBERT fine-tuning on RAW review text, 3-class rating labels.

Matches E1/E2 methodology exactly:
  - raw Review Text (NO stopword removal / lemmatization)
  - rating >= 4 positive | == 3 neutral | <= 2 negative
  - dropna(Review Text) -> 22,641 rows
  - stratified 80/20 split, random_state=42  (identical row assignment)

Experiments:
  --experiment baseline   plain cross-entropy
  --experiment weighted   class-weighted cross-entropy (weights from train freq)
  --labels rating|textual rating = original rating labels (default);
                          textual = rating-3 rows relabeled from
                          data/processed/corrected_rating3_labels.csv (E7B).
                          Split assignment stays identical either way
                          (stratification + seed unchanged).

Usage (from repo root):
  venv\\Scripts\\python.exe -u src\\experiments\\train_transformer.py --experiment baseline
  venv\\Scripts\\python.exe -u src\\experiments\\train_transformer.py --experiment baseline --max-steps 10   # smoke test
  venv\\Scripts\\python.exe -u src\\experiments\\train_transformer.py --labels textual                       # E7B
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
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
RAW_CSV = os.path.join(REPO_ROOT, "data", "raw", "reviews.csv")
CORRECTED_CSV = os.path.join(REPO_ROOT, "data", "processed", "corrected_rating3_labels.csv")
METRICS_DIR = os.path.join(REPO_ROOT, "results", "metrics")
MODELS_DIR = os.path.join(REPO_ROOT, "models")

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 128
SEED = 42
LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


def load_split():
    df = pd.read_csv(RAW_CSV)
    df = df.dropna(subset=["Review Text"]).reset_index(drop=True)
    df["label"] = df["Rating"].apply(
        lambda r: 0 if r <= 2 else (2 if r >= 4 else 1))  # neg=0, neu=1, pos=2
    idx_train, idx_test = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=SEED, stratify=df["label"])
    return df, idx_train, idx_test


class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.enc = tokenizer(list(texts), truncation=True, max_length=MAX_LEN,
                             padding="max_length", return_tensors="pt")
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


def full_eval(name, y_true, logits, elapsed, out):
    pred = np.argmax(logits, axis=1)
    acc = accuracy_score(y_true, pred)
    mf1 = f1_score(y_true, pred, average="macro")
    wf1 = f1_score(y_true, pred, average="weighted")
    p, r, f, s = precision_recall_fscore_support(y_true, pred, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(y_true, pred, labels=[0, 1, 2])
    out[name] = {
        "n_test": int(len(y_true)),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(mf1), 4),
        "weighted_f1": round(float(wf1), 4),
        "per_class": {LABELS[i]: {"precision": round(float(p[i]), 4),
                                  "recall": round(float(r[i]), 4),
                                  "f1": round(float(f[i]), 4),
                                  "support": int(s[i])} for i in range(3)},
        "confusion_rows_true_cols_pred": cm.tolist(),
        "train_time_seconds": round(elapsed, 1),
    }
    print(f"\n===== {name} =====")
    print(f"acc={acc:.4f} macro_f1={mf1:.4f} weighted_f1={wf1:.4f}")
    print(pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s},
                       index=LABELS).round(4).to_string())
    print(pd.DataFrame(cm, index=[f"true_{l}" for l in LABELS],
                       columns=[f"pred_{l}" for l in LABELS]).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=["baseline", "weighted"], default="baseline")
    ap.add_argument("--labels", choices=["rating", "textual"], default="rating",
                    help="rating: original labels | textual: model-corrected rating-3 labels (E7B)")
    ap.add_argument("--lr", type=float, default=None, help="override learning rate")
    ap.add_argument("--epochs", type=int, default=None, help="override num_train_epochs")
    ap.add_argument("--max-steps", type=int, default=-1, help="smoke-test override")
    args = ap.parse_args()
    lr = args.lr if args.lr is not None else 2e-5
    epochs = args.epochs if args.epochs is not None else 4

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")

    df, idx_train, idx_test = load_split()
    if args.labels == "textual":
        corr = pd.read_csv(CORRECTED_CSV)
        mapping = pd.Series(corr["textual_label"].map(LABEL2ID).values,
                            index=corr["row_index"])
        df["label_textual"] = df["label"].copy()
        df.loc[mapping.index, "label_textual"] = mapping.values
        n_changed = int((df["label"] != df["label_textual"]).sum())
        print(f"textual labels loaded: {len(corr)} rating-3 rows remapped "
              f"({n_changed} actually changed vs rating labels)")

    label_col = "label_textual" if args.labels == "textual" else "label"
    ytr = df.loc[idx_train, label_col].values.astype(int)
    yte = df.loc[idx_test, label_col].values.astype(int)
    yte_rating = df.loc[idx_test, "label"].values.astype(int)  # secondary eval reference
    print(f"rows={len(df)} train={len(idx_train)} test={len(idx_test)}")
    print("train dist:", np.bincount(ytr), "test dist:", np.bincount(yte))

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = ReviewDataset(df.loc[idx_train, "Review Text"], ytr, tok)
    test_ds = ReviewDataset(df.loc[idx_test, "Review Text"], yte, tok)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID)

    variant = args.experiment
    if args.labels == "textual":
        variant += "_textuallabels"
    if args.lr is not None or args.epochs is not None:
        variant += f"_lr{lr:g}_ep{epochs}"
    run_name = f"distilbert_3class_{variant}"
    out_dir = os.path.join(MODELS_DIR, run_name)

    targs = TrainingArguments(
        output_dir=out_dir,
        seed=SEED,
        data_seed=SEED,
        fp16=(device == "cuda"),
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,          # effective batch 32
        learning_rate=lr,
        num_train_epochs=epochs,
        warmup_steps=113,  # ~10% of 1,131 total steps (18,112 / 32 effective bs * 4 epochs)
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

    trainer_kwargs = dict(model=model, args=targs, train_dataset=train_ds,
                          eval_dataset=test_ds, compute_metrics=compute_metrics)
    if args.experiment == "weighted":
        w = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=ytr)
        w = torch.tensor(w, dtype=torch.float32).to(device)
        print("class weights:", w.tolist())
        trainer = WeightedTrainer(class_weights=w, **trainer_kwargs)
    else:
        trainer = Trainer(**trainer_kwargs)

    if args.max_steps > 0:
        trainer.train()
        print("SMOKE TEST OK")
        return

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    pred_out = trainer.predict(test_ds)
    os.makedirs(METRICS_DIR, exist_ok=True)
    metrics_path = os.path.join(METRICS_DIR, "transformer_experiments.json")
    results = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, encoding="utf-8") as fh:
            results = json.load(fh)
    full_eval(run_name, yte, pred_out.predictions, elapsed, results)
    if args.labels == "textual":
        full_eval(run_name + "__vs_rating_labels", yte_rating, pred_out.predictions,
                  elapsed, results)
    results[run_name]["train_args"] = {
        "model": MODEL_NAME, "max_len": MAX_LEN, "batch": 16, "grad_accum": 2,
        "lr": lr, "epochs": epochs, "fp16": True, "early_stopping": "macro_f1, patience 3",
        "device": device, "text": "raw Review Text (no TF-IDF preprocessing)",
        "label_mode": args.labels,
    }
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    pd.DataFrame({"y_true": yte, "y_pred": np.argmax(pred_out.predictions, axis=1)}
                 ).to_csv(os.path.join(METRICS_DIR, f"{run_name}_predictions.csv"), index=False)
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    print(f"\nSaved model -> {out_dir}")
    print(f"Saved metrics -> {metrics_path}")


if __name__ == "__main__":
    sys.exit(main())
