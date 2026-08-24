"""Export fp16 SHARDED deploy artifacts for the Streamlit app.

Why not fp32: 255 MB/model exceeds GitHub's 100 MB file cap (Streamlit
Community Cloud pulls code+weights from git).
Why not int8: torch.ao dynamic-quantized state_dicts don't round-trip through
load_state_dict in torch 2.x (quantized Linear expects a different key layout
than state_dict() emits) and pred agreement vs fp32 was only 81-86% on
ambiguous sentences.
fp16: exact same architecture, vanilla strict load_state_dict, ~50% size and
RAM cut, near-identical predictions. Shards capped at 90 MB so they can live
in git directly.

Outputs (per model): models/deploy/{doc,absa}/
  fp16_shard_000.pt ...   state_dict shards (strict-loadable)
  config.json, tokenizer files
Prints: sizes, CPU latency, RAM delta, fp16-vs-fp32 prediction agreement
on the human gold set.

Run: venv\\Scripts\\python.exe -u src\\experiments\\export_quantized_deploy.py
"""

import glob
import json
import os
import shutil
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = {
    "doc": os.path.join(REPO_ROOT, "models", "distilbert_3class_baseline"),
    "absa": os.path.join(REPO_ROOT, "models", "distilbert_absa_aspectconditioned_consensus"),
}
OUT_ROOT = os.path.join(REPO_ROOT, "models", "deploy")
KEEP_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json",
              "vocab.txt", "special_tokens_map.json"]
GOLD_XLSX = os.path.join(REPO_ROOT, "results", "e7b_path_b", "absa_gold_labeling.xlsx")
SHARD_BYTES = 90 * 1024 * 1024  # stay under GitHub's 100 MB hard cap


def save_sharded(state_dict, out_dir):
    for f in glob.glob(os.path.join(out_dir, "fp16_shard_*.pt")):
        os.remove(f)
    for f in glob.glob(os.path.join(out_dir, "int8_shard_*.pt")):
        os.remove(f)
    shards, cur, cur_bytes, idx = [], {}, 0, 0
    for k, v in state_dict.items():
        b = v.numel() * v.element_size()
        if cur and cur_bytes + b > SHARD_BYTES:
            path = os.path.join(out_dir, f"fp16_shard_{idx:03d}.pt")
            torch.save(cur, path)
            shards.append(path)
            cur, cur_bytes, idx = {}, 0, idx + 1
        cur[k] = v
        cur_bytes += b
    if cur:
        path = os.path.join(out_dir, f"fp16_shard_{idx:03d}.pt")
        torch.save(cur, path)
        shards.append(path)
    return shards


def load_fp16_model(model_dir):
    cfg = AutoConfig.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_config(cfg)
    sd = {}
    for path in sorted(glob.glob(os.path.join(model_dir, "fp16_shard_*.pt"))):
        sd.update(torch.load(path, map_location="cpu"))
    model.load_state_dict(sd)  # strict
    return model.half().eval()


def mb(path):
    return os.path.getsize(path) / 1e6


@torch.no_grad()
def infer(m, tok, texts, aspects=None):
    preds = []
    for i in range(0, len(texts), 64):
        if aspects is not None:
            enc = tok(texts[i:i + 64], aspects[i:i + 64], truncation=True,
                      max_length=96, padding=True, return_tensors="pt")
        else:
            enc = tok(texts[i:i + 64], truncation=True,
                      max_length=96, padding=True, return_tensors="pt")
        enc = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in enc.items()}
        preds.extend(m(**enc).logits.argmax(-1).tolist())
    return np.array(preds)


def main():
    gold = pd.read_excel(GOLD_XLSX, sheet_name="gold").dropna(subset=["score"])
    report = {}

    for name, mdir in SRC.items():
        out_dir = os.path.join(OUT_ROOT, name)
        os.makedirs(out_dir, exist_ok=True)

        tok = AutoTokenizer.from_pretrained(mdir)
        model = AutoModelForSequenceClassification.from_pretrained(mdir).eval()
        h_model = load_fp16_model(out_dir) if \
            glob.glob(os.path.join(out_dir, "fp16_shard_*.pt")) else None

        if h_model is None:
            sd = {k: v.half() for k, v in model.state_dict().items()}
            shards = save_sharded(sd, out_dir)
            for f in KEEP_FILES:
                src_f = os.path.join(mdir, f)
                if os.path.exists(src_f):
                    shutil.copy(src_f, os.path.join(out_dir, f))
            h_model = load_fp16_model(out_dir)

        texts = gold["sentence"].tolist()
        aspects = gold["aspect"].tolist() if name == "absa" else None

        t0 = time.perf_counter()
        p_fp = infer(model, tok, texts, aspects)
        t_fp = time.perf_counter() - t0
        t0 = time.perf_counter()
        p_h = infer(h_model, tok, texts, aspects)
        t_h = time.perf_counter() - t0

        agree = float((p_fp == p_h).mean())
        shards = sorted(glob.glob(os.path.join(out_dir, "fp16_shard_*.pt")))
        report[name] = {
            "fp32_mb": round(mb(os.path.join(mdir, "model.safetensors")), 1),
            "fp16_total_mb": round(sum(mb(s) for s in shards), 1),
            "n_shards": len(shards),
            "max_shard_mb": round(max(mb(s) for s in shards), 1),
            "fp32_batch91_seconds": round(t_fp, 2),
            "fp16_batch91_seconds": round(t_h, 2),
            "pred_agreement_vs_fp32": round(agree, 4),
        }
        print(f"{name}: fp32 {report[name]['fp32_mb']} MB -> fp16 "
              f"{report[name]['fp16_total_mb']} MB in {len(shards)} shards "
              f"(max {report[name]['max_shard_mb']} MB) | batch-of-91 cpu "
              f"{t_fp:.2f}s -> {t_h:.2f}s | pred agreement {agree:.1%}")

        # single-sample CPU latency (the number that matters for the live app)
        enc = tok(texts[7], truncation=True, max_length=96,
                  padding="max_length", return_tensors="pt")
        enc = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in enc.items()}
        with torch.no_grad():
            h_model(**enc)
            t0 = time.perf_counter()
            for _ in range(10):
                h_model(**enc)
        report[name]["fp16_single_ms"] = round((time.perf_counter() - t0) * 100, 1)
        print(f"  single-review fp16 latency: ~{report[name]['fp16_single_ms']:.0f} ms")

    import psutil
    proc = psutil.Process()
    base = proc.memory_info().rss / 1e6
    loaded = []
    for name in SRC:
        loaded.append(load_fp16_model(os.path.join(OUT_ROOT, name)))
    ram = proc.memory_info().rss / 1e6 - base
    report["ram_delta_both_models_loaded_mb"] = round(ram, 1)
    print(f"\nRAM delta loading BOTH fp16 models: ~{ram:.0f} MB "
          f"(on top of interpreter/torch baseline)")

    with open(os.path.join(OUT_ROOT, "export_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print("-> models/deploy/export_report.json")


if __name__ == "__main__":
    main()
