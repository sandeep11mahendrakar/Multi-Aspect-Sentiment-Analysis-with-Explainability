"""Model loading + prediction for the frozen E8 deploy artifacts.

Weights are FROZEN (E8 Phase 3). Loading replicates the verified fp16-shard
round-trip exactly: sorted fp16_shard_*.pt -> torch.load(cpu) -> strict
load_state_dict -> .float().eval() (fp32 compute, ~25 ms/review).

Source resolution per model:
  1. env DOC_MODEL_ID / ABSA_MODEL_ID (HF hub repo id) via snapshot_download
  2. local models/deploy/{doc,absa} fallback (offline dev)
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if os.path.join(REPO_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from sentence_splitter import split_sentences, tag_aspects  # noqa: E402

from apps.webapp.backend.scoring import LABELS

DEPLOY_DIR = os.path.join(REPO_ROOT, "models", "deploy")
MAX_LEN_DOC = 128   # matches training
MAX_LEN_ABSA = 96   # matches training
ASPECTS_SHOWN = ["quality", "price", "fit"]  # shipping dropped from ABSA training


@dataclass
class Models:
    doc_model: torch.nn.Module
    doc_tok: object
    absa_model: torch.nn.Module
    absa_tok: object


def _resolve_dir(env_key: str, local_name: str) -> str:
    hub_id = os.environ.get(env_key)
    if hub_id:
        from huggingface_hub import snapshot_download
        return snapshot_download(hub_id)
    d = os.path.join(DEPLOY_DIR, local_name)
    if not os.path.isdir(d):
        raise FileNotFoundError(
            f"{d} not found and {env_key} not set. Set the env var to a HF "
            "repo id containing fp16_shard_*.pt or restore models/deploy/.")
    return d


def _load_deploy(directory: str) -> Tuple[torch.nn.Module, object]:
    shard_paths = sorted(
        p for p in os.listdir(directory)
        if p.startswith("fp16_shard_") and p.endswith(".pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No fp16_shard_*.pt in {directory}")
    state_dict = {}
    for p in shard_paths:
        state_dict.update(torch.load(os.path.join(directory, p), map_location="cpu"))
    cfg = AutoConfig.from_pretrained(directory)
    model = AutoModelForSequenceClassification.from_config(cfg)
    model.load_state_dict(state_dict)  # strict — shards must cover everything
    tok = AutoTokenizer.from_pretrained(directory)
    return model.float().eval(), tok


def load_models() -> Models:
    doc_model, doc_tok = _load_deploy(_resolve_dir("DOC_MODEL_ID", "doc"))
    absa_model, absa_tok = _load_deploy(_resolve_dir("ABSA_MODEL_ID", "absa"))
    return Models(doc_model, doc_tok, absa_model, absa_tok)


@torch.no_grad()
def doc_probs(models: Models, texts: List[str]) -> np.ndarray:
    """Batched doc-level softmax probs, shape (len(texts), 3), order neg/neu/pos."""
    if not texts:
        return np.zeros((0, 3))
    enc = models.doc_tok(texts, truncation=True, max_length=MAX_LEN_DOC,
                         padding=True, return_tensors="pt")
    logits = models.doc_model(**enc).logits
    return torch.softmax(logits, dim=-1).numpy()


@torch.no_grad()
def absa_analyze(models: Models, text: str) -> Dict[str, Tuple[str, float]]:
    """Consensus ABSA over quality/price/fit; prob-mean across matching sentences."""
    per_aspect: Dict[str, List[np.ndarray]] = {}
    for sent in split_sentences(text):
        for aspect in tag_aspects(sent):
            if aspect not in ASPECTS_SHOWN:
                continue
            enc = models.absa_tok([sent], [aspect], truncation=True,
                                  max_length=MAX_LEN_ABSA, padding=True,
                                  return_tensors="pt")
            logits = models.absa_model(**enc).logits
            per_aspect.setdefault(aspect, []).append(torch.softmax(logits, dim=-1)[0].numpy())
    out: Dict[str, Tuple[str, float]] = {}
    for aspect, probs_list in per_aspect.items():
        mean = np.mean(probs_list, axis=0)
        idx = int(mean.argmax())
        out[aspect] = (LABELS[idx], float(mean[idx]))
    return out
