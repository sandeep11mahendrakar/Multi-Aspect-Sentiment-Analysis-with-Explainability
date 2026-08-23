"""Upload the frozen E8 deploy models to the Hugging Face Hub.

Uploads models/deploy/{doc,absa} as two PRIVATE model repos:
  <owner>/distilbert-sentiment-doc-gated
  <owner>/distilbert-absa-consensus

Only the verified fp16-shard artifacts ship (fp16_shard_*.pt + config +
tokenizer). int8_state_dict.pt is excluded: int8 dynamic quantization was
tried and REJECTED in E8 Phase 3 (state_dict does not round-trip).

Usage (token required, write scope):
    set HF_TOKEN=hf_...   (or: hf auth login)
    python src/experiments/upload_deploy_to_hf.py [--public]

--public flips the repos public AFTER verification (run once verified).
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from huggingface_hub import HfApi  # noqa: E402

OWNER = "sandeep11mahendrakar"

DOC_CARD = """\
---
license: mit
base_model: distilbert-base-uncased
tags:
- sentiment-analysis
- selective-prediction
- weak-labels
- distilbert
library_name: transformers
---

# DistilBERT Sentiment (doc-level, confidence-gated)

3-class doc-level review sentiment (negative / neutral / positive) with a
**selective-prediction gate**: predictions whose max softmax probability is
below **tau = 0.70** are abstained (UNSURE / MIXED) instead of forced.

## Measured operating point (held-out, n = 4,529)

| tau | coverage | committed acc | committed macroF1 | abstains |
|-----|----------|---------------|-------------------|----------|
| .70 | 86.2%    | 91.2%         | .715              | 13.8%    |
| .80 | 77.8%    | 95.1%         | .690              | 22.2%    |

Uncalibrated full-test baseline (same weights, no gate): acc .855 / macroF1 .695.

## Honest limitations

- **Weak-label provenance:** training labels are derived from review-star
  ratings (3 -> neutral, <=2 -> negative, >=4 -> positive), not human
  annotation of sentiment.
- **Annotation ceiling:** intra-annotator kappa on this dataset is ~0.15
  (collapsed agreement 50%) — absolute accuracy numbers must be read against
  that ceiling. The gate, not raw accuracy, is the product.
- **Abstained bucket** is dominated by genuinely ambiguous mixed reviews
  (e.g. "beautiful pattern but runs extremely small").
- **Committed bucket skews positive** as tau rises.

## Artifacts & loading

Weights are fp16 shards (`fp16_shard_000.pt`, `fp16_shard_001.pt`) that must
be merged before `load_state_dict` (strict):

```python
import torch, glob
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

state = {}
for p in sorted(glob.glob("fp16_shard_*.pt")):
    state.update(torch.load(p, map_location="cpu"))
cfg = AutoConfig.from_pretrained(".")
model = AutoModelForSequenceClassification.from_config(cfg)
model.load_state_dict(state)          # strict
tok = AutoTokenizer.from_pretrained(".")
model.float().eval()                  # fp32 compute
```

Verified round-trip: 100.0% prediction agreement with the pre-shard fp32 model.
int8 dynamic quantization was tried and rejected (does not round-trip).

Max sequence length at train time: 128. Labels: 0=negative, 1=neutral, 2=positive.
"""

ABSA_CARD = """\
---
license: mit
base_model: distilbert-base-uncased
tags:
- aspect-based-sentiment-analysis
- absa
- weak-labels
- distilbert
library_name: transformers
---

# DistilBERT ABSA (aspect consensus)

Aspect-level sentiment for **quality / price / fit** (shipping intentionally
excluded) on fashion e-commerce reviews. Input format: sentence + aspect pair.

## Provenance

Trained on aspect-sentiment pairs built by a **doc+sentence consensus filter**
(both the doc-level model and a sentence-level model must agree on the
sentence's aspect polarity) over weak labels derived from star ratings.

## Measured performance

- Human gold subset (n = 91 pairs): **acc .45 / macroF1 .45** — small sample,
  and the same ~0.15 annotation-kappa ceiling applies. Treat as directional.
- Consensus filtering trades recall for label cleanliness by design.

## Honest limitations

- Weak-label provenance (star-derived); consensus reduces but does not
  eliminate label noise.
- Only quality / price / fit aspects are supported.
- Max sequence length 96. Labels: 0=negative, 1=neutral, 2=positive.

## Artifacts & loading

Same fp16-shard format as the doc model (merge shards, strict
`load_state_dict`, `.float().eval()`). Verified 100.0% round-trip agreement.
"""


def _upload_one(api: HfApi, local_dir: str, repo_id: str, card: str, private: bool):
    patterns = ["fp16_shard_*.pt", "config.json", "tokenizer.json",
                "tokenizer_config.json"]
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=local_dir,
        allow_patterns=patterns,
    )
    print(f"uploaded {local_dir} -> {repo_id} (private={private})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", action="store_true",
                    help="flip repos public (only after verification)")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN env var not set (or run `hf auth login`). "
                 "A write-scope token is required.")

    api = HfApi(token=token)
    jobs = [
        (os.path.join(REPO_ROOT, "models", "deploy", "doc"),
         f"{OWNER}/distilbert-sentiment-doc-gated", DOC_CARD),
        (os.path.join(REPO_ROOT, "models", "deploy", "absa"),
         f"{OWNER}/distilbert-absa-consensus", ABSA_CARD),
    ]
    for local_dir, repo_id, card in jobs:
        if not os.path.isdir(local_dir):
            sys.exit(f"missing deploy dir: {local_dir}")
        _upload_one(api, local_dir, repo_id, card, private=not args.public)
    print("done.")


if __name__ == "__main__":
    main()
