# 🎯 Sentiment Analysis Pro — Gated DistilBERT + Aspect Sentiment

> Not just *what* customers feel — but *why*, and an honest **"I'm not sure"** when the model can't tell.

**Live demo (HF Space):** _pending first Space push_
**Models on the HF Hub:** _pending upload — see `src/experiments/upload_deploy_to_hf.py`_

---

## What This Project Does

Most sentiment systems force every review into positive / negative. This one doesn't:

- **Doc-level sentiment** with a fine-tuned DistilBERT, scored on a **1–10 scale** with − / ± / + bands
- **Selective prediction:** below confidence **τ = 0.70** the model **abstains** (`UNSURE / MIXED`) instead of guessing
- **Aspect-based sentiment** (quality / price / fit) via a consensus-filtered ABSA model
- **Batch CSV → scored CSV:** sorted by score, unsure rows sectioned at the bottom, summary dashboard included

---

## The Honest Numbers (measured, held-out n = 4,529)

| Operating point | Coverage | Committed acc | Committed macroF1 | Abstains |
|-----------------|----------|---------------|-------------------|----------|
| τ = 0.70 (deployed) | 86.2% | **91.2%** | .715 | 13.8% |
| τ = 0.80 | 77.8% | 95.1% | .690 | 22.2% |
| No gate (full test) | 100% | 85.5% | .695 | — |

- Labels are **weak** (derived from star ratings); intra-annotator κ ≈ 0.15 caps what "accuracy" can even mean here.
- The abstained bucket is dominated by genuinely ambiguous mixed reviews ("beautiful pattern but runs extremely small").
- **The gate is the product.** Forcing a prediction on ambiguous text is a lie; abstaining is a feature.

---

## Architecture

```
Review ──> Doc DistilBERT (fp16 shards, frozen) ──> probs ──> τ-gate ──> 1–10 score / UNSURE
                    │
                    └──> sentences ──> ABSA consensus (quality/price/fit) ──> aspect chips
```

- FastAPI backend + dependency-free vanilla JS frontend (no build step)
- Weights load from local `models/deploy/` or the HF Hub (`DOC_MODEL_ID` / `ABSA_MODEL_ID` env vars)
- fp16-shard round-trip verified at 100.0% prediction agreement; int8 quantization tried and rejected

---

## Run Locally

```bash
git clone https://github.com/sandeep11mahendrakar/sentiment-analysis-project.git
cd sentiment-analysis-project

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

uvicorn apps.webapp.backend.main:app --host 0.0.0.0 --port 7860
```

Open http://localhost:7860 — two tabs: **Single Review** and **Batch CSV** (max 5,000 rows per batch).

Tests: `python -m pytest apps\webapp\tests`

Deploy (HF Space, Docker SDK): push `apps/webapp/Dockerfile` with `SPACE_README.md` front-matter; models load from the Hub at boot.

---

## Dataset

Women's E-commerce Clothing Reviews (Kaggle) — ~23k real customer reviews.

---

## Project Structure

```
apps/webapp/        FastAPI backend + vanilla JS frontend + Dockerfile
src/                training / evaluation / experiment scripts
models/deploy/      frozen fp16 deploy weights (doc + absa)
results/            metrics, gating analysis, error analysis
docs/superpowers/   design spec + implementation plan for the webapp
PROJECT_MEMORY.md   full engineering log / source of truth
```

---

## Limitations (read before trusting any number)

- Weak-label provenance: stars → labels, not human sentiment annotation
- Annotation ceiling κ ≈ 0.15 — absolute accuracy must be read against it
- Committed bucket skews positive as τ rises
- ABSA gold-set performance is directional only (n = 91)
- English only; sarcasm and implicit sentiment remain hard

---

## Contact

**Sandeep Mahendrakar**
[LinkedIn](https://www.linkedin.com/in/sandeep-mahindrakar-336b972b9) · [GitHub](https://github.com/sandeep11mahendrakar)
