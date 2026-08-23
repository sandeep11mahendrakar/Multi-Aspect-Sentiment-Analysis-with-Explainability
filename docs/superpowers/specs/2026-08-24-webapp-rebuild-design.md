# Design: Web App Rebuild (FastAPI + vanilla JS) + HF Hub Deployment

Date: 2026-08-24
Status: Approved by Sandeep
Scope: UI/deployment layer only. Model weights and training artifacts are FROZEN — no retraining, no logic changes to models.

## Context

The Streamlit app (`apps/app.py`, E8 Phase 3) serves the gated doc-level DistilBERT
(`models/deploy/doc`) + consensus ABSA (`models/deploy/absa`). Sandeep wants a rebuilt,
modern, lightweight site around the same frozen weights with new features:
1–10 review scoring with −/±/+ bands, CSV-in/CSV-out batch pipeline sorted by score with
unsure rows sectioned at bottom, plus deployment where recruiters can see a live demo.

Decision: replace the Streamlit app entirely; deploy as a free Hugging Face Space.
Vercel rejected: serverless function bundle/memory limits do not fit torch + 665 MB
resident model weights.

## Architecture

```
apps/webapp/
  backend/
    main.py        # FastAPI app: serves static frontend + JSON API
    inference.py   # model loading (HF hub first, local models/deploy fallback) + predict
    scoring.py     # pure functions: probs -> score/band/unsure (no I/O)
  frontend/
    index.html     # single page, two tabs
    style.css      # CSS variables color theme, no framework
    app.js         # vanilla JS, fetch API
  tests/
    test_scoring.py
    test_api.py    # mocked inference
  Dockerfile       # HF Space (Docker SDK)
requirements-webapp.txt
```

Old `apps/app.py` is deleted. `apps/labeling_app.py` already abandoned — left as-is.

### Backend

FastAPI + uvicorn. Endpoints:

- `GET /` — serves `frontend/index.html`
- `POST /api/predict` — body `{text}` → `{score, band, sentiment, confidence, unsure, aspects: {quality, price, fit}, probs}`
- `POST /api/batch` — multipart CSV upload + optional `column` override → runs chunked inference → returns `{rows: [...], summary: {...}}`; frontend builds/downloads CSV client-side (keeps server stateless)
- `GET /api/health` — model load status

Model loading (inference.py):
- Doc-level: gated DistilBERT @ tau=0.70. HF hub id (private until verified) with local
  `models/deploy/doc` fallback so dev works offline.
- ABSA: consensus ABSA, same strategy.
- Reuse existing sentence splitting/tagging from `src/sentence_splitter.py` for aspect view;
  shipping stays excluded (standing decision).
- fp16-shard loading path preserved (round-trip verified in E8 Phase 3).

### Scoring (scoring.py — pure, unit-tested)

- Input: doc-level softmax probs `[p_neg, p_neu, p_pos]`.
- Unsure gate FIRST: if `max(prob) < 0.70` → `{unsure: true}`; no score emitted.
- Score: `1 + 9 × (p_pos + 0.5 × p_neu)` rounded to 1 decimal, range [1.0, 10.0].
- Bands: 1–3 Negative (−) · 4–6 Neutral (±) · 7–10 Positive (+).
- Rationale: tau=.70 is the measured operating point (coverage .862 | acc .912 on committed).

### CSV pipeline

Input: any CSV; text column auto-detected by candidate names
(`Review Text, review_text, review, text, Review`) with manual dropdown fallback if none match.
Row cap: 5,000 (hard limit with clear error; protects free-tier Space RAM/CPU).

Processing: chunked (batch 32) with per-chunk progress reported to frontend via chunked
JSON response or SSE; doc model runs on every row; ABSA runs per sentence of each row
(aspects requested in output — accepted slower runtime).

Output CSV (built client-side from API rows):
- Columns: `review_text | score | band | sentiment | confidence | quality | price | fit`
- Sorted by score DESCENDING.
- Unsure/mixed rows moved to bottom under a distinct section header row `## UNSURE / MIXED (model not confident enough to rate)` with their own columns (confidence kept, score blank).

Summary panel data: counts per band + unsure, mean committed score, coverage %.

### Frontend

Single page, two tabs, vanilla JS + CSS (no React, no build step):

- **Single Review tab:** textarea + example presets; result card = large colored score
  dial (red/amber/green via CSS vars), −/±/+ marker, confidence bar, aspect chips
  (quality/price/fit), UNSURE/MIXED badge state.
- **Batch CSV tab:** dropzone/upload, detected-column confirmation (or picker), progress
  bar during processing, summary panel (distribution bars, mean score, coverage %),
  download-results button.
- Disclosure footer: real measured stats (committed acc 91.2% @ coverage 86.2%; abstained
  bucket = ambiguous mixed reviews; kappa ceiling note). Honesty signals are part of the
  product per project rules.

### Hugging Face Hub

- Upload `models/deploy/doc` and `models/deploy/absa` as two model repos under
  Sandeep's account (fp16 shards as-is). Private first, flip public after verification.
- Model cards drafted with honest disclosures (gating tau, coverage/acc, weak-label
  provenance, kappa=.15 ceiling, shipping excluded).

### HF Space deployment

- Docker Space running uvicorn; loads both models from hub at boot; static frontend
  served by same process. Single public URL for recruiters.

## Error handling

- Empty/too-long input text → 422 with friendly message.
- CSV: missing file, undetectable column (without override), >5,000 rows, malformed CSV → explicit error surfaced in UI.
- Model load failure at boot → health endpoint reports degraded; UI shows banner.

## Testing

- `test_scoring.py`: score formula bounds, band edges (3/4, 6/7), unsure gating at tau boundary.
- `test_api.py`: `/api/predict` and `/api/batch` with mocked inference; CSV sorting order; unsure-section placement; row-cap rejection.
- Manual regression: "beautiful pattern but runs extremely small" → UNSURE (known correct abstention); clear pos/neg sanity cases.
- Local Docker build + run before pushing Space.

## Out of scope

- Any model/training changes, quantization retries, LIME/attribution, learned abstention.
