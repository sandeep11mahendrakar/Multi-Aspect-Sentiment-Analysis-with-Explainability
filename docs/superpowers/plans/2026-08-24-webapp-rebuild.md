# Web App Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Streamlit app with a FastAPI + vanilla JS web app around the FROZEN fp16 deploy weights, adding 1–10 scoring with −/±/+ bands, CSV-in/CSV-out (sorted, unsure sectioned at bottom), and HF Hub + Space deployment.

**Architecture:** FastAPI backend serves the static frontend and a small JSON/NDJSON API; pure `scoring.py` maps doc-model probs to score/band/unsure; `inference.py` loads both models from local `models/deploy/` with HF-hub override via env vars. Frontend is one page, two tabs, no build step.

**Tech Stack:** Python 3.12 venv, torch+transformers (existing), fastapi, uvicorn, python-multipart, pandas (CSV), vanilla HTML/CSS/JS.

**Spec:** `docs/superpowers/specs/2026-08-24-webapp-rebuild-design.md`

## Global Constraints

- Model weights/artifacts FROZEN — no retraining, no export changes.
- tau = 0.70 gate. Committed = max(prob) >= 0.70. Unsure/Mixed otherwise.
- Score formula: `1 + 9 * (p_pos + 0.5 * p_neu)`, 1 decimal.
- Bands by floor(score): 1–3 Negative(−) / 4–6 Neutral(±) / 7–10 Positive(+).
- Aspects shown: quality, price, fit ONLY (shipping excluded — standing decision).
- MAX_LEN_DOC=128, MAX_LEN_ABSA=96, batch 32, row cap 5000.
- CSV output columns: review_text, score, band, sentiment, confidence, quality, price, fit; sorted score DESC; unsure rows at bottom under header row `## UNSURE / MIXED (model not confident enough to rate)` with blank score.
- Run everything from repo root; venv at `venv\Scripts\activate`.
- seed42/model logic untouched.

---

### Task 1: scoring.py (pure functions) + unit tests

**Files:**
- Create: `apps/webapp/backend/scoring.py`
- Create: `apps/webapp/tests/test_scoring.py`

**Interfaces:**
- Produces:
  - `TAU: float = 0.70`
  - `LABELS = ["negative", "neutral", "positive"]`
  - `score_from_probs(probs: Sequence[float]) -> float` (probs order neg/neu/pos)
  - `band_from_score(score: float) -> str` returns `"negative"|"neutral"|"positive"`
  - `marker_from_band(band: str) -> str` returns `"−"|"±"|"+"`
  - `classify(probs) -> dict` → `{"unsure": bool, "score": float|None, "band": str|None, "sentiment": str|None, "confidence": float}` (confidence = max prob)

- [ ] Step 1: Write failing tests (bounds 1..10, band edges floor rule, tau boundary at exactly 0.70 = committed, unsure has score None)
- [ ] Step 2: Run pytest — FAIL
- [ ] Step 3: Implement scoring.py
- [ ] Step 4: pytest — PASS
- [ ] Step 5: Commit `feat(webapp): scoring module with 1-10 scale and unsure gating`

### Task 2: inference.py (model loading + predict)

**Files:**
- Create: `apps/webapp/backend/inference.py`

**Interfaces:**
- Consumes: `src/sentence_splitter.py` (`split_sentences`, `tag_aspects`)
- Produces:
  - `load_models() -> Models` (dataclass with doc_model, doc_tok, absa_model, absa_tok); source dirs resolved as: env `DOC_MODEL_ID`/`ABSA_MODEL_ID` (HF hub id) else local `models/deploy/{doc,absa}`
  - `doc_probs(texts: list[str]) -> np.ndarray` (batch)
  - `absa_analyze(text: str) -> dict[aspect, tuple[label, prob]]` (prob-mean aggregation like old app's deep-dive mode)

Loader replicates E8 Phase 3 round-trip exactly: sorted fp16_shard_*.pt → torch.load cpu → strict load_state_dict → .float().eval(). For hub ids use snapshot_download then same shard path.

- [ ] Step 1: Implement inference.py (loader + batch probs + absa aggregation)
- [ ] Step 2: Manual smoke: load locally, run 3 known texts, confirm clear-pos/clear-neg/"beautiful pattern but runs extremely small"→unsure
- [ ] Step 3: Commit `feat(webapp): frozen-weight inference loader (local + hub override)`

### Task 3: FastAPI main.py + API tests (mocked inference)

**Files:**
- Create: `apps/webapp/backend/main.py`
- Create: `apps/webapp/tests/test_api.py`
- Create: `apps/webapp/requirements.txt` (fastapi, uvicorn, python-multipart, pandas, numpy; torch/transformers from root requirements)

**Interfaces:**
- Consumes: scoring.classify, inference.load_models/doc_probs/absa_analyze
- Produces:
  - `GET /` → frontend/index.html
  - `GET /api/health` → `{status: "ok"|"loading"|"error", models: {...}}`
  - `POST /api/predict` `{text}` → `{unsure, score, band, marker, sentiment, confidence, probs, aspects:{quality:{sentiment,confidence}|null,...}}`
  - `POST /api/batch` multipart(file, column?) → NDJSON stream: `{"type":"meta","total":N,"columns":[...]}` | `{"type":"column_select","columns":[...]}` when auto-detect fails | `{"type":"progress","done":n,"total":N}` per chunk | `{"type":"done","rows":[...],"summary":{...}}`. Errors → `{"type":"error","message":...}`. Row cap 5000, empty-cell rows skipped.
  - Rows carry: text, unsure, score, band, marker, sentiment, confidence, aspects {quality,price,fit} each label|null
  - Summary: counts per band + unsure, mean committed score, coverage
- Tests monkeypatch inference functions; assert sort order desc, unsure placement last, cap rejection, column fallback event.

- [ ] Step 1: failing API tests
- [ ] Step 2: run — FAIL
- [ ] Step 3: implement main.py
- [ ] Step 4: pytest — PASS
- [ ] Step 5: commit `feat(webapp): FastAPI endpoints (predict, NDJSON batch, health)`

### Task 4: Frontend (index.html, style.css, app.js)

**Files:**
- Create: `apps/webapp/frontend/index.html`
- Create: `apps/webapp/frontend/style.css`
- Create: `apps/webapp/frontend/app.js`

Design: "Pattern Paper" palette carried over (ink #1C2430, accent #3A5BC7, green #1F9D66, red #C93B3B, amber #C77E14, paper #F6F7F9, hairline #E4E8EE) as CSS variables; grid-paper background; Bricolage Grotesque headings + Inter body + IBM Plex Mono numerals via Google Fonts. Two tabs (Single Review / Batch CSV). Single tab: textarea + 3 example preset chips, result card with big score dial (conic-gradient colored by band), −/±/+ marker, confidence bar, aspect chips, UNSURE badge state. Batch tab: dropzone, column-confirm UI if column_select event, progress bar from progress events, summary panel (distribution bars, mean score, coverage), preview table, Download CSV button (client-side build per spec column/sort/section rules), footer disclosure (τ=.70, committed acc 91.2% @ coverage 86.2%, kappa ceiling note).

- [ ] Step 1: write three files
- [ ] Step 2: serve via uvicorn, verify in browser (playwright): tabs render, single-review flow works end-to-end
- [ ] Step 3: commit `feat(webapp): vanilla JS frontend (single review + batch CSV)`

### Task 5: End-to-end verification (local, real models)

- [ ] Step 1: boot uvicorn headless, health 200
- [ ] Step 2: curl /api/predict regression trio (clear pos @ high conf, clear neg, mixed→unsure)
- [ ] Step 3: craft small test CSV (6 rows incl. 1-2 ambiguous), run through UI via playwright, download result CSV, verify sorting + unsure section + columns
- [ ] Step 4: pytest full suite green
- [ ] Step 5: commit any fixes `fix(webapp): e2e fixes`

### Task 6: Dockerfile for HF Space

**Files:**
- Create: `apps/webapp/Dockerfile`
- Create: `.dockerignore`

CPU-only torch install (`--index-url https://download.pytorch.org/whl/cpu`), copy backend+frontend+src/sentence_splitter.py, uvicorn on $PORT (Space sets 7860), README front-matter block for Space (sdk: docker, app_port: 7860) saved as `apps/webapp/SPACE_README.md`.

- [ ] Step 1: write Dockerfile + .dockerignore + SPACE_README.md
- [ ] Step 2: docker build + run + health check IF docker available locally; else record manual step
- [ ] Step 3: commit `feat(webapp): Dockerfile + Space config`

### Task 7: Hugging Face Hub upload + model cards

**Files:**
- Create: `src/experiments/upload_deploy_to_hf.py` (uploads models/deploy/{doc,absa} to `<owner>/...-doc-gated` / `-absa-consensus`, private=True, incl. model cards)
- Create: model card texts inside script (honest disclosures: tau .70, coverage .862/acc .9118, weak-label provenance, kappa=.15 ceiling, shipping excluded, fp16 shards + loader note)

BLOCKER: requires Sandeep's HF token (`hf auth login` or HF_TOKEN env). If absent at execution time, pause and ask.

- [ ] Step 1: check token availability
- [ ] Step 2: write upload script + cards
- [ ] Step 3: run upload once token available; verify re-download loads via DOC_MODEL_ID env
- [ ] Step 4: flip public after verification (with Sandeep's OK)
- [ ] Step 5: commit `feat: HF hub upload script + model cards`

### Task 8: Cleanup + docs

- [ ] Step 1: delete `apps/app.py` (replaced); keep labeling_app.py untouched (already abandoned but referenced in memory)
- [ ] Step 2: update root `requirements.txt`: remove streamlit/plotly/Pillow, add fastapi/uvicorn/python-multipart
- [ ] Step 3: README section: new app run instructions (`uvicorn apps.webapp.backend.main:app`), HF links (placeholders until Task 7 done)
- [ ] Step 4: commit `chore: retire streamlit app, point repo at webapp`

### Task 9: PROJECT_MEMORY.md update

- [ ] Step 1: add WEBAPP REBUILD section (what/where/how to run/deploy status), mark remaining issues #3 resolved
- [ ] Step 2: commit `docs: project memory — webapp rebuild`

## Execution notes

- Inline execution chosen by operator ("do the whole work phase-wise"); all tasks executed in-session with real-model verification at Tasks 2/5.

