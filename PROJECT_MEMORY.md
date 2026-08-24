# PROJECT MEMORY — Sentiment Analysis Project
> Context file for continuing work in future sessions. Update after major changes.
> Last updated: 2026-08-24 (webapp rebuild complete; model branded "Lilly · Fashion Muse"; HF push pending token)
>
> **RECONSTRUCTION NOTE (2026-08-24):** the original file was accidentally truncated during a
> botched in-place edit and no backup existed. This version was rebuilt from (a) verbatim
> fragments captured in-session, (b) the committed spec/plan under docs/superpowers/, and
> (c) the on-disk result artifacts in results/ — which remain the source of truth for every
> number below. Section coverage is slightly leaner than the original but all metrics are
> artifact-backed.

## FINAL RESULTS (the reference table for the report)

| Model | Eval | Metric | Value | Note |
|---|---|---|---|---|
| Doc-level baseline (distilbert_3class_baseline) | full test (n=4,529) acc / macroF1 | **.8547 / .6952** | rating-derived labels (results/metrics/transformer_experiments.json) |
| Confidence-gated baseline, tau=.70 (E8) | committed-bucket acc / macroF1 | **.9118 / .7148** | 86.2% coverage; T-scaled probs (results/e8_confidence_gating/phase1_calibration.json) |
| Consensus ABSA on human gold (Path B + E8 P2) | gold-set acc / macroF1 | **.4505 / .4466** | n=91 | Small sample; same kappa ceiling applies (results/e7b_path_b/absa_gold_eval_consensus.json) |
| Textual-labels retrain (Path A) | full test vs corrected labels | .9028 / .7609 | circular/inflated; .8545/.6819 vs original rating labels |

## CURRENT STATE (TL;DR for a new chat)
- Best model: `models/distilbert_3class_baseline_textuallabels/` — modest but real improvement on the
  human gold subset (macroF1 .3941 vs .3665 baseline; acc .3970 vs .3719) on inherently ambiguous rating-3 reviews.
  On full test set: acc .9028 / macroF1 .7609 vs CORRECTED labels (circular, inflated); .8545/.6819 vs original rating labels.
- DEPLOYED app (since 2026-08-24): FastAPI webapp in apps/webapp/ serving the GATED
  distilbert_3class_baseline (tau=.70) + consensus ABSA — see WEBAPP REBUILD section. Streamlit retired.
- Decision history: E7A showed rating-3 labels structurally noisy (28% truly neutral, κ=0.15); pre-registered rule
  briefly favored Path B, but Sandeep reviewed all 140 human-vs-model disagreements and judged the MODEL mostly correct
  → PATH A executed (relabel rating-3 with baseline model predictions, retrain).
- Model relabel was conservative: only 825/2823 rating-3 rows actually changed (model kept 70.8% neutral,
  vs Sandeep's blind human dist 41/28/31 neg/neu/pos).
- Path B ABSA trained on 34,955 weak pairs (max_len 96, aspect-conditioned); consensus-filtered
  version is the deployed ABSA artifact.

## Project Overview
- **Goal:** Multi-aspect sentiment analysis of Women's Clothing Reviews with explainability + web dashboard. Goal shifted from "max accuracy" to "technically defensible system" (see E6/E7A findings).

## Environment (verified working - do NOT rebuild)
- venv at repo root (`venv\Scripts\activate`), Python 3.12, torch CUDA, seed 42 everywhere.
- Run everything from repo root.

## Structure
- src/ — training/eval/experiment scripts (src/experiments/ has the one-offs)
- apps/webapp/ — the deployed app (backend + frontend + Dockerfile)
- models/ — training runs + models/deploy/{doc,absa} (frozen fp16 shards)
- data/, results/ — artifacts; docs/superpowers/ — spec + plan for the webapp rebuild

## E7A LABEL-NOISE DIAGNOSTIC (COMPLETE 2026-08-23 - the key result of the project so far)
Script: src/experiments/... -> results/e7a/ (e7a_analysis.json is the artifact).
- 200 rating-3 reviews sampled for human labeling.
  Labeled: 199/200 R1 (id=4 skipped) + 24/50 R2 kappa subset.
- Human dist on rating-3 (n=199): 41.2% neg / 28.1% neu / 30.7% pos — the "neutral" star-3
  class is NOT neutral under human judgment (28% truly neutral).
- Human-vs-model agreement .3719 (baseline) / .3970 (textuallabels retrain) on the subset;
  macroF1 .3665 / .3941.
- Intra-annotator kappa (24 pairs): 0.150, collapsed agreement 50%, exact score 12.5%, within−1 47.9%
  → the annotation task itself is near-chance for collapsed 3-class; κ≈0.15 is the CEILING argument.
- decision_hint recorded: "structurally noisy" — fed the Path A vs Path B decision.

## E7B RELABEL EXPERIMENT (COMPLETE 2026-08-23 - results recorded)
- PATH A executed: 825/2823 rating-3 rows relabeled with model predictions (conservative;
  model kept 70.8% neutral vs Sandeep's blind human dist 41/28/31).
- Retrain = distilbert_3class_baseline_textuallabels: full-test .9028/.7609 vs corrected
  (circular — inflated), .8545/.6819 vs original rating labels; gold subset .3970/.3941.
- Conclusion: relabeling helps the human-gold metric modestly; cannot fix the κ ceiling.

## E8 CONFIDENCE GATING PHASE 1 (COMPLETE 2026-08-23 - calibration + risk-coverage, no training)
Script: src/experiments/confidence_gating_phase1.py -> results/e8_confidence_gating/
1. Temperature scaling: T=0.895 fit on train-cal (n=1811). ECE small either way
   (cal raw .0109 / T .014); T-scaled adopted for gating (better coverage at same acc).
2. Risk–coverage (T-scaled, test n=4529):
   tau .70 -> .8616 | .9118   tau .80 -> .7783 | .9509   tau .90 -> .7236 | .9670
   (raw probs similar: tau .80 -> .7584 | .9590.)
3. HEADLINE CANDIDATE (DECIDED by Sandeep): operating point tau=.70 - coverage .8616 | acc .9118
   Writeup table MUST show tau .70 AND tau .80 side by side, each with coverage + macroF1:
     tau .70: cov 86.2% | acc 91.2% | macroF1 .715 | abstains 13.8%
     tau .80: cov 77.8% | acc 95.1% | macroF1 .690 | abstains 22.2%
4. Caveats recorded: (a) human ceiling on 3-class collapse is low
   (~kappa .15 territory); (b) committed bucket skews positive as tau rises; (c) abstention rate
   concentrates on genuinely mixed reviews (spot-checks confirm).

## PATH B / ABSA (scaffold + training COMPLETE 2026-08-23 - gold eval done 2026-08-23)
1. Pair build: sentence-split + keyword aspect tagging -> 34,955 weak pairs
   (fit 23,016 / quality 9,288 / price 2,107 / shipping 544).
2. Pair-noise estimate (pair_noise_estimate.json): overall proxy-vs-sentence disagreement 29.5%
   (shipping worst at 39.7% — one reason shipping was later dropped).
3. 8. DONE (E8 PHASE 2): aspect-tag audit + consensus-filter rebuild + ABSA retrain.
   Consensus = doc-level + sentence-level polarity must agree; rebuilt pairs CSV is
   data/processed/aspect_sentiment_pairs_consensus.csv.
4. Training (absa_training.json): aspect-conditioned DistilBERT, input "[CLS] aspect [SEP] sentence [SEP]",
   max_len 128, bs16xaccum2, lr 2e-5, fp16, 4 epochs, early stop macro_f1. ~8.4 min/run, VRAM ~2.6GB.
   max_len 96 for pairs; GroupShuffleSplit by review (no leakage); shipping excluded; class weights balanced.
   Weak held-out (sanity only, same proxy labels): acc .7358 / macroF1 .5675.
5. Human gold (n=91, Sandeep-labeled): raw pairs model vs consensus model evaluated;
   CONSENSUS is the deployed artifact: acc .4505 / macroF1 .4466 (absa_gold_eval_consensus.json).
   Directional only — small n and the κ ceiling apply.
6. SHIPPING DECISION: dropped from ABSA training (544 pairs, worst noise); showing a weak
   prediction in the same UI slot as quality/price/fit would fake consistency → excluded
   everywhere (app + model card) and stated as a limitation instead.
7. Doc-level tau=.70 routing of ABSA sentences tried:
   tau=.70 routing: acc .8267, macroF1 .5824 (small gain; possible later calibration layer only).
8. E7B deep-dive note: single-digit band predicted; still kappa-capped. Neutral over-prediction slightly reduced
   by the consensus filter (neutral recall .6364 on gold, precision low .2745 — known weakness).

## E6 ERROR ANALYSIS (verified 2026-08-22, error_analysis.json)
- 658/4529 errors (14.5%). Dominant flows: pos→neu (191), neg→neu (157), neu→pos (158).
- Errors skew LONG reviews (long-25% acc .8233 vs short-25% .8943) and contrast-heavy text.
- 21.4% of confident-wrong cases have conf > .8 — motivates E8 gating.
- neu→pos high-conf examples are rating-3 reviews that read clearly positive → weak-label noise, not model failure.

## E5 TRANSFORMER (verified 2026-08-22)
- distilbert-base-uncased, raw Review Text, max_len 128, bs16×accum2, lr 2e-5, 4 epochs,
  fp16, early stop macro_f1. Full-test .8547/.6952 (transformer_experiments.json).
- Variants: class-weighted .8417/.6947 (no gain), lr3e-5 .8565/.6919 (no macroF1 gain), ep2 .8496/.6656 (worse).
- Baseline (lr2e-5/ep4) kept as the project model.

## E2/E1 CLASSICAL (verified 2026-08-22)
- LR/TF-IDF era: retired from the app 2026-08-23; kept only as the E7A "Top Signals" history.
- lime/nltk/joblib/scikit-learn removed from requirements.

## DEPLOYMENT (E8 Phase 3 — DONE 2026-08-23, local verified / cloud push pending)
apps/app.py REWRITTEN: LR+TF-IDF fully replaced.
1. Doc-level = gated DistilBERT (models/deploy/doc, tau=.70). Below-threshold inputs get an
   explicit "MIXED / UNCERTAIN" badge + plain-language explanation (ambiguity even for humans;
   committed cases measured 91.2% acc). Batch mode adds Mixed count card + coverage metric.
2. Aspect view = consensus ABSA (models/deploy/absa), sentence-split + keyword-tagged
   (src/sentence_splitter), per-aspect prob-mean aggregation. SHIPPING REMOVED from UI
   (decided: dropped from ABSA training, 544 pairs — showing a TF-IDF prediction in the same
   slot would fake consistency; stated as limitation on-screen instead).
3. Hosting feasibility MEASURED, not assumed:
   - fp32 weights 255 MB/model -> EXCEEDS GitHub 100 MB file cap; cannot reach Streamlit
     Community Cloud via git at all without sharding/external hosting.
   - int8 dynamic quantization tried and REJECTED: state_dict does not round-trip
     (quantized Linear key mismatch) AND only 81-86% pred agreement vs fp32 on ambiguous gold.
   - fp16 SHARDED adopted: 134 MB/model in 2 shards (<100 MB each) -> git-safe; 100.0% pred
     agreement vs fp32 on the 91-row gold set; RAM both models resident ~665 MB (fp32 upcast).
   - Total est. RSS ~1.0-1.1 GB with torch+server baseline. FITS typical free containers
     (~2.8 GB) but NOT any hard-1GB tier. Knobs if needed: load .half() instead of
     .float() (~430 MB weights, but ~220 ms/review CPU); or upgrade tier.
4. LIME/explainability DECISION: dropped per-prediction attribution. Reasons recorded in-app:
   LIME surrogate matched the retired TF-IDF model; faithful gradient methods too slow for
   free CPU tier; multi-layer DistilBERT attention is not faithful attribution. Confidence +
   gating decision are the honesty signals. (lime/nltk/joblib/scikit-learn removed from
   requirements.txt; note old "Top Signals" section was coef-based, LIME was never actually
   wired in the old app.)
VERIFIED locally: loader round-trip strict-load OK;
predict path: clear-pos -> positive @1.00 | clear-neg -> negative @.92 |
"beautiful pattern but runs extremely small" -> MIXED (correct abstention).
NOTE 2026-08-24: apps/app.py (Streamlit) DELETED — replaced by apps/webapp (see below).

## WEBAPP REBUILD (COMPLETE 2026-08-24 — replaces Streamlit; HF push pending token)
BRANDING (2026-08-24, Sandeep's pick): model/app name = **Lilly**, category = **Fashion Muse**
(femme persona, universal utility — reads menswear too; stated on-site + in cards).
Everywhere consistent: site title "Lilly · Fashion Muse", HF repos
sandeep11mahendrakar/lilly-fashion-muse-doc-gated + .../lilly-fashion-muse-absa,
Dockerfile ENV defaults, upload script, model cards, README, CSV download name
lilly_review_scores.csv. Historical spec/plan docs under docs/superpowers/ left as dated records.
apps/app.py DELETED. New app: apps/webapp/ (FastAPI + vanilla JS, no build step).
- Run: `uvicorn apps.webapp.backend.main:app --port 7860` (repo root). Tests: `pytest apps\webapp\tests`.
- backend/scoring.py — pure: probs -> 1-10 score `1 + 9*(p_pos + 0.5*p_neu)`, bands by floor
  (1-3 neg / 4-6 neu / 7-10 pos), tau=.70 gate FIRST (unsure => no score). Unit-tested.
- backend/inference.py — frozen fp16-shard loader (local models/deploy fallback; env
  DOC_MODEL_ID / ABSA_MODEL_ID override via snapshot_download). ABSA per-sentence prob-mean,
  quality/price/fit only.
- backend/main.py — GET / (frontend), GET /api/health, POST /api/predict,
  POST /api/batch (NDJSON stream: meta | column_select | progress | done | error; row cap 5000,
  chunk 32, text column auto-detect + manual fallback).
- frontend/ — two tabs: Single Review (score dial, −/±/+ marker, confidence bar, aspect chips,
  UNSURE badge) and Batch CSV (dropzone, progress, summary panel, preview table, client-side
  CSV download: review_text|score|band|sentiment|confidence|quality|price|fit, sorted score
  DESC, unsure rows under `## UNSURE / MIXED` header). Disclosure footer with real metrics.
- Dockerfile + SPACE_README.md (sdk: docker, app_port 7860) ready for HF Space; docker not
  available locally — build happens on Space push (manual step: verify cold boot).
- src/experiments/upload_deploy_to_hf.py — uploads models/deploy/{doc,absa} (fp16 shards +
  config + tokenizer, int8 EXCLUDED) to sandeep11mahendrakar/lilly-fashion-muse-doc-gated
  and .../lilly-fashion-muse-absa, private first, honest model cards inline.
  BLOCKED ON: Sandeep's HF token (HF_TOKEN env or `hf auth login`); then verify
  re-download via env override, flip public with Sandeep's OK.
- E2E VERIFIED with real models: regression trio (clear-pos 10.0+, clear-neg 1.6-,
  "beautiful pattern but runs extremely small" -> UNSURE @0.55); 6-row CSV batch: sorted
  desc, unsure sectioned last, coverage/mean correct; CSV download format verified.

## Remaining engineering issues
1. notebooks have hardcoded paths; `data/proceeded` typo dir still exists alongside `data/processed`.
2. `src/preprocessing.py` doesn't match transformer preprocessing (irrelevant for transformer, relevant if LR used again).
3. RESOLVED (2026-08-24): Streamlit app retired entirely; FastAPI webapp serves the gated DistilBERT + consensus ABSA (see WEBAPP REBUILD).
4. RESOLVED by decision (2026-08-23): LIME dropped; confidence + tau-gate are the honesty signals.
5. RESOLVED (2026-08-23): consensus ABSA trained + deployed (Path B).
6. RESOLVED (2026-08-24): README rewritten for the webapp; merge conflicts gone with the cleanup commit.
7. PROJECT_MEMORY.md was truncated by accident on 2026-08-24 and reconstructed from artifacts
   (see RECONSTRUCTION NOTE at top) — some prose detail from sections E7A–E2 is leaner than the original.

## Do NOT (standing rules)
- No generic hyperparameter sweeps; no class_weight by default; no binary task; no LSTM; no external datasets yet;
  don't rebuild venv; keep seed42 split consistent; judge by macro-F1 (not accuracy alone);
  don't swap app model before evaluation work settles.

## Run Locally
```
venv\Scripts\activate
uvicorn apps.webapp.backend.main:app --port 7860    # webapp (FastAPI + vanilla JS)
python -m pytest apps\webapp\tests                  # scoring + API tests
# HF upload (needs token): set HF_TOKEN, then python src/experiments/upload_deploy_to_hf.py
```
