---
title: Sentiment Analysis Pro
emoji: 🎯
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Sentiment Analysis Pro

Gated DistilBERT review scorer: 1–10 scores with − / ± / + bands, aspect
chips (quality / price / fit), and an honest UNSURE / MIXED abstention when
the model's confidence is below τ = 0.70.

- **Single Review** — paste a review, get a live score dial, confidence bar, aspect chips.
- **Batch CSV** — drop a CSV, get sorted scored results with an UNSURE section at the bottom, plus a summary panel and downloadable CSV.

Operating point (held-out): **91.2% accuracy at 86.2% coverage**; the abstained
bucket is dominated by genuinely ambiguous mixed reviews. Inter-annotator
ceiling κ ≈ 0.15 — the gate, not raw accuracy, is the product.

Models: `distilbert-sentiment-doc-gated` + `distilbert-absa-consensus` (frozen fp16 shards, loaded from the HF Hub at boot).
