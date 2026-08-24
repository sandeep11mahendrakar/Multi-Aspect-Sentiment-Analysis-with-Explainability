---
title: Lilly · Fashion Muse
emoji: 🌸
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# Lilly · Fashion Muse

Lilly reads clothing reviews the way a stylist would. Gated DistilBERT review
scorer: 1–10 scores with − / ± / + bands, aspect chips (quality / price / fit),
and an honest **UNSURE / MIXED** abstention when her confidence is below
τ = 0.70. Trained on women's e-commerce reviews — reads any product review,
menswear included.

- **Single Review** — paste a review, get a live score dial, confidence bar, aspect chips.
- **Batch CSV** — drop a CSV, get sorted scored results with an UNSURE section at the bottom, plus a summary panel and downloadable CSV.

Operating point (held-out): **91.2% accuracy at 86.2% coverage**; the abstained
bucket is dominated by genuinely ambiguous mixed reviews. Inter-annotator
ceiling κ ≈ 0.15 — the gate, not raw accuracy, is the product.

Models: `lilly-fashion-muse-doc-gated` + `lilly-fashion-muse-absa` (frozen fp16 shards, loaded from the HF Hub at boot).
