"""Pure scoring logic: doc-model probs -> 1-10 score, band, unsure gate.

Frozen constants from E8 Phase 1 (do not tune): tau = 0.70 operating point
(coverage .862 | committed acc .9118 | macroF1 .7148).
"""

from typing import Dict, Optional, Sequence

TAU = 0.70
LABELS = ["negative", "neutral", "positive"]


def score_from_probs(probs: Sequence[float]) -> float:
    """Continuous review score in [1.0, 10.0].

    probs order: [p_negative, p_neutral, p_positive].
    """
    p_neg, p_neu, p_pos = float(probs[0]), float(probs[1]), float(probs[2])
    raw = 1.0 + 9.0 * (p_pos + 0.5 * p_neu)
    return round(min(max(raw, 1.0), 10.0), 1)


def band_from_score(score: float) -> str:
    """Floor rule: 1-3 negative / 4-6 neutral / 7-10 positive."""
    b = int(score) if score < 10.0 else 10
    if b <= 3:
        return "negative"
    if b <= 6:
        return "neutral"
    return "positive"


def marker_from_score(score: float) -> str:
    return {"negative": "\u2212", "neutral": "\u00b1", "positive": "+"}[band_from_score(score)]


def classify(probs: Sequence[float]) -> Dict:
    """Gate first (tau), then score/band. Unsure rows carry no score."""
    confidence = max(float(p) for p in probs)
    result = {
        "unsure": confidence < TAU,
        "score": None,
        "band": None,
        "sentiment": None,
        "confidence": confidence,
        "probs": [float(p) for p in probs],
    }
    if not result["unsure"]:
        score = score_from_probs(probs)
        band = band_from_score(score)
        result.update(
            score=score,
            band=band,
            marker=marker_from_score(score),
            sentiment=LABELS[int(max(range(3), key=lambda i: probs[i]))],
        )
    else:
        result["marker"] = None
    return result


def marker_from_result(result: Dict) -> Optional[str]:
    return result.get("marker")
