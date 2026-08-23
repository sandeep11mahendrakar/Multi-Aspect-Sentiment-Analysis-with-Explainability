import pytest

from backend.scoring import TAU, band_from_score, classify, marker_from_score, score_from_probs


def test_score_bounds():
    assert score_from_probs([1.0, 0.0, 0.0]) == 1.0
    assert score_from_probs([0.0, 0.0, 1.0]) == 10.0


def test_score_neutral_midpoint():
    assert score_from_probs([0.0, 1.0, 0.0]) == pytest.approx(5.5)


def test_score_one_decimal():
    assert score_from_probs([0.25, 0.25, 0.50]) == round(1 + 9 * (0.50 + 0.125), 1)


def test_band_floor_rule():
    assert band_from_score(1.0) == "negative"
    assert band_from_score(3.9) == "negative"
    assert band_from_score(4.0) == "neutral"
    assert band_from_score(6.9) == "neutral"
    assert band_from_score(7.0) == "positive"
    assert band_from_score(10.0) == "positive"


def test_marker():
    assert marker_from_score(2.0) == "\u2212"
    assert marker_from_score(5.0) == "\u00b1"
    assert marker_from_score(8.0) == "+"


def test_committed_positive():
    r = classify([0.05, 0.20, 0.75])
    assert not r["unsure"]
    assert r["sentiment"] == "positive"
    assert r["band"] == "positive"
    assert r["score"] == round(1 + 9 * (0.75 + 0.10), 1)
    assert r["confidence"] == pytest.approx(0.75)


def test_tau_boundary_is_committed():
    r = classify([0.30, 0.70, 0.00])
    assert not r["unsure"]
    assert r["sentiment"] == "neutral"


def test_below_tau_is_unsure_no_score():
    r = classify([0.45, 0.30, 0.25])
    assert r["unsure"]
    assert r["score"] is None
    assert r["band"] is None
    assert r["sentiment"] is None
    assert r["confidence"] == pytest.approx(0.45)


def test_tau_value_matches_e8_operating_point():
    assert TAU == 0.70
