import io
import json

import pytest
from fastapi.testclient import TestClient

import apps.webapp.backend.main as main
from apps.webapp.backend.scoring import classify


@pytest.fixture()
def client(monkeypatch):
    # Mock inference: deterministic probs per text.
    def fake_doc_probs(models, texts):
        table = {
            "pos": [0.05, 0.20, 0.75],
            "neg": [0.85, 0.10, 0.05],
            "unsure": [0.45, 0.30, 0.25],
        }
        import numpy as np
        return np.array([table.get(t, table["pos"]) for t in texts])

    def fake_absa(models, text):
        if "quality" in text:
            return {"quality": ("negative", 0.9)}
        return {}

    monkeypatch.setattr(main.inference, "doc_probs", fake_doc_probs)
    monkeypatch.setattr(main.inference, "absa_analyze", fake_absa)
    monkeypatch.setattr(main, "get_models", lambda: object())
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(main.api)
    return TestClient(app)


def test_predict_committed(client):
    r = client.post("/api/predict", json={"text": "pos"}).json()
    assert r["unsure"] is False
    assert r["score"] == round(1 + 9 * (0.75 + 0.5 * 0.20), 1)
    assert r["marker"] == "+"
    assert r["aspects"] == {}


def test_predict_unsure(client):
    r = client.post("/api/predict", json={"text": "unsure"}).json()
    assert r["unsure"] is True
    assert r["score"] is None


def _csv(rows):
    buf = io.StringIO()
    buf.write("id,review_text\n")
    for t in rows:
        buf.write(f"1,{t}\n")
    return io.BytesIO(buf.getvalue().encode("utf-8"))


def _events(resp):
    return [json.loads(l) for l in resp.text.splitlines() if l.strip()]


def test_batch_sorted_and_unsure_last(client):
    resp = client.post("/api/batch",
                       files={"file": ("t.csv", _csv(["pos", "neg", "unsure"]), "text/csv")})
    done = [e for e in _events(resp) if e["type"] == "done"][0]
    rows = done["rows"]
    n_unsure = sum(1 for r in rows if r["unsure"])
    committed_scores = [r["score"] for r in rows[: len(rows) - n_unsure]]
    assert committed_scores == sorted(committed_scores, reverse=True)
    assert all(r["unsure"] for r in rows[len(rows) - n_unsure:])


def test_batch_summary(client):
    resp = client.post("/api/batch",
                       files={"file": ("t.csv", _csv(["pos", "neg", "unsure"]), "text/csv")})
    done = [e for e in _events(resp) if e["type"] == "done"][0]
    s = done["summary"]
    assert s["counts"]["positive"] == 1
    assert s["counts"]["negative"] == 1
    assert s["counts"]["unsure"] == 1
    assert s["coverage"] == pytest.approx(2 / 3, abs=1e-3)


def test_batch_row_cap_rejected(client):
    big = io.BytesIO(b"review_text\n" + b"x\n" * 5001)
    resp = client.post("/api/batch", files={"file": ("big.csv", big, "text/csv")})
    events = _events(resp)
    assert any(e["type"] == "error" and "5,000" in e["message"] for e in events)


def test_batch_column_select_event(client):
    buf = io.BytesIO(b"a,b\nx,y\n")
    resp = client.post("/api/batch", files={"file": ("t.csv", buf, "text/csv")})
    events = _events(resp)
    assert events[0]["type"] == "column_select"
    assert set(events[0]["columns"]) == {"a", "b"}
