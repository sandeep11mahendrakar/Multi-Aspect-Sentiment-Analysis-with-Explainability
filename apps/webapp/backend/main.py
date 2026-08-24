"""FastAPI app for the Lilly - Fashion Muse webapp.

Serves the vanilla-JS frontend from frontend/ and a small JSON/NDJSON API
around the frozen E8 deploy models. Run from repo root:
    uvicorn apps.webapp.backend.main:app --host 0.0.0.0 --port 7860
"""

import io
import json
import os
from typing import Iterator, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from apps.webapp.backend import inference
from apps.webapp.backend.scoring import TAU, classify

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

ROW_CAP = 5000
CHUNK = 32
COLUMN_CANDIDATES = ["review_text", "review text", "reviewtext",
                     "review", "text", "reviews", "body", "comment"]

UNSURE_HEADER = "## UNSURE / MIXED (model not confident enough to rate)"

api = APIRouter()
_models = None


def get_models():
    global _models
    if _models is None:
        _models = inference.load_models()
    return _models


class PredictIn(BaseModel):
    text: str


def _row_payload(text: str, probs_row) -> dict:
    r = classify(probs_row)
    return {
        "text": text,
        "unsure": bool(r["unsure"]),
        "score": r["score"],
        "band": r["band"],
        "marker": r.get("marker"),
        "sentiment": r["sentiment"],
        "confidence": round(float(r["confidence"]), 4),
    }


@api.get("/api/health")
def health():
    try:
        get_models()
        return {"status": "ok", "tau": TAU}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "detail": str(e)}


@api.post("/api/predict")
def predict(body: PredictIn):
    text = body.text.strip()
    if not text:
        raise HTTPException(422, "Review text is empty.")
    if len(text) > 10000:
        raise HTTPException(422, "Review text too long (max 10,000 characters).")
    m = get_models()
    probs = inference.doc_probs(m, [text])[0]
    payload = _row_payload(text, probs)
    aspects = inference.absa_analyze(m, text)
    payload["aspects"] = {
        a: {"sentiment": s, "confidence": round(c, 4)}
        for a, (s, c) in aspects.items()
    }
    payload["probs"] = [round(float(p), 4) for p in probs]
    return payload


def _ndjson(obj) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"


def _detect_column(df: pd.DataFrame) -> Optional[str]:
    lowered = {c.lower().strip(): c for c in df.columns}
    for cand in COLUMN_CANDIDATES:
        if cand in lowered:
            return lowered[cand]
    return None


@api.post("/api/batch")
def batch(
    file: UploadFile = File(...),
    column: Optional[str] = Form(None),
) -> StreamingResponse:
    def stream() -> Iterator[str]:
        try:
            raw = file.file.read()
            try:
                df = pd.read_csv(io.BytesIO(raw))
            except Exception:
                yield _ndjson({"type": "error",
                               "message": "Could not parse the file as CSV."})
                return
            if df.empty:
                yield _ndjson({"type": "error", "message": "The CSV has no data rows."})
                return
            col = column if column else _detect_column(df)
            if col is None or col not in df.columns:
                yield _ndjson({"type": "column_select",
                               "columns": list(map(str, df.columns))})
                return
            texts = df[col].astype(str).tolist()
            texts = [t for t in texts if t.strip() and t.lower() != "nan"]
            total = len(texts)
            if total == 0:
                yield _ndjson({"type": "error",
                               "message": "The chosen column has no non-empty reviews."})
                return
            if total > ROW_CAP:
                yield _ndjson({
                    "type": "error",
                    "message": f"CSV has {total:,} reviews — limit is {ROW_CAP:,} per batch.",
                })
                return

            m = get_models()
            yield _ndjson({"type": "meta", "total": total, "column": str(col)})
            rows = []
            done = 0
            for i in range(0, total, CHUNK):
                chunk = texts[i:i + CHUNK]
                probs = inference.doc_probs(m, chunk)
                for t, p in zip(chunk, probs):
                    payload = _row_payload(t, p)
                    aspects = inference.absa_analyze(m, t)
                    payload["aspects"] = {
                        a: v[0] for a, v in aspects.items()}
                    rows.append(payload)
                done += len(chunk)
                yield _ndjson({"type": "progress", "done": min(done, total),
                               "total": total})

            committed = sorted(
                (r for r in rows if not r["unsure"]),
                key=lambda r: (-r["score"],))
            unsure = [r for r in rows if r["unsure"]]
            ordered = committed + unsure
            counts = {"positive": 0, "neutral": 0, "negative": 0, "unsure": len(unsure)}
            for r in committed:
                counts[r["band"]] += 1
            mean_score = (
                round(sum(r["score"] for r in committed) / len(committed), 2)
                if committed else None)
            summary = {
                "counts": counts,
                "total": total,
                "committed": len(committed),
                "coverage": round(len(committed) / total, 4),
                "mean_score": mean_score,
            }
            yield _ndjson({"type": "done", "rows": ordered, "summary": summary})
        except Exception as e:  # noqa: BLE001
            yield _ndjson({"type": "error", "message": f"Server error: {e}"})

    return StreamingResponse(stream(), media_type="application/x-ndjson")


app = FastAPI(title="Lilly - Fashion Muse", docs_url=None, redoc_url=None)
app.include_router(api)


@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/{static_file}")
def static(static_file: str):
    path = os.path.join(FRONTEND_DIR, static_file)
    if not os.path.isfile(path):
        raise HTTPException(404)
    return FileResponse(path)

