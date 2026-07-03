"""Minimal FastAPI inference backend for Project Sentinel.

Wraps the existing src/ pipeline: loads the Stage-2 LightGBM 24h model + metadata
at startup, builds one SHAP TreeExplainer, and serves clinical alerts. PoC only —
no auth, no DB. Mirrors notebook 07 (raw predict_proba, conformal abstention).

Run from project_sentinel/:
    uv run uvicorn backend.app:app --port 8000
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.explain import generate_clinical_alert
from src.models import load_models

_ROOT = Path(__file__).resolve().parent.parent  # project_sentinel/

# --- Load everything once at startup -----------------------------------------
_meta = json.loads((_ROOT / "models" / "model_metadata.json").read_text())
STAGE2_FEATURES: list[str] = _meta["stage2_features"]
THRESHOLD: float = _meta["alert_threshold"]
QHAT: float = _meta["conformal_qhat_stage2_24h"]

_stage1, _stage2 = load_models(str(_ROOT / "models"))
MODEL = _stage2[24]["lgbm"]["model"]
EXPLAINER = shap.TreeExplainer(MODEL)

# Test set kept in memory for the ward demo (small parquet). Index by stay id so
# alerts carry a real patient_id; probabilities precomputed once for stratifying.
_TEST = pd.read_parquet(_ROOT / "data" / "processed" / "test.parquet")
_X = _TEST[STAGE2_FEATURES].copy()
_X.index = _TEST["patient_id"].astype(str)
_PROBS = MODEL.predict_proba(_X)[:, 1]

app = FastAPI(title="Project Sentinel — SA-AKI inference")
app.add_middleware(  # ponytail: wide-open CORS, fine for a localhost PoC demo
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


def _alert(idx: int) -> dict:
    """Build a clinical alert for positional row `idx` of the in-memory test set."""
    row = _X.iloc[idx]
    sv = EXPLAINER.shap_values(_X.iloc[[idx]])
    sv = sv[1] if isinstance(sv, list) else sv
    return generate_clinical_alert(
        model=MODEL, explainer=EXPLAINER, shap_values_single=sv[0],
        patient_row=row, feature_names=STAGE2_FEATURES,
        prediction_prob=float(_PROBS[idx]), threshold=THRESHOLD, conformal_qhat=QHAT,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/patients")
def patients(n: int = 12) -> list[dict]:
    """A demo ward: `n` patients spread evenly across the risk spectrum."""
    order = np.argsort(_PROBS)  # low → high risk
    picks = order[np.linspace(0, len(order) - 1, n).round().astype(int)]
    return [_alert(int(i)) for i in picks]


@app.get("/sample")
def sample() -> dict[str, float]:
    """One raw patient-hour (feature dict) to prefill the scoring form."""
    return _X.iloc[int(_PROBS.argmax())].to_dict()  # the highest-risk row, for an interesting default


@app.post("/predict")
def predict(patient: dict[str, Any]) -> dict:
    """One patient-hour (keys = feature names) → clinical alert JSON.

    Accepts any JSON object: we slice to STAGE2_FEATURES ourselves, so extra
    columns from a raw export (patient_id, hospital_system, …) are ignored, and
    None is fine (unobserved labs are NaN by design — the model treats NaN as
    missing). Typing the body as float-only would 422 on those legitimate cases.
    """
    missing = [f for f in STAGE2_FEATURES if f not in patient]
    if missing:
        raise HTTPException(422, f"Missing {len(missing)} features, e.g. {missing[:5]}")

    # slice to the model's columns; astype float64 so None→NaN and all-null
    # columns are float (not object), which LightGBM requires.
    try:
        row = pd.DataFrame([patient])[STAGE2_FEATURES].astype("float64")
    except (ValueError, TypeError) as e:
        raise HTTPException(422, f"Non-numeric feature value: {e}")

    # carry a real id onto the alert if the row brought one (generate_clinical_alert
    # uses the Series .name as patient_id); falls back to the index otherwise.
    pid = patient.get("patient_id") or patient.get("stay_id")
    if pid is not None:
        row.index = [str(pid)]

    prob = float(MODEL.predict_proba(row)[:, 1][0])

    sv = EXPLAINER.shap_values(row)
    sv = sv[1] if isinstance(sv, list) else sv  # LightGBM binary → list[class0, class1]

    return generate_clinical_alert(
        model=MODEL,
        explainer=EXPLAINER,
        shap_values_single=sv[0],
        patient_row=row.iloc[0],
        feature_names=STAGE2_FEATURES,
        prediction_prob=prob,
        threshold=THRESHOLD,
        conformal_qhat=QHAT,
    )


# Serve the built React frontend at "/" so the whole demo is one URL in production
# (no CORS needed). Mounted last, after the API routes above, so those win; the SPA
# only catches everything else. ponytail: no SPA router here, so plain static serving
# is enough — no history-fallback middleware needed.
_DIST = _ROOT / "frontend" / "dist"
if _DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="frontend")
