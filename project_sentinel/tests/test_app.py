"""Integration tests for the FastAPI inference API — the product end to end.

Strategy: drive every endpoint through Starlette's TestClient (real request/response
cycle, real model + SHAP), then push the /predict endpoint with a range of inputs and
edge cases: valid, all-missing, malformed, extreme, and a carried patient id.

Bodies are built from /sample so the tests stay correct even if the 198-feature set
changes — /sample always returns exactly the columns /predict expects.
"""
from __future__ import annotations

import copy

import pytest
from fastapi.testclient import TestClient

from backend.app import app, STAGE2_FEATURES

client = TestClient(app)

VALID_LEVELS = {"HIGH", "MEDIUM", "LOW", "INDETERMINATE"}


def _assert_valid_alert(alert: dict):
    assert VALID_LEVELS.issuperset({alert["risk_level"]})
    assert 0.0 <= alert["risk_score"] <= 1.0
    assert isinstance(alert["top_contributors"], list) and len(alert["top_contributors"]) == 3
    assert alert["recommended_action"]


@pytest.fixture(scope="module")
def sample_body() -> dict:
    r = client.get("/sample")
    assert r.status_code == 200
    return r.json()


# ── Basic endpoints ──────────────────────────────────────────────────────────
def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_patients_spread_across_risk():
    r = client.get("/patients?n=8")
    assert r.status_code == 200
    alerts = r.json()
    assert len(alerts) == 8
    for a in alerts:
        _assert_valid_alert(a)
    # patients endpoint is meant to span the spectrum: scores should be non-constant
    scores = [a["risk_score"] for a in alerts]
    assert max(scores) > min(scores)


def test_sample_returns_full_feature_row(sample_body):
    assert set(sample_body) == set(STAGE2_FEATURES)  # exactly the model's inputs


# ── /predict happy path ──────────────────────────────────────────────────────
def test_predict_happy_path(sample_body):
    r = client.post("/predict", json=sample_body)
    assert r.status_code == 200
    _assert_valid_alert(r.json())


def test_predict_is_deterministic(sample_body):
    a = client.post("/predict", json=sample_body).json()
    b = client.post("/predict", json=sample_body).json()
    assert a["risk_score"] == b["risk_score"]


def test_predict_carries_patient_id(sample_body):
    body = copy.deepcopy(sample_body)
    body["patient_id"] = "WARD-42"          # extra key beyond the 198 features
    r = client.post("/predict", json=body)
    assert r.status_code == 200
    assert r.json()["patient_id"] == "WARD-42"


# ── /predict edge cases & bad input ──────────────────────────────────────────
def test_predict_all_missing_labs_still_scores(sample_body):
    # A brand-new admission with nothing charted yet: every feature None (-> NaN).
    # LightGBM treats NaN as missing, so this must score, not crash.
    body = {k: None for k in sample_body}
    r = client.post("/predict", json=body)
    assert r.status_code == 200
    _assert_valid_alert(r.json())


def test_predict_accepts_partial_input(sample_body):
    # A manual form or EHR feed rarely has all 205 features. Partial input is now
    # accepted — missing features are treated as NaN — and still returns an alert.
    body = copy.deepcopy(sample_body)
    body.pop(STAGE2_FEATURES[0])
    r = client.post("/predict", json=body)
    assert r.status_code == 200
    assert "risk_level" in r.json()

    # A tiny handful of fields must also work (the form's realistic minimum).
    r2 = client.post("/predict", json={"Creatinine": 3.2, "HR": 110, "Age": 72, "urine_rate": 0.3})
    assert r2.status_code == 200
    assert r2.json()["risk_level"] in {"HIGH", "MEDIUM", "LOW", "INDETERMINATE"}


def test_predict_non_numeric_rejected(sample_body):
    body = copy.deepcopy(sample_body)
    body[STAGE2_FEATURES[0]] = "not-a-number"
    r = client.post("/predict", json=body)
    assert r.status_code == 422


def test_predict_extreme_values_scores(sample_body):
    # Clinically extreme but numerically valid: shove every feature to a large value.
    body = {k: 9_999.0 for k in sample_body}
    r = client.post("/predict", json=body)
    assert r.status_code == 200
    _assert_valid_alert(r.json())
