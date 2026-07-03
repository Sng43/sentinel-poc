"""Unit tests for the clinical core logic — the rules that must never silently break.

Strategy: exercise the KDIGO labelling functions and the alert/conformal decision
logic in isolation, with hand-built inputs and known-correct outcomes. These guard
the two invariants that are easy to regress and dangerous if wrong:
  1. KDIGO creatinine + urine criteria (incl. the "0 mL/kg/h is anuria, not missing" rule)
  2. the risk-level thresholds and conformal abstention band
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.labels import check_kdigo_aki, check_kdigo_uop, compute_aki_state
from src.explain import generate_clinical_alert
from src.conformal import INDETERMINATE


# ── KDIGO creatinine channel ──────────────────────────────────────────────────
def test_kdigo_creatinine_relative_and_absolute():
    assert check_kdigo_aki(1.5, 1.0) is True      # 1.5x baseline  -> relative criterion
    assert check_kdigo_aki(1.31, 1.0) is True     # +0.31 mg/dL    -> absolute criterion
    assert check_kdigo_aki(1.2, 1.0) is False     # +0.2, 1.2x     -> neither
    assert check_kdigo_aki(float("nan"), 1.0) is False   # missing current -> not AKI
    assert check_kdigo_aki(1.5, float("nan")) is False   # missing baseline -> not AKI


# ── KDIGO urine channel — the "0 is anuria, not missing" invariant ───────────
def test_urine_zero_is_anuria_not_missing():
    # 3 OBSERVED hours at 0.0 mL/kg/h == anuria -> oliguria asserted (NOT skipped)
    assert check_kdigo_uop(np.array([0.0, 0.0, 0.0]), np.array([1, 1, 1])) is True
    # 3 observed hours all below 0.5 -> oliguria
    assert check_kdigo_uop(np.array([0.2, 0.3, 0.4]), np.array([1, 1, 1])) is True


def test_urine_missing_cannot_fabricate_oliguria():
    # Same zero rates but UNobserved (mask 0) -> ignored, never counted as anuria.
    assert check_kdigo_uop(np.array([0.0, 0.0, 0.0]), np.array([0, 0, 0])) is False
    # Too few observed hours (2 < the 3-hour minimum) -> cannot assert oliguria.
    assert check_kdigo_uop(np.array([0.0, 0.0]), np.array([1, 1])) is False
    # One observed hour above threshold breaks "all < 0.5".
    assert check_kdigo_uop(np.array([0.2, 0.3, 0.9]), np.array([1, 1, 1])) is False
    # Only the observed hours count: 3 anuric among unobserved noise -> oliguria.
    assert check_kdigo_uop(np.array([5.0, 0.0, 0.0, 0.0, 5.0]),
                           np.array([0, 1, 1, 1, 0])) is True


# ── Per-hour AKI state (creatinine OR urine), single synthetic patient ───────
def test_compute_aki_state_flags_creatinine_rise():
    df = pd.DataFrame({"ICULOS": [1, 2, 3, 4],
                       "Creatinine": [1.0, 1.0, 1.6, 1.7]})  # rises to 1.6x at hour 3
    state = compute_aki_state(df, baseline_cr=1.0)
    assert list(state) == [False, False, True, True]


# ── Risk-level thresholds + conformal abstention ─────────────────────────────
def _level(prob, qhat=None):
    feats = ["f0", "f1", "f2"]
    return generate_clinical_alert(
        model=None, explainer=None,
        shap_values_single=np.array([0.3, -0.1, 0.05]),
        patient_row=pd.Series([2.0, 1.0, 0.5], index=feats, name="TEST42"),
        feature_names=feats, prediction_prob=prob, threshold=0.35, conformal_qhat=qhat,
    )


def test_risk_levels_without_conformal():
    assert _level(0.90)["risk_level"] == "HIGH"     # >= 0.5
    assert _level(0.40)["risk_level"] == "MEDIUM"   # >= threshold 0.35
    assert _level(0.10)["risk_level"] == "LOW"      # below threshold


def test_conformal_abstention_band():
    # qhat 0.638 -> abstain band [0.362, 0.638]; 0.50 lands inside -> INDETERMINATE
    assert _level(0.50, 0.638)["risk_level"] == INDETERMINATE
    assert _level(0.95, 0.638)["risk_level"] == "HIGH"   # above the band -> decide
    assert _level(0.05, 0.638)["risk_level"] == "LOW"    # below the band -> decide


def test_alert_shape_and_id():
    alert = _level(0.90)
    assert set(alert) >= {"patient_id", "risk_score", "risk_level",
                          "aki_probability_24h", "top_contributors", "recommended_action"}
    assert alert["patient_id"] == "TEST42"
    assert len(alert["top_contributors"]) == 3
    assert alert["aki_probability_24h"] == "90%"
