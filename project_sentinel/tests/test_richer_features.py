"""Unit tests for the richer AKI features: partial SOFA + nephrotoxin exposure.

Hand-built inputs with known-correct outputs, guarding the two bits of non-trivial
scoring logic that are easy to regress:
  1. SOFA sub-score thresholds (and the "unmeasured = normal/0" convention)
  2. nephrotoxic-drug detection + hadm→stay mapping onto the ICU hour axis
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import add_sofa_features
from src.data_loader import _hourly_nephrotoxin


def test_sofa_partial_scoring():
    df = pd.DataFrame({
        "Platelets":       [200.0, 10.0, 120.0, np.nan],
        "Bilirubin_total": [0.5, 13.0, 2.5, np.nan],
        "MAP":             [80.0, 60.0, 72.0, np.nan],
        "Creatinine":      [1.0, 5.5, 2.5, np.nan],
    })
    out = add_sofa_features(df)

    # row 0 — all normal → 0
    assert out.loc[0, "sofa_partial"] == 0
    # row 1 — worst case: coag 4 + liver 4 + cardio 1 (MAP<70, vaso capped) + renal 4 = 13
    assert (out.loc[1, "sofa_coag"], out.loc[1, "sofa_liver"],
            out.loc[1, "sofa_cardio"], out.loc[1, "sofa_renal"]) == (4, 4, 1, 4)
    assert out.loc[1, "sofa_partial"] == 13
    # row 2 — mid: plt120→1, bili2.5→2, MAP72→0, cr2.5→2  = 5
    assert out.loc[2, "sofa_partial"] == 5
    # row 3 — unmeasured scores 0 (SOFA convention: assume normal)
    assert out.loc[3, "sofa_partial"] == 0


def test_nephrotoxin_detection_and_hour_mapping():
    presc = pd.DataFrame({
        "subject_id": [1, 1, 2],
        "hadm_id":    [10, 10, 20],
        "starttime":  pd.to_datetime(
            ["2020-01-01 05:00", "2020-01-01 02:00", "2020-01-01 00:00"]),
        "drug":       ["Ibuprofen 200mg", "Acetaminophen 500mg", "Gentamicin IV"],
    })
    stay_keys = pd.DataFrame({
        "subject_id": [1, 2], "hadm_id": [10, 20], "stay_id": [100, 200],
        "intime": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 00:00"]),
    })
    out = _hourly_nephrotoxin(presc, stay_keys)
    got = {(int(s), int(h)) for s, h in out[["stay_id", "ICULOS"]].values}

    assert (100, 5) in got          # ibuprofen → stay 100, hour 5
    assert (200, 0) in got          # gentamicin → stay 200, hour 0
    assert len(out) == 2            # acetaminophen is NOT nephrotoxic → excluded


if __name__ == "__main__":
    test_sofa_partial_scoring()
    test_nephrotoxin_detection_and_hour_mapping()
    print("OK — SOFA + nephrotoxin feature logic verified")
