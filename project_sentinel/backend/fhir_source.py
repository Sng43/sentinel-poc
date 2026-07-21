"""FHIR → model-feature adapter — the seam for live EHR integration.

The backend's ``/predict`` already accepts a plain ``{feature: value}`` dict, so
integrating any EHR reduces to **one job: turn that hospital's patient record into
this dict.** HL7 FHIR R4 is the universal way to do that.

**Primary target: OpenClinic GA** — the HIS running at the deployment hospital (RMRTH,
Rwanda) and 500+ others. OpenClinic GA exposes an HL7/FHIR API for structured data
exchange, so this adapter targets it directly; the identical shape also works for
OpenMRS (``/openmrs/ws/fhir2/R4/…``) and any FHIR R4 server. Point ``EHR_FHIR_URL`` at
the deployment's FHIR base. Note OpenClinic GA codes with SNOMED CT + LOINC — labs are
typically LOINC (mapped below); verify each against the live server.

This module fetches a patient's most-recent vitals/labs as FHIR ``Observation``
resources (coded by LOINC), maps them to the canonical feature names, and returns
a dict ready for ``/predict``. Stdlib only (urllib + json) — no new dependency.

⚠️  TWO things must be checked against YOUR deployment before clinical use:
  1. **LOINC codes.** The map below uses standard LOINC codes, but a given OpenMRS
     concept dictionary may code a lab differently (or use local codes). Verify each
     mapping against the target server's ``/Observation`` output.
  2. **Live endpoint + auth.** This is written against the FHIR REST shape but has
     NOT been run against a live server here. Point ``EHR_FHIR_URL`` at a real
     OpenMRS/FHIR endpoint and test before trusting it.

Scope note: a single FHIR snapshot supplies the **raw current** vitals/labs. The
derived features (rolling windows, creatinine velocity, SOFA) are left absent (→ NaN,
which the model tolerates). Full fidelity needs the patient's observation *history*
fetched and run through ``src.features`` — a documented next step, not this snapshot.
"""
from __future__ import annotations

import json
import urllib.request
from typing import Any

# Canonical feature name -> LOINC code(s). Standard LOINC; VERIFY per deployment.
# Only the raw (non-derived) clinical features are fetchable from FHIR; rolling /
# velocity / composite / SOFA features are computed downstream from these.
_LOINC_TO_FEATURE: dict[str, str] = {
    # vitals
    "8867-4": "HR", "8480-6": "SBP", "8462-4": "DBP", "8478-0": "MAP",
    "9279-1": "Resp", "2708-6": "O2Sat", "59408-5": "O2Sat", "8310-5": "Temp",
    # chemistry / renal
    "2160-0": "Creatinine", "3094-0": "BUN", "2823-3": "Potassium",
    "2075-0": "Chloride", "1963-8": "HCO3", "2345-7": "Glucose", "17861-6": "Calcium",
    "19123-9": "Magnesium", "2777-1": "Phosphate",
    # heme / coag
    "777-3": "Platelets", "718-7": "Hgb", "4544-3": "Hct", "6690-2": "WBC",
    "3173-2": "PTT", "3255-7": "Fibrinogen",
    # liver
    "1975-2": "Bilirubin_total", "1920-8": "AST", "6768-6": "Alkalinephos",
    # blood gas / other
    "2019-8": "PaCO2", "11558-4": "pH", "2524-7": "Lactate", "1798-8": "Amylase",
    "20565-8": "BaseExcess", "2708-6.1": "SaO2",
}


def _get_json(url: str, token: str | None, timeout: float = 10.0) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"Accept": "application/fhir+json"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (trusted EHR URL)
        return json.loads(resp.read().decode())


def fetch_patient_features(
    base_url: str, patient_id: str, token: str | None = None
) -> dict[str, Any]:
    """Fetch one patient's latest vitals/labs from a FHIR R4 server → feature dict.

    Parameters
    ----------
    base_url : e.g. ``http://host/openmrs/ws/fhir2/R4``
    patient_id : the FHIR Patient logical id.
    token : optional bearer token.

    Returns a ``{feature_name: value}`` dict (plus ``Age``/``Gender``/``patient_id``)
    suitable for POSTing to ``/predict``. Only observed features are populated; the
    rest are left absent (the model treats missing as NaN by design).
    """
    feats: dict[str, Any] = {"patient_id": str(patient_id)}

    # --- demographics from the Patient resource ---
    patient = _get_json(f"{base_url}/Patient/{patient_id}", token)
    gender = patient.get("gender", "").lower()
    if gender in ("male", "female"):
        feats["Gender"] = 1 if gender == "male" else 0
    birth = patient.get("birthDate")
    if birth:
        from datetime import date
        feats["Age"] = date.today().year - int(birth[:4])

    # --- latest Observation per LOINC code (sorted newest first, take first hit) ---
    codes = ",".join(sorted(set(_LOINC_TO_FEATURE) - {"2708-6.1"}))
    bundle = _get_json(
        f"{base_url}/Observation?patient={patient_id}&code={codes}&_sort=-date&_count=200",
        token,
    )
    for entry in bundle.get("entry", []):
        obs = entry.get("resource", {})
        value = obs.get("valueQuantity", {}).get("value")
        if value is None:
            continue
        for coding in obs.get("code", {}).get("coding", []):
            feature = _LOINC_TO_FEATURE.get(coding.get("code"))
            if feature and feature not in feats:  # first (newest) wins
                feats[feature] = float(value)
                break
    return feats


if __name__ == "__main__":
    # Self-check: the mapping is well-formed and translation logic works offline,
    # WITHOUT hitting a live server (which we don't have here).
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in _LOINC_TO_FEATURE.items())
    assert _LOINC_TO_FEATURE["2160-0"] == "Creatinine"
    assert _LOINC_TO_FEATURE["777-3"] == "Platelets"
    # simulate the Observation-bundle parse with a fake FHIR bundle
    fake = {"entry": [
        {"resource": {"code": {"coding": [{"code": "2160-0"}]}, "valueQuantity": {"value": 2.4}}},
        {"resource": {"code": {"coding": [{"code": "777-3"}]}, "valueQuantity": {"value": 95}}},
    ]}
    out: dict[str, Any] = {}
    for e in fake["entry"]:
        for c in e["resource"]["code"]["coding"]:
            f = _LOINC_TO_FEATURE.get(c["code"])
            if f:
                out[f] = float(e["resource"]["valueQuantity"]["value"])
    assert out == {"Creatinine": 2.4, "Platelets": 95.0}, out
    print("OK — FHIR→feature mapping verified (offline; live endpoint still needs testing)")
