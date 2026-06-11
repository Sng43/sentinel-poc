"""
explain.py — SHAP Explainability & Clinical Alert Generation for Project Sentinel.

Provides interpretable model explanations via SHAP (SHapley Additive
exPlanations) and generates structured clinical alerts suitable for
integration into electronic health record (EHR) dashboards.

Pipeline stages
───────────────
1. **SHAP summary plots** — global feature-importance bar and dot charts.
2. **Patient waterfall plots** — per-patient force decomposition.
3. **Clinical alerts** — structured JSON risk cards with top contributing
   features, human-readable names, and recommended actions.

All heavy plotting is performed with the ``Agg`` backend so the module
can run headless on CI / cloud workers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # noqa: E402  — must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

_DEFAULT_THRESHOLD: float = 0.35

_FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "creatinine_velocity_12h": "Creatinine velocity (last 12h)",
    "creatinine_velocity_6h": "Creatinine velocity (last 6h)",
    "creatinine_velocity_24h": "Creatinine velocity (last 24h)",
    "MAP_mean_6h": "Mean arterial pressure (6h avg)",
    "MAP_mean_12h": "Mean arterial pressure (12h avg)",
    "MAP_min_6h": "Mean arterial pressure (6h min)",
    "Lactate": "Lactate",
    "Lactate_max_12h": "Lactate (12h max)",
    "cr_above_baseline": "Creatinine above baseline",
    "bun_cr_ratio": "BUN/Creatinine ratio",
    "wbc_lactate_product": "WBC × Lactate product",
    "Creatinine": "Serum creatinine",
    "BUN": "Blood urea nitrogen",
    "WBC": "White blood cell count",
    "Heart_Rate": "Heart rate",
    "SBP": "Systolic blood pressure",
    "DBP": "Diastolic blood pressure",
    "SpO2": "Oxygen saturation (SpO₂)",
    "Temperature": "Body temperature",
    "Urine_Output": "Urine output",
    "urine_output_6h": "Urine output (6h total)",
    "urine_output_12h": "Urine output (12h total)",
    "GCS": "Glasgow Coma Scale",
    "Platelets": "Platelet count",
    "INR": "INR",
    "FiO2": "FiO₂",
    "PaO2_FiO2_ratio": "PaO₂/FiO₂ ratio",
    "vasopressor_flag": "Vasopressor use",
    "mechanical_ventilation": "Mechanical ventilation",
    "SOFA_score": "SOFA score",
    "age": "Age",
    "gender": "Gender",
    "weight": "Weight (kg)",
    "hours_since_admission": "Hours since admission",
}

_RISK_ACTIONS: dict[str, str] = {
    "HIGH": (
        "URGENT: Consider immediate nephrology consultation, initiate "
        "fluid balance monitoring, and review nephrotoxic medications"
    ),
    "MEDIUM": (
        "Consider nephrology review and fluid balance optimization. "
        "Increase creatinine monitoring frequency to q4h"
    ),
    "LOW": "Continue standard monitoring. Reassess in 6 hours",
}

# Features where a *positive* SHAP contribution generally corresponds to a
# *rising* or *high* clinical value, and vice-versa.
_RISING_WHEN_POSITIVE: set[str] = {
    "creatinine_velocity_12h",
    "creatinine_velocity_6h",
    "creatinine_velocity_24h",
    "Lactate",
    "Lactate_max_12h",
    "cr_above_baseline",
    "bun_cr_ratio",
    "wbc_lactate_product",
    "Creatinine",
    "BUN",
    "WBC",
    "Heart_Rate",
    "SOFA_score",
    "vasopressor_flag",
    "hours_since_admission",
    "INR",
    "FiO2",
    "mechanical_ventilation",
    "age",
}

_FALLING_WHEN_POSITIVE: set[str] = {
    "MAP_mean_6h",
    "MAP_mean_12h",
    "MAP_min_6h",
    "SBP",
    "DBP",
    "SpO2",
    "Urine_Output",
    "urine_output_6h",
    "urine_output_12h",
    "GCS",
    "Platelets",
    "PaO2_FiO2_ratio",
}


# ─── Plotting Style ──────────────────────────────────────────────────────────

def _apply_plot_style() -> None:
    """Apply a clean, professional Matplotlib style for clinical reports."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "axes.grid.axis": "x",
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


# ─── 1. SHAP Summary Plots ───────────────────────────────────────────────────

def generate_shap_summary(
    model: Any,
    X_test: pd.DataFrame,
    feature_names: list[str],
    output_dir: str,
) -> np.ndarray:
    """Compute SHAP values and generate global feature-importance plots.

    Parameters
    ----------
    model:
        A tree-based model compatible with ``shap.TreeExplainer``
        (e.g. XGBoost, LightGBM, scikit-learn GBT / RF).
    X_test:
        Test feature matrix.
    feature_names:
        Ordered list of feature names matching ``X_test`` columns.
    output_dir:
        Directory where plots are saved.

    Returns
    -------
    np.ndarray
        Raw SHAP values array (shape matches ``X_test``).
    """
    _apply_plot_style()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Computing SHAP values with TreeExplainer …")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # For binary classifiers the output may be a list [class_0, class_1].
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    # --- Bar plot (mean |SHAP|) -------------------------------------------
    fig_bar, ax_bar = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        sv,
        X_test,
        feature_names=feature_names,
        plot_type="bar",
        max_display=20,
        show=False,
    )
    plt.title("Global Feature Importance (mean |SHAP|)", fontsize=14, pad=12)
    plt.tight_layout()
    bar_path = out / "shap_summary_bar.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("Saved SHAP bar plot → %s", bar_path)

    # --- Dot plot (beeswarm distribution) ---------------------------------
    fig_dot, ax_dot = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        sv,
        X_test,
        feature_names=feature_names,
        plot_type="dot",
        max_display=20,
        show=False,
    )
    plt.title("SHAP Feature Impact Distribution", fontsize=14, pad=12)
    plt.tight_layout()
    dot_path = out / "shap_summary_dot.png"
    plt.savefig(dot_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("Saved SHAP dot plot → %s", dot_path)

    return sv


# ─── 2. Representative Patient Selection ─────────────────────────────────────

def select_representative_patients(
    X_test: pd.DataFrame | np.ndarray,
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    threshold: float = _DEFAULT_THRESHOLD,
) -> dict[str, int]:
    """Select three clinically representative patient indices.

    Parameters
    ----------
    X_test:
        Test feature matrix (used only for its index / length).
    y_true:
        Ground-truth binary labels (1 = AKI).
    y_prob:
        Predicted AKI probabilities.
    threshold:
        Classification threshold for positive prediction (default 0.35).

    Returns
    -------
    dict
        ``{'tp': idx, 'fn': idx, 'tn': idx}`` mapping to integer
        positional indices into ``X_test``.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    result: dict[str, int] = {}

    # True positive — highest predicted probability among TPs
    tp_mask = (y_true == 1) & (y_pred == 1)
    if tp_mask.any():
        result["tp"] = int(np.where(tp_mask)[0][np.argmax(y_prob[tp_mask])])
    else:
        logger.warning("No true positives found; selecting highest-risk positive.")
        pos_mask = y_true == 1
        result["tp"] = int(np.where(pos_mask)[0][np.argmax(y_prob[pos_mask])]) if pos_mask.any() else 0

    # False negative — highest predicted probability among FNs
    fn_mask = (y_true == 1) & (y_pred == 0)
    if fn_mask.any():
        result["fn"] = int(np.where(fn_mask)[0][np.argmax(y_prob[fn_mask])])
    else:
        logger.warning("No false negatives found; selecting closest-to-threshold positive.")
        pos_mask = y_true == 1
        if pos_mask.any():
            diffs = np.abs(y_prob[pos_mask] - threshold)
            result["fn"] = int(np.where(pos_mask)[0][np.argmin(diffs)])
        else:
            result["fn"] = 0

    # True negative — lowest predicted probability among TNs
    tn_mask = (y_true == 0) & (y_pred == 0)
    if tn_mask.any():
        result["tn"] = int(np.where(tn_mask)[0][np.argmin(y_prob[tn_mask])])
    else:
        logger.warning("No true negatives found; selecting lowest-risk negative.")
        neg_mask = y_true == 0
        result["tn"] = int(np.where(neg_mask)[0][np.argmin(y_prob[neg_mask])]) if neg_mask.any() else 0

    logger.info(
        "Representative patients → TP idx=%d (p=%.3f), FN idx=%d (p=%.3f), TN idx=%d (p=%.3f)",
        result["tp"], y_prob[result["tp"]],
        result["fn"], y_prob[result["fn"]],
        result["tn"], y_prob[result["tn"]],
    )
    return result


# ─── 3. Patient Waterfall Plot ────────────────────────────────────────────────

def plot_patient_waterfall(
    shap_values: np.ndarray | list[np.ndarray],
    explainer: shap.TreeExplainer,
    X_test: pd.DataFrame,
    patient_idx: int,
    patient_label: str,
    feature_names: list[str],
    output_path: str,
) -> None:
    """Render a SHAP waterfall plot for a single patient and save to disk.

    Parameters
    ----------
    shap_values:
        SHAP values array or list (binary classification returns a list).
    explainer:
        The ``shap.TreeExplainer`` instance used to produce ``shap_values``.
    X_test:
        Test feature matrix.
    patient_idx:
        Positional index of the patient in ``X_test``.
    patient_label:
        Human-readable label for the plot title (e.g. "True Positive").
    feature_names:
        Ordered list of feature names.
    output_path:
        File path for the saved PNG.
    """
    _apply_plot_style()

    # Resolve SHAP values for positive class when output is a list.
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    patient_sv = sv[patient_idx]

    # Resolve base (expected) value.
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1] if len(base_value) > 1 else base_value[0]

    # Build a shap.Explanation for the waterfall API.
    explanation = shap.Explanation(
        values=patient_sv,
        base_values=float(base_value),
        data=X_test.iloc[patient_idx].values if isinstance(X_test, pd.DataFrame) else X_test[patient_idx],
        feature_names=feature_names,
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.waterfall(explanation, max_display=15, show=False)
    plt.title(f"SHAP Waterfall — {patient_label}", fontsize=14, pad=12)
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("Saved waterfall plot → %s", out)


# ─── 4. Orchestrate Patient Explanations ─────────────────────────────────────

def generate_patient_explanations(
    model: Any,
    X_test: pd.DataFrame,
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    feature_names: list[str],
    output_dir: str,
) -> None:
    """Generate SHAP waterfall plots for representative patients.

    Selects a true positive, false negative, and true negative patient,
    then produces individual waterfall plots for each.

    Parameters
    ----------
    model:
        Trained tree-based model.
    X_test:
        Test feature matrix.
    y_true:
        Ground-truth binary labels.
    y_prob:
        Predicted AKI probabilities.
    feature_names:
        Ordered list of feature names.
    output_dir:
        Directory where patient plots are saved.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    patients = select_representative_patients(X_test, y_true, y_prob)

    label_map: dict[str, str] = {
        "tp": "True Positive (High Risk)",
        "fn": "False Negative (Medium Risk)",
        "tn": "True Negative (Low Risk)",
    }

    for key, idx in patients.items():
        plot_patient_waterfall(
            shap_values=shap_values,
            explainer=explainer,
            X_test=X_test,
            patient_idx=idx,
            patient_label=label_map[key],
            feature_names=feature_names,
            output_path=str(out / f"shap_patient_{key}.png"),
        )


# ─── 5. Clinical Alert Generation ────────────────────────────────────────────

def _humanise_feature(feature: str) -> str:
    """Map an internal feature name to a human-readable clinical name."""
    if feature in _FEATURE_DISPLAY_NAMES:
        return _FEATURE_DISPLAY_NAMES[feature]
    # Fallback: replace underscores and title-case
    return feature.replace("_", " ").capitalize()


def _infer_direction(feature: str, shap_value: float) -> str:
    """Determine clinical direction label from SHAP sign and feature context.

    Returns one of ``'RISING'``, ``'FALLING'``, ``'HIGH'``, ``'LOW'``,
    or ``'NORMAL'``.
    """
    if abs(shap_value) < 0.01:
        return "NORMAL"

    if feature in _RISING_WHEN_POSITIVE:
        return "RISING" if shap_value > 0 else "FALLING"
    if feature in _FALLING_WHEN_POSITIVE:
        return "FALLING" if shap_value > 0 else "RISING"

    # Unknown feature — use generic HIGH / LOW
    return "HIGH" if shap_value > 0 else "LOW"


def generate_clinical_alert(
    model: Any,
    explainer: shap.TreeExplainer,
    shap_values_single: np.ndarray,
    patient_row: pd.Series,
    feature_names: list[str],
    prediction_prob: float,
    threshold: float = _DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """Build a structured clinical alert dict for one patient.

    Parameters
    ----------
    model:
        Trained model (retained for forward-compatibility).
    explainer:
        ``shap.TreeExplainer`` instance.
    shap_values_single:
        1-D SHAP values for this patient (length = n_features).
    patient_row:
        Feature values for this patient (``pd.Series``).
    feature_names:
        Ordered list of feature names.
    prediction_prob:
        Predicted probability of AKI within 24 h.
    threshold:
        Classification threshold (default 0.35).

    Returns
    -------
    dict
        Structured alert with keys ``patient_id``, ``risk_score``,
        ``risk_level``, ``aki_probability_24h``, ``top_contributors``,
        and ``recommended_action``.
    """
    # --- Risk classification -----------------------------------------------
    if prediction_prob >= 0.5:
        risk_level = "HIGH"
    elif prediction_prob >= threshold:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # --- Top 3 contributors by |SHAP| -------------------------------------
    sv = np.asarray(shap_values_single).ravel()
    top_indices = np.argsort(np.abs(sv))[::-1][:3]

    contributors: list[dict[str, Any]] = []
    for idx in top_indices:
        feat = feature_names[idx]
        value = patient_row.iloc[idx] if isinstance(patient_row, pd.Series) else patient_row[idx]
        contributors.append(
            {
                "feature": _humanise_feature(feat),
                "direction": _infer_direction(feat, float(sv[idx])),
                "value": str(round(float(value), 4)) if not isinstance(value, str) else value,
                "shap_impact": round(float(sv[idx]), 4),
            }
        )

    # --- Patient ID -------------------------------------------------------
    patient_id = str(patient_row.name) if hasattr(patient_row, "name") and patient_row.name is not None else "UNKNOWN"

    return {
        "patient_id": patient_id,
        "risk_score": round(float(prediction_prob), 4),
        "risk_level": risk_level,
        "aki_probability_24h": f"{prediction_prob:.0%}",
        "top_contributors": contributors,
        "recommended_action": _RISK_ACTIONS[risk_level],
    }


# ─── 6. Batch Alert Generation ───────────────────────────────────────────────

def generate_sample_alerts(
    model: Any,
    X_test: pd.DataFrame,
    y_prob: np.ndarray | pd.Series,
    feature_names: list[str],
    output_path: str,
    n_alerts: int = 10,
) -> list[dict[str, Any]]:
    """Generate clinical alerts for a sample of test patients.

    Patients are selected to provide a balanced mix of HIGH, MEDIUM,
    and LOW risk levels.

    Parameters
    ----------
    model:
        Trained tree-based model.
    X_test:
        Test feature matrix.
    y_prob:
        Predicted AKI probabilities for all test patients.
    feature_names:
        Ordered list of feature names.
    output_path:
        Path to write the JSON alerts file.
    n_alerts:
        Number of alerts to generate (default 10).

    Returns
    -------
    list[dict]
        List of structured alert dicts.
    """
    y_prob = np.asarray(y_prob).ravel()
    n_patients = len(y_prob)

    # Select a stratified sample across risk tiers.
    high_idx = np.where(y_prob >= 0.5)[0]
    med_idx = np.where((y_prob >= _DEFAULT_THRESHOLD) & (y_prob < 0.5))[0]
    low_idx = np.where(y_prob < _DEFAULT_THRESHOLD)[0]

    n_per_tier = max(1, n_alerts // 3)
    selected: list[int] = []

    for tier_indices in (high_idx, med_idx, low_idx):
        if len(tier_indices) > 0:
            rng = np.random.default_rng(seed=42)
            chosen = rng.choice(
                tier_indices,
                size=min(n_per_tier, len(tier_indices)),
                replace=False,
            )
            selected.extend(chosen.tolist())

    # Fill remaining slots if any tier was under-represented.
    remaining = n_alerts - len(selected)
    if remaining > 0:
        all_idx = np.arange(n_patients)
        available = np.setdiff1d(all_idx, selected)
        if len(available) > 0:
            rng = np.random.default_rng(seed=42)
            extra = rng.choice(available, size=min(remaining, len(available)), replace=False)
            selected.extend(extra.tolist())

    selected = selected[:n_alerts]

    # Compute SHAP for selected patients.
    logger.info("Computing SHAP values for %d sample patients …", len(selected))
    explainer = shap.TreeExplainer(model)
    X_sample = X_test.iloc[selected] if isinstance(X_test, pd.DataFrame) else X_test[selected]
    shap_values = explainer.shap_values(X_sample)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    alerts: list[dict[str, Any]] = []
    for i, global_idx in enumerate(selected):
        patient_row = (
            X_test.iloc[global_idx] if isinstance(X_test, pd.DataFrame) else pd.Series(X_test[global_idx], index=feature_names)
        )
        alert = generate_clinical_alert(
            model=model,
            explainer=explainer,
            shap_values_single=sv[i],
            patient_row=patient_row,
            feature_names=feature_names,
            prediction_prob=float(y_prob[global_idx]),
        )
        alerts.append(alert)

    # Persist to JSON.
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(alerts, fh, indent=2, ensure_ascii=False)
    logger.info("Saved %d clinical alerts → %s", len(alerts), out)

    return alerts


# ─── 7. Full Explainability Pipeline ─────────────────────────────────────────

def run_explainability(
    model: Any,
    X_test: pd.DataFrame,
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    feature_names: list[str],
    output_dir: str,
) -> dict[str, Any]:
    """Execute the complete explainability pipeline.

    Steps
    -----
    1. Generate SHAP global summary plots (bar + dot).
    2. Generate per-patient waterfall plots for representative cases.
    3. Generate sample clinical alerts as JSON.

    Parameters
    ----------
    model:
        Trained tree-based model.
    X_test:
        Test feature matrix.
    y_true:
        Ground-truth binary labels.
    y_prob:
        Predicted AKI probabilities.
    feature_names:
        Ordered list of feature names.
    output_dir:
        Root output directory for all artefacts.

    Returns
    -------
    dict
        Summary containing:
        - ``top_features`` — list of top-10 features by mean |SHAP|.
        - ``alert_count`` — number of sample alerts generated.
        - ``output_dir`` — resolved output path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1 — SHAP summary
    logger.info("Step 1/3: Generating SHAP summary plots …")
    shap_values = generate_shap_summary(model, X_test, feature_names, str(out))

    # 2 — Patient waterfalls
    logger.info("Step 2/3: Generating patient waterfall plots …")
    generate_patient_explanations(model, X_test, y_true, y_prob, feature_names, str(out))

    # 3 — Clinical alerts
    logger.info("Step 3/3: Generating sample clinical alerts …")
    alerts = generate_sample_alerts(
        model, X_test, y_prob, feature_names, str(out / "sample_alerts.json")
    )

    # --- Summary statistics ------------------------------------------------
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    mean_abs_shap = np.abs(sv).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:10]
    top_features = [feature_names[i] for i in top_indices]

    summary: dict[str, Any] = {
        "top_features": top_features,
        "alert_count": len(alerts),
        "output_dir": str(out.resolve()),
    }

    logger.info(
        "Explainability pipeline complete — top features: %s",
        ", ".join(top_features[:5]),
    )
    return summary
