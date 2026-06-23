"""labels.py — KDIGO AKI Criteria & Prospective Horizon Labeling
=============================================================

Project Sentinel — Sepsis-Associated AKI Early Warning System

Converts the canonical hourly time-series (see ``data_loader`` for the schema)
into binary prediction targets at configurable forward-looking horizons
(default 6 h, 12 h, 24 h).

Labeling follows KDIGO Stage-1 AKI criteria, using **both** available channels:

  * **Serum creatinine**
      - Absolute increase ≥ 0.3 mg/dL from baseline, OR
      - Relative increase ≥ 1.5× baseline
  * **Urine output** (the SA-AKI addition, available now that we use MIMIC-IV)
      - < 0.5 mL/kg/h sustained for ≥ 6 h

A time step is considered "currently AKI" if **either** channel is met.

Mask-aware urine handling
-------------------------
Oliguria is only asserted from **observed** urine. A missing urine hour is *not*
treated as 0 mL/kg/h (anuria); the 6 h window must contain at least
``_OLIGURIA_MIN_OBS`` observed hours before an oliguria call is made. This is the
counterpart of the "never zero-fill urine output" rule enforced in the loader.

Key design decisions
---------------------
  * **Prevalent-case exclusion** — time steps already meeting AKI criteria are
    excluded so the model learns *onset*, not *persistence*.
  * **Insufficient-horizon exclusion** — the last hour of a stay is dropped
    because no full horizon window can be evaluated.
  * **Unobservable-outcome exclusion** — if the forward window has neither a
    creatinine measurement nor any urine observation, the label is
    indeterminate and the row is excluded from training.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

__all__ = [
    "check_kdigo_aki",
    "check_kdigo_uop",
    "compute_aki_state",
    "create_aki_horizon_labels",
    "apply_label_exclusions",
    "label_dataset",
    "print_label_distribution",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_HORIZONS: list[int] = [6, 12, 24]

_AKI_ABSOLUTE_THRESHOLD: float = 0.3  # mg/dL
_AKI_RELATIVE_FACTOR: float = 1.5

# Urine-output (oliguria) criterion.
_OLIGURIA_THRESHOLD: float = 0.5   # mL/kg/h
_OLIGURIA_WINDOW: int = 6          # hours of sustained low output
_OLIGURIA_MIN_OBS: int = 3         # min observed hours in window to assert oliguria

# Canonical column names (match data_loader's output schema).
_CREATININE_COL: str = "Creatinine"
_PATIENT_ID_COL: str = "patient_id"
_TIME_COL: str = "ICULOS"
_BASELINE_CR_COL: str = "baseline_creatinine"
_UOP_RATE_COL: str = "urine_rate"
_UOP_OBSERVED_COL: str = "urine_observed"


# =========================================================================
# 1.  Point criteria
# =========================================================================
def check_kdigo_aki(
    creatinine_current: float,
    creatinine_baseline: float,
) -> bool:
    """Evaluate KDIGO Stage-1 **creatinine** criteria for a single measurement.

    Returns ``True`` if creatinine has risen ≥ 0.3 mg/dL above baseline or to
    ≥ 1.5× baseline. Returns ``False`` when either input is NaN.

    Examples
    --------
    >>> check_kdigo_aki(1.5, 1.0)
    True
    >>> check_kdigo_aki(1.2, 1.0)
    False
    >>> check_kdigo_aki(float('nan'), 1.0)
    False
    """
    try:
        if (
            creatinine_current is None
            or creatinine_baseline is None
            or math.isnan(creatinine_current)
            or math.isnan(creatinine_baseline)
        ):
            return False
    except TypeError:
        return False

    absolute_increase = creatinine_current - creatinine_baseline
    relative_increase = creatinine_current >= (_AKI_RELATIVE_FACTOR * creatinine_baseline)
    return absolute_increase >= _AKI_ABSOLUTE_THRESHOLD or relative_increase


def check_kdigo_uop(
    window_rates: np.ndarray,
    window_observed: np.ndarray,
) -> bool:
    """Evaluate the KDIGO **urine-output** (oliguria) criterion for one window.

    Parameters
    ----------
    window_rates : np.ndarray
        Urine output rates (mL/kg/h) over the trailing ``_OLIGURIA_WINDOW`` hours.
    window_observed : np.ndarray
        Matching 0/1 mask — 1 where urine was actually charted that hour.

    Returns
    -------
    bool
        ``True`` if oliguria can be asserted: at least ``_OLIGURIA_MIN_OBS``
        observed hours in the window, **all** of which are below
        ``_OLIGURIA_THRESHOLD``. Missing hours are ignored (never counted as
        anuria), so sparse charting cannot fabricate an oliguria call.
    """
    observed_mask = np.asarray(window_observed).astype(bool)
    rates = np.asarray(window_rates, dtype=float)[observed_mask]
    if rates.size < _OLIGURIA_MIN_OBS:
        return False
    return bool(np.all(rates < _OLIGURIA_THRESHOLD))


# =========================================================================
# 2.  Per-patient current-AKI state (creatinine OR urine)
# =========================================================================
def compute_aki_state(
    patient_df: pd.DataFrame,
    baseline_cr: float,
) -> pd.Series:
    """Boolean Series: does each hour meet KDIGO Stage-1 (creatinine OR urine)?

    Assumes *patient_df* is a single patient sorted by ``_TIME_COL``.
    """
    n = len(patient_df)
    if n == 0:
        return pd.Series([], dtype=bool)

    # --- Creatinine channel ---
    if baseline_cr is None or (isinstance(baseline_cr, float) and math.isnan(baseline_cr)):
        cr_aki = pd.Series(False, index=patient_df.index)
    else:
        cr = patient_df[_CREATININE_COL].astype(float)
        cr_aki = (
            ((cr - baseline_cr) >= _AKI_ABSOLUTE_THRESHOLD)
            | (cr >= _AKI_RELATIVE_FACTOR * baseline_cr)
        ).fillna(False)

    # --- Urine channel (rolling, mask-aware) ---
    if _UOP_RATE_COL in patient_df.columns and _UOP_OBSERVED_COL in patient_df.columns:
        rates = patient_df[_UOP_RATE_COL].to_numpy(dtype=float)
        observed = patient_df[_UOP_OBSERVED_COL].fillna(0).to_numpy(dtype=int)
        uop_aki = np.zeros(n, dtype=bool)
        for i in range(n):
            lo = max(0, i - _OLIGURIA_WINDOW + 1)
            uop_aki[i] = check_kdigo_uop(rates[lo : i + 1], observed[lo : i + 1])
        uop_aki = pd.Series(uop_aki, index=patient_df.index)
    else:
        uop_aki = pd.Series(False, index=patient_df.index)

    return (cr_aki | uop_aki).astype(bool)


# =========================================================================
# 3.  create_aki_horizon_labels
# =========================================================================
def create_aki_horizon_labels(
    patient_df: pd.DataFrame,
    baseline_cr: float,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Create prospective AKI horizon labels for a single patient.

    For every time step *t* and horizon *h*, label ``1`` is assigned if:

    1. The patient does **not** currently meet AKI criteria at *t*
       (prevalent-case exclusion), and
    2. The patient **does** meet KDIGO criteria (creatinine OR urine) at any
       hour in the open-forward window ``(t, t + h]``.

    Adds ``aki_label_<h>h`` columns plus a helper ``_aki_current`` column.
    """
    if horizons is None:
        horizons = list(_DEFAULT_HORIZONS)

    df = patient_df.copy()
    if df.empty:
        for h in horizons:
            df[f"aki_label_{h}h"] = pd.Series(dtype="Int64")
        df["_aki_current"] = pd.Series(dtype="bool")
        return df

    df = df.sort_values(_TIME_COL).reset_index(drop=True)
    df["_aki_current"] = compute_aki_state(df, baseline_cr).to_numpy()

    hour_values = df[_TIME_COL].to_numpy()
    aki_current = df["_aki_current"].to_numpy()
    n = len(df)

    for h in horizons:
        labels = np.zeros(n, dtype=np.int64)
        for i in range(n):
            if aki_current[i]:
                labels[i] = 0  # prevalent → not an onset target
                continue
            current_hour = hour_values[i]
            mask = (hour_values > current_hour) & (hour_values <= current_hour + h)
            if not mask.any():
                labels[i] = 0
                continue
            labels[i] = int(aki_current[mask].any())
        df[f"aki_label_{h}h"] = labels

    return df


# =========================================================================
# 4.  apply_label_exclusions
# =========================================================================
def apply_label_exclusions(
    df: pd.DataFrame,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Mark rows that should be excluded from model training.

    Adds boolean ``valid_for_training`` — ``True`` only when none of these apply:

    1. **Prevalent AKI** — ``_aki_current`` is ``True``.
    2. **Insufficient horizon** — within 1 hour of the patient's last recorded
       hour, so no complete prediction window exists.
    3. **Unobservable outcome** — the forward window (up to the max horizon) has
       neither a creatinine measurement nor any urine observation, leaving the
       label indeterminate.
    """
    if horizons is None:
        horizons = list(_DEFAULT_HORIZONS)

    df = df.copy()
    max_horizon = max(horizons)

    if "_aki_current" in df.columns:
        excl_prevalent = df["_aki_current"].fillna(False).astype(bool)
    else:
        excl_prevalent = pd.Series(False, index=df.index)

    excl_horizon = pd.Series(False, index=df.index)
    excl_unobservable = pd.Series(False, index=df.index)

    has_uop = _UOP_OBSERVED_COL in df.columns

    for _pid, grp in df.groupby(_PATIENT_ID_COL):
        last_hour = grp[_TIME_COL].max()
        hours = grp[_TIME_COL]

        # Exclusion 2: within 1 hour of last recorded hour
        excl_horizon.loc[grp.index] = (hours >= (last_hour - 1)).to_numpy()

        # Exclusion 3: no creatinine AND no urine observation in forward window
        hour_arr = grp[_TIME_COL].to_numpy()
        cr_arr = grp[_CREATININE_COL].to_numpy(dtype=float)
        uop_obs_arr = (
            grp[_UOP_OBSERVED_COL].fillna(0).to_numpy(dtype=int)
            if has_uop else np.zeros(len(grp), dtype=int)
        )

        for idx, row_hour in zip(grp.index, hour_arr):
            fwd = (hour_arr > row_hour) & (hour_arr <= row_hour + max_horizon)
            fwd_cr = cr_arr[fwd]
            fwd_uop = uop_obs_arr[fwd]
            no_cr = fwd_cr.size == 0 or np.all(np.isnan(fwd_cr))
            no_uop = fwd_uop.size == 0 or fwd_uop.sum() == 0
            if no_cr and no_uop:
                excl_unobservable.at[idx] = True

    df["valid_for_training"] = ~(excl_prevalent | excl_horizon | excl_unobservable)
    return df


# =========================================================================
# 5.  label_dataset
# =========================================================================
def label_dataset(
    df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Orchestrate AKI labeling for an entire multi-patient dataset.

    Workflow: merge baseline creatinine → per-patient horizon labels
    (creatinine + urine) → exclusions → return labeled DataFrame.

    *baseline_df* must have columns ``patient_id`` and ``baseline_creatinine``.
    """
    if horizons is None:
        horizons = list(_DEFAULT_HORIZONS)

    if _BASELINE_CR_COL not in df.columns:
        df = df.merge(
            baseline_df[[_PATIENT_ID_COL, _BASELINE_CR_COL]],
            on=_PATIENT_ID_COL,
            how="left",
        )

    labeled_parts: list[pd.DataFrame] = []
    for pid, patient_group in df.groupby(_PATIENT_ID_COL):
        baseline_cr_vals = patient_group[_BASELINE_CR_COL].dropna().unique()
        baseline_cr = float(baseline_cr_vals[0]) if len(baseline_cr_vals) > 0 else float("nan")
        labeled_patient = create_aki_horizon_labels(
            patient_df=patient_group, baseline_cr=baseline_cr, horizons=horizons,
        )
        labeled_patient[_PATIENT_ID_COL] = pid
        labeled_parts.append(labeled_patient)

    if not labeled_parts:
        return df

    labeled_df = pd.concat(labeled_parts, ignore_index=True)

    if _BASELINE_CR_COL not in labeled_df.columns:
        labeled_df = labeled_df.merge(
            baseline_df[[_PATIENT_ID_COL, _BASELINE_CR_COL]],
            on=_PATIENT_ID_COL,
            how="left",
        )

    return apply_label_exclusions(labeled_df, horizons=horizons)


# =========================================================================
# 6.  print_label_distribution
# =========================================================================
def print_label_distribution(
    df: pd.DataFrame,
    horizons: list[int] | None = None,
) -> None:
    """Print positive/negative label distribution per horizon (valid rows only)."""
    if horizons is None:
        horizons = list(_DEFAULT_HORIZONS)

    valid = df[df["valid_for_training"]] if "valid_for_training" in df.columns else df
    if valid.empty:
        print("No valid training rows found.")
        return

    max_label_width = max(len(str(h)) for h in horizons) + 1

    for h in horizons:
        col = f"aki_label_{h}h"
        if col not in valid.columns:
            print(f"{str(h) + 'h':<{max_label_width}}: column '{col}' not found")
            continue

        n_pos = int((valid[col] == 1).sum())
        n_neg = int((valid[col] == 0).sum())
        counted = n_pos + n_neg
        if counted == 0:
            print(f"{str(h) + 'h':<{max_label_width}}: no valid labels")
            continue

        pct_pos = 100.0 * n_pos / counted
        pct_neg = 100.0 * n_neg / counted
        ratio_str = f"{n_neg / n_pos:.1f}:1" if n_pos > 0 else "inf:1"

        label_tag = f"{h}h:"
        print(
            f"{label_tag:<{max_label_width + 1}} "
            f"{pct_pos:.1f}% positive, {pct_neg:.1f}% negative (ratio: {ratio_str})"
        )
