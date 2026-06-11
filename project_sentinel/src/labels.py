"""
labels.py — KDIGO AKI Criteria & Prospective Horizon Labeling
=============================================================

Project Sentinel — Sepsis-Associated AKI Early Warning System

This module converts raw creatinine time-series into binary prediction
targets at configurable forward-looking horizons (default 6 h, 12 h, 24 h).

Labeling follows KDIGO Stage-1 AKI criteria:
  * Absolute increase ≥ 0.3 mg/dL from baseline   OR
  * Relative increase ≥ 1.5× baseline

Key design decisions:
  * **Prevalent-case exclusion** — time steps where the patient already
    meets AKI criteria are excluded so the model learns *onset*, not
    *persistence*.
  * **Insufficient-horizon exclusion** — the last hour of a patient stay
    is dropped because no full horizon window can be evaluated.
  * **Unobservable-outcome exclusion** — if no creatinine measurement
    exists anywhere in the forward window, the label is indeterminate and
    the row is excluded from training.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

__all__ = [
    "check_kdigo_aki",
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

# Column the pipeline expects for hourly creatinine values
_CREATININE_COL: str = "creatinine"
_PATIENT_ID_COL: str = "patient_id"
_TIME_COL: str = "hour"
_BASELINE_CR_COL: str = "baseline_creatinine"


# =========================================================================
# 1.  check_kdigo_aki
# =========================================================================
def check_kdigo_aki(
    creatinine_current: float,
    creatinine_baseline: float,
) -> bool:
    """Evaluate KDIGO Stage-1 AKI criteria for a single measurement pair.

    Parameters
    ----------
    creatinine_current : float
        The most recent serum creatinine value (mg/dL).
    creatinine_baseline : float
        The patient's baseline serum creatinine value (mg/dL).

    Returns
    -------
    bool
        ``True`` if KDIGO Stage-1 criteria are met, ``False`` otherwise.
        Returns ``False`` when either input is NaN.

    Examples
    --------
    >>> check_kdigo_aki(1.5, 1.0)
    True
    >>> check_kdigo_aki(1.2, 1.0)
    False
    >>> check_kdigo_aki(float('nan'), 1.0)
    False
    """
    # Guard against NaN / None / non-finite values
    try:
        if (
            creatinine_current is None
            or creatinine_baseline is None
            or math.isnan(creatinine_current)
            or math.isnan(creatinine_baseline)
        ):
            return False
    except TypeError:
        # Catches unexpected types that math.isnan cannot handle
        return False

    absolute_increase = creatinine_current - creatinine_baseline
    relative_increase = creatinine_current >= (_AKI_RELATIVE_FACTOR * creatinine_baseline)

    return absolute_increase >= _AKI_ABSOLUTE_THRESHOLD or relative_increase


# =========================================================================
# 2.  create_aki_horizon_labels
# =========================================================================
def create_aki_horizon_labels(
    patient_df: pd.DataFrame,
    baseline_cr: float,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Create prospective AKI horizon labels for a single patient.

    For every time step *t*, and for each horizon *h*, a label of ``1`` is
    assigned if:

    1. The patient does **not** currently meet AKI criteria at time *t*
       (prevalent-case exclusion).
    2. The patient **does** meet KDIGO AKI criteria at **any** time step in
       the open-forward window ``(t, t + h]`` (i.e. ``t+1`` through
       ``t+h`` inclusive).

    Parameters
    ----------
    patient_df : pd.DataFrame
        Time-series rows for a **single** patient.  Must contain at least
        columns ``creatinine`` and ``hour``.  Should be sorted by ``hour``.
    baseline_cr : float
        Baseline creatinine value (mg/dL) for this patient.
    horizons : list[int], optional
        Forward-looking windows in hours.  Defaults to ``[6, 12, 24]``.

    Returns
    -------
    pd.DataFrame
        A copy of *patient_df* with added columns ``aki_label_<h>h`` for
        each horizon and a helper column ``_aki_current`` indicating
        whether AKI criteria are currently met.
    """
    if horizons is None:
        horizons = list(_DEFAULT_HORIZONS)

    df = patient_df.copy()

    # Short-circuit: single-row or empty DataFrame
    if df.empty:
        for h in horizons:
            df[f"aki_label_{h}h"] = pd.Series(dtype="Int64")
        df["_aki_current"] = pd.Series(dtype="bool")
        return df

    # Ensure sorted by time
    df = df.sort_values(_TIME_COL).reset_index(drop=True)

    # --- Vectorised current-AKI flag ---
    cr_values = df[_CREATININE_COL].astype(float)

    # Handle NaN baseline → no AKI can be assessed
    if baseline_cr is None or (isinstance(baseline_cr, float) and math.isnan(baseline_cr)):
        df["_aki_current"] = False
        for h in horizons:
            df[f"aki_label_{h}h"] = 0
        return df

    aki_abs = (cr_values - baseline_cr) >= _AKI_ABSOLUTE_THRESHOLD
    aki_rel = cr_values >= (_AKI_RELATIVE_FACTOR * baseline_cr)
    # NaN creatinine → not AKI
    df["_aki_current"] = (aki_abs | aki_rel).fillna(False).astype(bool)

    # --- Build hour → index mapping for look-ahead ---
    hour_values = df[_TIME_COL].values  # numpy array
    n = len(df)

    for h in horizons:
        labels = np.zeros(n, dtype=np.int64)

        for i in range(n):
            # Prevalent case: already has AKI → label stays 0
            if df["_aki_current"].iat[i]:
                labels[i] = 0
                continue

            current_hour = hour_values[i]

            # Forward window: (current_hour, current_hour + h]
            mask = (hour_values > current_hour) & (hour_values <= current_hour + h)
            future_slice = df.loc[mask]

            if future_slice.empty:
                labels[i] = 0
                continue

            # Check if ANY future time step in window meets AKI criteria
            future_cr = future_slice[_CREATININE_COL].astype(float)
            future_aki = (
                ((future_cr - baseline_cr) >= _AKI_ABSOLUTE_THRESHOLD)
                | (future_cr >= _AKI_RELATIVE_FACTOR * baseline_cr)
            ).fillna(False)

            labels[i] = int(future_aki.any())

        df[f"aki_label_{h}h"] = labels

    return df


# =========================================================================
# 3.  apply_label_exclusions
# =========================================================================
def apply_label_exclusions(
    df: pd.DataFrame,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Mark rows that should be excluded from model training.

    Three exclusion rules are applied per patient:

    1. **Prevalent AKI** — the ``_aki_current`` column is ``True``.
    2. **Insufficient horizon** — the time step is within 1 hour of the
       patient's last recorded hour, so no complete prediction window
       exists.
    3. **Unobservable outcome** — there is no creatinine measurement
       anywhere in the forward horizon window for the *maximum* requested
       horizon, making the label indeterminate.

    A new boolean column ``valid_for_training`` is added.  It is ``True``
    only when **none** of the above exclusions apply.

    Parameters
    ----------
    df : pd.DataFrame
        Labeled DataFrame (output of :func:`create_aki_horizon_labels` or
        :func:`label_dataset`).  Must contain ``_aki_current``,
        ``creatinine``, ``hour``, and ``patient_id`` columns.
    horizons : list[int], optional
        Horizons used during labeling.  Defaults to ``[6, 12, 24]``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an added ``valid_for_training`` column.
    """
    if horizons is None:
        horizons = list(_DEFAULT_HORIZONS)

    df = df.copy()
    max_horizon = max(horizons)

    # --- Exclusion 1: prevalent AKI ---
    if "_aki_current" in df.columns:
        excl_prevalent = df["_aki_current"].fillna(False).astype(bool)
    else:
        excl_prevalent = pd.Series(False, index=df.index)

    # --- Per-patient exclusions (horizon + observability) ---
    excl_horizon = pd.Series(False, index=df.index)
    excl_unobservable = pd.Series(False, index=df.index)

    for _pid, grp in df.groupby(_PATIENT_ID_COL):
        last_hour = grp[_TIME_COL].max()
        hours = grp[_TIME_COL]

        # Exclusion 2: within 1 hour of last recorded hour
        mask_horizon = hours >= (last_hour - 1)
        excl_horizon.loc[grp.index] = mask_horizon.values

        # Exclusion 3: no creatinine in forward window [t+1, t+max_horizon]
        hour_arr = grp[_TIME_COL].values
        cr_arr = grp[_CREATININE_COL].values

        for idx, row_hour in zip(grp.index, hour_arr):
            forward_mask = (hour_arr > row_hour) & (hour_arr <= row_hour + max_horizon)
            forward_cr = cr_arr[forward_mask]

            # If no future measurements at all, or all are NaN → unobservable
            if len(forward_cr) == 0 or np.all(np.isnan(forward_cr.astype(float))):
                excl_unobservable.at[idx] = True

    df["valid_for_training"] = ~(excl_prevalent | excl_horizon | excl_unobservable)
    return df


# =========================================================================
# 4.  label_dataset
# =========================================================================
def label_dataset(
    df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Orchestrate AKI labeling for an entire multi-patient dataset.

    Workflow:
      1. Merge baseline creatinine values from *baseline_df*.
      2. Apply :func:`create_aki_horizon_labels` per patient.
      3. Apply :func:`apply_label_exclusions`.
      4. Return the fully labeled DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Hourly clinical time-series.  Must contain at least ``patient_id``,
        ``hour``, and ``creatinine`` columns.
    baseline_df : pd.DataFrame
        One row per patient with columns ``patient_id`` and
        ``baseline_creatinine``.
    horizons : list[int], optional
        Forward-looking windows in hours.  Defaults to ``[6, 12, 24]``.

    Returns
    -------
    pd.DataFrame
        Labeled DataFrame with ``aki_label_<h>h`` columns, ``_aki_current``,
        and ``valid_for_training``.
    """
    if horizons is None:
        horizons = list(_DEFAULT_HORIZONS)

    # --- Merge baseline creatinine ---
    if _BASELINE_CR_COL not in df.columns:
        df = df.merge(
            baseline_df[[_PATIENT_ID_COL, _BASELINE_CR_COL]],
            on=_PATIENT_ID_COL,
            how="left",
        )

    # --- Per-patient labeling ---
    labeled_parts: list[pd.DataFrame] = []

    for pid, patient_group in df.groupby(_PATIENT_ID_COL):
        baseline_cr_vals = patient_group[_BASELINE_CR_COL].dropna().unique()
        baseline_cr: float = float(baseline_cr_vals[0]) if len(baseline_cr_vals) > 0 else float("nan")

        labeled_patient = create_aki_horizon_labels(
            patient_df=patient_group,
            baseline_cr=baseline_cr,
            horizons=horizons,
        )
        # Preserve patient_id (it may already be there, but ensure it)
        labeled_patient[_PATIENT_ID_COL] = pid
        labeled_parts.append(labeled_patient)

    if not labeled_parts:
        return df

    labeled_df = pd.concat(labeled_parts, ignore_index=True)

    # Carry over baseline creatinine
    if _BASELINE_CR_COL not in labeled_df.columns:
        labeled_df = labeled_df.merge(
            baseline_df[[_PATIENT_ID_COL, _BASELINE_CR_COL]],
            on=_PATIENT_ID_COL,
            how="left",
        )

    # --- Apply exclusions ---
    labeled_df = apply_label_exclusions(labeled_df, horizons=horizons)

    return labeled_df


# =========================================================================
# 5.  print_label_distribution
# =========================================================================
def print_label_distribution(
    df: pd.DataFrame,
    horizons: list[int] | None = None,
) -> None:
    """Print label distribution statistics for each horizon.

    Only rows where ``valid_for_training == True`` are counted.

    Output format per horizon::

        6h:  X% positive, Y% negative (ratio: Z:1)

    Parameters
    ----------
    df : pd.DataFrame
        Labeled DataFrame produced by :func:`label_dataset`.
    horizons : list[int], optional
        Horizons to report.  Defaults to ``[6, 12, 24]``.
    """
    if horizons is None:
        horizons = list(_DEFAULT_HORIZONS)

    if "valid_for_training" in df.columns:
        valid = df[df["valid_for_training"]]
    else:
        valid = df

    if valid.empty:
        print("No valid training rows found.")
        return

    # Determine padding for alignment
    max_label_width = max(len(str(h)) for h in horizons) + 1  # +1 for 'h'

    for h in horizons:
        col = f"aki_label_{h}h"
        if col not in valid.columns:
            print(f"{str(h) + 'h':<{max_label_width}}: column '{col}' not found")
            continue

        total = valid[col].notna().sum()
        if total == 0:
            print(f"{str(h) + 'h':<{max_label_width}}: no valid labels")
            continue

        n_pos = (valid[col] == 1).sum()
        n_neg = (valid[col] == 0).sum()
        counted = n_pos + n_neg

        pct_pos = 100.0 * n_pos / counted if counted > 0 else 0.0
        pct_neg = 100.0 * n_neg / counted if counted > 0 else 0.0

        if n_pos > 0:
            ratio = n_neg / n_pos
            ratio_str = f"{ratio:.1f}:1"
        else:
            ratio_str = "inf:1"

        label_tag = f"{h}h:"
        print(
            f"{label_tag:<{max_label_width + 1}} "
            f"{pct_pos:.1f}% positive, {pct_neg:.1f}% negative "
            f"(ratio: {ratio_str})"
        )
