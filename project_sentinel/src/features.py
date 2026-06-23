"""Feature engineering pipeline for Project Sentinel.

This module implements the complete feature engineering pipeline for the
Sepsis-Associated AKI Early Warning System.  It transforms raw clinical
time-series data (vitals, labs, demographics) into a rich feature set
suitable for sequential classification models.

Pipeline order (orchestrated by ``engineer_features``):
    1. Missingness indicators  (before any imputation)
    2. Rolling window statistics  (on original *sparse* data)
    3. Forward / backward fill imputation
    4. Creatinine velocity & baseline ratios
    5. Composite / interaction features
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column group constants
# ---------------------------------------------------------------------------

VITAL_COLUMNS: list[str] = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp",
]

LAB_COLUMNS: list[str] = [
    "BaseExcess", "HCO3", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]

DEMO_COLUMNS: list[str] = ["Age", "Gender"]

# Key lab columns whose rolling statistics are computed alongside vitals.
_KEY_LAB_COLUMNS: list[str] = [
    "Creatinine", "BUN", "Lactate", "WBC", "Platelets",
    "Glucose", "Potassium", "Hgb", "Hct",
]

# Columns that should never appear as model features: identifiers, timestamps,
# and — critically — the labels/meta added downstream. Including any aki_label_*,
# _aki_current, or valid_for_training column would leak the target into training.
_EXCLUDE_FROM_FEATURES: set[str] = {
    "patient_id", "subject_id", "ICULOS", "t_sepsis", "SepsisLabel",
    "hospital_system", "AKI_label", "AKI_stage",
    "_aki_current", "valid_for_training",
}

STAGE_1_FEATURES: list[str] = [
    "HR", "MAP", "O2Sat", "Temp", "Resp", "SBP",
    "HR_mean_6h", "MAP_mean_6h", "MAP_std_6h",
    "HR_std_6h", "SBP_mean_6h", "Resp_mean_6h",
    "hours_since_sepsis", "Age", "Gender",
]


# ---------------------------------------------------------------------------
# 1. Missingness indicators
# ---------------------------------------------------------------------------

def add_missingness_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary observation flags for each lab column.

    For every column in ``LAB_COLUMNS`` that exists in *df*, a new column
    ``{col}_observed`` is created with value 1 where the lab value is
    present and 0 where it is missing.  These indicators capture the
    informative-missingness signal (labs are ordered *because* a clinician
    suspects something) without leaking the lab value itself.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or interim DataFrame that contains at least some lab columns.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with ``*_observed`` columns appended.
    """
    df = df.copy()
    present_labs = [c for c in LAB_COLUMNS if c in df.columns]
    if not present_labs:
        logger.warning("No LAB_COLUMNS found in DataFrame — skipping missingness indicators.")
        return df

    for col in present_labs:
        df[f"{col}_observed"] = df[col].notna().astype(np.int8)

    logger.info("Added %d missingness indicator columns.", len(present_labs))
    return df


# ---------------------------------------------------------------------------
# 2. Imputation
# ---------------------------------------------------------------------------

def impute_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values using clinically-appropriate fill strategies.

    * **Labs** — forward-fill then backward-fill (within patient), each
      with ``limit=4`` hours.  Both directions are used because labs are
      drawn sporadically and the last-known value is often the best proxy.
    * **Vitals** — forward-fill only with ``limit=4``.  Vitals are charted
      regularly; a backward fill would leak future information.

    .. important::

        Call this **after** computing rolling statistics on the original
        sparse data so that window aggregations reflect true observation
        density rather than imputed values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame (should already contain rolling features).

    Returns
    -------
    pd.DataFrame
        DataFrame with labs and vitals forward-/backward-filled.
    """
    df = df.copy()
    assert "patient_id" in df.columns, "impute_values requires the canonical 'patient_id' column"
    grouped = df.groupby("patient_id", sort=False)

    # --- Labs: ffill + bfill ---
    lab_cols = [c for c in LAB_COLUMNS if c in df.columns]
    if lab_cols:
        df[lab_cols] = grouped[lab_cols].transform(
            lambda s: s.ffill(limit=4).bfill(limit=4)
        )

    # --- Vitals: ffill only ---
    vital_cols = [c for c in VITAL_COLUMNS if c in df.columns]
    if vital_cols:
        df[vital_cols] = grouped[vital_cols].transform(
            lambda s: s.ffill(limit=4)
        )

    n_remaining = df[lab_cols + vital_cols].isna().sum().sum() if (lab_cols or vital_cols) else 0
    logger.info(
        "Imputation complete — %d residual NaNs across labs + vitals.",
        n_remaining,
    )
    return df


# ---------------------------------------------------------------------------
# 3. Rolling window features
# ---------------------------------------------------------------------------

def add_rolling_features(
    df: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute rolling window statistics grouped by patient.

    For each variable in ``VITAL_COLUMNS + _KEY_LAB_COLUMNS`` and each
    window size, four rolling aggregates are produced:

    * ``{v}_mean_{w}h`` — rolling mean  (``min_periods=1``)
    * ``{v}_std_{w}h``  — rolling std   (``min_periods=2``)
    * ``{v}_min_{w}h``  — rolling min   (``min_periods=1``)
    * ``{v}_max_{w}h``  — rolling max   (``min_periods=1``)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least ``patient_id`` and clinical columns.
    windows : list[int], optional
        Window sizes in hours.  Defaults to ``[6, 12]``.

    Returns
    -------
    pd.DataFrame
        DataFrame augmented with rolling-window columns.
    """
    if windows is None:
        windows = [6, 12]

    df = df.copy()
    rolling_vars = [c for c in VITAL_COLUMNS + _KEY_LAB_COLUMNS if c in df.columns]

    if not rolling_vars:
        logger.warning("No rolling-eligible columns found — skipping rolling features.")
        return df

    if "patient_id" not in df.columns:
        logger.warning("'patient_id' not found — computing rolling stats ungrouped.")
        for w in windows:
            for v in rolling_vars:
                _compute_rolling_for_series(df, v, w)
        return df

    grouped = df.groupby("patient_id", sort=False)

    # Build all rolling columns first, then concat once — assigning them one at
    # a time fragments the DataFrame and triggers pandas PerformanceWarnings.
    new_cols: dict[str, pd.Series] = {}
    for w in windows:
        for v in rolling_vars:
            rolling = grouped[v].rolling(window=w, min_periods=1)
            new_cols[f"{v}_mean_{w}h"] = rolling.mean().droplevel(0)
            new_cols[f"{v}_min_{w}h"] = rolling.min().droplevel(0)
            new_cols[f"{v}_max_{w}h"] = rolling.max().droplevel(0)
            new_cols[f"{v}_std_{w}h"] = grouped[v].rolling(window=w, min_periods=2).std().droplevel(0)

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    logger.info("Added %d rolling-window features (windows=%s).", len(new_cols), windows)
    return df


def _compute_rolling_for_series(
    df: pd.DataFrame, v: str, w: int,
) -> None:
    """Compute rolling stats in-place for a single series (no grouping)."""
    rolling = df[v].rolling(window=w, min_periods=1)
    df[f"{v}_mean_{w}h"] = rolling.mean()
    df[f"{v}_min_{w}h"] = rolling.min()
    df[f"{v}_max_{w}h"] = rolling.max()

    rolling_std = df[v].rolling(window=w, min_periods=2)
    df[f"{v}_std_{w}h"] = rolling_std.std()


# ---------------------------------------------------------------------------
# 4. Creatinine velocity & baseline ratios
# ---------------------------------------------------------------------------

def add_creatinine_velocity(
    df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    window_hours: int = 12,
) -> pd.DataFrame:
    """Derive creatinine velocity and baseline-relative features.

    New columns:

    * ``delta_Cr``               — Cr(t) − Cr(t − *window_hours*)
    * ``creatinine_velocity_12h`` — delta_Cr / *window_hours*  (mg/dL/hr)
    * ``cr_above_baseline``      — Cr(t) − baseline_creatinine
    * ``cr_ratio_baseline``      — Cr(t) / baseline_creatinine

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame with ``Creatinine`` and ``patient_id``.
    baseline_df : pd.DataFrame
        Must contain ``patient_id`` and ``baseline_creatinine``.
    window_hours : int
        Look-back period for the velocity calculation (default 12).

    Returns
    -------
    pd.DataFrame
        DataFrame with creatinine-derived columns appended.
    """
    df = df.copy()

    if "Creatinine" not in df.columns:
        logger.warning("'Creatinine' column not found — skipping creatinine velocity.")
        df["delta_Cr"] = np.nan
        df["creatinine_velocity_12h"] = np.nan
        df["cr_above_baseline"] = np.nan
        df["cr_ratio_baseline"] = np.nan
        return df

    # --- Delta and velocity ---
    if "patient_id" in df.columns:
        df["delta_Cr"] = df.groupby("patient_id", sort=False)["Creatinine"].transform(
            lambda s: s - s.shift(window_hours)
        )
    else:
        df["delta_Cr"] = df["Creatinine"] - df["Creatinine"].shift(window_hours)

    df["creatinine_velocity_12h"] = df["delta_Cr"] / window_hours

    # --- Baseline ratios ---
    if baseline_df is not None and "baseline_creatinine" in baseline_df.columns:
        baseline_map = baseline_df.set_index("patient_id")["baseline_creatinine"]

        if "patient_id" in df.columns:
            bl = df["patient_id"].map(baseline_map)
        else:
            bl = pd.Series(np.nan, index=df.index)

        # Guard against zero / NaN baselines
        bl_safe = bl.replace(0, np.nan)

        df["cr_above_baseline"] = df["Creatinine"] - bl
        df["cr_ratio_baseline"] = df["Creatinine"] / bl_safe
    else:
        logger.warning(
            "baseline_df missing or lacks 'baseline_creatinine' — "
            "setting baseline features to NaN."
        )
        df["cr_above_baseline"] = np.nan
        df["cr_ratio_baseline"] = np.nan

    logger.info("Creatinine velocity features added (window=%dh).", window_hours)
    return df


# ---------------------------------------------------------------------------
# 5. Composite / interaction features
# ---------------------------------------------------------------------------

def add_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create clinically-motivated composite and interaction features.

    New columns:

    * ``renal_perf_pressure``          — MAP (CVP unavailable in this dataset)
    * ``map_cr_velocity_interaction``  — MAP × creatinine_velocity_12h
    * ``wbc_lactate_product``          — WBC × Lactate
    * ``anion_gap_proxy``              — Chloride + HCO3  (simplified proxy)
    * ``bun_cr_ratio``                 — BUN / Creatinine
    * ``hours_since_sepsis``           — ICULOS − t_sepsis

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the required source columns.

    Returns
    -------
    pd.DataFrame
        DataFrame augmented with composite features.
    """
    df = df.copy()

    # Renal perfusion pressure (CVP not available → just MAP)
    if "MAP" in df.columns:
        df["renal_perf_pressure"] = df["MAP"]
    else:
        df["renal_perf_pressure"] = np.nan

    # MAP × creatinine velocity interaction
    map_vals = df.get("MAP", pd.Series(np.nan, index=df.index))
    cr_vel = df.get("creatinine_velocity_12h", pd.Series(np.nan, index=df.index))
    df["map_cr_velocity_interaction"] = map_vals * cr_vel

    # WBC × Lactate product
    wbc = df.get("WBC", pd.Series(np.nan, index=df.index))
    lactate = df.get("Lactate", pd.Series(np.nan, index=df.index))
    df["wbc_lactate_product"] = wbc * lactate

    # Anion-gap proxy (simplified: Chloride + HCO3)
    chloride = df.get("Chloride", pd.Series(np.nan, index=df.index))
    hco3 = df.get("HCO3", pd.Series(np.nan, index=df.index))
    df["anion_gap_proxy"] = chloride + hco3

    # BUN / Creatinine ratio — guard division by zero
    bun = df.get("BUN", pd.Series(np.nan, index=df.index))
    cr = df.get("Creatinine", pd.Series(np.nan, index=df.index))
    cr_safe = cr.replace(0, np.nan)
    df["bun_cr_ratio"] = bun / cr_safe

    # Hours since sepsis onset
    iculos = df.get("ICULOS", pd.Series(np.nan, index=df.index))
    t_sepsis = df.get("t_sepsis", pd.Series(np.nan, index=df.index))
    df["hours_since_sepsis"] = iculos - t_sepsis

    logger.info("Composite features added.")
    return df


# ---------------------------------------------------------------------------
# 6. Full pipeline orchestrator
# ---------------------------------------------------------------------------

def engineer_features(
    df: pd.DataFrame,
    baseline_df: pd.DataFrame,
) -> pd.DataFrame:
    """Orchestrate the complete feature-engineering pipeline.

    Execution order:

    1. **Missingness indicators** — capture observation patterns.
    2. **Rolling features** — computed on *original* sparse data so that
       window aggregates reflect true observation density.
    3. **Imputation** — fill remaining gaps *after* rolling stats.
    4. **Creatinine velocity** — requires imputed creatinine values.
    5. **Composite features** — interactions that depend on imputed data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw/interim clinical DataFrame.
    baseline_df : pd.DataFrame
        Per-patient baseline creatinine values (columns: ``patient_id``,
        ``baseline_creatinine``).

    Returns
    -------
    pd.DataFrame
        Fully feature-engineered DataFrame ready for modelling.
    """
    logger.info("Starting feature engineering pipeline …")
    n_rows_in, n_cols_in = df.shape

    df = add_missingness_indicators(df)
    df = add_rolling_features(df)           # on sparse data
    df = impute_values(df)                  # after rolling
    df = add_creatinine_velocity(df, baseline_df)
    df = add_composite_features(df)

    logger.info(
        "Feature engineering complete: %d → %d columns (%d rows).",
        n_cols_in, df.shape[1], df.shape[0],
    )
    return df


# ---------------------------------------------------------------------------
# 7. Feature manifest
# ---------------------------------------------------------------------------

def _classify_feature(name: str) -> str:
    """Return a human-readable category for a feature column."""
    if name in VITAL_COLUMNS:
        return "vital"
    if name in LAB_COLUMNS:
        return "lab"
    if name in DEMO_COLUMNS:
        return "demographic"
    if name.endswith("_observed"):
        return "missingness"
    if any(
        name.endswith(f"_{w}h") for w in (6, 12)
    ):
        return "derived"
    if name in {
        "hours_since_sepsis", "ICULOS", "delta_Cr",
        "creatinine_velocity_12h", "cr_above_baseline", "cr_ratio_baseline",
    }:
        return "temporal"
    if name in {
        "renal_perf_pressure", "map_cr_velocity_interaction",
        "wbc_lactate_product", "anion_gap_proxy", "bun_cr_ratio",
    }:
        return "composite"
    return "other"


def create_feature_manifest(
    df: pd.DataFrame,
    output_path: str | Path,
) -> pd.DataFrame:
    """Build and save a manifest describing every feature column.

    The manifest contains one row per feature with:

    * ``name``       — column name
    * ``category``   — vital / lab / derived / temporal / composite / other
    * ``pct_missing``— percentage of NaN values (before imputation snapshot)
    * ``mean``, ``std``, ``min``, ``max``

    Parameters
    ----------
    df : pd.DataFrame
        The feature-engineered DataFrame (pre- or post-imputation).
    output_path : str | Path
        Destination CSV path.

    Returns
    -------
    pd.DataFrame
        The manifest DataFrame.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Only include numeric columns that are not identifiers / targets.
    numeric_cols = df.select_dtypes(include="number").columns
    feature_cols = [c for c in numeric_cols if c not in _EXCLUDE_FROM_FEATURES]

    records: list[dict[str, Any]] = []
    for col in feature_cols:
        series = df[col]
        records.append({
            "name": col,
            "category": _classify_feature(col),
            "pct_missing": round(series.isna().mean() * 100, 2),
            "mean": round(series.mean(), 4) if series.notna().any() else np.nan,
            "std": round(series.std(), 4) if series.notna().any() else np.nan,
            "min": round(series.min(), 4) if series.notna().any() else np.nan,
            "max": round(series.max(), 4) if series.notna().any() else np.nan,
        })

    manifest = pd.DataFrame(records)
    manifest.to_csv(output_path, index=False)
    logger.info("Feature manifest saved to %s (%d features).", output_path, len(manifest))
    return manifest


# ---------------------------------------------------------------------------
# 8. Temporal / site-level split
# ---------------------------------------------------------------------------

def temporal_split(
    df: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    output_dir: str | Path = "data/processed",
    aki_label_col: str = "AKI_label",
) -> dict[str, pd.DataFrame]:
    """Create chronological train / val / test splits + site holdout.

    Splitting strategy:

    1. **Site holdout** — all patients from ``hospital_system == 'B'`` are
       removed and reserved as an external validation cohort.
    2. **Chronological split** — remaining patients are sorted by their
       earliest ``ICULOS`` (proxy for admission order) and divided:

       * Train: first 70 % of patients
       * Validation: next 15 %
       * Test: last 15 %

    Each split is saved as a Parquet file under *output_dir*.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame.
    val_ratio : float
        Fraction of patients for validation (default 0.15).
    test_ratio : float
        Fraction of patients for test (default 0.15).
    output_dir : str | Path
        Directory for Parquet output files.
    aki_label_col : str
        Column name for the AKI binary label (default ``'AKI_label'``).

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: ``'train'``, ``'val'``, ``'test'``, ``'site_holdout'``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "patient_id" not in df.columns:
        raise ValueError("DataFrame must contain 'patient_id' for splitting.")

    # ------------------------------------------------------------------
    # 1. Separate site-B holdout
    # ------------------------------------------------------------------
    if "hospital_system" in df.columns:
        site_holdout = df[df["hospital_system"] == "B"].copy()
        remaining = df[df["hospital_system"] != "B"].copy()
    else:
        logger.warning(
            "'hospital_system' column not found — no site holdout produced."
        )
        site_holdout = pd.DataFrame(columns=df.columns)
        remaining = df.copy()

    # ------------------------------------------------------------------
    # 2. Sort patients chronologically by first ICULOS
    # ------------------------------------------------------------------
    if "ICULOS" not in remaining.columns:
        raise ValueError("DataFrame must contain 'ICULOS' for temporal ordering.")

    first_icu = (
        remaining.groupby("patient_id", sort=False)["ICULOS"]
        .min()
        .sort_values()
    )
    sorted_patients = first_icu.index.tolist()

    n_patients = len(sorted_patients)
    n_test = max(1, int(round(n_patients * test_ratio)))
    n_val = max(1, int(round(n_patients * val_ratio)))
    n_train = n_patients - n_val - n_test

    if n_train <= 0:
        raise ValueError(
            f"Not enough patients ({n_patients}) for the requested split ratios."
        )

    train_ids = set(sorted_patients[:n_train])
    val_ids = set(sorted_patients[n_train : n_train + n_val])
    test_ids = set(sorted_patients[n_train + n_val :])

    splits: dict[str, pd.DataFrame] = {
        "train": remaining[remaining["patient_id"].isin(train_ids)].copy(),
        "val": remaining[remaining["patient_id"].isin(val_ids)].copy(),
        "test": remaining[remaining["patient_id"].isin(test_ids)].copy(),
        "site_holdout": site_holdout,
    }

    # ------------------------------------------------------------------
    # 3. Print summary & save
    # ------------------------------------------------------------------
    for name, split_df in splits.items():
        n_pat = split_df["patient_id"].nunique()
        n_row = len(split_df)

        if aki_label_col in split_df.columns and n_row > 0:
            pct_pos = split_df[aki_label_col].mean() * 100
            aki_str = f"{pct_pos:.1f}% AKI-positive"
        else:
            aki_str = "AKI label unavailable"

        logger.info(
            "  %-14s  %5d patients  |  %7d rows  |  %s",
            name, n_pat, n_row, aki_str,
        )
        print(
            f"  {name:<14s}  {n_pat:>5d} patients  |  {n_row:>7d} rows  |  {aki_str}"
        )

        out_path = output_dir / f"{name}.parquet"
        split_df.to_parquet(out_path, index=False)
        logger.info("  → saved to %s", out_path)

    return splits


# ---------------------------------------------------------------------------
# 9. Feature list helpers
# ---------------------------------------------------------------------------

def get_stage1_features() -> list[str]:
    """Return the curated feature list for the Stage-1 screening model.

    Stage 1 uses a small, low-latency feature set drawn from vitals,
    basic rolling statistics, sepsis timing, and demographics.
    """
    return list(STAGE_1_FEATURES)


def get_all_features(df: pd.DataFrame) -> list[str]:
    """Return all engineered feature columns for the Stage-2 model.

    Every numeric column in *df* is included **except** identifiers,
    timestamps, and the labels/meta added downstream — anything in
    ``_EXCLUDE_FROM_FEATURES`` plus every ``aki_label_*`` horizon column.
    This is what prevents the target from leaking into the feature set.

    Parameters
    ----------
    df : pd.DataFrame
        A feature-engineered DataFrame.

    Returns
    -------
    list[str]
        Sorted list of feature column names.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    features = sorted(
        c for c in numeric_cols
        if c not in _EXCLUDE_FROM_FEATURES and not c.startswith("aki_label_")
    )
    return features
