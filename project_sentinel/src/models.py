"""Model training, hyperparameter tuning, and calibration for Project Sentinel.

This module provides utilities for:
- Training LightGBM and XGBoost classifiers for Sepsis-Associated AKI prediction
- Hyperparameter tuning via Optuna (Bayesian optimisation)
- Probability calibration with isotonic regression
- Two-stage model training across multiple prediction horizons (6h, 12h, 24h)
- Serialisation / deserialisation of trained artefacts
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default hyper-parameters
# ---------------------------------------------------------------------------

_MIN_POS_SAMPLES: int = 5
"""Minimum number of positive samples required to train a model."""


def get_lgbm_params() -> dict[str, Any]:
    """Return default LightGBM hyper-parameters for binary AKI prediction.

    Returns
    -------
    dict[str, Any]
        Dictionary of LightGBM constructor keyword arguments.
    """
    return {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "class_weight": "balanced",
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "verbose": -1,
    }


def get_xgb_params(pos_weight: float) -> dict[str, Any]:
    """Return default XGBoost hyper-parameters for binary AKI prediction.

    Parameters
    ----------
    pos_weight : float
        Value for ``scale_pos_weight`` (typically *n_neg / n_pos*).

    Returns
    -------
    dict[str, Any]
        Dictionary of XGBoost constructor keyword arguments.
    """
    return {
        "objective": "binary:logistic",
        "eval_metric": ["auc", "logloss"],
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "scale_pos_weight": pos_weight,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
    }


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def compute_pos_weight(y: np.ndarray) -> float:
    """Compute class-imbalance ratio *n_negative / n_positive*.

    Parameters
    ----------
    y : np.ndarray
        Binary label array (0/1).

    Returns
    -------
    float
        Ratio suitable for ``scale_pos_weight`` in XGBoost.

    Raises
    ------
    ValueError
        If *y* contains no positive examples.
    """
    y = np.asarray(y)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0:
        raise ValueError("Cannot compute pos_weight: no positive examples in y.")
    return n_neg / n_pos


def _check_sufficient_positives(y: np.ndarray, context: str = "") -> bool:
    """Return ``True`` if there are enough positive samples for training.

    Logs a warning and returns ``False`` otherwise.
    """
    n_pos = int(np.sum(np.asarray(y) == 1))
    if n_pos < _MIN_POS_SAMPLES:
        logger.warning(
            "Insufficient positive samples (%d) for %s – skipping training.",
            n_pos,
            context or "this horizon",
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Single-model training
# ---------------------------------------------------------------------------


def train_lgbm(
    X_train: pd.DataFrame | np.ndarray,
    y_train: np.ndarray,
    X_val: pd.DataFrame | np.ndarray,
    y_val: np.ndarray,
    params: dict[str, Any] | None = None,
) -> lgb.LGBMClassifier:
    """Train a LightGBM binary classifier with early stopping.

    Parameters
    ----------
    X_train, y_train : array-like
        Training features and binary labels.
    X_val, y_val : array-like
        Validation features and labels used for early stopping.
    params : dict, optional
        LightGBM hyper-parameters.  Falls back to :func:`get_lgbm_params`.

    Returns
    -------
    lgb.LGBMClassifier
        Fitted LightGBM model.
    """
    params = params or get_lgbm_params()
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    logger.info(
        "LightGBM trained – best iteration %s, best score %s",
        model.best_iteration_,
        model.best_score_,
    )
    return model


def train_xgb(
    X_train: pd.DataFrame | np.ndarray,
    y_train: np.ndarray,
    X_val: pd.DataFrame | np.ndarray,
    y_val: np.ndarray,
    params: dict[str, Any] | None = None,
) -> xgb.XGBClassifier:
    """Train an XGBoost binary classifier with early stopping.

    Parameters
    ----------
    X_train, y_train : array-like
        Training features and binary labels.
    X_val, y_val : array-like
        Validation features and labels used for early stopping.
    params : dict, optional
        XGBoost hyper-parameters.  Falls back to :func:`get_xgb_params` with
        a ``pos_weight`` computed from *y_train*.

    Returns
    -------
    xgb.XGBClassifier
        Fitted XGBoost model.
    """
    if params is None:
        pw = compute_pos_weight(y_train)
        params = get_xgb_params(pos_weight=pw)

    # xgboost >= 2.0 moved early_stopping_rounds from .fit() to the constructor
    model = xgb.XGBClassifier(early_stopping_rounds=50, **params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    logger.info(
        "XGBoost trained – best iteration %s, best score %s",
        model.best_iteration,
        model.best_score,
    )
    return model


# ---------------------------------------------------------------------------
# Optuna tuning
# ---------------------------------------------------------------------------


def optuna_tune_lgbm(
    X_train: pd.DataFrame | np.ndarray,
    y_train: np.ndarray,
    X_val: pd.DataFrame | np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
) -> dict[str, Any]:
    """Use Optuna to tune LightGBM hyper-parameters, maximising AUROC.

    The following hyper-parameters are searched:

    * ``num_leaves`` – [20, 150]
    * ``learning_rate`` – [0.01, 0.3] (log-uniform)
    * ``min_child_samples`` – [5, 50]
    * ``subsample`` – [0.5, 1.0]
    * ``colsample_bytree`` – [0.5, 1.0]
    * ``reg_alpha`` – [1e-3, 10] (log-uniform)
    * ``reg_lambda`` – [1e-3, 10] (log-uniform)

    Parameters
    ----------
    X_train, y_train : array-like
        Training data.
    X_val, y_val : array-like
        Validation data (used for AUROC evaluation).
    n_trials : int, optional
        Number of Optuna trials.  Default **50**.

    Returns
    -------
    dict[str, Any]
        Best hyper-parameter dictionary, ready to pass to :func:`train_lgbm`.
    """

    def _objective(trial: optuna.Trial) -> float:
        trial_params = get_lgbm_params()  # start from defaults
        trial_params.update(
            {
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.5, 1.0
                ),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", 1e-3, 10.0, log=True
                ),
            }
        )

        model = lgb.LGBMClassifier(**trial_params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        y_prob = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_prob)

    # Silence Optuna's default logging to keep output clean.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="maximize")
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(
        "Optuna tuning complete – best AUROC %.4f in %d trials",
        study.best_value,
        n_trials,
    )

    # Merge best trial params back into defaults so the result is a complete
    # parameter dict that can be passed directly to ``train_lgbm``.
    best_params = get_lgbm_params()
    best_params.update(study.best_params)
    return best_params


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibrate_model(
    model: lgb.LGBMClassifier | xgb.XGBClassifier,
    X_val: pd.DataFrame | np.ndarray,
    y_val: np.ndarray,
) -> IsotonicRegression:
    """Fit an isotonic-regression calibrator on validation predictions.

    Parameters
    ----------
    model : LGBMClassifier or XGBClassifier
        A trained binary classifier.
    X_val, y_val : array-like
        Validation features and labels.

    Returns
    -------
    IsotonicRegression
        Fitted calibrator.  Call ``calibrator.predict(prob)`` to obtain
        calibrated probabilities.
    """
    y_prob = model.predict_proba(X_val)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(y_prob, y_val)
    logger.info("Isotonic calibrator fitted on %d validation samples.", len(y_val))
    return calibrator


# ---------------------------------------------------------------------------
# Multi-horizon stage training
# ---------------------------------------------------------------------------


def train_stage1_models(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    feature_cols: list[str],
    horizons: list[int] | None = None,
) -> dict[int, dict[str, Any]]:
    """Train Stage-1 LightGBM models (one per prediction horizon).

    Stage-1 uses a restricted feature set to provide rapid early alerts.

    Parameters
    ----------
    train_data, val_data : pd.DataFrame
        Must contain *feature_cols* **and** columns named
        ``aki_label_{horizon}h`` for each horizon.
    feature_cols : list[str]
        Stage-1 feature column names.
    horizons : list[int], optional
        Prediction horizons in hours.  Default ``[6, 12, 24]``.

    Returns
    -------
    dict[int, dict[str, Any]]
        ``{horizon: {'model': LGBMClassifier, 'calibrator': IsotonicRegression}}``
        Horizons with insufficient positive samples are **omitted**.
    """
    if horizons is None:
        horizons = [6, 12, 24]

    results: dict[int, dict[str, Any]] = {}
    X_train = train_data[feature_cols]
    X_val = val_data[feature_cols]

    for h in horizons:
        label_col = f"aki_label_{h}h"
        logger.info("Stage 1 – training %dh horizon model …", h)

        y_train = train_data[label_col].values
        y_val = val_data[label_col].values

        if not _check_sufficient_positives(y_train, context=f"Stage-1 {h}h (train)"):
            continue
        if not _check_sufficient_positives(y_val, context=f"Stage-1 {h}h (val)"):
            continue

        model = train_lgbm(X_train, y_train, X_val, y_val)
        calibrator = calibrate_model(model, X_val, y_val)
        results[h] = {"model": model, "calibrator": calibrator}

    return results


def _apply_feature_dropout(
    X: pd.DataFrame, p: float, seed: int = 42
) -> pd.DataFrame:
    """Augment training data for missing-feature robustness.

    Returns ``concat(X_clean, X_masked)`` where the masked copy has each cell set
    to NaN with probability *p*. Training on this teaches the model to predict
    when features are absent — the reality at resource-limited hospitals that lack
    the full lab panel (see the feature-tier degradation analysis). The label is
    unchanged for the masked copy (same patient, some values hidden), so callers
    duplicate ``y`` alongside.

    ponytail: per-*cell* dropout — a simple proxy. The realistic case is whole
    labs missing (per-column, per-patient); upgrade to column-group masking if the
    tier analysis shows this isn't enough.
    """
    rng = np.random.default_rng(seed)
    masked = X.to_numpy(dtype="float64", copy=True)
    masked[rng.random(masked.shape) < p] = np.nan
    masked_df = pd.DataFrame(masked, columns=X.columns)
    return pd.concat([X.reset_index(drop=True), masked_df], ignore_index=True)


def train_stage2_models(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    feature_cols: list[str],
    horizons: list[int] | None = None,
    n_optuna_trials: int = 50,
    feature_dropout: float = 0.0,
) -> dict[int, dict[str, dict[str, Any]]]:
    """Train Stage-2 LightGBM + XGBoost models with Optuna tuning.

    Optuna tuning is performed **once** for the 24 h horizon.  The resulting
    hyper-parameters are reused for the 6 h and 12 h horizons to save compute.

    Parameters
    ----------
    train_data, val_data : pd.DataFrame
        Must contain *feature_cols* **and** columns named
        ``aki_label_{horizon}h`` for each horizon.
    feature_cols : list[str]
        Stage-2 (full) feature column names.
    horizons : list[int], optional
        Prediction horizons in hours.  Default ``[6, 12, 24]``.
    n_optuna_trials : int, optional
        Number of Optuna trials for LightGBM tuning.  Default **50**.

    Returns
    -------
    dict[int, dict[str, dict[str, Any]]]
        Nested dict::

            {
                horizon: {
                    'lgbm': {'model': LGBMClassifier, 'calibrator': IsotonicRegression},
                    'xgb':  {'model': XGBClassifier,  'calibrator': IsotonicRegression},
                }
            }

        Horizons with insufficient positive samples are **omitted**.
    """
    if horizons is None:
        horizons = [6, 12, 24]

    results: dict[int, dict[str, dict[str, Any]]] = {}
    X_train = train_data[feature_cols]
    X_val = val_data[feature_cols]

    # Missing-feature robustness: fit on clean + randomly-masked copies so the model
    # degrades gracefully when a hospital lacks labs. Validation/calibration stay
    # CLEAN (we calibrate for the real, fully-observed distribution).
    X_fit = X_train
    _augmented = feature_dropout and feature_dropout > 0.0
    if _augmented:
        logger.info("Feature-dropout augmentation: p=%.2f (train set doubled).", feature_dropout)
        X_fit = _apply_feature_dropout(X_train, feature_dropout)

    # ------------------------------------------------------------------
    # Optuna tuning on 24h horizon (reused for other horizons)
    # ------------------------------------------------------------------
    tuned_lgbm_params: dict[str, Any] | None = None
    tuning_horizon = 24

    label_col_24 = f"aki_label_{tuning_horizon}h"
    if label_col_24 in train_data.columns:
        y_train_24 = train_data[label_col_24].values
        y_val_24 = val_data[label_col_24].values
        if _check_sufficient_positives(
            y_train_24, context=f"Stage-2 Optuna {tuning_horizon}h"
        ) and _check_sufficient_positives(
            y_val_24, context=f"Stage-2 Optuna {tuning_horizon}h (val)"
        ):
            logger.info(
                "Running Optuna tuning (%d trials) on %dh horizon …",
                n_optuna_trials,
                tuning_horizon,
            )
            tuned_lgbm_params = optuna_tune_lgbm(
                X_train, y_train_24, X_val, y_val_24, n_trials=n_optuna_trials
            )
    else:
        logger.warning(
            "Label column '%s' not found – Optuna tuning skipped.", label_col_24
        )

    # ------------------------------------------------------------------
    # Train per-horizon models
    # ------------------------------------------------------------------
    for h in horizons:
        label_col = f"aki_label_{h}h"
        logger.info("Stage 2 – training %dh horizon models …", h)

        if label_col not in train_data.columns:
            logger.warning("Label column '%s' not found – skipping.", label_col)
            continue

        y_train = train_data[label_col].values
        y_val = val_data[label_col].values

        if not _check_sufficient_positives(y_train, context=f"Stage-2 {h}h (train)"):
            continue
        if not _check_sufficient_positives(y_val, context=f"Stage-2 {h}h (val)"):
            continue

        # match y to the (possibly doubled) fit set; masked copy shares the label
        y_fit = np.concatenate([y_train, y_train]) if _augmented else y_train

        # ---- LightGBM ------------------------------------------------
        lgbm_model = train_lgbm(
            X_fit, y_fit, X_val, y_val, params=tuned_lgbm_params
        )
        lgbm_calibrator = calibrate_model(lgbm_model, X_val, y_val)

        # ---- XGBoost --------------------------------------------------
        xgb_model = train_xgb(X_fit, y_fit, X_val, y_val)
        xgb_calibrator = calibrate_model(xgb_model, X_val, y_val)

        results[h] = {
            "lgbm": {"model": lgbm_model, "calibrator": lgbm_calibrator},
            "xgb": {"model": xgb_model, "calibrator": xgb_calibrator},
        }

    return results


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_models(
    stage1_models: dict[int, dict[str, Any]],
    stage2_models: dict[int, dict[str, dict[str, Any]]],
    output_dir: str,
) -> None:
    """Serialise all trained models and calibrators to *output_dir*.

    Directory layout::

        output_dir/
        ├── stage1_lgbm_6h.pkl
        ├── stage1_lgbm_12h.pkl
        ├── stage1_lgbm_24h.pkl
        ├── stage2_lgbm_6h.pkl
        ├── stage2_lgbm_12h.pkl
        ├── stage2_lgbm_24h.pkl
        ├── stage2_xgb_6h.pkl
        ├── stage2_xgb_12h.pkl
        ├── stage2_xgb_24h.pkl
        └── calibrators/
            ├── stage1_lgbm_6h_calibrator.pkl
            ├── …
            ├── stage2_lgbm_6h_calibrator.pkl
            └── stage2_xgb_24h_calibrator.pkl

    Parameters
    ----------
    stage1_models : dict
        Output of :func:`train_stage1_models`.
    stage2_models : dict
        Output of :func:`train_stage2_models`.
    output_dir : str
        Root directory for saved artefacts.
    """
    out = Path(output_dir)
    cal_dir = out / "calibrators"
    out.mkdir(parents=True, exist_ok=True)
    cal_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1
    for h, artefacts in stage1_models.items():
        model_path = out / f"stage1_lgbm_{h}h.pkl"
        cal_path = cal_dir / f"stage1_lgbm_{h}h_calibrator.pkl"
        joblib.dump(artefacts["model"], model_path)
        joblib.dump(artefacts["calibrator"], cal_path)
        logger.info("Saved Stage-1 %dh model → %s", h, model_path)

    # Stage 2
    for h, families in stage2_models.items():
        for family_name in ("lgbm", "xgb"):
            if family_name not in families:
                continue
            model_path = out / f"stage2_{family_name}_{h}h.pkl"
            cal_path = cal_dir / f"stage2_{family_name}_{h}h_calibrator.pkl"
            joblib.dump(families[family_name]["model"], model_path)
            joblib.dump(families[family_name]["calibrator"], cal_path)
            logger.info("Saved Stage-2 %s %dh model → %s", family_name, h, model_path)


def load_models(
    output_dir: str,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, dict[str, Any]]]]:
    """Deserialise models and calibrators previously saved by :func:`save_models`.

    Parameters
    ----------
    output_dir : str
        Root directory containing ``.pkl`` artefacts.

    Returns
    -------
    tuple[dict, dict]
        ``(stage1_models, stage2_models)`` in the same nested-dict format
        produced by :func:`train_stage1_models` and :func:`train_stage2_models`.

    Raises
    ------
    FileNotFoundError
        If *output_dir* does not exist.
    """
    out = Path(output_dir)
    if not out.exists():
        raise FileNotFoundError(f"Model directory not found: {output_dir}")

    cal_dir = out / "calibrators"

    # Stage 1
    stage1: dict[int, dict[str, Any]] = {}
    for model_path in sorted(out.glob("stage1_lgbm_*h.pkl")):
        # Extract horizon from filename, e.g. "stage1_lgbm_6h.pkl" → 6
        stem = model_path.stem  # "stage1_lgbm_6h"
        h = int(stem.split("_")[-1].replace("h", ""))
        cal_path = cal_dir / f"stage1_lgbm_{h}h_calibrator.pkl"

        model = joblib.load(model_path)
        calibrator = joblib.load(cal_path) if cal_path.exists() else None
        stage1[h] = {"model": model, "calibrator": calibrator}
        logger.info("Loaded Stage-1 %dh model from %s", h, model_path)

    # Stage 2
    stage2: dict[int, dict[str, dict[str, Any]]] = {}
    for family_name in ("lgbm", "xgb"):
        for model_path in sorted(out.glob(f"stage2_{family_name}_*h.pkl")):
            stem = model_path.stem
            h = int(stem.split("_")[-1].replace("h", ""))
            cal_path = cal_dir / f"stage2_{family_name}_{h}h_calibrator.pkl"

            model = joblib.load(model_path)
            calibrator = joblib.load(cal_path) if cal_path.exists() else None

            if h not in stage2:
                stage2[h] = {}
            stage2[h][family_name] = {"model": model, "calibrator": calibrator}
            logger.info(
                "Loaded Stage-2 %s %dh model from %s", family_name, h, model_path
            )

    return stage1, stage2
