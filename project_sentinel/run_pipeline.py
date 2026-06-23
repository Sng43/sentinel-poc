"""Master runner for the Project Sentinel SA-AKI pipeline (MIMIC-IV demo).

Mirrors notebooks 01–07 as a single command-line pipeline. Each phase reads the
artifacts written by the previous one, so phases can be run individually (given
their prerequisites exist) or all at once.

Usage
-----
    uv run python run_pipeline.py --all
    uv run python run_pipeline.py --phase 3
    uv run python run_pipeline.py --from 5            # run phases 5..7
    uv run python run_pipeline.py --data-dir data/raw/mimic_iv_demo --all

Phases
------
    1  acquisition   load MIMIC-IV tables → hourly cohort → sepsis onset
    2  cohort        inclusion criteria → baseline creatinine → prevalent-AKI exclusion
    3  features      missingness → rolling → impute → velocity → composites
    4  labels        KDIGO labels (creatinine + urine) → chronological split
    5  train         Stage-1 + Stage-2 models (+ Optuna, isotonic calibration)
    6  evaluate      AUROC/AUPRC/ECE, ROC/PR/calibration/DCA, subgroups
    7  explain       SHAP summaries, waterfalls, clinical-alert JSON

Note: phases 5–7 import LightGBM/XGBoost/SHAP, which need OpenMP (`libomp`).
Phases 1–4 have no such dependency.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("run_pipeline")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
RAW_MIMIC = DATA_DIR / "raw" / "mimic_iv_demo"
INTERIM = DATA_DIR / "interim"
PROCESSED = DATA_DIR / "processed"
MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("outputs")

HORIZONS = [6, 12, 24]


def _require(path: Path, phase: int) -> bool:
    if not path.exists():
        logger.warning("Phase %d prerequisite missing: %s — skipping.", phase, path)
        return False
    return True


# ---------------------------------------------------------------------------
# Phase 1 — Data acquisition
# ---------------------------------------------------------------------------
def run_phase1(data_dir: Path) -> None:
    logger.info("=== Phase 1: Data acquisition (MIMIC-IV demo) ===")
    from src.data_loader import (
        load_mimic_demo, build_hourly_cohort, identify_sepsis_onset, print_data_summary,
    )

    if not _require(data_dir, 1):
        logger.warning("Run: python ../mimic_demo_project/download_mimic_demo.py %s", data_dir)
        return

    tables = load_mimic_demo(str(data_dir))
    hourly = build_hourly_cohort(tables)
    hourly = identify_sepsis_onset(hourly, tables)
    print_data_summary(hourly)

    INTERIM.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(INTERIM / "hourly_cohort.parquet", index=False)
    logger.info("Phase 1 done → %s", INTERIM / "hourly_cohort.parquet")


# ---------------------------------------------------------------------------
# Phase 2 — Cohort definition
# ---------------------------------------------------------------------------
def run_phase2() -> None:
    logger.info("=== Phase 2: Cohort definition ===")
    from src.data_loader import apply_inclusion_criteria, define_cohort

    src = INTERIM / "hourly_cohort.parquet"
    if not _require(src, 2):
        return

    hourly = pd.read_parquet(src)
    included = apply_inclusion_criteria(hourly, min_age=18, min_hours=8)
    cohort_df, baseline_df = define_cohort(included)

    INTERIM.mkdir(parents=True, exist_ok=True)
    cohort_df.to_parquet(INTERIM / "cohort_defined.parquet", index=False)
    baseline_df.to_parquet(INTERIM / "baseline_creatinine.parquet", index=False)
    logger.info("Phase 2 done → cohort_defined.parquet + baseline_creatinine.parquet")


# ---------------------------------------------------------------------------
# Phase 3 — Feature engineering
# ---------------------------------------------------------------------------
def run_phase3() -> None:
    logger.info("=== Phase 3: Feature engineering ===")
    from src.features import engineer_features, create_feature_manifest

    cohort_path = INTERIM / "cohort_defined.parquet"
    baseline_path = INTERIM / "baseline_creatinine.parquet"
    if not (_require(cohort_path, 3) and _require(baseline_path, 3)):
        return

    cohort_df = pd.read_parquet(cohort_path)
    baseline_df = pd.read_parquet(baseline_path)

    features_df = engineer_features(cohort_df, baseline_df)
    INTERIM.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(INTERIM / "engineered_features.parquet", index=False)
    create_feature_manifest(features_df, INTERIM / "feature_manifest.csv")
    logger.info("Phase 3 done → engineered_features.parquet (%d cols)", features_df.shape[1])


# ---------------------------------------------------------------------------
# Phase 4 — Label engineering + split
# ---------------------------------------------------------------------------
def run_phase4() -> None:
    logger.info("=== Phase 4: Label engineering + split ===")
    from src.labels import label_dataset, print_label_distribution
    from src.features import temporal_split

    cohort_path = INTERIM / "cohort_defined.parquet"
    features_path = INTERIM / "engineered_features.parquet"
    baseline_path = INTERIM / "baseline_creatinine.parquet"
    if not (_require(cohort_path, 4) and _require(features_path, 4) and _require(baseline_path, 4)):
        return

    cohort_df = pd.read_parquet(cohort_path)        # raw measurements (un-imputed)
    features_df = pd.read_parquet(features_path)    # engineered (imputed) features
    baseline_df = pd.read_parquet(baseline_path)

    # Derive KDIGO labels from RAW creatinine/urine, never the imputed features —
    # ffill+bfill imputation would leak future measurements into the label.
    labeled = label_dataset(cohort_df, baseline_df, horizons=HORIZONS)
    print_label_distribution(labeled, HORIZONS)

    # Attach the label columns onto the engineered features by (patient_id, hour).
    label_cols = ["patient_id", "ICULOS", "_aki_current", "valid_for_training"] + [
        f"aki_label_{h}h" for h in HORIZONS
    ]
    merged = features_df.merge(labeled[label_cols], on=["patient_id", "ICULOS"], how="inner")

    valid = merged[merged["valid_for_training"]].copy()
    logger.info("Valid training rows: %d of %d", len(valid), len(merged))

    PROCESSED.mkdir(parents=True, exist_ok=True)
    temporal_split(
        valid, val_ratio=0.15, test_ratio=0.15,
        output_dir=str(PROCESSED), aki_label_col="aki_label_6h",
    )
    logger.info("Phase 4 done → train/val/test/site_holdout in %s", PROCESSED)


# ---------------------------------------------------------------------------
# Phase 5 — Model training
# ---------------------------------------------------------------------------
def run_phase5(n_optuna_trials: int = 50) -> None:
    logger.info("=== Phase 5: Model training ===")
    from src.models import train_stage1_models, train_stage2_models, save_models
    from src.features import get_stage1_features, get_all_features

    train_path = PROCESSED / "train.parquet"
    val_path = PROCESSED / "val.parquet"
    if not (_require(train_path, 5) and _require(val_path, 5)):
        return

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    stage1_feats = get_stage1_features()
    stage2_feats = get_all_features(train_df)

    stage1 = train_stage1_models(train_df, val_df, feature_cols=stage1_feats, horizons=HORIZONS)
    stage2 = train_stage2_models(
        train_df, val_df, feature_cols=stage2_feats,
        horizons=HORIZONS, n_optuna_trials=n_optuna_trials,
    )

    save_models(stage1, stage2, str(MODELS_DIR))
    logger.info("Phase 5 done → models in %s", MODELS_DIR)


# ---------------------------------------------------------------------------
# Phase 6 — Evaluation
# ---------------------------------------------------------------------------
def run_phase6() -> None:
    logger.info("=== Phase 6: Evaluation ===")
    from src.evaluation import run_full_evaluation
    from src.models import load_models
    from src.features import get_all_features

    test_path = PROCESSED / "test.parquet"
    if not (_require(test_path, 6) and _require(MODELS_DIR, 6)):
        return

    test_df = pd.read_parquet(test_path)
    _stage1, stage2 = load_models(str(MODELS_DIR))
    if not stage2:
        logger.warning("No Stage-2 models found — skipping evaluation.")
        return

    s2_feats = get_all_features(test_df)
    h = 24 if 24 in stage2 else sorted(stage2)[-1]
    models_dict = {
        f"stage2_lgbm_{h}h": stage2[h]["lgbm"]["model"],
        f"stage2_xgb_{h}h": stage2[h]["xgb"]["model"],
    }
    subgroups = (test_df["Age"] >= 65).map({True: "elderly", False: "adult"})
    run_full_evaluation(
        models_dict, test_df[s2_feats], test_df[f"aki_label_{h}h"].values,
        subgroups=subgroups, output_dir=str(OUTPUTS_DIR),
    )
    logger.info("Phase 6 done → %s", OUTPUTS_DIR)


# ---------------------------------------------------------------------------
# Phase 7 — Explainability
# ---------------------------------------------------------------------------
def run_phase7() -> None:
    logger.info("=== Phase 7: Explainability ===")
    from src.explain import run_explainability
    from src.models import load_models
    from src.features import get_all_features

    test_path = PROCESSED / "test.parquet"
    if not (_require(test_path, 7) and _require(MODELS_DIR, 7)):
        return

    test_df = pd.read_parquet(test_path)
    _stage1, stage2 = load_models(str(MODELS_DIR))
    if not stage2:
        logger.warning("No Stage-2 models found — skipping explainability.")
        return

    s2_feats = get_all_features(test_df)
    h = 24 if 24 in stage2 else sorted(stage2)[-1]
    model = stage2[h]["lgbm"]["model"]
    y_prob = model.predict_proba(test_df[s2_feats])[:, 1]

    run_explainability(
        model=model, X_test=test_df[s2_feats], y_true=test_df[f"aki_label_{h}h"].values,
        y_prob=y_prob, feature_names=s2_feats,
        output_dir=str(OUTPUTS_DIR / f"explain_stage2_{h}h"),
    )
    logger.info("Phase 7 done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
_PHASES = {1: run_phase1, 2: run_phase2, 3: run_phase3, 4: run_phase4,
           5: run_phase5, 6: run_phase6, 7: run_phase7}


def main() -> None:
    parser = argparse.ArgumentParser(description="Project Sentinel SA-AKI pipeline runner.")
    parser.add_argument("--phase", type=int, choices=range(1, 8), help="Run a single phase.")
    parser.add_argument("--from", dest="from_phase", type=int, choices=range(1, 8),
                        help="Run from this phase through phase 7.")
    parser.add_argument("--all", action="store_true", help="Run all phases 1–7.")
    parser.add_argument("--data-dir", default=str(RAW_MIMIC), help="MIMIC-IV demo directory.")
    parser.add_argument("--optuna-trials", type=int, default=50,
                        help="Optuna trials for Stage-2 tuning (phase 5).")
    args = parser.parse_args()

    if args.all:
        phases = list(range(1, 8))
    elif args.from_phase:
        phases = list(range(args.from_phase, 8))
    elif args.phase:
        phases = [args.phase]
    else:
        parser.print_help()
        sys.exit(1)

    for p in phases:
        try:
            if p == 1:
                run_phase1(Path(args.data_dir))
            elif p == 5:
                run_phase5(n_optuna_trials=args.optuna_trials)
            else:
                _PHASES[p]()
        except Exception:  # keep going so one phase's failure doesn't abort the rest
            logger.exception("Phase %d failed.", p)


if __name__ == "__main__":
    main()
