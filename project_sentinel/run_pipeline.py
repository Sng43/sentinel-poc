import argparse
import logging
import sys
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("outputs")

def run_phase2():
    logger.info("--- Running Phase 2: Data Loading & Cohort Definition ---")
    try:
        from project_sentinel.src.data_loader import load_physionet_2019, define_cohort
        df = load_physionet_2019(str(DATA_DIR))
        cohort_df, baseline_df = define_cohort(df, output_dir=str(INTERIM_DIR))
        logger.info("Phase 2 completed successfully.")
    except ImportError as e:
        logger.warning(f"Skipping Phase 2 due to missing module: {e}")
    except Exception as e:
        logger.error(f"Phase 2 execution failed: {e}")

def run_phase3():
    logger.info("--- Running Phase 3: Prospective Horizon Labeling ---")
    try:
        from project_sentinel.src.labels import label_dataset, print_label_distribution
        cohort_path = INTERIM_DIR / "cohort_defined.parquet"
        baseline_path = INTERIM_DIR / "baseline_creatinine.parquet"
        
        if not cohort_path.exists() or not baseline_path.exists():
            logger.warning("Phase 3 prerequisites not found. Skipping.")
            return

        cohort_df = pd.read_parquet(cohort_path)
        baseline_df = pd.read_parquet(baseline_path)
        
        horizons = [6, 12, 24]
        labeled_df = label_dataset(cohort_df, baseline_df, horizons=horizons)
        print_label_distribution(labeled_df, horizons=horizons)
        
        out_path = INTERIM_DIR / "labeled_cohort.parquet"
        labeled_df.to_parquet(out_path, index=False)
        logger.info(f"Phase 3 completed successfully. Saved to {out_path}")
    except ImportError as e:
        logger.warning(f"Skipping Phase 3 due to missing module: {e}")
    except Exception as e:
        logger.error(f"Phase 3 execution failed: {e}")

def run_phase4():
    logger.info("--- Running Phase 4: Feature Engineering ---")
    try:
        from project_sentinel.src.features import engineer_features, temporal_split, create_feature_manifest
        labeled_path = INTERIM_DIR / "labeled_cohort.parquet"
        baseline_path = INTERIM_DIR / "baseline_creatinine.parquet"
        
        if not labeled_path.exists() or not baseline_path.exists():
            logger.warning("Phase 4 prerequisites not found. Skipping.")
            return
            
        labeled_df = pd.read_parquet(labeled_path)
        baseline_df = pd.read_parquet(baseline_path)
        
        features_df = engineer_features(labeled_df, baseline_df)
        
        manifest_path = PROCESSED_DIR / "feature_manifest.csv"
        create_feature_manifest(features_df, manifest_path)
        
        if "valid_for_training" in features_df.columns:
            features_df = features_df[features_df["valid_for_training"]]
            
        splits = temporal_split(
            features_df, 
            val_ratio=0.15, 
            test_ratio=0.15, 
            output_dir=str(PROCESSED_DIR),
            aki_label_col="aki_label_6h"
        )
        logger.info("Phase 4 completed successfully.")
    except ImportError as e:
        logger.warning(f"Skipping Phase 4 due to missing module: {e}")
    except Exception as e:
        logger.error(f"Phase 4 execution failed: {e}")

def run_phase5():
    logger.info("--- Running Phase 5: Modeling ---")
    try:
        from project_sentinel.src.models import train_stage1_models, train_stage2_models, save_models
        from project_sentinel.src.features import get_stage1_features, get_all_features
        
        train_path = PROCESSED_DIR / "train.parquet"
        val_path = PROCESSED_DIR / "val.parquet"
        
        if not train_path.exists() or not val_path.exists():
            logger.warning("Phase 5 prerequisites not found. Skipping.")
            return
            
        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)
        
        stage1_feats = get_stage1_features()
        all_feats = get_all_features(train_df)
        
        logger.info("Training Stage 1...")
        stage1_models = train_stage1_models(
            train_df, val_df, 
            features=stage1_feats, 
            target_col="aki_label_6h", 
            tune=False
        )
        save_models(stage1_models, MODELS_DIR / "stage1", prefix="stage1")
        
        logger.info("Training Stage 2...")
        stage2_models = train_stage2_models(
            train_df, val_df, 
            features=all_feats, 
            target_cols=["aki_label_6h", "aki_label_12h", "aki_label_24h"],
            tune=False
        )
        for t, models_dict in stage2_models.items():
            save_models(models_dict, MODELS_DIR / f"stage2_{t}", prefix=f"stage2_{t}")
            
        logger.info("Phase 5 completed successfully.")
    except ImportError as e:
        logger.warning(f"Skipping Phase 5 due to missing module: {e}")
    except Exception as e:
        logger.error(f"Phase 5 execution failed: {e}")

def run_phase6():
    logger.info("--- Running Phase 6: Evaluation ---")
    try:
        from project_sentinel.src.evaluation import run_full_evaluation
        from project_sentinel.src.models import load_models
        from project_sentinel.src.features import get_stage1_features
        
        test_path = PROCESSED_DIR / "test.parquet"
        if not test_path.exists():
            logger.warning("Phase 6 prerequisites not found. Skipping.")
            return
            
        test_df = pd.read_parquet(test_path)
        
        try:
            stage1_models = load_models(MODELS_DIR / "stage1")
            stage1_feats = get_stage1_features()
            X_test = test_df[stage1_feats]
            y_test = test_df["aki_label_6h"].values
            run_full_evaluation(stage1_models, X_test, y_test, output_dir=str(OUTPUTS_DIR / "stage1"))
            logger.info("Phase 6 completed successfully.")
        except Exception as inner_e:
            logger.warning(f"Could not evaluate models: {inner_e}")
            
    except ImportError as e:
        logger.warning(f"Skipping Phase 6 due to missing module: {e}")
    except Exception as e:
        logger.error(f"Phase 6 execution failed: {e}")

def run_phase7():
    logger.info("--- Running Phase 7: Explainability / Model Export ---")
    try:
        from project_sentinel.src.explain import run_explainability
        from project_sentinel.src.models import load_models
        from project_sentinel.src.features import get_stage1_features
        
        test_path = PROCESSED_DIR / "test.parquet"
        if not test_path.exists():
            logger.warning("Phase 7 prerequisites not found. Skipping.")
            return
            
        test_df = pd.read_parquet(test_path)
        
        try:
            stage1_models = load_models(MODELS_DIR / "stage1")
            stage1_feats = get_stage1_features()
            # We assume LGBM is available
            if "lgbm" in stage1_models:
                run_explainability(
                    model=stage1_models["lgbm"],
                    df=test_df,
                    features=stage1_feats,
                    target_col="aki_label_6h",
                    output_dir=str(OUTPUTS_DIR / "explain_stage1")
                )
                logger.info("Phase 7 completed successfully.")
            else:
                logger.warning("LGBM model not found for explainability.")
        except Exception as inner_e:
            logger.warning(f"Could not run explainability: {inner_e}")
            
    except ImportError as e:
        logger.warning(f"Skipping Phase 7 due to missing module: {e}")
    except Exception as e:
        logger.error(f"Phase 7 execution failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Master runner script for Sentinel POC Pipeline.")
    parser.add_argument("--phase", type=int, choices=[2, 3, 4, 5, 6, 7], help="Run a specific phase.")
    parser.add_argument("--all", action="store_true", help="Run all phases sequentially.")
    args = parser.parse_args()

    if not args.phase and not args.all:
        parser.print_help()
        sys.exit(1)

    phases_to_run = [2, 3, 4, 5, 6, 7] if args.all else [args.phase]

    for phase in phases_to_run:
        if phase == 2:
            run_phase2()
        elif phase == 3:
            run_phase3()
        elif phase == 4:
            run_phase4()
        elif phase == 5:
            run_phase5()
        elif phase == 6:
            run_phase6()
        elif phase == 7:
            run_phase7()

if __name__ == "__main__":
    main()
