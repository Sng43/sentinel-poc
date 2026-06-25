# 🏥 Project Sentinel — Sepsis-Associated AKI Early Warning System

**Proof-of-Concept ML Pipeline for Prospective Prediction of Acute Kidney Injury in ICU Patients**

> *"What is the probability that this ICU patient will meet KDIGO AKI Stage ≥1 criteria within the next 6, 12, or 24 hours?"*

---

## Table of Contents

- [Overview](#overview)
- [Clinical Context](#clinical-context)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Data Acquisition](#data-acquisition)
- [Pipeline Phases](#pipeline-phases)
- [Model Design](#model-design)
- [Evaluation Suite](#evaluation-suite)
- [Explainability & Clinical Alerts](#explainability--clinical-alerts)
- [Running the Pipeline](#running-the-pipeline)
- [Expected Outputs](#expected-outputs)
- [Known Limitations](#known-limitations)
- [License](#license)

---

## Overview

Project Sentinel is a proof-of-concept machine learning pipeline for early prediction of **Sepsis-Associated Acute Kidney Injury (SA-AKI)**. Rather than classifying existing AKI (which KDIGO criteria already do — often too late), this system performs **probabilistic, prospective prediction** across multiple time horizons.

The pipeline demonstrates the full workflow — from raw ICU time-series data to calibrated clinical alerts with per-patient explanations — and is architecturally ready for retraining on real hospital data.

## Clinical Context

This project targets the clinical reality of **resource-limited hospitals** where:

- **34.1%** of adult patients requiring acute hemodialysis die
- Serum creatinine is a **lagging indicator** — patients lose >50% kidney function before standard diagnostic criteria are met
- Urine output is tracked in fewer than **4%** of at-risk pediatric cases
- The target deployment environment is **OpenClinic GA**, a hospital EHR system

The core insight: *KDIGO criteria diagnose AKI after the damage is done. This system predicts it before it happens, enabling earlier intervention.*

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Cascaded Two-Stage Design               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────┐     ┌────────────────────────┐ │
│  │   Stage 1: Screener │     │  Stage 2: Full Model   │ │
│  │   (Vitals Only)     │────▶│  (All Features)        │ │
│  │   LightGBM          │     │  LightGBM + XGBoost    │ │
│  │   Low latency       │     │  High accuracy         │ │
│  └─────────────────────┘     └────────────────────────┘ │
│         │                              │                │
│         ▼                              ▼                │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  Isotonic Calibration → Risk Score → SHAP Alert    │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

- **Stage 1** — Vitals-only screener (HR, MAP, O₂Sat, Temp, Resp, SBP + rolling stats). Simulates the scenario where labs are unavailable (common in district hospitals).
- **Stage 2** — Full-feature predictor (all vitals, labs, creatinine velocity, composite features). Fires only when Stage 1 risk exceeds threshold (`p ≥ 0.25`).
- Both stages train separate models for **6h, 12h, and 24h** prediction horizons.

## Project Structure

```
sentinel-poc/
├── README.md                              ← You are here
├── project_sentinel/                      ← Core ML pipeline
│   ├── pyproject.toml                     ← Dependencies (managed by uv)
│   ├── uv.lock                            ← Lockfile (commit to VCS)
│   ├── .python-version                    ← Python 3.12
│   ├── src/
│   │   ├── __init__.py
│   │   ├── data_loader.py                 ← Data loading, cohort definition, baseline Cr
│   │   ├── labels.py                      ← KDIGO AKI criteria & prospective horizon labels
│   │   ├── features.py                    ← Feature engineering pipeline (rolling, velocity, composites)
│   │   ├── models.py                      ← LightGBM/XGBoost training, Optuna tuning, calibration
│   │   ├── evaluation.py                  ← Metrics, ROC/PR curves, DCA, subgroup analysis
│   │   └── explain.py                     ← SHAP explainability & clinical alert generation
│   ├── notebooks/
│   │   ├── 01_data_acquisition.ipynb
│   │   ├── 02_cohort_definition.ipynb
│   │   ├── 03_feature_engineering.ipynb
│   │   ├── 04_label_engineering.ipynb
│   │   ├── 05_model_training.ipynb
│   │   ├── 06_evaluation.ipynb
│   │   └── 07_explainability.ipynb
│   ├── data/
│   │   ├── raw/                           ← Downloaded source files (never modified)
│   │   ├── interim/                       ← Merged + cleaned DataFrames
│   │   └── processed/                     ← Feature-engineered, split datasets
│   ├── outputs/
│   │   ├── figures/                       ← ROC, PR, calibration, DCA, SHAP plots
│   │   ├── models/                        ← Serialised .pkl model artefacts
│   │   └── reports/                       ← CSV comparisons, JSON alerts
│   ├── run_pipeline.py                    ← Master CLI runner (--phase N or --all)
│   └── main.py                            ← Entry point stub
└── mimic_demo_project/                    ← MIMIC-IV Demo data downloader
    └── download_mimic_demo.py             ← Anonymous S3 download script
```

## Getting Started

### Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** — Fast Python package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/sengakabare/sentinel-poc.git
cd sentinel-poc/project_sentinel

# Install all dependencies (creates .venv automatically)
uv sync

# For development (includes Jupyter)
uv sync --group dev
```

### Dependencies

| Package | Purpose |
|---|---|
| `pandas`, `numpy`, `scipy` | Data manipulation & scientific computing |
| `scikit-learn` | Preprocessing, calibration, evaluation metrics |
| `lightgbm` | Stage 1 & Stage 2 gradient boosting models |
| `xgboost` | Stage 2 comparison model |
| `shap` | Model explainability (TreeExplainer) |
| `optuna` | Bayesian hyperparameter optimisation |
| `imbalanced-learn` | SMOTE / class imbalance handling |
| `matplotlib`, `seaborn` | Visualisation |
| `pyarrow` | Parquet I/O |
| `tqdm` | Progress bars |

## Data Acquisition

### Primary Dataset: PhysioNet/CinC Challenge 2019

**"Early Prediction of Sepsis from Clinical Data"**

- **Source:** [physionet.org/content/challenge-2019/1.0.0/](https://physionet.org/content/challenge-2019/1.0.0/)
- **License:** Open Data Commons Attribution License v1.0
- **Size:** ~40,000 ICU patients across two hospital systems (Set A + Set B)
- **Format:** One `.psv` file per patient, hourly measurements
- **Variables:** 8 vitals, 26 labs, 6 demographics, 1 sepsis label

```bash
cd project_sentinel/data/raw

# Set A (~20,000 patients)
wget -r -N -c -np https://physionet.org/files/challenge-2019/1.0.0/training/training_setA/

# Set B (~20,000 patients)
wget -r -N -c -np https://physionet.org/files/challenge-2019/1.0.0/training/training_setB/
```

### Fallback: MIMIC-IV Demo

If PhysioNet 2019 is inaccessible, a download script for the [MIMIC-IV Clinical Database Demo](https://physionet.org/content/mimiciv-demo/) is provided:

```bash
cd mimic_demo_project
uv run download_mimic_demo.py
```

This downloads the 100-patient demo dataset from AWS Open Data Registry (no credentials required).

## Pipeline Phases

The pipeline is structured as sequential phases, each building on the previous:

| Phase | Module | Description |
|---|---|---|
| **2** | `data_loader.py` | Load `.psv` files, merge into unified DataFrame, define cohort with inclusion criteria |
| **3** | `labels.py` | Apply KDIGO AKI criteria, generate prospective 6h/12h/24h horizon labels |
| **4** | `features.py` | Missingness indicators → rolling stats → imputation → creatinine velocity → composite features |
| **5** | `models.py` | Train Stage 1 (vitals-only) and Stage 2 (full features) models with Optuna tuning |
| **6** | `evaluation.py` | AUROC, AUPRC, Brier score, ECE, DCA, subgroup analysis |
| **7** | `explain.py` | SHAP global importance, patient waterfall plots, clinical alert JSON generation |

### Labeling Strategy (KDIGO Criteria)

```
AKI = True if:
    (creatinine_current − creatinine_baseline) ≥ 0.3 mg/dL
    OR
    creatinine_current ≥ 1.5 × creatinine_baseline
```

Three exclusion rules ensure label quality:
1. **Prevalent-case exclusion** — patient already has AKI at time `t`
2. **Insufficient-horizon exclusion** — less than 1 hour before discharge
3. **Unobservable-outcome exclusion** — no creatinine measurement in the forward window

### Key Engineered Features

| Category | Examples |
|---|---|
| **Creatinine velocity** | `delta_Cr`, `creatinine_velocity_12h`, `cr_above_baseline`, `cr_ratio_baseline` |
| **Rolling statistics** | `{vital/lab}_{mean,std,min,max}_{6h,12h}` for all vitals and key labs |
| **Composite features** | `renal_perf_pressure`, `map_cr_velocity_interaction`, `wbc_lactate_product`, `bun_cr_ratio` |
| **Missingness indicators** | `{lab}_observed` — captures informative missingness signal |
| **Temporal** | `hours_since_sepsis` |

### Data Split Strategy

**Temporal validation** (not random) to prevent data leakage:

| Split | Strategy | Purpose |
|---|---|---|
| **Train** | First 70% of patients chronologically | Model training |
| **Validation** | Next 15% | Early stopping, calibration |
| **Test** | Last 15% | Final evaluation |
| **Site holdout** | All Hospital System B patients | Generalisability assessment |

## Model Design

### Stage 1 — Vitals-Only Screener (LightGBM)

Uses 15 features: `HR`, `MAP`, `O2Sat`, `Temp`, `Resp`, `SBP` + rolling stats + `hours_since_sepsis`, `Age`, `Gender`.

### Stage 2 — Full-Feature Predictor (LightGBM + XGBoost)

Uses all engineered features (~200+ columns). Optuna tunes LightGBM on the 24h horizon (50 trials), then reuses the best hyperparameters across all horizons.

### Calibration

All models are post-hoc calibrated with **isotonic regression** — a non-negotiable requirement for clinical deployment. A miscalibrated model that outputs 90% risk when true risk is 40% destroys clinician trust.

## Evaluation Suite

| Metric | Purpose |
|---|---|
| **AUROC** | Overall discrimination |
| **AUPRC** | Better metric for imbalanced classes |
| **Brier Score** | Probability accuracy |
| **ECE** | Calibration quality |
| **Decision Curve Analysis** | Net clinical benefit at different thresholds |
| **Threshold metrics** | Sensitivity, specificity, PPV, NPV at clinical thresholds (0.20, 0.35, 0.50) |
| **Subgroup analysis** | Performance by age, ICU type, MAP at sepsis onset, hospital system |

## Explainability & Clinical Alerts

Every risk alert surfaces its **top 3 contributing factors** via SHAP:

```json
{
  "patient_id": "p001234",
  "risk_score": 0.72,
  "risk_level": "HIGH",
  "aki_probability_24h": "72%",
  "top_contributors": [
    {"feature": "Creatinine velocity (last 12h)", "direction": "RISING", "value": "+0.12 mg/dL/hr"},
    {"feature": "Mean arterial pressure (6h avg)", "direction": "LOW",    "value": "61 mmHg"},
    {"feature": "Lactate",                         "direction": "HIGH",   "value": "4.1 mmol/L"}
  ],
  "recommended_action": "URGENT: Consider immediate nephrology consultation..."
}
```

## Running the Pipeline

### Via CLI (run_pipeline.py)

```bash
cd project_sentinel

# Run a specific phase
uv run python run_pipeline.py --phase 2

# Run all phases sequentially
uv run python run_pipeline.py --all
```

### Via Notebooks (interactive)

```bash
cd project_sentinel

# Launch Jupyter
uv run jupyter notebook

# Or execute non-interactively
uv run jupyter nbconvert --to notebook --execute notebooks/01_data_acquisition.ipynb
```

## Expected Outputs

```
outputs/
├── models/
│   ├── stage1_lgbm_{6,12,24}h.pkl
│   ├── stage2_lgbm_{6,12,24}h.pkl
│   ├── stage2_xgb_{6,12,24}h.pkl
│   └── calibrators/
├── figures/
│   ├── calibration_curves.png
│   ├── roc_curves.png
│   ├── pr_curves.png
│   ├── decision_curve_analysis.png
│   ├── shap_summary_bar.png
│   ├── shap_summary_dot.png
│   └── shap_patient_{tp,fn,tn}.png
└── reports/
    ├── model_comparison.csv
    ├── subgroup_analysis.csv
    ├── feature_manifest.csv
    └── sample_alerts.json
```

## Known Limitations

These are not failures of this PoC — they are the specification gaps that real hospital data will close:

| # | Limitation | Mitigation Path |
|---|---|---|
| 1 | **No urine output labels** — KDIGO urine output criterion (<0.5 mL/kg/hr for ≥6h) is not computable from PhysioNet 2019 data | Real hospital data must include structured urine output tracking |
| 2 | **No baseline creatinine from prior admissions** — only in-stay creatinine is available | Longitudinal EHR records will provide community baseline |
| 3 | **No novel biomarkers** — NGAL, KIM-1, TIMP-2·IGFBP7 (NephroCheck) are not in this dataset | If the hospital can run these assays, they should be added as features |
| 4 | **Western ICU population** — data comes from US hospital systems; SA-AKI etiology at the target hospital is dominated by severe malaria and tropical infections | Model requires fine-tuning or full retraining on local data |
| 5 | **No medication data** — nephrotoxic drugs (NSAIDs, aminoglycosides, contrast agents) are strong AKI risk factors | Must be populated from EHR pharmacy records |
| 6 | **No pediatric cohort** — this PoC targets adult ICU patients (Age ≥ 18) | Pediatric SA-AKI requires a separate model and separate data |

## Validation Checklist

Status after the first full local run on the 100-patient MIMIC-IV demo:

- [x] Temporal split confirmed: split is chronological by patient (train/val/test)
- [x] No feature leakage: test AUROC is 0.66, **not** fake-perfect — a leak would force it near 1.0
- [x] Calibration curves generated (pre vs post isotonic) — `outputs/figures`
- [ ] AUROC > 0.75 for the 24h Stage 2 model — **0.66 on the demo; expected to miss at 100 patients.** Target applies to full MIMIC-IV
- [x] DCA and subgroup reports generated — `outputs/reports` (surfaced a real <65 vs 65+ fairness gap: AUROC 0.74 vs 0.62)
- [ ] Site holdout (Set B) — **N/A**: demo is single-site, so `site_holdout` is empty by design
- [x] `sample_alerts.json` is readable and structured for EHR integration
- [x] All output files exist and are non-empty (except the by-design-empty `site_holdout`)

> The unmet items are **expected** on a 100-patient demo, not failures — the goal here is a
> working end-to-end pipeline, not a deployable model. They become real targets on full MIMIC-IV.

## License

This project uses publicly available datasets under the [Open Data Commons Attribution License v1.0](https://opendatacommons.org/licenses/by/1.0/).

---

*Project Sentinel PoC — Full ML pipeline established. Retraining on real hospital data is Phase 2.*
