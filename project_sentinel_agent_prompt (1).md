# PROJECT SENTINEL — AGENT BUILD PROMPT v1.0
**Proof-of-Concept: Sepsis-Associated AKI Early Warning System**
**Executing Agent:** Claude Code (or equivalent autonomous coding executor)
**Status:** Pre-hospital-data phase — using public ICU datasets as clinical surrogate

---

## 0. MISSION BRIEF

You are building a proof-of-concept (PoC) ML pipeline for **Project Sentinel**, an early warning system for **Sepsis-Associated Acute Kidney Injury (SA-AKI)**. The clinical context is a resource-limited East African hospital (RMRTH, Rwanda) where:

- 34.1% of adult patients requiring acute hemodialysis die
- Serum creatinine is a lagging indicator — patients lose >50% kidney function before standard diagnostic criteria are met
- Urine output is tracked in fewer than 4% of at-risk pediatric cases
- The target system is **OpenClinic GA**, a hospital EHR running since 2007

**The core task is not classification of existing AKI — KDIGO already does that, too late. The task is probabilistic, prospective prediction:**

> *"What is the probability that this ICU patient will meet KDIGO AKI Stage ≥1 criteria within the next 6, 12, or 24 hours?"*

You will build this pipeline using **publicly available ICU data** as a clinical surrogate. You are not inventing data. You are demonstrating that the full pipeline — from raw time-series to calibrated alert — is architecturally sound and ready for re-training on RMRTH's OpenClinic GA data once it becomes available.

---

## 1. TECH STACK

### 1.1 Required Packages

Initialize the project and install all dependencies using `uv`. This creates a `pyproject.toml` and `uv.lock` automatically — do not create a `requirements.txt`.

```bash
# Initialize the project (run once at the repo root)
uv init project_sentinel
cd project_sentinel

# Add all runtime dependencies
uv add pandas numpy scipy scikit-learn xgboost lightgbm shap \
        matplotlib seaborn pyarrow tqdm imbalanced-learn optuna

# Add Jupyter as a dev dependency
uv add --dev jupyter notebook ipykernel
```

To sync the environment on a fresh clone (e.g., on the hospital workstation):
```bash
uv sync
```

To run any script or notebook within the managed environment:
```bash
uv run python src/data_loader.py
uv run jupyter notebook
```

> **Note:** `calibration-curve` is not a standalone package — it is part of `scikit-learn` (`sklearn.calibration`). No separate install needed.

### 1.2 Project Structure

Create this directory layout before writing any code:

```
project_sentinel/
├── data/
│   ├── raw/              # Downloaded source files (never modified)
│   ├── interim/          # Merged + cleaned DataFrames
│   └── processed/        # Feature-engineered, label-attached, split datasets
├── notebooks/
│   ├── 01_data_acquisition.ipynb
│   ├── 02_cohort_definition.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_label_engineering.ipynb
│   ├── 05_model_training.ipynb
│   ├── 06_evaluation.ipynb
│   └── 07_explainability.ipynb
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── labels.py
│   ├── models.py
│   ├── evaluation.py
│   └── explain.py
├── outputs/
│   ├── figures/
│   ├── models/           # Saved .pkl and .json model artifacts
│   └── reports/
├── pyproject.toml        # Managed by uv — do not edit manually
├── uv.lock               # Auto-generated lockfile — commit to version control
└── README.md
```

---

## 2. DATA ACQUISITION

### 2.1 Primary Dataset: PhysioNet/CinC Challenge 2019

**"Early Prediction of Sepsis from Clinical Data"**

- URL: `https://physionet.org/content/challenge-2019/1.0.0/`
- License: Open Data Commons Attribution License v1.0 — freely downloadable, no credentialing required
- Size: ~40,000 ICU patients across two hospital systems (Set A + Set B)
- Format: One `.psv` (pipe-separated) file per patient, hourly measurements
- **This is your primary dataset. Always try this first.**

**Download procedure:**

```bash
cd project_sentinel/data/raw

# Set A (~20,000 patients)
wget -r -N -c -np https://physionet.org/files/challenge-2019/1.0.0/training/training_setA/
# Set B (~20,000 patients)
wget -r -N -c -np https://physionet.org/files/challenge-2019/1.0.0/training/training_setB/
```

If `wget` fails due to auth, try the direct `.zip` download links from the PhysioNet page, or use the Kaggle mirror:
- Kaggle dataset name: `"salikhussaini49/prediction-of-sepsis"` or search `"PhysioNet sepsis 2019 challenge"`
- Use the Kaggle API: `kaggle datasets download salikhussaini49/prediction-of-sepsis`

### 2.2 Variables in This Dataset

Each `.psv` file has one row per ICU hour. Columns:

| Category | Variables |
|---|---|
| **Vitals** | HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2 |
| **Labs** | BaseExcess, HCO3, FiO2, pH, PaCO2, SaO2, AST, BUN, Alkalinephos, Calcium, Chloride, **Creatinine**, Bilirubin_direct, **Glucose**, **Lactate**, Magnesium, Phosphate, **Potassium**, Bilirubin_total, TroponinI, **Hct**, **Hgb**, PTT, **WBC**, Fibrinogen, **Platelets** |
| **Demographics** | Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS |
| **Label** | SepsisLabel (0 = no sepsis, 1 = sepsis onset at this hour, per Sepsis-3) |

Note: Lab measurements are sparse — many cells will be NaN. This is realistic and expected. Your pipeline must handle this.

### 2.3 Fallback Dataset (If PhysioNet 2019 Is Inaccessible)

Use **MIMIC-III Clinical Database Demo**:
- URL: `https://physionet.org/content/mimiciii-demo/1.4/`
- No credentialing required
- 100 ICU patients, full structured EHR data (admits, labs, vitals, diagnoses, notes)
- Download all CSV tables

If using MIMIC-III Demo, adapt the pipeline to extract equivalent features from:
- `LABEVENTS.csv` (creatinine, BUN, WBC, etc.)
- `CHARTEVENTS.csv` (vitals: HR, MAP, urine output)
- `DIAGNOSES_ICD.csv` (sepsis ICD codes for cohort definition: ICD-9 995.91, 995.92, or ICD-10 A41.x)
- `ADMISSIONS.csv` + `ICUSTAYS.csv` (admission timestamps)

Document clearly in `README.md` which dataset was used and why.

---

## 3. DATA LOADING & MERGING

In `src/data_loader.py`, implement:

```python
def load_physionet_2019(data_dir: str) -> pd.DataFrame:
    """
    Reads all .psv patient files from Set A and Set B.
    Adds a 'patient_id' column derived from filename.
    Adds a 'hospital_system' column (A or B) — critical for site-level validation.
    Returns a single merged DataFrame with MultiIndex (patient_id, ICULOS).
    """
```

Requirements:
- Use `tqdm` for progress bar during file loading
- Add `patient_id` from filename (e.g., `p000001`)
- Add `hospital_system` column (`'A'` or `'B'`)
- Parse `ICULOS` as integer (hours since ICU admission)
- Do NOT drop NaN values at this stage — handle missingness later
- Save merged DataFrame to `data/interim/merged_cohort.parquet`
- Print a data summary: total patients, total rows, % missingness per column

---

## 4. COHORT DEFINITION

In `notebooks/02_cohort_definition.ipynb` and `src/data_loader.py`:

### 4.1 Inclusion Criteria

Define your study cohort as:

```python
INCLUSION_CRITERIA = {
    "sepsis_patients_only": True,       # SepsisLabel == 1 at any point in stay
    "min_icu_stay_hours": 6,            # At least 6h of data before sepsis onset
    "min_creatinine_measurements": 2,   # Need ≥2 creatinine values to compute velocity
    "age_adult": True,                  # Age >= 18 (align with RMRTH adult cohort)
}
```

### 4.2 Cohort Construction Logic

1. Identify all patients where `SepsisLabel` becomes `1` at some point → these are your sepsis patients
2. Record `t_sepsis` = the first hour where `SepsisLabel = 1` for each patient
3. Define analysis window: from `t_sepsis` onward (SA-AKI occurs after sepsis onset, not before)
4. Exclude patients who already meet AKI criteria at `t_sepsis` (prevalent AKI — you are predicting incident AKI)
5. For patients meeting inclusion criteria, retain all rows from `t_sepsis` onward as candidate prediction time points

### 4.3 Baseline Creatinine

For each patient, define:

```python
def compute_baseline_creatinine(patient_df):
    """
    Baseline = lowest creatinine value in the first 24h of ICU stay.
    If no creatinine in first 24h, use first available value.
    If no creatinine at all, exclude patient.
    """
```

Save baseline creatinine to a lookup dict / patient-level DataFrame.

---

## 5. LABEL ENGINEERING

This is the most critical methodological step. **Do not label this as simple binary AKI classification.**

In `src/labels.py`, implement:

### 5.1 KDIGO AKI Criteria (Simplified for Available Data)

Using the creatinine-based KDIGO criterion (the only one computable from this dataset):

```
AKI = True if:
    (creatinine_current - creatinine_baseline) >= 0.3 mg/dL
    OR
    creatinine_current >= 1.5 × creatinine_baseline
```

Note: Urine output criterion (< 0.5 mL/kg/hr for ≥6h) cannot be applied — document this limitation explicitly in the notebook.

### 5.2 Prospective Horizon Labels

For each patient at each time step `t`, generate three binary labels:

```python
def create_aki_horizon_labels(patient_df, baseline_cr, horizons=[6, 12, 24]):
    """
    For time step t, label = 1 if:
        - Patient does NOT currently have AKI (exclude prevalent cases)
        - AND patient meets KDIGO AKI criteria at any time in [t+1, t+horizon]
    
    Returns columns: 'aki_label_6h', 'aki_label_12h', 'aki_label_24h'
    """
```

### 5.3 Label Exclusions

At each time step `t`, exclude from training:
- Time steps where the patient already meets AKI criteria (they already have it)
- Time steps within 1 hour of ICU discharge (insufficient horizon)
- Time steps with no creatinine measurement in the entire forward horizon window (the outcome is unobservable)

### 5.4 Class Imbalance Assessment

After labeling, print:
```
Label distribution per horizon:
  6h:  X% positive, Y% negative (ratio: Z:1)
  12h: X% positive, Y% negative
  24h: X% positive, Y% negative
```

Expect roughly 80–90% negative class. Plan to use `scale_pos_weight` in XGBoost or `class_weight='balanced'` in LightGBM, or SMOTE from `imbalanced-learn`.

---

## 6. FEATURE ENGINEERING

In `src/features.py`. This is where clinical domain knowledge is operationalized as model inputs.

### 6.1 Missingness Strategy

Apply **forward-fill then backward-fill** for labs (they are measured intermittently). For vitals (measured hourly), forward-fill only up to 4 hours. After that, use a separate missingness indicator column:

```python
for col in lab_columns:
    df[f'{col}_observed'] = df[col].notna().astype(int)  # Missingness indicator
    df[col] = df[col].ffill(limit=4).bfill(limit=4)
```

Do NOT impute before computing rolling statistics — compute rolling stats on the original sparse data, then impute.

### 6.2 Absolute Value Features (at time t)

Include all available vitals and labs as-is at time `t`. These are your baseline static features.

### 6.3 Temporal/Dynamic Features (Rolling Windows)

For each numeric variable `v`, compute over last 6h and 12h:

```python
rolling_features = {
    f'{v}_mean_6h':   df.groupby('patient_id')[v].transform(lambda x: x.rolling(6,  min_periods=1).mean()),
    f'{v}_mean_12h':  df.groupby('patient_id')[v].transform(lambda x: x.rolling(12, min_periods=1).mean()),
    f'{v}_std_6h':    df.groupby('patient_id')[v].transform(lambda x: x.rolling(6,  min_periods=2).std()),
    f'{v}_min_6h':    df.groupby('patient_id')[v].transform(lambda x: x.rolling(6,  min_periods=1).min()),
    f'{v}_max_6h':    df.groupby('patient_id')[v].transform(lambda x: x.rolling(6,  min_periods=1).max()),
}
```

### 6.4 Creatinine Velocity — The Most Important Feature

Implement explicitly:

```python
def creatinine_velocity(patient_df, window_hours=12):
    """
    delta_Cr = creatinine(t) - creatinine(t - window_hours)
    velocity = delta_Cr / window_hours  (mg/dL per hour)
    
    Also compute:
    - cr_above_baseline: creatinine(t) - baseline_creatinine
    - cr_ratio_baseline: creatinine(t) / baseline_creatinine
    """
```

These three creatinine derivatives are likely to be the highest-importance features. Ensure they are always computed before model training.

### 6.5 Composite/Interaction Features

Implement these clinically grounded composite features:

```python
# Renal perfusion pressure proxy
df['renal_perf_pressure'] = df['MAP'] - df.get('CVP', 0)  # CVP may not be available

# Hemodynamic-renal coupling
df['map_cr_velocity_interaction'] = df['MAP'] * df['creatinine_velocity_12h']

# Inflammatory-metabolic axis
df['wbc_lactate_product'] = df['WBC'] * df['Lactate']

# Acid-base status
df['anion_gap_proxy'] = df['Chloride'] + df['HCO3']  # simplified

# BUN-Creatinine ratio (indicator of prerenal vs intrinsic AKI)
df['bun_cr_ratio'] = df['BUN'] / df['Creatinine'].replace(0, np.nan)

# Time since sepsis onset
df['hours_since_sepsis'] = df['ICULOS'] - df['t_sepsis']
```

### 6.6 Feature List Finalization

After feature engineering, produce a feature manifest file `outputs/feature_manifest.csv` with:
- Feature name
- Category (vital / lab / derived / temporal / composite)
- % missing before imputation
- Mean, std, min, max

This manifest will guide the OpenClinic GA feature extraction spec when real hospital data becomes available.

---

## 7. TRAIN / VALIDATION / TEST SPLIT

**Critical: Do NOT use random split. Use temporal validation.**

```python
def temporal_split(df, val_ratio=0.15, test_ratio=0.15):
    """
    Sort patients by their ICU admission time (ICULOS start as proxy for admission).
    
    - Train set:  First 70% of patients (chronologically)
    - Val set:    Next 15% of patients
    - Test set:   Last 15% of patients (held out entirely)
    
    This prevents data leakage from shared temporal trends across patients.
    Additionally, hold out ALL patients from hospital_system='B' as a 
    site-level generalization holdout (simulates deploying at RMRTH after 
    training on external data).
    """
```

Print a split summary:
```
Temporal split summary:
  Train:       N patients, M rows, K% AKI positive
  Val:         N patients, M rows, K% AKI positive
  Test:        N patients, M rows, K% AKI positive
  Site holdout (Set B): N patients (for generalization analysis)
```

Save all four splits to `data/processed/` as parquet files.

---

## 8. MODEL ARCHITECTURE — CASCADED TWO-STAGE DESIGN

Implement a **cascaded model** optimized for the LMIC deployment constraint: the first stage is cheap and runs on minimal inputs; the second stage adds richer features when risk exceeds a threshold.

### 8.1 Stage 1 — Vitals-Only Screener (LightGBM)

Train on only the vital signs subset (HR, MAP, O2Sat, Temp, Resp, SBP + their rolling stats). This model simulates the scenario where labs are unavailable (very common in district hospitals).

```python
STAGE_1_FEATURES = [
    'HR', 'MAP', 'O2Sat', 'Temp', 'Resp', 'SBP',
    'HR_mean_6h', 'MAP_mean_6h', 'MAP_std_6h',
    'hours_since_sepsis',
    'Age', 'Gender'
]
```

Train three separate Stage 1 models (for 6h, 12h, 24h horizons).

### 8.2 Stage 2 — Full-Feature Predictor (LightGBM + XGBoost)

Train on the complete feature set from Section 6. This model fires only when Stage 1 probability exceeds `threshold_stage1 = 0.25` (tune this on validation set).

```python
STAGE_2_FEATURES = ALL_ENGINEERED_FEATURES  # from Section 6
```

Train three separate Stage 2 models (for 6h, 12h, 24h horizons).

### 8.3 Model Training Configuration

For LightGBM:
```python
lgbm_params = {
    'objective': 'binary',
    'metric': ['auc', 'binary_logloss'],
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'class_weight': 'balanced',
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'verbose': -1
}
```

Use early stopping with `validation_split` as callback. Save best model checkpoint.

For XGBoost (Stage 2 only, for comparison):
```python
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': ['auc', 'logloss'],
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'scale_pos_weight': compute_pos_weight(y_train),  # handles class imbalance
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42
}
```

### 8.4 Hyperparameter Tuning

Use **Optuna** for automated hyperparameter search on the 24h horizon model (this will be the primary deployment model):

```python
import optuna

def objective(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    }
    # Train and return validation AUROC

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

### 8.5 Model Calibration

After training, apply **isotonic regression calibration** to all final models:

```python
from sklearn.calibration import CalibratedClassifierCV

# Or post-hoc:
from sklearn.isotonic import IsotonicRegression
calibrator = IsotonicRegression(out_of_bounds='clip')
calibrator.fit(val_probs, val_labels)
calibrated_probs = calibrator.predict(test_probs)
```

Calibration is non-negotiable for clinical deployment. A miscalibrated model that outputs 90% risk when true risk is 40% will destroy clinician trust.

Save all trained models (Stage 1 and Stage 2, all 3 horizons, LightGBM and XGBoost) to `outputs/models/`.

---

## 9. EVALUATION SUITE

In `src/evaluation.py`. Implement all of the following. Report results for ALL models (Stage 1, Stage 2 LightGBM, Stage 2 XGBoost) at ALL horizons (6h, 12h, 24h) on the test set AND site holdout set.

### 9.1 Primary Metrics

```python
def evaluate_model(y_true, y_prob):
    return {
        'auroc':  roc_auc_score(y_true, y_prob),
        'auprc':  average_precision_score(y_true, y_prob),  # better for imbalanced
        'brier':  brier_score_loss(y_true, y_prob),
        'ece':    expected_calibration_error(y_true, y_prob, n_bins=10),
    }
```

Implement `expected_calibration_error()` manually (it's not in sklearn directly):
```python
def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if mask.sum() > 0:
            acc = y_true[mask].mean()
            conf = y_prob[mask].mean()
            ece += mask.sum() * abs(acc - conf)
    return ece / len(y_true)
```

### 9.2 Calibration Curve

Plot reliability diagrams (calibration curves) for all models before and after isotonic calibration. Save to `outputs/figures/calibration_curves.png`.

### 9.3 ROC and PR Curves

Plot ROC and Precision-Recall curves for all models at all horizons on the same axes. Save to `outputs/figures/roc_curves.png` and `outputs/figures/pr_curves.png`.

### 9.4 Decision Curve Analysis (Net Benefit)

This is the most clinically relevant metric. Implement DCA:

```python
def decision_curve_analysis(y_true, y_prob, thresholds=np.arange(0.01, 0.50, 0.01)):
    """
    Computes Net Benefit = TP_rate - FP_rate × (threshold / (1 - threshold))
    
    Plot three curves:
    1. Model net benefit
    2. "Treat all" net benefit (baseline)
    3. "Treat none" net benefit (= 0)
    
    The model is clinically useful where its curve exceeds both baselines.
    """
```

Save to `outputs/figures/decision_curve_analysis.png`. Annotate the threshold range where the model shows net clinical benefit.

### 9.5 Threshold-Dependent Metrics at Clinical Decision Points

At three clinically defined probability thresholds (0.20, 0.35, 0.50), report:

```
Threshold = 0.20 (high sensitivity, screen not to miss):
  Sensitivity, Specificity, PPV, NPV, FPR, FNR, F1

Threshold = 0.35 (balanced):
  ...

Threshold = 0.50 (high specificity, avoid alarm fatigue):
  ...
```

### 9.6 Subgroup Analysis

Stratify test set performance by:
- Age groups: 18–40, 41–60, 61–80, 80+
- ICU type (Unit1 vs Unit2 if available in the dataset)
- Sepsis severity proxy: high vs low MAP at sepsis onset (< 65 mmHg vs ≥ 65 mmHg)
- Hospital system: Set A vs Set B (generalizability analysis)

Generate a subgroup performance table: `outputs/reports/subgroup_analysis.csv`

### 9.7 Comparative Summary Table

Produce a final summary table `outputs/reports/model_comparison.csv`:

| Model | Horizon | AUROC | AUPRC | ECE | Net Benefit (0.2) |
|---|---|---|---|---|---|
| Stage 1 (vitals only) | 6h | ... | ... | ... | ... |
| Stage 2 LightGBM | 6h | ... | ... | ... | ... |
| Stage 2 XGBoost | 6h | ... | ... | ... | ... |
| Stage 1 (vitals only) | 12h | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... |

---

## 10. EXPLAINABILITY LAYER

In `src/explain.py`. Clinicians will not adopt a black-box. Every risk alert must surface its top 3 contributing factors.

### 10.1 Global Feature Importance (SHAP)

```python
import shap

def generate_shap_summary(model, X_test, feature_names, output_path):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Summary bar plot (mean |SHAP|)
    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                      plot_type='bar', max_display=20, show=False)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    # Summary dot plot (distribution)
    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                      max_display=20, show=False)
    plt.savefig(output_path.replace('.png', '_dot.png'), dpi=150, bbox_inches='tight')
```

### 10.2 Individual Patient Explanations (Waterfall Plots)

Select 3 representative patients from the test set:
- **High-risk true positive**: patient who developed AKI, correctly predicted
- **Medium-risk false negative**: patient who developed AKI, model missed
- **Low-risk true negative**: patient who did not develop AKI, correctly predicted

For each:
```python
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[patient_idx],
        base_values=explainer.expected_value,
        data=X_test.iloc[patient_idx],
        feature_names=feature_names
    )
)
```

Save to `outputs/figures/shap_patient_[id].png`.

### 10.3 Top-3 Feature Alert Format

Implement a function that mimics what the EHR alert would show:

```python
def generate_clinical_alert(model, explainer, patient_row, threshold=0.35):
    """
    Returns a structured dict matching the proposed RMRTH alert format:
    
    {
        'patient_id': '...',
        'risk_score': 0.72,
        'risk_level': 'HIGH',         # HIGH / MEDIUM / LOW
        'aki_probability_24h': 72%,
        'top_contributors': [
            {'feature': 'Creatinine velocity (last 12h)',  'direction': 'RISING', 'value': '+0.12 mg/dL/hr'},
            {'feature': 'Mean arterial pressure (6h avg)', 'direction': 'LOW',    'value': '61 mmHg'},
            {'feature': 'Lactate',                         'direction': 'HIGH',   'value': '4.1 mmol/L'},
        ],
        'recommended_action': 'Consider nephrology review and fluid balance optimization'
    }
    """
```

Generate alerts for 10 test patients and save to `outputs/reports/sample_alerts.json`. This artifact directly demonstrates the clinical workflow integration.

---

## 11. OUTPUTS & DELIVERABLES

By the end of the build, the following must exist:

```
outputs/
├── models/
│   ├── stage1_lgbm_6h.pkl
│   ├── stage1_lgbm_12h.pkl
│   ├── stage1_lgbm_24h.pkl
│   ├── stage2_lgbm_6h.pkl
│   ├── stage2_lgbm_12h.pkl
│   ├── stage2_lgbm_24h.pkl
│   ├── stage2_xgb_6h.pkl
│   ├── stage2_xgb_12h.pkl
│   ├── stage2_xgb_24h.pkl
│   └── calibrators/          # Isotonic calibration objects
├── figures/
│   ├── calibration_curves.png
│   ├── roc_curves.png
│   ├── pr_curves.png
│   ├── decision_curve_analysis.png
│   ├── shap_summary_bar.png
│   ├── shap_summary_dot.png
│   ├── shap_patient_tp.png
│   ├── shap_patient_fn.png
│   └── shap_patient_tn.png
├── reports/
│   ├── model_comparison.csv
│   ├── subgroup_analysis.csv
│   ├── feature_manifest.csv
│   └── sample_alerts.json
└── README.md                  # Auto-generated pipeline summary
```

### 11.1 Auto-Generated README

At the end of execution, generate a `README.md` summarizing:
- Dataset used (name, size, # patients, date downloaded)
- Cohort definition (inclusion criteria, final N)
- Label distribution at each horizon
- Best model (by AUROC on test set) at each horizon with metrics
- Top 5 features by SHAP importance for the 24h model
- Known limitations relative to RMRTH deployment context

---

## 12. KNOWN LIMITATIONS TO DOCUMENT

The agent must explicitly document these limitations in `README.md` — they are not failures of this PoC, they are the specification gaps that real RMRTH data will close:

1. **No urine output labels**: KDIGO urine output criterion (< 0.5 mL/kg/hr for ≥6h) is not computable from PhysioNet 2019 data. RMRTH data must include structured urine output tracking.

2. **No baseline creatinine from prior admissions**: PhysioNet 2019 only has in-stay creatinine. True KDIGO requires community baseline. OpenClinic GA longitudinal records will address this.

3. **No novel biomarkers**: NGAL, KIM-1, and TIMP-2·IGFBP7 (NephroCheck) are not in this dataset. These are high-value early biomarkers described in the clinical literature. If RMRTH can run these assays, they should be added as features.

4. **Western ICU population**: PhysioNet 2019 data comes from US hospital systems. SA-AKI etiology at RMRTH is dominated by severe malaria and tropical infections — the model will require fine-tuning or retraining on RMRTH data before clinical deployment.

5. **No medication data**: Nephrotoxic drugs (NSAIDs, aminoglycosides, contrast agents) are strong AKI risk factors. The pipeline has placeholder columns for these; they must be populated from OpenClinic GA pharmacy records.

6. **No pediatric cohort**: This PoC targets adult ICU patients (Age ≥ 18). Pediatric SA-AKI (relevant to the district hospital audit showing 76.3% KDIGO Stage 3 at presentation) requires a separate model and separate data.

---

## 13. EXECUTION ORDER

Run notebooks in this sequence using `uv run` to ensure the managed environment is used:

```bash
uv run jupyter nbconvert --to notebook --execute notebooks/01_data_acquisition.ipynb
uv run jupyter nbconvert --to notebook --execute notebooks/02_cohort_definition.ipynb
uv run jupyter nbconvert --to notebook --execute notebooks/03_feature_engineering.ipynb
uv run jupyter nbconvert --to notebook --execute notebooks/04_label_engineering.ipynb
uv run jupyter nbconvert --to notebook --execute notebooks/05_model_training.ipynb
uv run jupyter nbconvert --to notebook --execute notebooks/06_evaluation.ipynb
uv run jupyter nbconvert --to notebook --execute notebooks/07_explainability.ipynb
```

For interactive development, launch the notebook server with:
```bash
uv run jupyter notebook
```

Each notebook must:
- Start by loading from the previous step's parquet file
- End by saving its outputs to parquet/pkl/csv/json as appropriate
- Print a completion summary with key numbers

---

## 14. FINAL VALIDATION CHECKLIST

Before declaring the PoC complete, verify:

- [ ] Temporal split is confirmed: no patient appears in both train and test
- [ ] No feature leakage: no future information (post-`t`) is used to predict at time `t`
- [ ] Calibration curves show pre vs post calibration improvement
- [ ] AUROC > 0.75 for the 24h Stage 2 model on the test set (if below this, debug feature engineering)
- [ ] SHAP summary plots confirm creatinine velocity is among the top 5 features
- [ ] DCA shows net benefit above "treat all" baseline for threshold range 0.15–0.40
- [ ] Site holdout (Set B) AUROC is within 0.05 of test set AUROC (generalizability check)
- [ ] `sample_alerts.json` is readable and structured for EHR integration
- [ ] All output files listed in Section 11 exist and are non-empty
- [ ] `README.md` is populated with actual results, not placeholders

---

*End of Project Sentinel Agent Build Prompt v1.0*
*This PoC establishes the full ML pipeline. Re-training on OpenClinic GA data from RMRTH is Phase 2.*
