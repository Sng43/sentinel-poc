---
name: sentinel-poc-codebase-state
description: Refactor progress — MIMIC-IV pivot done through notebooks; data pipeline verified
metadata:
  type: project
---

The sentinel-poc codebase was originally three mutually-inconsistent layers (src / notebooks / run_pipeline) generated piecemeal and never run end-to-end. The MIMIC-IV pivot refactor is now well underway (June 2026):

**Done & verified on real MIMIC-IV demo data (136 stays, 12,413 patient-hours, 52 septic):**
- `src/data_loader.py` — full rewrite: reads relational MIMIC-IV demo (`hosp/`+`icu/`), reshapes to canonical hourly schema, urine output with mask, Sepsis-3 suspicion-of-infection proxy (SOFA≥2 still a TODO), `define_cohort` returns `(cohort_df, baseline_df)`.
- `src/labels.py` — full rewrite: aligned to canonical caps schema (`Creatinine`/`ICULOS` via module constants), and **added the UOP KDIGO criterion** (<0.5 mL/kg/h ≥6h, mask-aware — see [[sentinel-poc-uop-mask-bug]]). Label rates rise with horizon (6h 12% → 24h 26%) — sensible.
- `src/features.py` — already used caps schema so works as-is; fixed a rolling-features fragmentation `PerformanceWarning` (build cols then concat). engineer_features: 42→203 cols; all 15 stage-1 feats present; urine flows into the 199 stage-2 feats.
- Notebooks 01–07 all re-wired to the real src API + canonical schema, correct order (label BEFORE temporal_split).
- `mimic_demo_project/download_mimic_demo.py` — rewritten to use the open-access physionet.org zip via stdlib urllib (NO boto3; the `mimic-iv-demo/` S3 prefix on physionet-open does not exist — only `-meds`/`-fhir` variants do). Demo downloaded to `project_sentinel/data/raw/mimic_iv_demo/`.

**Canonical schema:** `patient_id` (=MIMIC stay_id, str), `ICULOS` (hours since ICU admit), `t_sepsis`, vitals/labs in PhysioNet-style caps (`HR`,`MAP`,`Creatinine`,...), `urine_rate`+`urine_observed`(mask)+`weight_kg`, `Age`,`Gender`,`hospital_system`(='A', single-site demo). Documented at top of data_loader.py.

- `run_pipeline.py` — ✅ rewritten to match the verified src API (phases 1–7 mirror the notebooks; renumbered). Phases 1–4 verified on real data via `--all`; 5–7 fail gracefully (caught per-phase) without libomp.
- **Label-leakage fix:** labels are derived from the RAW cohort (`cohort_defined.parquet`), then merged onto the imputed engineered features by `(patient_id, ICULOS)`. Labeling the imputed features leaked future values via ffill+bfill. Applied in both notebook 04 and run_pipeline phase 4.
- Root `CLAUDE.md` written (full operational guide). `.gitignore` added; data/ + `__pycache__` untracked via `git rm --cached` (43 staged deletions, **not committed** — left for the user). NOTE: a commit `e4fe1a5 "context for project"` (not made by me) had committed the 16MB MIMIC demo data + .pyc + refactored src.

**Not yet verified / remaining:** model training (notebook 05 / `models.py`) + eval (06) + explain (07) couldn't run locally — LightGBM needs `libomp` (`brew install libomp`); runs fine on Colab. Not started: backend (FastAPI), frontend (React), OpenClinic mock. See [[sentinel-poc-overview]].
