---
name: sentinel-poc-overview
description: What Project Sentinel is, its goal, deadline, and data pivot
metadata:
  type: project
---

Project Sentinel (sentinel-poc) is a proof-of-concept ML pipeline for **prospective prediction of Sepsis-Associated AKI (SA-AKI)** — "probability this ICU patient meets KDIGO AKI Stage ≥1 within next 6/12/24h". Clinical target: resource-limited Rwandan hospital (RMRTH) running OpenClinic GA. Owner is building it as a capstone/demo (user is "Emmy"/Sng43).

**Deadline: demo on June 25 2026** (work started ~June 22). Resources: MIMIC-IV demo (100 patients), free Google Colab (T4), Google Antigravity IDE, Claude + ChatGPT.

**Data pivot (key decision):** moving from PhysioNet 2019 Challenge (flat PSV, NO urine output) → **MIMIC-IV demo** (relational tables, HAS urine output → enables full KDIGO staging incl. UOP criterion <0.5 mL/kg/h ≥6h). Approach = Plan B: keep what's salvageable, refactor the rest. See [[sentinel-poc-codebase-state]] and [[sentinel-poc-uop-mask-bug]].

Architecture: cascaded two-stage — Stage 1 vitals-only LightGBM screener → Stage 2 full-feature LightGBM+XGBoost (Optuna-tuned), isotonic calibration, SHAP clinical alerts. Planned additions: FastAPI backend + React frontend + OpenClinic GA mock layer.
