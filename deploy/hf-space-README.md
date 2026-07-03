---
title: Project Sentinel — SA-AKI Early Warning
emoji: 🏥
colorFrom: red
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Project Sentinel — live demo

Prospective early-warning system for **Sepsis-Associated Acute Kidney Injury (SA-AKI)**
in ICU patients. This Space serves the FastAPI inference backend and the React ward
dashboard from a single container.

- **Score a patient-hour** → calibrated 24 h AKI risk + KDIGO-aware clinical alert.
- **SHAP explanation** → the top 3 contributing factors behind every score.
- **Conformal abstention** → borderline cases return `INDETERMINATE` instead of a
  false-confident number.

Trained on the open-access MIMIC-IV demo (proof-of-concept). Source code and full
documentation: https://github.com/Sng43/sentinel-poc
