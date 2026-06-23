---
name: sentinel-poc-uop-mask-bug
description: Never fill missing urine output with 0; use a missingness mask
metadata:
  type: feedback
---

When handling urine output (UOP) in sentinel-poc: **never fill missing UOP with sentinel 0.**

**Why:** 0 mL/kg/h urine output means anuria — a severe KDIGO Stage-3 AKI signal, NOT "missing data". Filling absent UOP with 0 teaches the model "0 = ignore this slot"; then real zeros (severe kidney failure) get ignored — a silent bug that only surfaces as garbage stage-2/3 predictions.

**How to apply:** represent UOP missingness as a separate **mask channel** (mask=0 absent), impute absent values with a neutral statistic (population median, not 0), and keep the masking scheme identical across any pretraining/fine-tuning stages. This generalizes the existing missingness-indicator approach in `features.py`. Same principle rejects the proposed "fill UOP with 0 dummy column" two-stage pretraining hack. See [[sentinel-poc-overview]].
