"""Data loading and cohort definition for Project Sentinel (MIMIC-IV demo).

This module replaces the original PhysioNet-2019 flat-PSV loader. It reads the
**relational** MIMIC-IV clinical database demo (v2.2) — separate ``hosp/`` and
``icu/`` tables of ``.csv.gz`` files — and reshapes them into a single hourly
time-series DataFrame, one row per (ICU stay, hour-since-admission).

Why the rewrite
---------------
The SA-AKI task needs **urine output**, which the PhysioNet-2019 Challenge set
does not contain. The MIMIC-IV demo does (``icu/outputevents``), so we can build
the full KDIGO criteria (creatinine **and** urine output) downstream.

Canonical output schema  (the contract every downstream module relies on)
-------------------------------------------------------------------------
Identifiers / time:
    ``patient_id``   : str   — one ICU stay (uses MIMIC ``stay_id``); the unit of analysis
    ``subject_id``   : int   — the underlying patient (for joins/debugging)
    ``ICULOS``       : int   — hours since ICU admission (0, 1, 2, …); the time axis
    ``t_sepsis``     : float — ICULOS hour of sepsis onset (NaN if never septic)
    ``hospital_system`` : str — site id for the holdout split (demo is single-site → 'A')

Vitals  (named to match ``features.VITAL_COLUMNS``):
    ``HR``, ``O2Sat``, ``Temp``, ``SBP``, ``MAP``, ``DBP``, ``Resp``

Labs  (named to match ``features.LAB_COLUMNS``):
    ``Creatinine``, ``BUN``, ``Lactate``, ``WBC``, ``Platelets``, ``Hgb``, ``Hct``,
    ``Glucose``, ``Potassium``, ``Chloride``, ``HCO3``, ``Bilirubin_total``,
    ``Calcium``, ``Magnesium``, ``Phosphate``, ``AST``, ``Alkalinephos``,
    ``PaCO2``, ``pH``, ``SaO2``, ``BaseExcess``, ``PTT``, ``Fibrinogen``

Urine output  (the SA-AKI addition):
    ``urine_rate``       : float — mL/kg/hr for that hour (NaN where unobserved)
    ``urine_observed``   : int   — 1 if any urine charted that hour, else 0  (mask channel)
    ``weight_kg``        : float — admission/daily weight used to scale urine

Demographics  (subset of ``features.DEMO_COLUMNS``):
    ``Age``, ``Gender``  (Gender encoded 1=M, 0=F)

.. important::
   **Urine output is never zero-filled.** A literal 0 mL/kg/hr means *anuria*
   (severe KDIGO-3 AKI), not "missing". Absent hours are left as NaN with
   ``urine_observed = 0``; downstream imputation must respect the mask. See the
   project note on the zero-fill bug.

Downstream alignment note
-------------------------
``labels.py`` currently expects lowercase ``creatinine`` / ``hour`` via its
module constants ``_CREATININE_COL`` / ``_TIME_COL``. Point those at
``"Creatinine"`` / ``"ICULOS"`` (a 2-line change) so it consumes this schema.
``features.py`` already uses these capitalised names.
"""
from __future__ import annotations

import os
import glob
import logging
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MIMIC-IV itemid → canonical-variable maps
# ---------------------------------------------------------------------------
# chartevents (icu) item ids. Several ids map to the same concept (e.g. arterial
# vs non-invasive BP); we coalesce them under one canonical name.
_VITAL_ITEMIDS: dict[str, list[int]] = {
    "HR":     [220045],
    "SBP":    [220179, 220050],   # NBP systolic, ABP systolic
    "DBP":    [220180, 220051],   # NBP diastolic, ABP diastolic
    "MAP":    [220181, 220052, 225312],  # NBP mean, ABP mean, ART BP mean
    "Resp":   [220210, 224690],
    "O2Sat":  [220277],           # SpO2
    "Temp_C": [223762],           # Temperature Celsius
    "Temp_F": [223761],           # Temperature Fahrenheit (converted → C)
}

# labevents (hosp) item ids → canonical lab name (matches features.LAB_COLUMNS).
_LAB_ITEMIDS: dict[str, list[int]] = {
    "Creatinine":       [50912],
    "BUN":              [51006],
    "Lactate":          [50813],
    "WBC":              [51301, 51300],
    "Platelets":        [51265],
    "Hgb":              [51222],
    "Hct":              [51221],
    "Glucose":          [50931, 50809],
    "Potassium":        [50971, 50822],
    "Chloride":         [50902, 50806],
    "HCO3":             [50882],
    "Bilirubin_total":  [50885],
    "Calcium":          [50893],
    "Magnesium":        [50960],
    "Phosphate":        [50970],
    "AST":              [50878],
    "Alkalinephos":     [50863],
    "PaCO2":            [50818],
    "pH":               [50820, 50831],
    "SaO2":             [50817],
    "BaseExcess":       [50802],
    "PTT":              [51275],
    "Fibrinogen":       [51214],
}

# outputevents (icu) urine-output item ids. 227488 (GU irrigant *in*) is
# subtracted from the hourly total; everything else is urine *out*.
_URINE_OUT_ITEMIDS: list[int] = [
    226559,  # Foley
    226560,  # Void
    226561,  # Condom Cath
    226584,  # Ileoconduit
    226563,  # Suprapubic
    226564,  # R Nephrostomy
    226565,  # L Nephrostomy
    226567,  # Straight Cath
    226557,  # R Ureteral Stent
    226558,  # L Ureteral Stent
    227489,  # GU Irrigant/Urine Volume Out
]
_URINE_IRRIGANT_ITEMID: int = 227488  # GU Irrigant Volume (in) → subtract

# chartevents weight item ids (kg unless noted).
_WEIGHT_KG_ITEMIDS: list[int] = [226512, 224639]   # Admission Weight (Kg), Daily Weight
_WEIGHT_LB_ITEMID: int = 226531                     # Admission Weight (lbs)

# Antibiotic name fragments used for the suspicion-of-infection sepsis proxy.
_ANTIBIOTIC_KEYWORDS: tuple[str, ...] = (
    "vancomycin", "piperacillin", "tazobactam", "cefepime", "ceftriaxone",
    "meropenem", "ciprofloxacin", "levofloxacin", "metronidazole", "azithromycin",
    "ampicillin", "gentamicin", "ceftazidime", "imipenem", "linezolid",
    "aztreonam", "clindamycin", "tobramycin", "doxycycline", "cefazolin",
)

# chartevents/labevents are streamed in chunks of this many rows rather than
# loaded whole. Full MIMIC-IV chartevents is ~330M rows and labevents is
# ~150M+ rows — a single pd.read_csv risks OOM on a standard ~13 GB Colab
# runtime. The demo (12k rows) runs through the same code path in one chunk.
_CHUNK_ROWS = 5_000_000


# ---------------------------------------------------------------------------
# Low-level table loading
# ---------------------------------------------------------------------------
def _find_table(data_dir: str, name: str) -> Optional[str]:
    """Locate a MIMIC ``{name}.csv.gz`` anywhere under *data_dir* (case-insensitive)."""
    matches = glob.glob(os.path.join(data_dir, "**", f"{name}.csv.gz"), recursive=True)
    if not matches:
        matches = glob.glob(os.path.join(data_dir, "**", f"{name}.csv"), recursive=True)
    return matches[0] if matches else None


def _read_table(
    data_dir: str,
    name: str,
    usecols: Optional[list[str]] = None,
    parse_dates: Optional[list[str]] = None,
    required: bool = True,
) -> pd.DataFrame:
    """Read a single MIMIC table by logical name, returning empty DF if optional + missing."""
    path = _find_table(data_dir, name)
    if path is None:
        msg = f"MIMIC table '{name}.csv.gz' not found under {data_dir}"
        if required:
            raise FileNotFoundError(msg)
        logger.warning("%s — skipping (optional).", msg)
        return pd.DataFrame()
    logger.info("Reading %s", os.path.basename(path))
    return pd.read_csv(path, usecols=usecols, parse_dates=parse_dates, low_memory=False)


def load_mimic_demo(data_dir: str) -> dict[str, pd.DataFrame]:
    """Load the MIMIC-IV demo tables needed for the SA-AKI pipeline.

    Parameters
    ----------
    data_dir : str
        Path to the extracted demo (parent of the ``hosp/`` and ``icu/`` dirs).
        Tables are located recursively, so either the dataset root or its parent
        works.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: ``patients``, ``icustays``, ``outputevents`` (required);
        ``prescriptions``, ``microbiologyevents`` (optional, used for the
        sepsis proxy). ``chartevents``/``labevents`` are deliberately absent —
        they are streamed from disk by :func:`build_hourly_cohort` instead.
    """
    tables = {
        "patients": _read_table(
            data_dir, "patients",
            usecols=["subject_id", "gender", "anchor_age"],
        ),
        "icustays": _read_table(
            data_dir, "icustays",
            usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"],
            parse_dates=["intime", "outtime"],
        ),
        # chartevents and labevents are NOT eagerly loaded here — at full
        # MIMIC-IV scale they risk OOM. build_hourly_cohort() streams them
        # directly from disk in chunks (see _stream_chartevents / _pivot_labs).
        "outputevents": _read_table(
            data_dir, "outputevents",
            usecols=["stay_id", "charttime", "itemid", "value"],
            parse_dates=["charttime"],
        ),
        # Optional — only for the Sepsis-3 suspicion-of-infection proxy.
        "prescriptions": _read_table(
            data_dir, "prescriptions",
            usecols=["subject_id", "hadm_id", "starttime", "drug"],
            parse_dates=["starttime"], required=False,
        ),
        "microbiologyevents": _read_table(
            data_dir, "microbiologyevents",
            usecols=["subject_id", "hadm_id", "charttime", "chartdate"],
            parse_dates=["charttime", "chartdate"], required=False,
        ),
    }
    return tables


# ---------------------------------------------------------------------------
# Reshaping helpers (long → hourly wide, per ICU stay)
# ---------------------------------------------------------------------------
def _hour_offset(charttime: pd.Series, intime: pd.Series) -> pd.Series:
    """Integer hours from ICU admission (floored). Negative values = pre-ICU."""
    return ((charttime - intime).dt.total_seconds() // 3600).astype("Int64")


def _stream_chartevents(
    data_dir: str, stay_intime: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Single chunked pass over ``chartevents.csv.gz`` producing both:

    * hourly vitals — wide frame keyed ``(stay_id, ICULOS)``
    * per-stay weight (kg) — used to scale urine output

    Combined into one function so the (potentially ~330M-row) file is only
    streamed once. Reads in ``_CHUNK_ROWS``-row chunks, accumulating partial
    sum/count per group, so memory stays bounded regardless of dataset size —
    the demo (12k rows) and full MIMIC-IV both go through this same path.
    """
    path = _find_table(data_dir, "chartevents")
    if path is None:
        logger.warning("chartevents table not found under %s.", data_dir)
        return (
            pd.DataFrame(columns=["stay_id", "ICULOS"]),
            pd.DataFrame(columns=["stay_id", "weight_kg"]),
        )

    item_to_name: dict[int, str] = {}
    for name, ids in _VITAL_ITEMIDS.items():
        for i in ids:
            item_to_name[i] = name
    weight_items = set(_WEIGHT_KG_ITEMIDS) | {_WEIGHT_LB_ITEMID}
    relevant_items = set(item_to_name) | weight_items

    vital_partials: list[pd.DataFrame] = []
    weight_rows: list[pd.DataFrame] = []

    reader = pd.read_csv(
        path,
        usecols=["stay_id", "charttime", "itemid", "valuenum"],
        parse_dates=["charttime"],
        chunksize=_CHUNK_ROWS,
        low_memory=False,
    )
    for chunk in reader:
        relevant = chunk[chunk["itemid"].isin(relevant_items)]
        if relevant.empty:
            continue

        # --- weights (kg, converting lb→kg; plausibility filter) -----------
        w = relevant[relevant["itemid"].isin(weight_items)][["stay_id", "itemid", "valuenum"]].copy()
        if not w.empty:
            lb_mask = w["itemid"] == _WEIGHT_LB_ITEMID
            w.loc[lb_mask, "valuenum"] = w.loc[lb_mask, "valuenum"] * 0.453592
            w = w[(w["valuenum"] > 20) & (w["valuenum"] < 400)]
            if not w.empty:
                weight_rows.append(w[["stay_id", "valuenum"]])

        # --- vitals ----------------------------------------------------------
        ce = relevant[relevant["itemid"].isin(item_to_name)].merge(
            stay_intime, on="stay_id", how="inner"
        )
        if ce.empty:
            continue
        ce = ce.assign(ICULOS=_hour_offset(ce["charttime"], ce["intime"]))
        ce = ce[ce["ICULOS"] >= 0].copy()
        ce["var"] = ce["itemid"].map(item_to_name)

        # Convert Fahrenheit → Celsius then collapse the two temp channels.
        f_mask = ce["var"] == "Temp_F"
        ce.loc[f_mask, "valuenum"] = (ce.loc[f_mask, "valuenum"] - 32.0) * 5.0 / 9.0
        ce.loc[ce["var"].isin(["Temp_C", "Temp_F"]), "var"] = "Temp"

        vital_partials.append(
            ce.groupby(["stay_id", "ICULOS", "var"])["valuenum"].agg(["sum", "count"])
        )

    if vital_partials:
        totals = pd.concat(vital_partials).groupby(level=[0, 1, 2]).sum()
        vitals = (totals["sum"] / totals["count"]).unstack("var").reset_index()
    else:
        vitals = pd.DataFrame(columns=["stay_id", "ICULOS"])

    if weight_rows:
        all_weights = pd.concat(weight_rows, ignore_index=True)
        weights = all_weights.groupby("stay_id")["valuenum"].median().reset_index()
        weights = weights.rename(columns={"valuenum": "weight_kg"})
    else:
        weights = pd.DataFrame(columns=["stay_id", "weight_kg"])

    return vitals, weights


def _pivot_labs(data_dir: str, stay_keys: pd.DataFrame) -> pd.DataFrame:
    """Hourly mean of each lab per stay → wide frame keyed (stay_id, ICULOS).

    Labs are charted at the hadm level, so we map them onto ICU stays via
    (subject_id, hadm_id) and the stay's admission time. Streams
    ``labevents.csv.gz`` in ``_CHUNK_ROWS``-row chunks — full MIMIC-IV
    labevents is well over 100M rows, so a single-shot read risks OOM.
    """
    path = _find_table(data_dir, "labevents")
    if path is None:
        logger.warning("labevents table not found under %s.", data_dir)
        return pd.DataFrame(columns=["stay_id", "ICULOS"])

    item_to_name: dict[int, str] = {}
    for name, ids in _LAB_ITEMIDS.items():
        for i in ids:
            item_to_name[i] = name

    partials: list[pd.DataFrame] = []
    reader = pd.read_csv(
        path,
        usecols=["subject_id", "hadm_id", "charttime", "itemid", "valuenum"],
        parse_dates=["charttime"],
        chunksize=_CHUNK_ROWS,
        low_memory=False,
    )
    for chunk in reader:
        le = chunk[chunk["itemid"].isin(item_to_name)]
        if le.empty:
            continue
        le = le.merge(stay_keys, on=["subject_id", "hadm_id"], how="inner")
        if le.empty:
            continue
        le = le.assign(ICULOS=_hour_offset(le["charttime"], le["intime"]))
        le = le[le["ICULOS"] >= 0].copy()
        le["var"] = le["itemid"].map(item_to_name)

        partials.append(
            le.groupby(["stay_id", "ICULOS", "var"])["valuenum"].agg(["sum", "count"])
        )

    if not partials:
        return pd.DataFrame(columns=["stay_id", "ICULOS"])

    totals = pd.concat(partials).groupby(level=[0, 1, 2]).sum()
    wide = (totals["sum"] / totals["count"]).unstack("var").reset_index()
    return wide


def _hourly_urine(
    outputevents: pd.DataFrame,
    stay_intime: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    """Hourly urine output rate (mL/kg/hr) + observation mask per stay.

    Returns columns: ``stay_id``, ``ICULOS``, ``urine_rate``, ``urine_observed``.
    Hours with no charted urine are simply absent here (→ NaN after the merge,
    ``urine_observed`` filled to 0). They are **never** treated as 0 mL/kg/hr.
    """
    if outputevents.empty:
        return pd.DataFrame(columns=["stay_id", "ICULOS", "urine_rate", "urine_observed"])

    oe = outputevents[
        outputevents["itemid"].isin(_URINE_OUT_ITEMIDS + [_URINE_IRRIGANT_ITEMID])
    ].copy()
    oe = oe.merge(stay_intime, on="stay_id", how="inner")
    oe["ICULOS"] = _hour_offset(oe["charttime"], oe["intime"])
    oe = oe[oe["ICULOS"] >= 0]

    # Irrigant flowing in is logged as urine volume out → subtract it.
    oe["signed_value"] = np.where(
        oe["itemid"] == _URINE_IRRIGANT_ITEMID, -oe["value"], oe["value"]
    )

    hourly = oe.groupby(["stay_id", "ICULOS"])["signed_value"].sum().reset_index()
    hourly = hourly.rename(columns={"signed_value": "urine_ml"})
    hourly["urine_ml"] = hourly["urine_ml"].clip(lower=0)  # net negative = charting noise
    hourly["urine_observed"] = 1

    # Scale to mL/kg/hr using the stay weight (fallback 70 kg if unknown).
    hourly = hourly.merge(weights, on="stay_id", how="left")
    hourly["weight_kg"] = hourly["weight_kg"].fillna(70.0)
    hourly["urine_rate"] = hourly["urine_ml"] / hourly["weight_kg"]

    return hourly[["stay_id", "ICULOS", "urine_rate", "urine_observed", "weight_kg"]]


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------
def build_hourly_cohort(tables: dict[str, pd.DataFrame], data_dir: str) -> pd.DataFrame:
    """Assemble the canonical hourly time-series frame from raw MIMIC tables.

    One row per (ICU stay, hour-since-admission), spanning hour 0 through the
    stay's length-of-stay. See the module docstring for the exact schema.

    Parameters
    ----------
    tables : dict
        Output of :func:`load_mimic_demo` (small tables, already in memory).
    data_dir : str
        Same directory passed to :func:`load_mimic_demo`. chartevents and
        labevents are streamed from here directly (see :func:`_stream_chartevents`
        / :func:`_pivot_labs`) rather than held fully in memory.
    """
    icustays = tables["icustays"].copy()
    patients = tables["patients"].copy()
    if icustays.empty:
        logger.warning("No ICU stays found — returning empty cohort.")
        return pd.DataFrame()

    icustays["stay_id"] = icustays["stay_id"].astype(int)
    stay_intime = icustays[["stay_id", "intime"]]
    stay_keys = icustays[["subject_id", "hadm_id", "stay_id", "intime"]]

    # --- Build the dense (stay_id, ICULOS) grid -----------------------------
    icustays["n_hours"] = np.ceil(icustays["los"].fillna(0) * 24).astype(int).clip(lower=1)
    grid_parts = [
        pd.DataFrame({"stay_id": sid, "ICULOS": np.arange(n)})
        for sid, n in zip(icustays["stay_id"], icustays["n_hours"])
    ]
    grid = pd.concat(grid_parts, ignore_index=True)

    # --- Reshape each source stream (chartevents/labevents streamed from disk) -
    vitals, weights = _stream_chartevents(data_dir, stay_intime)
    labs = _pivot_labs(data_dir, stay_keys)
    urine = _hourly_urine(tables["outputevents"], stay_intime, weights)

    # --- Merge everything onto the grid -------------------------------------
    df = grid.merge(vitals, on=["stay_id", "ICULOS"], how="left")
    df = df.merge(labs, on=["stay_id", "ICULOS"], how="left")
    df = df.merge(urine, on=["stay_id", "ICULOS"], how="left")

    # Urine mask: charted → 1, otherwise 0 (NEVER fill the *rate* with 0).
    df["urine_observed"] = df["urine_observed"].fillna(0).astype(int)

    # --- Attach static demographics -----------------------------------------
    static = icustays[["stay_id", "subject_id"]].merge(
        patients[["subject_id", "gender", "anchor_age"]], on="subject_id", how="left"
    )
    static = static.rename(columns={"anchor_age": "Age"})
    static["Gender"] = (static["gender"].astype(str).str.upper() == "M").astype(int)
    df = df.merge(static[["stay_id", "subject_id", "Age", "Gender"]], on="stay_id", how="left")

    # --- Canonical identifiers / site --------------------------------------
    df["patient_id"] = df["stay_id"].astype(str)
    df["hospital_system"] = "A"   # demo is single-site; eICU adds 'B' later

    # Guarantee every canonical clinical column exists even if absent in demo.
    for col in ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]:
        if col not in df.columns:
            df[col] = np.nan
    df = df.sort_values(["patient_id", "ICULOS"]).reset_index(drop=True)
    logger.info(
        "Built hourly cohort: %d stays, %d patient-hours, %d columns.",
        df["patient_id"].nunique(), len(df), df.shape[1],
    )
    return df


# ---------------------------------------------------------------------------
# Cohort criteria
# ---------------------------------------------------------------------------
def apply_inclusion_criteria(df: pd.DataFrame, min_age: int = 18, min_hours: int = 8) -> pd.DataFrame:
    """Keep adult stays (Age ≥ *min_age*) with at least *min_hours* of ICU data."""
    if df.empty:
        return df
    initial = df["patient_id"].nunique()

    if "Age" in df.columns:
        df = df[df["Age"].fillna(0) >= min_age]

    stay_lengths = df.groupby("patient_id")["ICULOS"].max()
    valid = stay_lengths[stay_lengths >= min_hours].index
    df = df[df["patient_id"].isin(valid)]

    logger.info("Inclusion criteria: %d → %d stays.", initial, df["patient_id"].nunique())
    return df


def identify_sepsis_onset(
    df: pd.DataFrame,
    tables: Optional[dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Add ``t_sepsis`` (ICULOS hour of sepsis onset) per stay.

    Uses a **suspicion-of-infection** proxy for Sepsis-3: the time of the first
    IV antibiotic order that is paired with a culture (antibiotic within 72 h of
    a culture, or culture within 24 h of an antibiotic), mapped onto the ICU
    hour axis. This is the standard "suspected infection" half of Sepsis-3.

    .. note::
       The SOFA-increase ≥ 2 component is **not** yet applied — this is a
       documented simplification for the demo. Refine once the full data is in
       and a SOFA computation is available. If ``prescriptions`` /
       ``microbiologyevents`` are unavailable, every included stay is treated as
       septic from hour 0 (so the pipeline still runs end-to-end on minimal
       tables).
    """
    if df.empty:
        return df

    df = df.copy()
    presc = (tables or {}).get("prescriptions", pd.DataFrame())
    micro = (tables or {}).get("microbiologyevents", pd.DataFrame())

    if presc.empty or micro.empty:
        logger.warning(
            "prescriptions/microbiologyevents unavailable — using fallback "
            "t_sepsis = 0 for all included stays."
        )
        df["t_sepsis"] = 0.0
        return df

    # First qualifying antibiotic order per hadm.
    presc = presc.copy()
    presc["drug_l"] = presc["drug"].astype(str).str.lower()
    abx = presc[presc["drug_l"].str.contains("|".join(_ANTIBIOTIC_KEYWORDS), na=False)]
    abx_first = abx.groupby(["subject_id", "hadm_id"])["starttime"].min().reset_index()
    abx_first = abx_first.rename(columns={"starttime": "abx_time"})

    # First culture per hadm (charttime if present else chartdate).
    micro = micro.copy()
    micro["cx_time"] = micro["charttime"].fillna(micro["chartdate"])
    cx_first = micro.groupby(["subject_id", "hadm_id"])["cx_time"].min().reset_index()

    susp = abx_first.merge(cx_first, on=["subject_id", "hadm_id"], how="inner")
    # Pair antibiotic + culture within the standard window.
    delta = (susp["abx_time"] - susp["cx_time"]).dt.total_seconds() / 3600.0
    paired = susp[(delta.between(-24, 72))].copy()
    paired["infection_time"] = paired[["abx_time", "cx_time"]].min(axis=1)

    # Map infection_time onto the ICU hour axis for each stay in that hadm.
    icustays = tables["icustays"][["subject_id", "hadm_id", "stay_id", "intime"]].copy()
    icustays["stay_id"] = icustays["stay_id"].astype(int)
    onset = icustays.merge(paired[["subject_id", "hadm_id", "infection_time"]],
                           on=["subject_id", "hadm_id"], how="left")
    onset["t_sepsis"] = ((onset["infection_time"] - onset["intime"]).dt.total_seconds() // 3600)
    onset["patient_id"] = onset["stay_id"].astype(str)
    # Onset must fall at or after ICU admission to count for this stay.
    onset.loc[onset["t_sepsis"] < 0, "t_sepsis"] = np.nan

    df = df.merge(onset[["patient_id", "t_sepsis"]], on="patient_id", how="left")
    n_septic = df.loc[df["t_sepsis"].notna(), "patient_id"].nunique()
    logger.info("Sepsis onset identified for %d stays.", n_septic)
    return df


def compute_baseline_creatinine(patient_df: pd.DataFrame) -> float:
    """Baseline serum creatinine for one stay (mg/dL).

    Uses the minimum creatinine at/ before sepsis onset; falls back to the
    first-24h minimum, then the overall minimum.
    """
    if "Creatinine" not in patient_df.columns:
        return np.nan
    cr = patient_df["Creatinine"].dropna()
    if cr.empty:
        return np.nan

    if "t_sepsis" in patient_df.columns and not pd.isna(patient_df["t_sepsis"].iloc[0]):
        t_sepsis = patient_df["t_sepsis"].iloc[0]
        pre = patient_df[patient_df["ICULOS"] <= t_sepsis]["Creatinine"].dropna()
        if not pre.empty:
            return float(pre.min())

    first_24h = patient_df[patient_df["ICULOS"] <= 24]["Creatinine"].dropna()
    if not first_24h.empty:
        return float(first_24h.min())
    return float(cr.min())


def define_cohort(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Finalise the cohort and compute per-stay baseline creatinine.

    Excludes stays with **prevalent AKI** at sepsis onset (already meeting KDIGO
    Stage-1 creatinine criteria), so the model learns *onset* not *persistence*.

    Returns
    -------
    (cohort_df, baseline_df)
        ``cohort_df``   — hourly rows for included stays, with ``baseline_creatinine``.
        ``baseline_df`` — one row per stay: ``patient_id``, ``baseline_creatinine``.
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(columns=["patient_id", "baseline_creatinine"])

    df = df.copy()
    if "t_sepsis" not in df.columns:
        df["t_sepsis"] = 0.0

    # Baseline creatinine per stay.
    baselines = (
        df.groupby("patient_id", group_keys=False)
        .apply(compute_baseline_creatinine)
        .reset_index(name="baseline_creatinine")
    )
    df = df.merge(baselines, on="patient_id", how="left")

    # Prevalent-AKI exclusion at t_sepsis.
    def _prevalent_aki(pdf: pd.DataFrame) -> bool:
        t_sepsis = pdf["t_sepsis"].iloc[0]
        baseline = pdf["baseline_creatinine"].iloc[0]
        if pd.isna(t_sepsis) or pd.isna(baseline):
            return False
        at_sepsis = pdf[pdf["ICULOS"] == t_sepsis]["Creatinine"].dropna()
        if at_sepsis.empty:
            return False
        cr = at_sepsis.iloc[0]
        return bool(cr >= baseline + 0.3 or cr >= baseline * 1.5)

    aki_flags = df.groupby("patient_id").apply(_prevalent_aki)
    excluded = aki_flags[aki_flags].index
    cohort_df = df[~df["patient_id"].isin(excluded)].copy()

    logger.info(
        "Final cohort: %d stays (excluded %d for prevalent AKI).",
        cohort_df["patient_id"].nunique(), len(excluded),
    )
    return cohort_df, baselines


def print_data_summary(df: pd.DataFrame) -> None:
    """Print a brief summary of the hourly cohort."""
    if df.empty:
        print("Dataset is empty.")
        return
    print("-" * 44)
    print("MIMIC-IV DEMO COHORT SUMMARY")
    print("-" * 44)
    print(f"ICU stays:          {df['patient_id'].nunique():,}")
    print(f"Patient-hours:      {len(df):,}")
    if "t_sepsis" in df.columns:
        septic = df.loc[df["t_sepsis"].notna(), "patient_id"].nunique()
        print(f"Septic stays:       {septic:,}")
    if "Creatinine" in df.columns:
        print(f"Creatinine values:  {df['Creatinine'].notna().sum():,}")
    if "urine_observed" in df.columns:
        print(f"Hours w/ urine:     {int(df['urine_observed'].sum()):,} "
              f"({100 * df['urine_observed'].mean():.1f}%)")
    print("-" * 44)
