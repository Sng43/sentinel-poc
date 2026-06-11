"""
Data loading and cohort definition for Project Sentinel.
"""
import os
import glob
import logging
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_physionet_2019(data_dir: str) -> pd.DataFrame:
    """
    Loads PhysioNet 2019 CSV/PSV files into a single DataFrame.
    Caches to parquet to speed up subsequent loads.
    """
    cache_path = os.path.join(data_dir, "physionet_2019_cache.parquet")
    if os.path.exists(cache_path):
        logger.info(f"Loading data from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    logger.info(f"Loading data from raw files in {data_dir}")
    all_files = glob.glob(os.path.join(data_dir, "**", "*.psv"), recursive=True)
    if not all_files:
        logger.warning(f"No .psv files found in {data_dir}")
        return pd.DataFrame()

    df_list = []
    for file in tqdm(all_files, desc="Reading PSV files"):
        try:
            temp_df = pd.read_csv(file, sep="|")
            # Extract patient ID from filename (e.g., p000001.psv)
            patient_id = os.path.basename(file).split(".")[0]
            temp_df["Patient_ID"] = patient_id
            df_list.append(temp_df)
        except Exception as e:
            logger.error(f"Error reading {file}: {e}")

    if not df_list:
        return pd.DataFrame()

    full_df = pd.concat(df_list, ignore_index=True)
    
    # Ensure ICULOS is integer and handle NaN
    if 'ICULOS' in full_df.columns:
        full_df['ICULOS'] = full_df['ICULOS'].fillna(0).astype(int)

    logger.info(f"Saving combined data to cache: {cache_path}")
    full_df.to_parquet(cache_path, index=False)
    
    return full_df


def apply_inclusion_criteria(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters patients based on inclusion criteria:
    - Age >= 18
    - At least 8 hours of ICU stay
    """
    logger.info("Applying inclusion criteria...")
    if df.empty:
        return df
        
    initial_patients = df['Patient_ID'].nunique()
    
    # Age criteria
    if 'Age' in df.columns:
        df = df[df['Age'] >= 18]
        
    # Minimum 8 hours ICU stay
    stay_lengths = df.groupby('Patient_ID')['ICULOS'].max()
    valid_patients = stay_lengths[stay_lengths >= 8].index
    df = df[df['Patient_ID'].isin(valid_patients)]
    
    final_patients = df['Patient_ID'].nunique()
    logger.info(f"Patients before: {initial_patients}, after: {final_patients}")
    
    return df


def identify_sepsis_onset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies the time of sepsis onset (t_sepsis) for each patient.
    """
    logger.info("Identifying sepsis onset...")
    if df.empty or 'SepsisLabel' not in df.columns:
        return df

    # Find the first row where SepsisLabel == 1 for each patient
    sepsis_rows = df[df['SepsisLabel'] == 1].groupby('Patient_ID').first().reset_index()
    sepsis_times = sepsis_rows[['Patient_ID', 'ICULOS']].rename(columns={'ICULOS': 't_sepsis'})
    
    df = df.merge(sepsis_times, on='Patient_ID', how='left')
    return df


def compute_baseline_creatinine(patient_df: pd.DataFrame) -> float:
    """
    Computes baseline creatinine for a single patient.
    Uses the minimum creatinine value prior to t_sepsis or within the first 24h.
    """
    if 'Creatinine' not in patient_df.columns:
        return np.nan
        
    cr_values = patient_df['Creatinine'].dropna()
    if cr_values.empty:
        return np.nan
        
    # If we have t_sepsis, look at values before it
    if 't_sepsis' in patient_df.columns and not pd.isna(patient_df['t_sepsis'].iloc[0]):
        t_sepsis = patient_df['t_sepsis'].iloc[0]
        pre_sepsis = patient_df[patient_df['ICULOS'] <= t_sepsis]['Creatinine'].dropna()
        if not pre_sepsis.empty:
            return float(pre_sepsis.min())
            
    # Fallback to first 24h min
    first_24h = patient_df[patient_df['ICULOS'] <= 24]['Creatinine'].dropna()
    if not first_24h.empty:
        return float(first_24h.min())
        
    # Fallback to overall min
    return float(cr_values.min())


def define_cohort(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Defines the final cohort by applying exclusion criteria and computing baselines.
    Excludes patients with prevalent AKI at t_sepsis.
    Returns: (cohort_df, excluded_df)
    """
    logger.info("Defining final cohort...")
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
        
    df = identify_sepsis_onset(df)
    
    # Calculate baseline creatinine for all patients
    tqdm.pandas(desc="Computing baselines")
    baselines = df.groupby('Patient_ID').progress_apply(compute_baseline_creatinine).reset_index(name='baseline_cr')
    df = df.merge(baselines, on='Patient_ID', how='left')
    
    # Identify prevalent AKI at t_sepsis
    # KDIGO criteria: Cr increase by >= 0.3 mg/dl or >= 1.5x baseline
    def has_prevalent_aki(pdf):
        if pd.isna(pdf['t_sepsis'].iloc[0]) or pd.isna(pdf['baseline_cr'].iloc[0]):
            return False
            
        t_sepsis = pdf['t_sepsis'].iloc[0]
        baseline = pdf['baseline_cr'].iloc[0]
        
        at_sepsis = pdf[pdf['ICULOS'] == t_sepsis]['Creatinine']
        if at_sepsis.empty or pd.isna(at_sepsis.iloc[0]):
            return False
            
        cr_sepsis = at_sepsis.iloc[0]
        return cr_sepsis >= baseline + 0.3 or cr_sepsis >= baseline * 1.5

    aki_status = df.groupby('Patient_ID').apply(has_prevalent_aki).reset_index(name='prevalent_aki')
    df = df.merge(aki_status, on='Patient_ID', how='left')
    
    # Split into cohort and excluded
    # Exclude if prevalent AKI is True
    excluded_patients = df[df['prevalent_aki'] == True]['Patient_ID'].unique()
    
    cohort_df = df[~df['Patient_ID'].isin(excluded_patients)].copy()
    excluded_df = df[df['Patient_ID'].isin(excluded_patients)].copy()
    
    # Drop intermediate columns to clean up
    for col in ['prevalent_aki']:
        if col in cohort_df.columns:
            cohort_df = cohort_df.drop(columns=[col])
        if col in excluded_df.columns:
            excluded_df = excluded_df.drop(columns=[col])
            
    logger.info(f"Final cohort size: {cohort_df['Patient_ID'].nunique()} patients")
    logger.info(f"Excluded due to prevalent AKI: {len(excluded_patients)} patients")
    
    return cohort_df, excluded_df


def print_data_summary(df: pd.DataFrame) -> None:
    """
    Prints a brief summary of the dataset.
    """
    if df.empty:
        print("Dataset is empty.")
        return
        
    num_patients = df['Patient_ID'].nunique()
    total_hours = len(df)
    
    print("-" * 40)
    print("DATASET SUMMARY")
    print("-" * 40)
    print(f"Total Patients:     {num_patients:,}")
    print(f"Total Patient Hours: {total_hours:,}")
    
    if 'SepsisLabel' in df.columns:
        sepsis_patients = df[df['SepsisLabel'] == 1]['Patient_ID'].nunique()
        print(f"Patients with Sepsis: {sepsis_patients:,} ({sepsis_patients/num_patients*100:.1f}%)")
        
    if 'Creatinine' in df.columns:
        cr_measurements = df['Creatinine'].notna().sum()
        print(f"Creatinine Measurements: {cr_measurements:,}")
        
    print("-" * 40)
