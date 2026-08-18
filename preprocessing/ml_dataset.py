"""
ML Dataset Preparation

Purpose:
--------
Prepare the existing processed scheduler dataset
for CPU requirement prediction.

Target:
-------
CPU_Requirement

Excluded:
---------
Process_ID       -> unique identifier
Completion_Time  -> future information
Burst_Time       -> derived using Completion_Time
Required_Cores   -> derived directly from CPU_Requirement
"""

import pandas as pd


# =========================================================
# FILE PATHS
# =========================================================

INPUT_FILE = (
    "C:/Users/gvkva/Desktop/My ongoing projects/"
    "Machine Learning Project/Adaptive-Multi-Core-CPU-Scheduler-RL/"
    "data/processed/processedscheduler_dataset.csv"
)

OUTPUT_FILE = (
    "C:/Users/gvkva/Desktop/My ongoing projects/"
    "Machine Learning Project/Adaptive-Multi-Core-CPU-Scheduler-RL/"
    "data/processed/ml_cpu_dataset.csv"
)


# =========================================================
# LOAD DATA
# =========================================================

print("=" * 60)
print("LOADING PROCESSED DATASET")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df)}")


# =========================================================
# SELECT ML COLUMNS
# =========================================================

feature_columns = [
    "Task_ID",
    "Job_ID",
    "PC_ID",
    "Arrival_Time",
    "Memory_Requirement",
]

target_column = "CPU_Requirement"

ml_df = df[feature_columns + [target_column]].copy()


# =========================================================
# REMOVE INVALID TARGET VALUES
# =========================================================

ml_df = ml_df.dropna(subset=[target_column])

ml_df = ml_df[
    ml_df[target_column] >= 0
]


# =========================================================
# SAVE ML DATASET
# =========================================================

ml_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================================
# DISPLAY RESULTS
# =========================================================

print("\n" + "=" * 60)
print("ML DATASET CREATED")
print("=" * 60)

print(f"Rows: {len(ml_df)}")
print(f"Columns: {len(ml_df.columns)}")

print("\nColumns:")
print(ml_df.columns.tolist())

print("\nFirst 5 rows:")
print(ml_df.head())

print("\nMissing values:")
print(ml_df.isnull().sum())

print("\nTarget statistics:")
print(ml_df[target_column].describe())

print("\nSaved to:")
print(OUTPUT_FILE)