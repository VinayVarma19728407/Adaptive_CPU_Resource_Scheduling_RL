"""
Create Leakage-Free Historical Features

Purpose:
--------
Create historical CPU usage features using only records that
occurred BEFORE the current process.

Target:
-------
CPU_Requirement

Historical features:
--------------------
Task historical CPU statistics
PC historical CPU statistics

Important:
----------
The current row's CPU_Requirement is NEVER included in its
historical statistics.
"""

import pandas as pd


# =========================================================
# FILE PATHS
# =========================================================

INPUT_FILE = (
    "C:/Users/gvkva/Desktop/My ongoing projects/"
    "Machine Learning Project/Adaptive-Multi-Core-CPU-Scheduler-RL/"
    "data/processed/ml_cpu_dataset.csv"
)

OUTPUT_FILE = (
    "C:/Users/gvkva/Desktop/My ongoing projects/"
    "Machine Learning Project/Adaptive-Multi-Core-CPU-Scheduler-RL/"
    "data/processed/ml_cpu_historical.csv"
)


# =========================================================
# LOAD DATA
# =========================================================

print("=" * 60)
print("LOADING ML DATASET")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df)}")


# =========================================================
# SORT CHRONOLOGICALLY
# =========================================================

print("\n" + "=" * 60)
print("SORTING BY ARRIVAL TIME")
print("=" * 60)

df = df.sort_values(
    ["Arrival_Time"],
    kind="mergesort"
).reset_index(drop=True)


# =========================================================
# TASK HISTORICAL FEATURES
# =========================================================

print("\n" + "=" * 60)
print("CREATING TASK HISTORICAL FEATURES")
print("=" * 60)

task_group = df.groupby("Task_ID")["CPU_Requirement"]

df["Task_Previous_Count"] = (
    task_group.cumcount()
)

df["Task_Historical_Mean"] = (
    task_group.transform(
        lambda x: x.shift(1).expanding().mean()
    )
)

df["Task_Historical_Std"] = (
    task_group.transform(
        lambda x: x.shift(1).expanding().std()
    )
)

df["Task_Historical_Min"] = (
    task_group.transform(
        lambda x: x.shift(1).expanding().min()
    )
)

df["Task_Historical_Max"] = (
    task_group.transform(
        lambda x: x.shift(1).expanding().max()
    )
)


# =========================================================
# PC HISTORICAL FEATURES
# =========================================================

print("\n" + "=" * 60)
print("CREATING PC HISTORICAL FEATURES")
print("=" * 60)

pc_group = df.groupby("PC_ID")["CPU_Requirement"]

df["PC_Previous_Count"] = (
    pc_group.cumcount()
)

df["PC_Historical_Mean"] = (
    pc_group.transform(
        lambda x: x.shift(1).expanding().mean()
    )
)

df["PC_Historical_Std"] = (
    pc_group.transform(
        lambda x: x.shift(1).expanding().std()
    )
)

df["PC_Historical_Min"] = (
    pc_group.transform(
        lambda x: x.shift(1).expanding().min()
    )
)

df["PC_Historical_Max"] = (
    pc_group.transform(
        lambda x: x.shift(1).expanding().max()
    )
)


# =========================================================
# MISSING HISTORICAL VALUES
# =========================================================

print("\n" + "=" * 60)
print("HANDLING INITIAL HISTORICAL VALUES")
print("=" * 60)

historical_columns = [
    "Task_Historical_Mean",
    "Task_Historical_Std",
    "Task_Historical_Min",
    "Task_Historical_Max",
    "PC_Historical_Mean",
    "PC_Historical_Std",
    "PC_Historical_Min",
    "PC_Historical_Max",
]

for column in historical_columns:

    df[column] = df[column].fillna(
        df["CPU_Requirement"].mean()
    )


# =========================================================
# SAVE
# =========================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================================
# RESULTS
# =========================================================

print("\n" + "=" * 60)
print("HISTORICAL FEATURE DATASET CREATED")
print("=" * 60)

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

print("\nNew historical features:")

for column in [
    "Task_Previous_Count",
    "Task_Historical_Mean",
    "Task_Historical_Std",
    "Task_Historical_Min",
    "Task_Historical_Max",
    "PC_Previous_Count",
    "PC_Historical_Mean",
    "PC_Historical_Std",
    "PC_Historical_Min",
    "PC_Historical_Max",
]:
    print(f"  - {column}")

print("\nFirst 5 rows:")
print(df.head())

print("\nSaved to:")
print(OUTPUT_FILE)