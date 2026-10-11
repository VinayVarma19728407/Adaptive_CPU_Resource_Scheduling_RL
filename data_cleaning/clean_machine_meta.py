"""
Clean Machine Metadata Files.

This script implements the cleaning pipeline for physical node specifications (machine_meta). It standardizes machine IDs,
core counts, memory capacities, GPU attributes, and node status values.
"""

import os
import pandas as pd

PROJECT_PATH = ("C:/Users/gvkva/Desktop/My ongoing projects/""Machine Learning Project/""Adaptive-Multi-Core-CPU-Scheduler-RL")

RAW_MACHINE_META_PATH = os.path.join(PROJECT_PATH,"dataset","raw","machine_meta.csv")
CLEANED_DATASET_PATH = os.path.join(PROJECT_PATH,"dataset","cleaned")
CLEAN_MACHINE_META_PATH = os.path.join(CLEANED_DATASET_PATH,"machine_meta.csv")

os.makedirs(CLEANED_DATASET_PATH,exist_ok=True)

MACHINE_META_COLUMNS = ["machine_id","time_stamp","failure_domain_1","failure_domain_2","cpu_num","mem_size","status"]

machine_meta = pd.read_csv(RAW_MACHINE_META_PATH,header=None,names=MACHINE_META_COLUMNS)

machine_meta["machine_id"] = machine_meta["machine_id"].astype("string").str.strip().str.replace(r"^m_","",regex=True)
machine_meta["status"] = machine_meta["status"].astype("string").str.strip()
machine_meta["time_stamp"] = pd.to_numeric(machine_meta["time_stamp"],errors="coerce")
machine_meta["failure_domain_1"] = pd.to_numeric(machine_meta["failure_domain_1"],errors="coerce")
machine_meta["failure_domain_2"] = pd.to_numeric(machine_meta["failure_domain_2"],errors="coerce")
machine_meta["cpu_num"] = pd.to_numeric(machine_meta["cpu_num"],errors="coerce")
machine_meta["mem_size"] = pd.to_numeric(machine_meta["mem_size"],errors="coerce")

original_rows = len(machine_meta)
original_missing_failure_domain_2 = machine_meta["failure_domain_2"].isna().sum()

filled_failure_domain_2 = 0

for i in machine_meta.index[machine_meta["failure_domain_2"].isna()]:
    previous_index = i - 1
    next_index = i + 1
    previous_match = previous_index in machine_meta.index and machine_meta.loc[previous_index,"machine_id"] == machine_meta.loc[i,"machine_id"]
    next_match = next_index in machine_meta.index and machine_meta.loc[next_index,"machine_id"] == machine_meta.loc[i,"machine_id"]
    previous_value = machine_meta.loc[previous_index,"failure_domain_2"] if previous_match else pd.NA
    next_value = machine_meta.loc[next_index,"failure_domain_2"] if next_match else pd.NA
    if previous_match and pd.notna(previous_value) and next_match and pd.notna(next_value):
        if previous_value == next_value:
            machine_meta.loc[i,"failure_domain_2"] = previous_value
            filled_failure_domain_2 += 1
    elif previous_match and pd.notna(previous_value):
        machine_meta.loc[i,"failure_domain_2"] = previous_value
        filled_failure_domain_2 += 1
    elif next_match and pd.notna(next_value):
        machine_meta.loc[i,"failure_domain_2"] = next_value
        filled_failure_domain_2 += 1

machine_meta = machine_meta[machine_meta["status"] != "IMPORT_INSTALLING"].copy()
machine_meta = machine_meta.drop_duplicates()
cleaned_rows = len(machine_meta)
remaining_missing_failure_domain_2 = machine_meta["failure_domain_2"].isna().sum()
machine_meta.to_csv(CLEAN_MACHINE_META_PATH,index=False)

print("Machine meta cleaning completed.")
print(f"Original rows: {original_rows}")
print(f"Cleaned rows: {cleaned_rows}")
print(f"Rows removed: {original_rows - cleaned_rows}")
print(f"Original missing failure_domain_2: {original_missing_failure_domain_2}")
print(f"Filled failure_domain_2 values: {filled_failure_domain_2}")
print(f"Remaining missing failure_domain_2: {remaining_missing_failure_domain_2}")
print(f"Output: {CLEAN_MACHINE_META_PATH}")