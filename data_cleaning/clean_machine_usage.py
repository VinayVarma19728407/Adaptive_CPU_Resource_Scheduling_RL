"""
============================================================
File: clean_machine_usage.py
Module: data_cleaning
============================================================
Implements the cleaning pipeline for the Alibaba cluster
machine_usage dataset. The machine_usage table contains
time-series resource utilization measurements sampled at
regular intervals for each physical cluster node.

Cleaning operations performed:
  - Reads the raw machine_usage.csv in memory-efficient chunks.
  - Strips the "m_" prefix from machine_id values to normalize
    identifiers (matching the convention in machine_meta).
  - Removes rows where cpu_util_percent is missing (NaN), as CPU
    utilization is a mandatory field for the RL environment.
  - Removes rows where disk_io_percent falls outside the valid
    physical range [0, 100] (negative or above 100).
  - Drops the mem_gps and mkpi columns entirely, as they are not
    used in the downstream scheduling environment.
  - All remaining fields (mem_util_percent, net_in, net_out) are
    preserved as-is without normalization or imputation.

Key Components:
  - Chunk-based streaming pipeline using pandas read_csv with
    chunksize=250000.
  - Column-drop for mem_gps and mkpi to reduce output file size.

Inputs:
  - dataset/raw/machine_usage.csv: raw Alibaba cluster machine usage
    time-series trace.
  - Expected columns (9): machine_id, time_stamp, cpu_util_percent,
    mem_util_percent, mem_gps, mkpi, net_in, net_out, disk_io_percent.

Outputs:
  - dataset/cleaned/machine_usage.csv: cleaned machine usage records
    with normalized machine_id, invalid rows removed, and unused
    columns (mem_gps, mkpi) dropped.
  - Output columns (7): machine_id, time_stamp, cpu_util_percent,
    mem_util_percent, net_in, net_out, disk_io_percent.

Project Module:
  Part of the data_cleaning stage of the Adaptive Multi-Core CPU
  Scheduler RL pipeline. Feeds into the RL environment's cluster
  state representation.
============================================================
"""

import os
import pandas as pd

PROJECT_PATH = ("C:/Users/gvkva/Desktop/My ongoing projects/""Machine Learning Project/""Adaptive-Multi-Core-CPU-Scheduler-RL")
RAW_MACHINE_USAGE_PATH = os.path.join(PROJECT_PATH,"dataset","raw","machine_usage.csv")
CLEANED_DATASET_PATH = os.path.join(PROJECT_PATH,"dataset","cleaned")
CLEAN_MACHINE_USAGE_PATH = os.path.join(CLEANED_DATASET_PATH,"machine_usage.csv")

os.makedirs(CLEANED_DATASET_PATH,exist_ok=True)

CHUNK_SIZE = 250000
columns = ["machine_id","time_stamp","cpu_util_percent","mem_util_percent","mem_gps","mkpi","net_in","net_out","disk_io_percent"]
dtype_map = {
    "machine_id":"string",
    "time_stamp":"int64",
    "cpu_util_percent":"float32",
    "mem_util_percent":"float32",
    "mem_gps":"float32",
    "mkpi":"float32",
    "net_in":"float32",
    "net_out":"float32",
    "disk_io_percent":"float32"
}

print("DATASET PATH")
print(RAW_MACHINE_USAGE_PATH)

if not os.path.exists(RAW_MACHINE_USAGE_PATH):
    raise FileNotFoundError(RAW_MACHINE_USAGE_PATH)
file_size_bytes = os.path.getsize(RAW_MACHINE_USAGE_PATH)
file_size_gb = file_size_bytes / (1024 ** 3)
print("Dataset size in GB:",round(file_size_gb,2))

original_rows = 0
cleaned_rows = 0
removed_cpu_rows = 0
removed_disk_rows = 0
processed_chunks = 0
first_chunk = True
reader = pd.read_csv(RAW_MACHINE_USAGE_PATH,header=None,names=columns,chunksize=CHUNK_SIZE,dtype=dtype_map,low_memory=False)

with open(CLEAN_MACHINE_USAGE_PATH,"w",newline="",encoding="utf-8") as output_file:

    for chunk in reader:
        processed_chunks += 1
        original_rows += len(chunk)

        chunk["machine_id"] = (chunk["machine_id"].str.strip().str.replace(r"^m_","",regex=True))

        missing_cpu = chunk["cpu_util_percent"].isna()
        removed_cpu_rows += int(missing_cpu.sum())
        chunk = chunk.loc[~missing_cpu].copy()

        invalid_disk = (chunk["disk_io_percent"].lt(0) |chunk["disk_io_percent"].gt(100))

        removed_disk_rows += int(invalid_disk.sum())
        chunk = chunk.loc[~invalid_disk].copy()
        chunk = chunk.drop(columns=["mem_gps","mkpi"])
        cleaned_rows += len(chunk)
        if first_chunk:
            chunk.to_csv(output_file,index=False)
            first_chunk = False
        else:
            chunk.to_csv(output_file,index=False,header=False)
        if processed_chunks % 20 == 0:
            print("Processed chunks:",processed_chunks,"| Rows processed:",original_rows)

print("MACHINE USAGE CLEANING COMPLETED")
print("Original rows:",original_rows)
print("Cleaned rows:",cleaned_rows)
print("Rows removed due to missing cpu_util_percent:",removed_cpu_rows)
print("Rows removed due to invalid disk_io_percent:",removed_disk_rows)
print("Total rows removed:",original_rows - cleaned_rows)
print("Removed columns: mem_gps, mkpi")
print("Normalized machine_id values: m_xxxx -> xxxx")
print("Output:",CLEAN_MACHINE_USAGE_PATH)