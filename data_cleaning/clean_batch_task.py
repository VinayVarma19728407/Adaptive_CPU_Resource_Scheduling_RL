"""
============================================================
File: clean_batch_task.py
Module: data_cleaning
============================================================
Implements the cleaning pipeline for the Alibaba cluster
batch_task metadata dataset. The batch_task table contains
one record per task with planned resource requirements and
scheduling lifecycle timestamps.

Cleaning operations performed:
  - Reads the raw batch_task.csv in memory-efficient chunks.
  - Detects and removes duplicate header rows embedded mid-file.
  - Fills missing instance_num values with 1 (default single instance).
  - Fills missing plan_cpu values with the global mode of plan_cpu
    computed in a first pass over the entire dataset.
  - Fills missing plan_mem values with the global mode of plan_mem
    computed in a first pass over the entire dataset.
  - Removes records where end_time < start_time (invalid duration).
  - Retains records where end_time == start_time (zero-duration tasks).
  - Retains all task status values without filtering.

Key Components:
  - Two-pass pipeline: first pass computes global modes for plan_cpu
    and plan_mem; second pass applies all cleaning transformations.
  - Chunk-based streaming using pandas read_csv with chunksize=250000.

Inputs:
  - dataset/raw/batch_task.csv: raw Alibaba cluster batch_task trace.
  - Expected columns (9): task_name, instance_num, job_name, task_type,
    status, start_time, end_time, plan_cpu, plan_mem.

Outputs:
  - dataset/cleaned/batch_task.csv: cleaned batch_task records with
    missing resource fields filled, invalid-duration rows removed,
    and duplicate header rows eliminated.

Project Module:
  Part of the data_cleaning stage of the Adaptive Multi-Core CPU
  Scheduler RL pipeline. Feeds into preprocess_batch_task.py.
============================================================
"""

import os
import pandas as pd

PROJECT_PATH = ("C:/Users/gvkva/Desktop/My ongoing projects/""Machine Learning Project/""Adaptive-Multi-Core-CPU-Scheduler-RL")
RAW_BATCH_TASK_PATH = os.path.join(PROJECT_PATH,"dataset","raw","batch_task.csv")
CLEANED_DATASET_PATH = os.path.join(PROJECT_PATH,"dataset","cleaned")
CLEAN_BATCH_TASK_PATH = os.path.join(CLEANED_DATASET_PATH,"batch_task.csv")

os.makedirs(CLEANED_DATASET_PATH,exist_ok=True)

CHUNK_SIZE = 250000

columns = ["task_name","instance_num","job_name","task_type","status","start_time","end_time","plan_cpu","plan_mem"]
dtype_map = {
    "task_name":"string",
    "instance_num":"float32",
    "job_name":"string",
    "task_type":"string",
    "status":"string",
    "start_time":"float64",
    "end_time":"float64",
    "plan_cpu":"float32",
    "plan_mem":"float32"
}

print("DATASET PATH")
print(RAW_BATCH_TASK_PATH)

if not os.path.exists(RAW_BATCH_TASK_PATH):
    raise FileNotFoundError(RAW_BATCH_TASK_PATH)

file_size_bytes = os.path.getsize(RAW_BATCH_TASK_PATH)
file_size_gb = file_size_bytes / (1024 ** 3)
print("Dataset size in GB:",round(file_size_gb,2))
plan_cpu_counts = {}
plan_mem_counts = {}

processed_chunks = 0
original_rows = 0
duplicate_header_rows = 0

reader = pd.read_csv(RAW_BATCH_TASK_PATH,header=None,names=columns,chunksize=CHUNK_SIZE,dtype=dtype_map,low_memory=False)

for chunk in reader:
    processed_chunks += 1
    original_rows += len(chunk)
    duplicate_header = chunk["task_name"].eq("task_name")
    duplicate_header_rows += int(duplicate_header.sum())
    chunk = chunk.loc[~duplicate_header].copy()
    cpu_counts = chunk["plan_cpu"].dropna().value_counts()
    for value,count in cpu_counts.items():
        plan_cpu_counts[value] = plan_cpu_counts.get(value,0) + int(count)
    mem_counts = chunk["plan_mem"].dropna().value_counts()
    for value,count in mem_counts.items():
        plan_mem_counts[value] = plan_mem_counts.get(value,0) + int(count)
    if processed_chunks % 20 == 0:
        print("Mode calculation chunks:",processed_chunks,"| Rows processed:",original_rows)
if len(plan_cpu_counts) == 0:
    raise ValueError("No valid plan_cpu values found.")
if len(plan_mem_counts) == 0:
    raise ValueError("No valid plan_mem values found.")
plan_cpu_mode = max(plan_cpu_counts,key=plan_cpu_counts.get)
plan_mem_mode = max(plan_mem_counts,key=plan_mem_counts.get)
print("Global plan_cpu mode:",plan_cpu_mode)
print("Global plan_mem mode:",plan_mem_mode)
original_rows = 0
cleaned_rows = 0
removed_invalid_time_rows = 0
filled_instance_rows = 0
filled_plan_cpu_rows = 0
filled_plan_mem_rows = 0
processed_chunks = 0

first_chunk = True

reader = pd.read_csv(RAW_BATCH_TASK_PATH,header=None,names=columns,chunksize=CHUNK_SIZE,dtype=dtype_map,low_memory=False)

with open(CLEAN_BATCH_TASK_PATH,"w",newline="",encoding="utf-8") as output_file:
    for chunk in reader:
        processed_chunks += 1
        original_rows += len(chunk)

        duplicate_header = chunk["task_name"].eq("task_name")
        chunk = chunk.loc[~duplicate_header].copy()

        missing_instance = chunk["instance_num"].isna()
        filled_instance_rows += int(missing_instance.sum())
        chunk.loc[missing_instance,"instance_num"] = 1

        missing_plan_cpu = chunk["plan_cpu"].isna()
        filled_plan_cpu_rows += int(missing_plan_cpu.sum())
        chunk.loc[missing_plan_cpu,"plan_cpu"] = plan_cpu_mode

        missing_plan_mem = chunk["plan_mem"].isna()
        filled_plan_mem_rows += int(missing_plan_mem.sum())
        chunk.loc[missing_plan_mem,"plan_mem"] = plan_mem_mode

        invalid_time = chunk["end_time"] < chunk["start_time"]
        removed_invalid_time_rows += int(invalid_time.sum())
        chunk = chunk.loc[~invalid_time].copy()

        cleaned_rows += len(chunk)

        if first_chunk:
            chunk.to_csv(output_file,index=False)
            first_chunk = False
        else:
            chunk.to_csv(output_file,index=False,header=False)

        if processed_chunks % 20 == 0:
            print("Cleaning chunks:",processed_chunks,"| Rows processed:",original_rows)

print("BATCH TASK CLEANING COMPLETED")
print("Original rows:",original_rows)
print("Cleaned rows:",cleaned_rows)
print("Rows removed due to end_time < start_time:",removed_invalid_time_rows)
print("Rows filled for missing instance_num:",filled_instance_rows)
print("Rows filled for missing plan_cpu:",filled_plan_cpu_rows)
print("Rows filled for missing plan_mem:",filled_plan_mem_rows)
print("plan_cpu mode used:",plan_cpu_mode)
print("plan_mem mode used:",plan_mem_mode)
print("end_time == start_time rows retained.")
print("All status values retained.")
print("Output:",CLEAN_BATCH_TASK_PATH)