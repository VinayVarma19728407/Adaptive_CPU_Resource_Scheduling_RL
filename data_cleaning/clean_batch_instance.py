"""
============================================================
File: clean_batch_instance.py
Module: data_cleaning
============================================================
Implements the cleaning pipeline for the raw Alibaba cluster
batch_instance trace dataset. The raw data is stored in a
compressed tar.gz archive and is processed in memory-efficient
chunks without full extraction to disk.

Cleaning operations performed:
  - Reads the raw batch_instance.csv from inside batch_instance.tar.gz
    via streaming (no full extraction).
  - Detects and removes duplicate header rows embedded mid-file
    (artifact of how the Alibaba dataset is distributed).
  - Removes records where end_time < start_time (invalid duration).
  - Retains records where end_time == start_time (zero duration).
  - Preserves all missing values (NaN) for downstream preprocessing.
  - Preserves all original column values and identifiers without
    modification, clipping, or normalization.
  - Validates the output file structure and row counts after writing.

Key Components:
  - Chunk-based streaming pipeline using pandas read_csv with chunksize.
  - tarfile streaming to avoid extracting the multi-GB archive.
  - Output validation: verifies column names and row counts match
    expectations before finalizing.

Inputs:
  - batch_instance.tar.gz: compressed Alibaba cluster batch_instance
    trace file located one level above the project root.
  - Expected columns (14): instance_name, task_name, job_name,
    task_type, status, start_time, end_time, machine_id, seq_no,
    total_seq_no, cpu_avg, cpu_max, mem_avg, mem_max.

Outputs:
  - dataset/cleaned/batch_instance.csv: cleaned batch instance records
    with duplicate headers and invalid-duration rows removed.
    All other fields retained as-is for preprocessing.

Project Module:
  Part of the data_cleaning stage of the Adaptive Multi-Core CPU
  Scheduler RL pipeline. Feeds into data_preprocessing.
============================================================
"""

import os
import tarfile
import pandas as pd

PROJECT_PATH = ("C:/Users/gvkva/Desktop/My ongoing projects/""Machine Learning Project/""Adaptive-Multi-Core-CPU-Scheduler-RL")
BATCH_INSTANCE_TAR_PATH = os.path.join("C:/Users/gvkva/Desktop/My ongoing projects/""Machine Learning Project","batch_instance.tar.gz")
BATCH_INSTANCE_FILE_NAME = "batch_instance.csv"
CLEANED_DATASET_PATH = os.path.join(PROJECT_PATH,"dataset","cleaned")
CLEAN_BATCH_INSTANCE_PATH = os.path.join(CLEANED_DATASET_PATH,"batch_instance.csv")

os.makedirs(CLEANED_DATASET_PATH,exist_ok=True)

CHUNK_SIZE = 250000

columns = ["instance_name","task_name","job_name","task_type","status","start_time","end_time","machine_id","seq_no","total_seq_no","cpu_avg","cpu_max","mem_avg","mem_max"]

dtype_map = {
    "instance_name":"string",
    "task_name":"string",
    "job_name":"string",
    "task_type":"string",
    "status":"string",
    "start_time":"string",
    "end_time":"string",
    "machine_id":"string",
    "seq_no":"string",
    "total_seq_no":"string",
    "cpu_avg":"string",
    "cpu_max":"string",
    "mem_avg":"string",
    "mem_max":"string"
}

print("DATASET PATH")
print(BATCH_INSTANCE_TAR_PATH)

if not os.path.exists(BATCH_INSTANCE_TAR_PATH):
    raise FileNotFoundError(BATCH_INSTANCE_TAR_PATH)

print("\nOUTPUT PATH")
print(CLEAN_BATCH_INSTANCE_PATH)

if os.path.exists(CLEAN_BATCH_INSTANCE_PATH):
    os.remove(CLEAN_BATCH_INSTANCE_PATH)

print("\nOPENING TAR.GZ FILE")

with tarfile.open(BATCH_INSTANCE_TAR_PATH,"r:gz") as tar:
    matching_member = None

    for member in tar:
        if member.isfile() and os.path.basename(member.name) == BATCH_INSTANCE_FILE_NAME:
            matching_member = member
            break

    if matching_member is None:
        raise FileNotFoundError("batch_instance.csv was not found inside the tar.gz file.")

    print("CSV file found inside tar.gz:",matching_member.name)
    print("Uncompressed CSV size in GB:",round(matching_member.size / (1024 ** 3),2))

    csv_file = tar.extractfile(matching_member)

    if csv_file is None:
        raise IOError("Unable to access batch_instance.csv inside the tar.gz file.")

    original_rows = 0
    cleaned_rows = 0
    removed_negative_duration_rows = 0
    processed_chunks = 0
    first_output = True

    try:
        reader = pd.read_csv(
            csv_file,
            header=None,
            names=columns,
            chunksize=CHUNK_SIZE,
            dtype=dtype_map,
            low_memory=False,
            on_bad_lines="error"
        )

        with open(CLEAN_BATCH_INSTANCE_PATH,"w",newline="",encoding="utf-8") as output_file:
            for chunk in reader:
                processed_chunks += 1
                original_rows += len(chunk)

                header_rows = chunk["instance_name"].astype("string").str.strip().eq("instance_name")
                header_rows &= chunk["task_name"].astype("string").str.strip().eq("task_name")
                header_rows &= chunk["job_name"].astype("string").str.strip().eq("job_name")

                data_chunk = chunk.loc[~header_rows].copy()

                start_time = pd.to_numeric(data_chunk["start_time"],errors="coerce")
                end_time = pd.to_numeric(data_chunk["end_time"],errors="coerce")

                invalid_duration_mask = start_time.notna() & end_time.notna() & (end_time < start_time)
                removed_negative_duration_rows += int(invalid_duration_mask.sum())

                if invalid_duration_mask.any():
                    data_chunk = data_chunk.loc[~invalid_duration_mask].copy()

                cleaned_rows += len(data_chunk)

                if len(data_chunk) > 0:
                    data_chunk.to_csv(output_file,index=False,header=first_output)
                    first_output = False

                del data_chunk
                del chunk

                if processed_chunks % 20 == 0:
                    print("Processed chunks:",processed_chunks,"| Physical rows processed:",original_rows,"| Cleaned data rows written:",cleaned_rows)

    finally:
        csv_file.close()

if processed_chunks == 0:
    if os.path.exists(CLEAN_BATCH_INSTANCE_PATH):
        os.remove(CLEAN_BATCH_INSTANCE_PATH)
    raise ValueError("No rows were processed from batch_instance.csv.")

if cleaned_rows == 0:
    if os.path.exists(CLEAN_BATCH_INSTANCE_PATH):
        os.remove(CLEAN_BATCH_INSTANCE_PATH)
    raise ValueError("No data rows were written. Cleaning stopped.")

print("\nBATCH INSTANCE CLEANING COMPLETED")
print("Physical rows processed:",original_rows)
print("Cleaned data rows written:",cleaned_rows)
print("Rows removed due to duplicate header rows:",original_rows - cleaned_rows - removed_negative_duration_rows)
print("Rows removed due to end_time < start_time:",removed_negative_duration_rows)
print("Processed chunks:",processed_chunks)

print("\nCLEANING POLICY")
print("The complete CSV stream is read, including both header rows.")
print("Duplicate header rows are detected and removed as non-data records.")
print("Rows with end_time < start_time are removed.")
print("Rows with end_time == start_time are retained.")
print("Missing values are preserved for preprocessing.")
print("No outlier values are deleted or modified.")
print("Original identifiers and dataset columns are preserved.")

print("\nOUTPUT VALIDATION")

output_rows = 0

with open(CLEAN_BATCH_INSTANCE_PATH,"r",encoding="utf-8",newline="") as output_file:
    output_header = output_file.readline().rstrip("\r\n")
    output_columns = output_header.split(",")

    for _ in output_file:
        output_rows += 1

if output_columns != columns:
    os.remove(CLEAN_BATCH_INSTANCE_PATH)
    raise ValueError("Output column structure does not match the expected batch_instance schema.")

if output_rows != cleaned_rows:
    os.remove(CLEAN_BATCH_INSTANCE_PATH)
    raise ValueError("Output row count does not match the number of cleaned rows written.")

print("Output data rows verified:",output_rows)
print("Output columns verified:",len(output_columns))
print("Output validation passed.")
print("Output file size in GB:",round(os.path.getsize(CLEAN_BATCH_INSTANCE_PATH) / (1024 ** 3),2))
print("Output:",CLEAN_BATCH_INSTANCE_PATH)

print("\nCLEANED DATASET STAGE COMPLETED")
