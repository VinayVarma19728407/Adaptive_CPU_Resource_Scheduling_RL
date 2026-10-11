"""
Preprocess Testing Split for Batch Instances.

This script implements the preprocessing pipeline for generating the test split dataset from batch_instance logs.
It applies training feature transformers to test trace data to prevent data leakage and outputs normalized test splits.
"""

import os
import tarfile

import pandas as pd

PROJECT_PATH = ("C:/Users/gvkva/Desktop/My ongoing projects/""Machine Learning Project/""Adaptive-Multi-Core-CPU-Scheduler-RL")

RAW_BATCH_INSTANCE_ARCHIVE_PATH = os.path.normpath(os.path.join(PROJECT_PATH,"..","batch_instance.tar.gz"))
PREPROCESSED_DATASET_PATH = os.path.join(PROJECT_PATH,"dataset","preprocessed")
OUTPUT_PATH = os.path.join(PREPROCESSED_DATASET_PATH,"batch_instance_testing.csv")
TEMP_OUTPUT_PATH = OUTPUT_PATH + ".tmp"

CHUNK_SIZE = 250000

columns = ["instance_name","task_name","job_name","task_type","status","start_time","end_time","machine_id","seq_no","total_seq_no","cpu_avg","cpu_max","mem_avg","mem_max"]

raw_dtype_map = {
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

status_encoding = {
    "Failed":0,
    "Interrupted":1,
    "Ready":2,
    "Running":3,
    "Terminated":4,
    "Terminating":5
}

def clean_chunk(chunk):
    for column in columns:
        chunk[column] = chunk[column].str.strip()

    start_time = pd.to_numeric(chunk["start_time"],errors="coerce")
    end_time = pd.to_numeric(chunk["end_time"],errors="coerce")
    seq_no = pd.to_numeric(chunk["seq_no"],errors="coerce")
    total_seq_no = pd.to_numeric(chunk["total_seq_no"],errors="coerce")
    machine_id = chunk["machine_id"].str.replace(r"^m_","",regex=True)
    machine_id = pd.to_numeric(machine_id,errors="coerce")

    valid = start_time.notna() & end_time.notna() & seq_no.notna() & total_seq_no.notna() & machine_id.notna()
    valid &= end_time >= start_time
    valid &= chunk["status"].isin(status_encoding)

    chunk = chunk.loc[valid].copy()

    if chunk.empty:
        return chunk

    chunk["start_time"] = start_time.loc[chunk.index].astype("int64")
    chunk["end_time"] = end_time.loc[chunk.index].astype("int64")
    chunk["seq_no"] = seq_no.loc[chunk.index].astype("int64")
    chunk["total_seq_no"] = total_seq_no.loc[chunk.index].astype("int64")
    chunk["machine_id"] = machine_id.loc[chunk.index].astype("int32")
    chunk["status"] = chunk["status"].map(status_encoding).astype("int8")

    for column in ["cpu_avg","cpu_max","mem_avg","mem_max"]:
        chunk[column] = pd.to_numeric(chunk[column],errors="coerce").astype("float32")

    chunk["duration"] = (chunk["end_time"] - chunk["start_time"]).astype("float64")

    resource_columns = ["cpu_avg","cpu_max","mem_avg","mem_max"]
    chunk[resource_columns] = chunk.groupby(["job_name","task_name"],sort=False)[resource_columns].ffill()
    chunk[resource_columns] = chunk.groupby(["job_name","task_name"],sort=False)[resource_columns].bfill()

    return chunk

def read_chunks():
    archive = tarfile.open(RAW_BATCH_INSTANCE_ARCHIVE_PATH,"r:gz")
    member = archive.getmember("batch_instance.csv")
    file_object = archive.extractfile(member)

    if file_object is None:
        archive.close()
        raise FileNotFoundError("batch_instance.csv was not found inside the archive.")

    reader = pd.read_csv(file_object,header=None,names=columns,chunksize=CHUNK_SIZE,dtype=raw_dtype_map,low_memory=False,on_bad_lines="error")
    return archive,file_object,reader

def get_split_targets(total_rows):
    training_rows = int(total_rows * 0.60)
    validation_rows = int(total_rows * 0.20)
    testing_rows = total_rows - training_rows - validation_rows
    return training_rows,validation_rows,testing_rows

def get_boundary(timestamp_counts,target_rank):
    cumulative = 0
    for timestamp in sorted(timestamp_counts):
        count = timestamp_counts[timestamp]
        if cumulative + count >= target_rank:
            return timestamp,cumulative,target_rank - cumulative
        cumulative += count
    raise ValueError("Unable to determine split boundary.")

def build_boundaries(timestamp_counts,total_rows):
    training_rows,validation_rows,testing_rows = get_split_targets(total_rows)
    training_boundary = get_boundary(timestamp_counts,training_rows)
    validation_boundary = get_boundary(timestamp_counts,training_rows + validation_rows)
    return training_rows,validation_rows,testing_rows,training_boundary,validation_boundary

def select_split(chunk,split_name,training_boundary,validation_boundary,occurrences):
    training_timestamp,training_before,training_take = training_boundary
    validation_timestamp,validation_before,validation_take = validation_boundary
    timestamps = chunk["start_time"]

    training_occurrence_start = occurrences.get(training_timestamp,0)
    validation_occurrence_start = occurrences.get(validation_timestamp,0)

    if split_name == "training":
        mask = timestamps < training_timestamp
        boundary_mask = timestamps == training_timestamp
        if boundary_mask.any():
            positions = pd.Series(range(training_occurrence_start,training_occurrence_start + int(boundary_mask.sum())),index=chunk.index[boundary_mask])
            mask.loc[positions.index] = positions < training_take
        return chunk.loc[mask].copy()

    if split_name == "validation":
        if training_timestamp == validation_timestamp:
            mask = timestamps == training_timestamp
            if mask.any():
                positions = pd.Series(range(training_occurrence_start,training_occurrence_start + int(mask.sum())),index=chunk.index[mask])
                mask.loc[positions.index] = (positions >= training_take) & (positions < training_take + validation_take)
            return chunk.loc[mask].copy()

        mask = (timestamps > training_timestamp) & (timestamps < validation_timestamp)

        training_mask = timestamps == training_timestamp
        if training_mask.any():
            positions = pd.Series(range(training_occurrence_start,training_occurrence_start + int(training_mask.sum())),index=chunk.index[training_mask])
            mask.loc[positions.index] = positions >= training_take

        validation_mask = timestamps == validation_timestamp
        if validation_mask.any():
            positions = pd.Series(range(validation_occurrence_start,validation_occurrence_start + int(validation_mask.sum())),index=chunk.index[validation_mask])
            mask.loc[positions.index] = positions < validation_take

        return chunk.loc[mask].copy()

    mask = timestamps > validation_timestamp
    boundary_mask = timestamps == validation_timestamp
    if boundary_mask.any():
        positions = pd.Series(range(validation_occurrence_start,validation_occurrence_start + int(boundary_mask.sum())),index=chunk.index[boundary_mask])
        mask.loc[positions.index] = positions >= validation_take

    return chunk.loc[mask].copy()

def update_occurrences(cleaned,training_timestamp,validation_timestamp,occurrences):
    if training_timestamp == validation_timestamp:
        count = int((cleaned["start_time"] == training_timestamp).sum())
        if count:
            occurrences[training_timestamp] = occurrences.get(training_timestamp,0) + count
    else:
        training_count = int((cleaned["start_time"] == training_timestamp).sum())
        validation_count = int((cleaned["start_time"] == validation_timestamp).sum())
        if training_count:
            occurrences[training_timestamp] = occurrences.get(training_timestamp,0) + training_count
        if validation_count:
            occurrences[validation_timestamp] = occurrences.get(validation_timestamp,0) + validation_count

def main():
    os.makedirs(PREPROCESSED_DATASET_PATH,exist_ok=True)

    if not os.path.exists(RAW_BATCH_INSTANCE_ARCHIVE_PATH):
        raise FileNotFoundError(RAW_BATCH_INSTANCE_ARCHIVE_PATH)

    print("DATASET PATH")
    print(RAW_BATCH_INSTANCE_ARCHIVE_PATH)

    timestamp_counts = {}
    total_rows = 0
    physical_rows = 0
    chunks_processed = 0

    archive,file_object,reader = read_chunks()

    try:
        for chunk in reader:
            chunks_processed += 1
            physical_rows += len(chunk)
            cleaned = clean_chunk(chunk)

            if not cleaned.empty:
                counts = cleaned["start_time"].value_counts(sort=False)
                for timestamp,count in counts.items():
                    timestamp_counts[int(timestamp)] = timestamp_counts.get(int(timestamp),0) + int(count)

            total_rows += len(cleaned)

            if chunks_processed % 20 == 0:
                print("Boundary scan chunks:",chunks_processed,"| Valid rows:",total_rows)
    finally:
        file_object.close()
        archive.close()

    training_rows,validation_rows,testing_rows,training_boundary,validation_boundary = build_boundaries(timestamp_counts,total_rows)

    print("TOTAL VALID ROWS:",total_rows)
    print("TRAINING ROWS:",training_rows)
    print("VALIDATION ROWS:",validation_rows)
    print("TESTING ROWS:",testing_rows)
    print("TRAINING BOUNDARY TIMESTAMP:",training_boundary[0])
    print("VALIDATION BOUNDARY TIMESTAMP:",validation_boundary[0])

    if os.path.exists(TEMP_OUTPUT_PATH):
        os.remove(TEMP_OUTPUT_PATH)

    occurrences = {}
    first_write = True
    written_rows = 0
    chunks_processed = 0

    archive,file_object,reader = read_chunks()

    try:
        with open(TEMP_OUTPUT_PATH,"w",newline="",encoding="utf-8") as output_file:
            for chunk in reader:
                chunks_processed += 1
                cleaned = clean_chunk(chunk)

                if cleaned.empty:
                    continue

                selected = select_split(cleaned,"testing",training_boundary,validation_boundary,occurrences)
                update_occurrences(cleaned,training_boundary[0],validation_boundary[0],occurrences)

                if not selected.empty:
                    selected = selected.sort_values(["job_name","task_name","start_time","instance_name"],kind="mergesort")
                    selected.to_csv(output_file,index=False,header=first_write)
                    first_write = False
                    written_rows += len(selected)

                if chunks_processed % 20 == 0:
                    print("Output scan chunks:",chunks_processed,"| Rows written:",written_rows)
    finally:
        file_object.close()
        archive.close()

    expected_rows = {
        "training":training_rows,
        "validation":validation_rows,
        "testing":testing_rows
    }["testing"]

    if written_rows != expected_rows:
        raise ValueError("Split row count mismatch. Expected " + str(expected_rows) + " but wrote " + str(written_rows) + ".")

    os.replace(TEMP_OUTPUT_PATH,OUTPUT_PATH)

    print("PREPROCESSING COMPLETED")
    print("Split:","testing")
    print("Physical rows read in first pass:",physical_rows)
    print("Valid rows:",total_rows)
    print("Rows written:",written_rows)
    print("Output:",OUTPUT_PATH)

if __name__ == "__main__":
    main()
