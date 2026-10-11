"""
Preprocess Batch Task Features.

This script preprocesses task-level attributes from batch_task logs. It encodes categorical features, constructs task lookup
tables, and formats task resource request vectors for input into the scheduling environment.
"""

import os
import pandas as pd

PROJECT_PATH = ("C:/Users/gvkva/Desktop/My ongoing projects/""Machine Learning Project/""Adaptive-Multi-Core-CPU-Scheduler-RL")
CLEAN_BATCH_TASK_PATH = os.path.join(PROJECT_PATH,"dataset","cleaned","batch_task.csv")
PREPROCESSED_DATASET_PATH = os.path.join(PROJECT_PATH,"dataset","preprocessed")
PREPROCESS_BATCH_TASK_PATH = os.path.join(PREPROCESSED_DATASET_PATH,"batch_task.csv")
os.makedirs(PREPROCESSED_DATASET_PATH,exist_ok=True)

CHUNK_SIZE = 250000
columns = ["task_name","instance_num","job_name","task_type","status","start_time","end_time","plan_cpu","plan_mem"]

dtype_map = {
    "task_name":"string",
    "instance_num":"float32",
    "job_name":"string",
    "task_type":"string",
    "status":"string",
    "start_time":"int64",
    "end_time":"int64",
    "plan_cpu":"float32",
    "plan_mem":"float32"
}

print("DATASET PATH")
print(CLEAN_BATCH_TASK_PATH)
if not os.path.exists(CLEAN_BATCH_TASK_PATH):
    raise FileNotFoundError(CLEAN_BATCH_TASK_PATH)

print("BUILDING GLOBAL ENCODINGS")

task_name_values = set()
job_name_values = set()
task_type_values = set()
processed_rows = 0
processed_chunks = 0
reader = pd.read_csv(CLEAN_BATCH_TASK_PATH,header=0,chunksize=CHUNK_SIZE,dtype=dtype_map,low_memory=False)
for chunk in reader:
    processed_chunks += 1
    processed_rows += len(chunk)
    task_name_values.update(chunk["task_name"].dropna().tolist())
    job_name_values.update(chunk["job_name"].dropna().tolist())
    task_type_values.update(chunk["task_type"].dropna().tolist())
    if processed_chunks % 20 == 0:
        print("Encoding scan chunks:",processed_chunks,"| Rows processed:",processed_rows)
task_name_encoding = {value:index for index,value in enumerate(sorted(task_name_values))}
job_name_encoding = {value:index for index,value in enumerate(sorted(job_name_values))}
task_type_encoding = {value:index for index,value in enumerate(sorted(task_type_values))}
status_encoding = {
    "Waiting":0,
    "Running":1,
    "Terminated":2,
    "Failed":3
}

print("TASK TYPE ENCODING")
for key,value in task_type_encoding.items():
    print(key,"=",value)
print("STATUS ENCODING")
for key,value in status_encoding.items():
    print(key,"=",value)
print("IDENTIFIER ENCODING")
print("task_name_id = sorted global task_name mapping starting from 0")
print("job_name_id = sorted global job_name mapping starting from 0")
processed_rows = 0
processed_chunks = 0
first_chunk = True
reader = pd.read_csv(CLEAN_BATCH_TASK_PATH,header=0,chunksize=CHUNK_SIZE,dtype=dtype_map,low_memory=False)
with open(PREPROCESS_BATCH_TASK_PATH,"w",newline="",encoding="utf-8") as output_file:

    for chunk in reader:
        processed_chunks += 1
        chunk["task_name_id"] = chunk["task_name"].map(task_name_encoding).astype("int32")
        chunk["job_name_id"] = chunk["job_name"].map(job_name_encoding).astype("int32")
        chunk["task_type_id"] = chunk["task_type"].map(task_type_encoding).astype("int16")
        unknown_status = chunk.loc[~chunk["status"].isin(status_encoding),"status"].dropna().unique()
        if len(unknown_status) > 0:
            raise ValueError("Unknown batch task status values: " + str(sorted(unknown_status.tolist())))
        chunk["status_id"] = chunk["status"].map(status_encoding).astype("int8")
        chunk["duration"] = chunk["end_time"] - chunk["start_time"]
        processed_rows += len(chunk)
        if first_chunk:
            chunk.to_csv(output_file,index=False)
            first_chunk = False
        else:
            chunk.to_csv(output_file,index=False,header=False)

        if processed_chunks % 20 == 0:
            print("Preprocessing chunks:",processed_chunks,"| Rows processed:",processed_rows)

print("BATCH TASK PREPROCESSING COMPLETED")
print("Rows processed:",processed_rows)
print("Columns added: task_name_id, job_name_id, task_type_id, status_id, duration")
print("Original task_name and job_name values preserved.")
print("TASK TYPE ENCODING")
for key,value in task_type_encoding.items():
    print(key,"=",value)
print("STATUS ENCODING")
for key,value in status_encoding.items():
    print(key,"=",value)
print("Output:",PREPROCESS_BATCH_TASK_PATH)
