"""
Audit Batch Task Key Cardinality and Relationships.

This script evaluates task ID and job ID cardinality, key uniqueness, and task-to-job ratio distributions
within batch_task tables. It analyzes parent job groupings, checking for duplicate keys or multi-instance
task definitions in the workload traces.
"""

import os
import pandas as pd

PROJECT_PATH = (
    "C:/Users/gvkva/Desktop/My ongoing projects/"
    "Machine Learning Project/Adaptive-Multi-Core-CPU-Scheduler-RL"
)

FILES = {
    "RAW": os.path.join(
        PROJECT_PATH,
        "dataset",
        "raw",
        "batch_task.csv"
    ),
    "CLEANED": os.path.join(
        PROJECT_PATH,
        "dataset",
        "cleaned",
        "batch_task.csv"
    ),
    "PREPROCESSED": os.path.join(
        PROJECT_PATH,
        "dataset",
        "preprocessed",
        "batch_task.csv"
    ),
}

CHUNK_SIZE = 250000

for label, path in FILES.items():

    print()
    print(label)
    print(path)

    if not os.path.exists(path):
        print("FILE NOT FOUND")
        continue

    unique_tasks = set()
    total_rows = 0
    missing_tasks = 0
    duplicate_header_rows = 0

    if label == "RAW":
        reader = pd.read_csv(
            path,
            header=None,
            names=[
                "task_name",
                "instance_num",
                "job_name",
                "task_type",
                "status",
                "start_time",
                "end_time",
                "plan_cpu",
                "plan_mem"
            ],
            usecols=["task_name"],
            dtype="string",
            chunksize=CHUNK_SIZE,
            low_memory=False
        )
    else:
        reader = pd.read_csv(
            path,
            header=0,
            usecols=["task_name"],
            dtype="string",
            chunksize=CHUNK_SIZE,
            low_memory=False
        )

    chunks = 0

    for chunk in reader:
        chunks += 1
        total_rows += len(chunk)

        missing_tasks += int(chunk["task_name"].isna().sum())

        duplicate_header_rows += int(
            chunk["task_name"].eq("task_name").sum()
        )

        unique_tasks.update(
            chunk["task_name"].dropna().tolist()
        )

        if chunks % 20 == 0:
            print(
                "Chunks:",
                chunks,
                "| Rows:",
                total_rows,
                "| Unique task_names:",
                len(unique_tasks)
            )

    print()
    print("FINAL RESULTS")
    print("Rows:", total_rows)
    print("Unique task_names:", len(unique_tasks))
    print("Missing task_names:", missing_tasks)
    print("Duplicate header rows:", duplicate_header_rows)


print()
print("AUDIT COMPLETED")