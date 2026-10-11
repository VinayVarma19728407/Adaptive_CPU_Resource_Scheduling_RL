"""
Audit Task, Instance, and Job Foreign Key Relationships (Version 2).

This script performs an advanced relational integrity audit between tasks, jobs, and instances using memory-efficient
chunked scanning. It maps composite join keys (job_name, task_name) across large trace files to resolve foreign key
mismatches across dataset splits.
"""

import os
import pandas as pd
from collections import defaultdict

CHUNK_SIZE = 250_000

BATCH_TASK = "dataset/preprocessed/batch_task.csv"

SPLITS = {
    "training": "dataset/preprocessed/batch_instance_training.csv",
    "validation": "dataset/preprocessed/batch_instance_validation.csv",
    "testing": "dataset/preprocessed/batch_instance_testing.csv",
}

OUTPUT_DIR = "dataset/relationship_audit"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "task_instance_job_relationship_summary_v2.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)



# BUILD REFERENCE SETS FROM BATCH_TASK
print("\nBUILDING SET-BASED BATCH_TASK REFERENCE MAPPINGS")

task_jobs = defaultdict(set)
task_types = defaultdict(set)

total_batch_task_rows = 0

usecols = ["task_name", "job_name", "task_type"]

for chunk in pd.read_csv(
    BATCH_TASK,
    usecols=usecols,
    chunksize=CHUNK_SIZE,
    low_memory=False
):
    total_batch_task_rows += len(chunk)

    for task_name, job_name, task_type in zip(
        chunk["task_name"],
        chunk["job_name"],
        chunk["task_type"]
    ):
        if pd.notna(task_name):

            if pd.notna(job_name):
                task_jobs[task_name].add(job_name)

            if pd.notna(task_type):
                task_types[task_name].add(task_type)


unique_task_names = len(task_jobs)
unique_job_names = len(
    set(
        job
        for jobs in task_jobs.values()
        for job in jobs
    )
)

multi_job_tasks = {
    task: jobs
    for task, jobs in task_jobs.items()
    if len(jobs) > 1
}

multi_type_tasks = {
    task: types
    for task, types in task_types.items()
    if len(types) > 1
}

print(f"batch_task rows: {total_batch_task_rows}")
print(f"unique task_names: {unique_task_names}")
print(f"unique job_names: {unique_job_names}")

print(
    f"task_names mapping to multiple job_names: "
    f"{len(multi_job_tasks)}"
)

print(
    f"task_names mapping to multiple task_types: "
    f"{len(multi_type_tasks)}"
)



# DISPLAY MULTI-JOB TASKS
print("\nMULTI-JOB TASK EXAMPLES")

shown = 0

for task_name, jobs in multi_job_tasks.items():

    print(
        f"task_name={task_name} | "
        f"job_names={sorted(map(str, jobs))}"
    )

    shown += 1

    if shown >= 20:
        break

# DISPLAY MULTI-TYPE TASKS
print("\nMULTI-TASK-TYPE TASKS")
for task_name, types in multi_type_tasks.items():
    print(
        f"task_name={task_name} | "
        f"task_types={sorted(map(str, types))}"
    )



# SCAN BATCH_INSTANCE
print("\nSCANNING BATCH_INSTANCE WITH SET-BASED RELATIONSHIP LOGIC")


results = []


for split_name, filepath in SPLITS.items():

    print(f"SPLIT: {split_name}")
    print(filepath)

    total_rows = 0
    matched_task_rows = 0
    unmatched_task_rows = 0

    true_job_mismatch_rows = 0
    true_task_type_mismatch_rows = 0

    invalid_seq_rows = 0
    seq_greater_than_total_rows = 0

    negative_duration_rows = 0
    zero_duration_rows = 0
    positive_duration_rows = 0

    unique_tasks = set()
    unique_jobs = set()

    job_examples = []
    type_examples = []

    reader = pd.read_csv(
        filepath,
        usecols=[
            "task_name",
            "job_name",
            "task_type",
            "seq_no",
            "total_seq_no",
            "duration"
        ],
        chunksize=CHUNK_SIZE,
        low_memory=False
    )

    chunk_number = 0

    for chunk in reader:

        chunk_number += 1
        total_rows += len(chunk)

        unique_tasks.update(
            chunk["task_name"].dropna().unique()
        )

        unique_jobs.update(
            chunk["job_name"].dropna().unique()
        )

        for row in chunk.itertuples(index=False):

            task_name = row.task_name
            job_name = row.job_name
            task_type = row.task_type
            seq_no = row.seq_no
            total_seq_no = row.total_seq_no
            duration = row.duration

            # TASK RELATIONSHIP
            if task_name not in task_jobs:

                unmatched_task_rows += 1

                continue

            matched_task_rows += 1

            valid_jobs = task_jobs[task_name]

            if job_name not in valid_jobs:

                true_job_mismatch_rows += 1

                if len(job_examples) < 20:
                    job_examples.append(
                        (
                            task_name,
                            job_name,
                            sorted(map(str, valid_jobs))
                        )
                    )


            # TASK TYPE RELATIONSHIP


            valid_types = task_types.get(task_name, set())

            if task_type not in valid_types:

                true_task_type_mismatch_rows += 1

                if len(type_examples) < 20:
                    type_examples.append(
                        (
                            task_name,
                            task_type,
                            sorted(map(str, valid_types))
                        )
                    )

            # SEQUENCE VALIDATION


            if pd.isna(seq_no) or pd.isna(total_seq_no):

                invalid_seq_rows += 1

            else:

                if seq_no < 1 or total_seq_no < 1:
                    invalid_seq_rows += 1

                if seq_no > total_seq_no:
                    seq_greater_than_total_rows += 1


            # DURATION VALIDATION

            if pd.isna(duration):
                continue

            if duration < 0:
                negative_duration_rows += 1

            elif duration == 0:
                zero_duration_rows += 1

            else:
                positive_duration_rows += 1

        if chunk_number % 100 == 0:
            print(
                f"Chunks: {chunk_number} | "
                f"Rows: {total_rows:,} | "
                f"Unmatched tasks: {unmatched_task_rows:,} | "
                f"True job mismatches: {true_job_mismatch_rows:,} | "
                f"True type mismatches: {true_task_type_mismatch_rows:,}"
            )

    print(f"\nRESULTS FOR {split_name}")

    print(f"Rows: {total_rows:,}")
    print(f"Unique task_names: {len(unique_tasks):,}")
    print(f"Unique job_names: {len(unique_jobs):,}")

    print(f"Matched task rows: {matched_task_rows:,}")
    print(f"Unmatched task rows: {unmatched_task_rows:,}")

    print(
        f"TRUE job-name mismatch rows: "
        f"{true_job_mismatch_rows:,}"
    )

    print(
        f"TRUE task-type mismatch rows: "
        f"{true_task_type_mismatch_rows:,}"
    )

    print(f"Invalid sequence rows: {invalid_seq_rows:,}")
    print(
        f"seq_no > total_seq_no rows: "
        f"{seq_greater_than_total_rows:,}"
    )

    print(f"Negative duration rows: {negative_duration_rows:,}")
    print(f"Zero duration rows: {zero_duration_rows:,}")
    print(f"Positive duration rows: {positive_duration_rows:,}")

    results.append({
        "split": split_name,
        "rows": total_rows,
        "unique_task_names": len(unique_tasks),
        "unique_job_names": len(unique_jobs),
        "matched_task_rows": matched_task_rows,
        "unmatched_task_rows": unmatched_task_rows,
        "true_job_mismatch_rows": true_job_mismatch_rows,
        "true_task_type_mismatch_rows":
            true_task_type_mismatch_rows,
        "invalid_seq_rows": invalid_seq_rows,
        "seq_greater_than_total_rows":
            seq_greater_than_total_rows,
        "negative_duration_rows":
            negative_duration_rows,
        "zero_duration_rows":
            zero_duration_rows,
        "positive_duration_rows":
            positive_duration_rows
    })



# OVERALL SUMMARY
summary = pd.DataFrame(results)
print("OVERALL SET-BASED RELATIONSHIP SUMMARY")
print(summary.to_string(index=False))

summary.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSummary saved to:")
print(os.path.abspath(OUTPUT_FILE))


# FINAL STATUS
print("\nSET-BASED TASK / INSTANCE / JOB AUDIT COMPLETED")