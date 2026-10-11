"""
Targeted Investigation of Specific Job Identifier Mismatches.

This script performs targeted diagnosis on specific anomalous job mismatch cases identified during preprocessing.
It isolates problem job IDs, traces execution timelines, and evaluates alternative composite join key strategies.
"""

import pandas as pd
from pathlib import Path

BASE = Path("dataset")

BATCH_TASK = BASE / "preprocessed" / "batch_task.csv"

SPLITS = {
    "training": BASE / "preprocessed" / "batch_instance_training.csv",
    "validation": BASE / "preprocessed" / "batch_instance_validation.csv",
    "testing": BASE / "preprocessed" / "batch_instance_testing.csv"
}

CHUNK_SIZE = 250_000
TARGET_PATTERNS = 30


print("Building task job relationships")

task_jobs = {}

for chunk in pd.read_csv(
    BATCH_TASK,
    usecols=["task_name", "job_name"],
    chunksize=CHUNK_SIZE,
    low_memory=False
):

    chunk = chunk.dropna(
        subset=["task_name", "job_name"]
    )

    for task_name, group in chunk.groupby(
        "task_name",
        sort=False
    ):

        jobs = set(
            group["job_name"].astype(str)
        )

        if task_name in task_jobs:
            task_jobs[task_name].update(jobs)
        else:
            task_jobs[task_name] = jobs


print(
    f"Unique task relationships loaded: {len(task_jobs):,}"
)

print("Task job relationship map ready")


for split_name, split_path in SPLITS.items():

    print()
    print(f"Scanning {split_name}")

    chunks_scanned = 0
    rows_scanned = 0
    unique_mismatch_tasks = 0

    processed_tasks = set()

    pattern_counts = {}

    for chunk in pd.read_csv(
        split_path,
        usecols=["task_name", "job_name"],
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        chunks_scanned += 1
        rows_scanned += len(chunk)

        known = chunk[
            chunk["task_name"].isin(task_jobs)
        ]

        if known.empty:
            continue

        for row in known.itertuples(index=False):

            task_name = row.task_name
            instance_job = str(row.job_name)

            if task_name in processed_tasks:
                continue

            valid_jobs = task_jobs[task_name]

            if instance_job not in valid_jobs:

                processed_tasks.add(task_name)

                unique_mismatch_tasks += 1

                valid_job_count = len(valid_jobs)

                if valid_job_count == 1:
                    category = "ONE_VALID_JOB"

                elif valid_job_count == 2:
                    category = "TWO_VALID_JOBS"

                elif valid_job_count <= 5:
                    category = "THREE_TO_FIVE_VALID_JOBS"

                elif valid_job_count <= 10:
                    category = "SIX_TO_TEN_VALID_JOBS"

                else:
                    category = "MORE_THAN_TEN_VALID_JOBS"

                pattern_counts[category] = (
                    pattern_counts.get(category, 0) + 1
                )

                print(
                    f"Pattern {unique_mismatch_tasks:02d}: "
                    f"{category} | "
                    f"valid_job_count {valid_job_count} | "
                    f"mismatch_outside_valid_set YES"
                )

                if unique_mismatch_tasks >= TARGET_PATTERNS:
                    break

        if unique_mismatch_tasks >= TARGET_PATTERNS:
            break

        if chunks_scanned % 20 == 0:

            print(
                f"Progress | "
                f"chunks {chunks_scanned} | "
                f"rows {rows_scanned:,} | "
                f"patterns {unique_mismatch_tasks}"
            )

    print()
    print(f"{split_name} summary")

    print(
        f"Chunks scanned: {chunks_scanned:,}"
    )

    print(
        f"Rows scanned: {rows_scanned:,}"
    )

    print(
        f"Representative mismatch tasks: "
        f"{unique_mismatch_tasks:,}"
    )

    print("Relationship pattern summary")

    for category, count in sorted(
        pattern_counts.items(),
        key=lambda item: item[1],
        reverse=True
    ):

        print(
            f"{category}: {count}"
        )


print()
print("Targeted relationship analysis complete")
print("No task names were printed")
print("No job names were printed")