"""
Investigate Job and Task Identifier Mismatches.

This script investigates root causes of key mismatches between job_name and task_name identifiers in batch trace tables.
It performs set intersection, key frequency analysis, and substring matching on unmatched identifiers.
"""

import os
import pandas as pd
from collections import defaultdict, Counter

CHUNK_SIZE = 250_000

BATCH_TASK = "dataset/preprocessed/batch_task.csv"

SPLITS = {
    "training": "dataset/preprocessed/batch_instance_training.csv",
    "validation": "dataset/preprocessed/batch_instance_validation.csv",
    "testing": "dataset/preprocessed/batch_instance_testing.csv",
}

OUTPUT_DIR = "dataset/relationship_audit"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "job_mismatch_investigation.csv"
)


# BUILD task_name -> VALID JOB NAMES
print("\nBUILDING TASK -> VALID JOB SETS")

task_jobs = defaultdict(set)

for chunk in pd.read_csv(
    BATCH_TASK,
    usecols=["task_name", "job_name"],
    chunksize=CHUNK_SIZE,
    low_memory=False
):
    for task_name, job_name in zip(
        chunk["task_name"],
        chunk["job_name"]
    ):
        if pd.notna(task_name) and pd.notna(job_name):
            task_jobs[task_name].add(job_name)

print(f"Task names: {len(task_jobs):,}")



# INVESTIGATE MISMATCHES
mismatch_counter = Counter()
mismatch_task_counter = Counter()
mismatch_split_counter = Counter()

examples = []

for split_name, filepath in SPLITS.items():
    print(f"SCANNING {split_name}")
    print(filepath)

    total_mismatches = 0

    reader = pd.read_csv(
        filepath,
        usecols=["task_name", "job_name"],
        chunksize=CHUNK_SIZE,
        low_memory=False
    )

    for chunk_number, chunk in enumerate(reader, start=1):

        for task_name, job_name in zip(
            chunk["task_name"],
            chunk["job_name"]
        ):

            if task_name not in task_jobs:
                continue

            valid_jobs = task_jobs[task_name]

            if job_name not in valid_jobs:

                total_mismatches += 1

                key = (
                    task_name,
                    job_name
                )

                mismatch_counter[key] += 1
                mismatch_task_counter[task_name] += 1
                mismatch_split_counter[split_name] += 1

                if len(examples) < 100:
                    examples.append({
                        "split": split_name,
                        "task_name": task_name,
                        "batch_instance_job_name": job_name,
                        "valid_batch_task_job_names":
                            "|".join(
                                sorted(
                                    map(str, valid_jobs)
                                )
                            )
                    })

        if chunk_number % 100 == 0:
            print(
                f"Chunks: {chunk_number} | "
                f"True job mismatches: "
                f"{total_mismatches:,}"
            )

    print(
        f"{split_name} mismatches: "
        f"{total_mismatches:,}"
    )



# TOP MISMATCH COMBINATIONS
print("TOP 50 TASK/JOB MISMATCH COMBINATIONS")

rows = []

for (task_name, instance_job), count in (
    mismatch_counter.most_common(50)
):

    valid_jobs = task_jobs.get(task_name, set())

    rows.append({
        "task_name": task_name,
        "batch_instance_job_name": instance_job,
        "valid_batch_task_job_names":
            "|".join(
                sorted(
                    map(str, valid_jobs)
                )
            ),
        "mismatch_rows": count
    })

top_df = pd.DataFrame(rows)

if len(top_df) > 0:
    print(top_df.to_string(index=False))



# TOP TASKS BY MISMATCH COUNT
print("TOP 50 TASK_NAMES BY MISMATCH COUNT")

for task_name, count in mismatch_task_counter.most_common(50):

    print(
        f"task_name={task_name} | "
        f"mismatch_rows={count:,} | "
        f"valid_jobs="
        f"{sorted(map(str, task_jobs[task_name]))}"
    )



# SPLIT SUMMARY
print("MISMATCHES BY SPLIT")

for split_name, count in mismatch_split_counter.items():
    print(
        f"{split_name}: {count:,}"
    )



# SAVE TOP 50
top_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:")
print(os.path.abspath(OUTPUT_FILE))



# SAVE FIRST 100 RAW EXAMPLES
examples_file = os.path.join(
    OUTPUT_DIR,
    "job_mismatch_examples.csv"
)

pd.DataFrame(examples).to_csv(
    examples_file,
    index=False
)

print("Examples saved:")
print(os.path.abspath(examples_file))

print("\nJOB MISMATCH INVESTIGATION COMPLETED")