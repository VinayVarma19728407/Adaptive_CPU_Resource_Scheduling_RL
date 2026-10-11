"""
Investigate Job and Task Identifier Mismatches (Version 2).

This script refines job mismatch investigation to handle edge cases in trace naming conventions, task index formatting,
missing task suffixes, and multi-tenant workload job naming structures across Alibaba trace files.
"""

import pandas as pd
from collections import Counter
from pathlib import Path


# CONFIGURATION
BASE = Path("dataset")

BATCH_TASK = BASE / "preprocessed" / "batch_task.csv"

SPLITS = {
    "training": BASE / "preprocessed" / "batch_instance_training.csv",
    "validation": BASE / "preprocessed" / "batch_instance_validation.csv",
    "testing": BASE / "preprocessed" / "batch_instance_testing.csv",
}

CHUNK_SIZE = 250_000

# Number of top mismatch task/job combinations to report
TOP_N = 50



# STEP 1 — BUILD task_name -> VALID job_name SET
print("BUILDING TASK -> VALID JOB RELATIONSHIPS")

task_jobs = {}

for chunk in pd.read_csv(
    BATCH_TASK,
    usecols=["task_name", "job_name"],
    chunksize=CHUNK_SIZE,
    low_memory=False
):
    chunk = chunk.dropna(subset=["task_name", "job_name"])

    for task_name, group in chunk.groupby("task_name", sort=False):
        jobs = set(group["job_name"].astype(str))
        
        if task_name in task_jobs:
            task_jobs[task_name].update(jobs)
        else:
            task_jobs[task_name] = jobs

print(f"Unique task_names in batch_task: {len(task_jobs):,}")

# Convert to frozenset so the mapping is stable/read-only during scanning.
task_jobs = {
    task: frozenset(jobs)
    for task, jobs in task_jobs.items()
}

print("TASK -> JOB RELATIONSHIP MAP READY")



# STEP 2 — SCAN EACH SPLIT
overall_summary = []

for split_name, split_path in SPLITS.items():

    print()
    print(f"SCANNING {split_name.upper()}")
    print(split_path)

    total_rows = 0
    known_task_rows = 0
    unmatched_task_rows = 0
    matching_job_rows = 0
    mismatching_job_rows = 0

    # Count mismatch combinations:
    # (task_name, batch_instance_job_name)
    mismatch_pairs = Counter()

    # Count mismatching rows by task_name
    mismatch_tasks = Counter()

    # Count mismatching rows by instance job_name
    mismatch_jobs = Counter()

    chunk_number = 0

    for chunk in pd.read_csv(
        split_path,
        usecols=["task_name", "job_name"],
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        chunk_number += 1
        total_rows += len(chunk)

        # Determine whether task_name exists in batch_task

        valid_jobs = chunk["task_name"].map(task_jobs)

        known_mask = valid_jobs.notna()

        known_task_rows += int(known_mask.sum())
        unmatched_task_rows += int((~known_mask).sum())


        # Only investigate rows whose task_name exists
        # in batch_task.


        known = chunk.loc[known_mask, ["task_name", "job_name"]].copy()
        known_valid_jobs = valid_jobs.loc[known_mask]

        # Check whether batch_instance job_name belongs to
        # the valid job_name set for that task_name.
        #
        # This is the vectorized equivalent of:
        #
        # job_name not in task_jobs[task_name]

        job_values = known["job_name"].astype(str).to_numpy()
        allowed_sets = known_valid_jobs.to_numpy()

        mismatch_flags = [
            job not in allowed
            for job, allowed in zip(job_values, allowed_sets)
        ]

        mismatch_mask = pd.Series(
            mismatch_flags,
            index=known.index
        )

        matching_count = int((~mismatch_mask).sum())
        mismatch_count = int(mismatch_mask.sum())

        matching_job_rows += matching_count
        mismatching_job_rows += mismatch_count


        # Aggregate only mismatching rows.


        if mismatch_count > 0:

            mismatches = known.loc[
                mismatch_mask,
                ["task_name", "job_name"]
            ].copy()

            # Count by task_name
            task_counts = mismatches["task_name"].value_counts()

            for task_name, count in task_counts.items():
                mismatch_tasks[task_name] += int(count)

            # Count by job_name
            job_counts = mismatches["job_name"].astype(str).value_counts()

            for job_name, count in job_counts.items():
                mismatch_jobs[job_name] += int(count)

            # Count task_name + batch_instance job_name
            pair_counts = (
                mismatches
                .assign(job_name=mismatches["job_name"].astype(str))
                .groupby(["task_name", "job_name"])
                .size()
            )

            for (task_name, job_name), count in pair_counts.items():
                mismatch_pairs[(task_name, job_name)] += int(count)

        # Progress

        if chunk_number % 100 == 0:
            print(
                f"Chunks: {chunk_number} | "
                f"Rows: {total_rows:,} | "
                f"Job mismatches: {mismatching_job_rows:,}"
            )

    # SPLIT SUMMARY
    print()
    print(f"{split_name.upper()} SUMMARY")

    print(f"Total rows:              {total_rows:,}")
    print(f"Known task rows:         {known_task_rows:,}")
    print(f"Unmatched task rows:     {unmatched_task_rows:,}")
    print(f"Matching job rows:       {matching_job_rows:,}")
    print(f"TRUE job mismatches:     {mismatching_job_rows:,}")

    if known_task_rows > 0:
        rate = (
            mismatching_job_rows /
            known_task_rows
        ) * 100

        print(f"Mismatch rate:           {rate:.6f}%")


    # TOP MISMATCH TASK/JOB COMBINATIONS
    print()
    print(f"TOP {TOP_N} MISMATCH TASK/JOB COMBINATIONS")
    print("-" * 70)

    top_pairs = mismatch_pairs.most_common(TOP_N)

    for rank, ((task_name, job_name), count) in enumerate(
        top_pairs,
        start=1
    ):
        print(
            f"{rank:02d}. "
            f"count={count:,} | "
            f"task_name={task_name} | "
            f"batch_instance_job_name={job_name}"
        )


    # TOP MISMATCH TASK NAMES

    print()
    print(f"TOP {TOP_N} TASK NAMES WITH JOB MISMATCHES")

    for rank, (task_name, count) in enumerate(
        mismatch_tasks.most_common(TOP_N),
        start=1
    ):
        valid_jobs = sorted(task_jobs.get(task_name, []))

        print(
            f"{rank:02d}. "
            f"count={count:,} | "
            f"task_name={task_name} | "
            f"valid_batch_task_jobs={valid_jobs}"
        )


    # TOP BATCH_INSTANCE JOB NAMES INVOLVED IN MISMATCHES
    print()
    print(f"TOP {TOP_N} BATCH_INSTANCE JOB NAMES IN MISMATCHES")

    for rank, (job_name, count) in enumerate(
        mismatch_jobs.most_common(TOP_N),
        start=1
    ):
        print(
            f"{rank:02d}. "
            f"count={count:,} | "
            f"batch_instance_job_name={job_name}"
        )

    overall_summary.append({
        "split": split_name,
        "total_rows": total_rows,
        "known_task_rows": known_task_rows,
        "unmatched_task_rows": unmatched_task_rows,
        "matching_job_rows": matching_job_rows,
        "mismatching_job_rows": mismatching_job_rows,
    })



# FINAL SUMMARY
print()
print("FINAL SUMMARY")

summary_df = pd.DataFrame(overall_summary)

print(summary_df.to_string(index=False))

print()
print("INVESTIGATION COMPLETE")