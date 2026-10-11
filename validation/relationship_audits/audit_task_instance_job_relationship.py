"""
Audit Task, Instance, and Job Foreign Key Relationships.

This script audits relational foreign-key integrity between parent jobs (batch_task), child tasks, and task
execution instances (batch_instance). It checks referential integrity, orphan instance presence, and job-level
instance count consistency across data files.
"""

import os
import pandas as pd

PROJECT_PATH = (
    "C:/Users/gvkva/Desktop/My ongoing projects/"
    "Machine Learning Project/Adaptive-Multi-Core-CPU-Scheduler-RL"
)

BATCH_TASK = os.path.join(
    PROJECT_PATH,
    "dataset",
    "preprocessed",
    "batch_task.csv"
)

BATCH_INSTANCE_FILES = {
    "training": os.path.join(
        PROJECT_PATH,
        "dataset",
        "preprocessed",
        "batch_instance_training.csv"
    ),
    "validation": os.path.join(
        PROJECT_PATH,
        "dataset",
        "preprocessed",
        "batch_instance_validation.csv"
    ),
    "testing": os.path.join(
        PROJECT_PATH,
        "dataset",
        "preprocessed",
        "batch_instance_testing.csv"
    ),
}

OUTPUT_DIR = os.path.join(
    PROJECT_PATH,
    "dataset",
    "relationship_audit"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

CHUNK_SIZE = 250000



# 1. BUILD TASK -> JOB / TASK_TYPE REFERENCE


print()
print("BUILDING BATCH_TASK REFERENCE MAPPINGS")


batch_task = pd.read_csv(
    BATCH_TASK,
    usecols=[
        "task_name",
        "job_name",
        "task_type",
    ],
    dtype={
        "task_name": "string",
        "job_name": "string",
        "task_type": "string",
    },
    low_memory=False,
)

batch_task["task_name"] = batch_task["task_name"].str.strip()
batch_task["job_name"] = batch_task["job_name"].str.strip()

print("batch_task rows:", len(batch_task))
print("unique task_names:", batch_task["task_name"].nunique())
print("unique job_names:", batch_task["job_name"].nunique())



# Check whether batch_task itself maps task_name consistently


task_job_counts = (
    batch_task
    .groupby("task_name")["job_name"]
    .nunique(dropna=False)
)

task_type_counts = (
    batch_task
    .groupby("task_name")["task_type"]
    .nunique(dropna=False)
)

inconsistent_task_job = set(
    task_job_counts[
        task_job_counts > 1
    ].index
)

inconsistent_task_type = set(
    task_type_counts[
        task_type_counts > 1
    ].index
)

print()
print("batch_task task_names mapping to multiple job_names:",
      len(inconsistent_task_job))

print("batch_task task_names mapping to multiple task_types:",
      len(inconsistent_task_type))


# Create reference dictionaries


task_job_reference = (
    batch_task
    .drop_duplicates("task_name")
    .set_index("task_name")["job_name"]
    .to_dict()
)

task_type_reference = (
    batch_task
    .drop_duplicates("task_name")
    .set_index("task_name")["task_type"]
    .to_dict()
)

valid_task_names = set(task_job_reference.keys())


# 2. SCAN BATCH_INSTANCE


print()
print("SCANNING BATCH_INSTANCE STRUCTURAL RELATIONSHIPS")

summary_rows = []

all_inconsistent_job_examples = []
all_inconsistent_type_examples = []

for split_name, path in BATCH_INSTANCE_FILES.items():

    print()
    print("SPLIT:", split_name)
    print(path)

    total_rows = 0

    matched_task_rows = 0
    unmatched_task_rows = 0

    job_mismatch_rows = 0
    type_mismatch_rows = 0

    invalid_seq_rows = 0
    seq_greater_than_total_rows = 0

    negative_duration_rows = 0
    zero_duration_rows = 0
    positive_duration_rows = 0

    unique_tasks = set()
    unique_jobs = set()

    reader = pd.read_csv(
        path,
        usecols=[
            "task_name",
            "job_name",
            "task_type",
            "seq_no",
            "total_seq_no",
            "duration",
        ],
        dtype={
            "task_name": "string",
            "job_name": "string",
            "task_type": "string",
            "seq_no": "int64",
            "total_seq_no": "int64",
            "duration": "float64",
        },
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    chunks = 0

    for chunk in reader:

        chunks += 1
        total_rows += len(chunk)

        chunk["task_name"] = (
            chunk["task_name"]
            .astype("string")
            .str.strip()
        )

        chunk["job_name"] = (
            chunk["job_name"]
            .astype("string")
            .str.strip()
        )

        unique_tasks.update(
            chunk["task_name"].dropna().tolist()
        )

        unique_jobs.update(
            chunk["job_name"].dropna().tolist()
        )


        # Task existence


        task_exists = chunk["task_name"].isin(
            valid_task_names
        )

        matched_task_rows += int(task_exists.sum())
        unmatched_task_rows += int((~task_exists).sum())

        # Compare job_name and task_type only for matched tasks


        matched = chunk.loc[task_exists].copy()

        if len(matched) > 0:

            expected_job = (
                matched["task_name"]
                .map(task_job_reference)
            )

            expected_type = (
                matched["task_name"]
                .map(task_type_reference)
            )

            job_mismatch = (
                matched["job_name"]
                != expected_job
            )

            type_mismatch = (
                matched["task_type"]
                != expected_type
            )

            job_mismatch_rows += int(
                job_mismatch.sum()
            )

            type_mismatch_rows += int(
                type_mismatch.sum()
            )

            # Keep only a small number of examples
            if len(all_inconsistent_job_examples) < 100:
                examples = matched.loc[
                    job_mismatch,
                    [
                        "task_name",
                        "job_name",
                    ],
                ]

                all_inconsistent_job_examples.extend(
                    examples.head(
                        100 - len(all_inconsistent_job_examples)
                    ).to_dict("records")
                )

            if len(all_inconsistent_type_examples) < 100:
                examples = matched.loc[
                    type_mismatch,
                    [
                        "task_name",
                        "task_type",
                    ],
                ]

                all_inconsistent_type_examples.extend(
                    examples.head(
                        100 - len(all_inconsistent_type_examples)
                    ).to_dict("records")
                )


        # Sequence validation


        invalid_seq = (
            (chunk["seq_no"] < 1)
            |
            (chunk["total_seq_no"] < 1)
        )

        seq_greater_than_total = (
            chunk["seq_no"]
            >
            chunk["total_seq_no"]
        )

        invalid_seq_rows += int(
            invalid_seq.sum()
        )

        seq_greater_than_total_rows += int(
            seq_greater_than_total.sum()
        )

        # Duration validation


        negative = chunk["duration"] < 0
        zero = chunk["duration"] == 0
        positive = chunk["duration"] > 0

        negative_duration_rows += int(
            negative.sum()
        )

        zero_duration_rows += int(
            zero.sum()
        )

        positive_duration_rows += int(
            positive.sum()
        )

        if chunks % 100 == 0:
            print(
                "Chunks:",
                chunks,
                "| Rows:",
                total_rows,
                "| Task mismatches:",
                job_mismatch_rows,
                "| Type mismatches:",
                type_mismatch_rows,
            )

    print()
    print("RESULTS FOR", split_name)
    print("Rows:", total_rows)
    print("Unique task_names:", len(unique_tasks))
    print("Unique job_names:", len(unique_jobs))
    print("Matched task rows:", matched_task_rows)
    print("Unmatched task rows:", unmatched_task_rows)
    print("Job-name mismatch rows:", job_mismatch_rows)
    print("Task-type mismatch rows:", type_mismatch_rows)
    print("Invalid sequence rows:", invalid_seq_rows)
    print("seq_no > total_seq_no rows:", seq_greater_than_total_rows)
    print("Negative duration rows:", negative_duration_rows)
    print("Zero duration rows:", zero_duration_rows)
    print("Positive duration rows:", positive_duration_rows)

    summary_rows.append({
        "split": split_name,
        "rows": total_rows,
        "unique_task_names": len(unique_tasks),
        "unique_job_names": len(unique_jobs),
        "matched_task_rows": matched_task_rows,
        "unmatched_task_rows": unmatched_task_rows,
        "job_mismatch_rows": job_mismatch_rows,
        "task_type_mismatch_rows": type_mismatch_rows,
        "invalid_seq_rows": invalid_seq_rows,
        "seq_greater_than_total_rows": seq_greater_than_total_rows,
        "negative_duration_rows": negative_duration_rows,
        "zero_duration_rows": zero_duration_rows,
        "positive_duration_rows": positive_duration_rows,
    })


# 3. SAVE RESULTS


summary = pd.DataFrame(summary_rows)

summary_path = os.path.join(
    OUTPUT_DIR,
    "task_instance_job_relationship_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)

print()
print("OVERALL SUMMARY")

print(summary.to_string(index=False))

print()
print("Summary saved to:")
print(summary_path)


# 4. EXAMPLES OF ANY STRUCTURAL MISMATCHES


print()
print("MISMATCH EXAMPLES")

if all_inconsistent_job_examples:
    print()
    print("JOB-NAME MISMATCH EXAMPLES:")
    print(
        pd.DataFrame(
            all_inconsistent_job_examples
        ).to_string(index=False)
    )
else:
    print("No job-name mismatch examples.")

if all_inconsistent_type_examples:
    print()
    print("TASK-TYPE MISMATCH EXAMPLES:")
    print(
        pd.DataFrame(
            all_inconsistent_type_examples
        ).to_string(index=False)
    )
else:
    print("No task-type mismatch examples.")


print()
print("TASK / INSTANCE / JOB AUDIT COMPLETED")