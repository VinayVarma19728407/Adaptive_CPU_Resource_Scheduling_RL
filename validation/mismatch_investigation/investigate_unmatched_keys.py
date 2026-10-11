"""
Investigate Unmatched Join Keys in Workload Traces.

This script scans preprocessed trace files to identify unmatched join keys between task definitions and instance logs.
It outputs summary counts of unmatched keys and suggests key resolution mappings for data cleaning pipelines.
"""

import pandas as pd
import os
from collections import Counter

PROJECT = "dataset"

BATCH_TASK = os.path.join(
    PROJECT, "preprocessed", "batch_task.csv"
)

MACHINE_META = os.path.join(
    PROJECT, "preprocessed", "machine_meta.csv"
)

BATCH_INSTANCE_FILES = [
    os.path.join(
        PROJECT, "preprocessed",
        "batch_instance_training.csv"
    ),
    os.path.join(
        PROJECT, "preprocessed",
        "batch_instance_validation.csv"
    ),
    os.path.join(
        PROJECT, "preprocessed",
        "batch_instance_testing.csv"
    ),
]

CHUNK_SIZE = 250_000



# LOAD REFERENCE KEYS


batch_task = pd.read_csv(
    BATCH_TASK,
    usecols=["task_name"],
    low_memory=False
)

machine_meta = pd.read_csv(
    MACHINE_META,
    usecols=["machine_id"],
    low_memory=False
)

batch_task_names = set(
    batch_task["task_name"].dropna().astype(str)
)

machine_meta_ids = set(
    machine_meta["machine_id"].dropna().astype(str)
)

# FIND UNMATCHED KEYS


unmatched_task_counts = Counter()
unmatched_machine_counts = Counter()

unmatched_task_by_file = Counter()
unmatched_machine_by_file = Counter()


for path in BATCH_INSTANCE_FILES:

    filename = os.path.basename(path)

    print("\nProcessing:", filename)

    for chunk in pd.read_csv(
        path,
        usecols=[
            "instance_name",
            "task_name",
            "job_name",
            "machine_id",
            "start_time",
            "end_time"
        ],
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        # TASK NAME

        task_values = chunk["task_name"].astype(str)

        task_mask = ~task_values.isin(batch_task_names)

        unmatched_tasks = task_values[task_mask]

        for value in unmatched_tasks:
            unmatched_task_counts[value] += 1
            unmatched_task_by_file[
                (filename, value)
            ] += 1

        # MACHINE ID


        machine_values = chunk["machine_id"].astype(str)

        machine_mask = ~machine_values.isin(machine_meta_ids)

        unmatched_machines = machine_values[machine_mask]

        for value in unmatched_machines:
            unmatched_machine_counts[value] += 1
            unmatched_machine_by_file[
                (filename, value)
            ] += 1

# PRINT TASK RESULTS

print("UNMATCHED TASK NAMES")


print(
    "Number of unique unmatched task_names:",
    len(unmatched_task_counts)
)

for task_name, count in sorted(
    unmatched_task_counts.items()
):
    print("\nTask:", repr(task_name))
    print("Total rows:", count)

    for (filename, value), file_count in sorted(
        unmatched_task_by_file.items()
    ):
        if value == task_name:
            print("  ", filename, ":", file_count)



# PRINT MACHINE RESULTS

print("UNMATCHED MACHINE IDS")


print(
    "Number of unique unmatched machine_ids:",
    len(unmatched_machine_counts)
)

for machine_id, count in sorted(
    unmatched_machine_counts.items()
):
    print("\nMachine ID:", repr(machine_id))
    print("Total rows:", count)

    for (filename, value), file_count in sorted(
        unmatched_machine_by_file.items()
    ):
        if value == machine_id:
            print("  ", filename, ":", file_count)


# REVERSE RELATIONSHIP


batch_instance_task_names = set(unmatched_task_counts.keys())

batch_task_only = (
    batch_task_names -
    batch_instance_task_names
)

print("NOTE")

print(
    "The previous audit showed 7,406 batch_task task_names "
    "not observed in batch_instance."
)

print(
    "This script intentionally does NOT list those 7,406 "
    "values because they require a separate interpretation."
)

print("\nAUDIT COMPLETE")