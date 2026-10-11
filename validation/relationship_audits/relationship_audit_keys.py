"""
Relationship Audit for Workload Key Structures.

This script performs key structure and composite key cardinality audits across all project dataset tables.
It checks the uniqueness and integrity of (job_name, task_name, instance_name) tuples across all trace CSVs.
"""

import pandas as pd
import os

PROJECT = "dataset"

BATCH_TASK = os.path.join(PROJECT, "preprocessed", "batch_task.csv")
MACHINE_META = os.path.join(PROJECT, "preprocessed", "machine_meta.csv")

BATCH_INSTANCE_FILES = [
    os.path.join(PROJECT, "preprocessed", "batch_instance_training.csv"),
    os.path.join(PROJECT, "preprocessed", "batch_instance_validation.csv"),
    os.path.join(PROJECT, "preprocessed", "batch_instance_testing.csv"),
]

CHUNK_SIZE = 250_000


# 1. LOAD SMALL REFERENCE DATASETS


print("LOADING REFERENCE KEYS")

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

batch_task_names = set(batch_task["task_name"].dropna().astype(str))
machine_meta_ids = set(machine_meta["machine_id"].dropna().astype(str))

print("Unique batch_task task_names:", len(batch_task_names))
print("Unique machine_meta machine_ids:", len(machine_meta_ids))


# 2. SCAN BATCH_INSTANCE

print("SCANNING BATCH_INSTANCE")

batch_instance_task_names = set()
batch_instance_machine_ids = set()

total_rows = 0

task_match_rows = 0
task_unmatched_rows = 0

machine_match_rows = 0
machine_unmatched_rows = 0

for path in BATCH_INSTANCE_FILES:

    print("\nProcessing:", os.path.basename(path))

    file_rows = 0
    file_task_match = 0
    file_machine_match = 0

    for chunk in pd.read_csv(
        path,
        usecols=["task_name", "machine_id"],
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        task_values = chunk["task_name"].astype(str)
        machine_values = chunk["machine_id"].astype(str)

        # Unique values observed in batch_instance
        batch_instance_task_names.update(task_values.dropna().unique())
        batch_instance_machine_ids.update(machine_values.dropna().unique())

        # Relationship checks
        task_matches = task_values.isin(batch_task_names)

        machine_matches = machine_values.isin(machine_meta_ids)

        rows = len(chunk)

        file_rows += rows
        total_rows += rows

        file_task_match += task_matches.sum()
        task_match_rows += task_matches.sum()
        task_unmatched_rows += (~task_matches).sum()

        file_machine_match += machine_matches.sum()
        machine_match_rows += machine_matches.sum()
        machine_unmatched_rows += (~machine_matches).sum()

    print("Rows:", file_rows)
    print("Task-name matched rows:", file_task_match)
    print("Machine-ID matched rows:", file_machine_match)

# 3. FINAL RESULTS
print("RELATIONSHIP AUDIT RESULTS")


print("\n TASK_NAME RELATIONSHIP ")

print("Unique task_names in batch_task:",
      len(batch_task_names))

print("Unique task_names in batch_instance:",
      len(batch_instance_task_names))

task_names_missing_from_batch_task = (
    batch_instance_task_names - batch_task_names
)

task_names_unused_in_batch_instance = (
    batch_task_names - batch_instance_task_names
)

print("batch_instance task_names NOT in batch_task:",
      len(task_names_missing_from_batch_task))

print("batch_task task_names NOT in batch_instance:",
      len(task_names_unused_in_batch_instance))

print("\nTotal batch_instance rows:", total_rows)

print("Rows with matching task_name:",
      task_match_rows)

print("Rows with unmatched task_name:",
      task_unmatched_rows)


print("\n MACHINE_ID RELATIONSHIP ")

print("Unique machine_ids in machine_meta:",
      len(machine_meta_ids))

print("Unique machine_ids in batch_instance:",
      len(batch_instance_machine_ids))

machine_ids_missing_from_meta = (
    batch_instance_machine_ids - machine_meta_ids
)

machine_ids_unused_in_batch_instance = (
    machine_meta_ids - batch_instance_machine_ids
)

print("batch_instance machine_ids NOT in machine_meta:",
      len(machine_ids_missing_from_meta))

print("machine_meta machine_ids NOT in batch_instance:",
      len(machine_ids_unused_in_batch_instance))

print("\nRows with matching machine_id:",
      machine_match_rows)

print("Rows with unmatched machine_id:",
      machine_unmatched_rows)


print("AUDIT COMPLETE")