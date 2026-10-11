"""
Inspect Unmatched Task and Instance Records.

This script performs detailed inspection and diagnosis of unmatched or orphan records between task definitions
and instance execution logs. It filters unmatched job/task keys, samples problematic records, and outputs trace logs
for data cleaning and key resolution.
"""

import os
import pandas as pd

PROJECT_PATH = (
    "C:/Users/gvkva/Desktop/My ongoing projects/"
    "Machine Learning Project/Adaptive-Multi-Core-CPU-Scheduler-RL"
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

BATCH_TASK = os.path.join(
    PROJECT_PATH,
    "dataset",
    "preprocessed",
    "batch_task.csv"
)

MACHINE_META = os.path.join(
    PROJECT_PATH,
    "dataset",
    "preprocessed",
    "machine_meta.csv"
)

CHUNK_SIZE = 250000

UNMATCHED_TASK_NAMES = {
    "J12_5_9_10_11_13_14",
    "J14_1_2_4_5_13",
    "J20_9_11_12_19_23",
    "J22_9_13_14_18",
    "J37_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_",
    "J5_2_3_4_7_9_11",
    "M22_2_4_6_8_10_12_14_16_18_21",
    "task_NTUwNzc2NzM4MjY5NTYwMDI1OA==",
}

UNMATCHED_MACHINE_ID = "2739"

OUTPUT_DIR = os.path.join(
    PROJECT_PATH,
    "dataset",
    "relationship_audit"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# 1. Inspect unmatched task names in batch_instance


print()
print("INSPECTING UNMATCHED TASK NAMES")


task_output = os.path.join(
    OUTPUT_DIR,
    "unmatched_task_records.csv"
)

first_write = True
total_unmatched_rows = 0

usecols = [
    "instance_name",
    "task_name",
    "job_name",
    "task_type",
    "status",
    "start_time",
    "end_time",
    "machine_id",
    "seq_no",
    "total_seq_no",
    "cpu_avg",
    "cpu_max",
    "mem_avg",
    "mem_max",
    "duration",
]

for split_name, path in BATCH_INSTANCE_FILES.items():

    print()
    print("Scanning:", split_name)
    print(path)

    reader = pd.read_csv(
        path,
        usecols=usecols,
        dtype={
            "instance_name": "string",
            "task_name": "string",
            "job_name": "string",
            "task_type": "int64",
            "status": "int64",
            "start_time": "int64",
            "end_time": "int64",
            "machine_id": "string",
            "seq_no": "int64",
            "total_seq_no": "int64",
            "cpu_avg": "float64",
            "cpu_max": "float64",
            "mem_avg": "float64",
            "mem_max": "float64",
            "duration": "float64",
        },
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    split_rows = 0

    for chunk in reader:

        mask = chunk["task_name"].isin(UNMATCHED_TASK_NAMES)

        matched = chunk.loc[mask].copy()

        if len(matched) > 0:
            matched.insert(0, "split", split_name)

            matched.to_csv(
                task_output,
                mode="w" if first_write else "a",
                header=first_write,
                index=False,
            )

            first_write = False

            total_unmatched_rows += len(matched)
            split_rows += len(matched)

    print(
        "Unmatched rows found in",
        split_name,
        ":",
        split_rows
    )

print()
print("TOTAL UNMATCHED TASK ROWS:", total_unmatched_rows)
print("Output:", task_output)


# 2. Count unmatched task names individually


print()

print("UNMATCHED TASK NAME COUNTS")
if os.path.exists(task_output):

    result = pd.read_csv(
        task_output,
        dtype={
            "task_name": "string",
            "split": "string",
        },
        usecols=["split", "task_name"],
    )

    counts = (
        result
        .groupby(["task_name", "split"])
        .size()
        .reset_index(name="rows")
        .sort_values(["task_name", "split"])
    )

    print(counts.to_string(index=False))

else:
    print("No unmatched task records found.")


# 3. Inspect machine 2739


print()
print("INSPECTING UNMATCHED MACHINE 2739")

machine_output = os.path.join(
    OUTPUT_DIR,
    "unmatched_machine_2739_records.csv"
)

first_write = True
machine_rows = 0

for split_name, path in BATCH_INSTANCE_FILES.items():

    print()
    print("Scanning:", split_name)

    reader = pd.read_csv(
        path,
        usecols=usecols,
        dtype={
            "instance_name": "string",
            "task_name": "string",
            "job_name": "string",
            "task_type": "int64",
            "status": "int64",
            "start_time": "int64",
            "end_time": "int64",
            "machine_id": "string",
            "seq_no": "int64",
            "total_seq_no": "int64",
            "cpu_avg": "float64",
            "cpu_max": "float64",
            "mem_avg": "float64",
            "mem_max": "float64",
            "duration": "float64",
        },
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    split_rows = 0

    for chunk in reader:

        mask = (
            chunk["machine_id"]
            .astype("string")
            .str.strip()
            .eq(UNMATCHED_MACHINE_ID)
        )

        matched = chunk.loc[mask].copy()

        if len(matched) > 0:

            matched.insert(0, "split", split_name)

            matched.to_csv(
                machine_output,
                mode="w" if first_write else "a",
                header=first_write,
                index=False,
            )

            first_write = False

            machine_rows += len(matched)
            split_rows += len(matched)

    print(
        "Machine 2739 rows in",
        split_name,
        ":",
        split_rows
    )

print()
print("TOTAL MACHINE 2739 ROWS:", machine_rows)
print("Output:", machine_output)


# 4. Check machine 2739 in machine_meta


print()
print("CHECKING MACHINE 2739 IN MACHINE META")


machine_meta_found = False

reader = pd.read_csv(
    MACHINE_META,
    dtype={
        "machine_id": "string",
        "time_stamp": "int64",
        "failure_domain_1": "int64",
        "failure_domain_2": "float64",
        "cpu_num": "int32",
        "mem_size": "float64",
        "status": "int8",
    },
    low_memory=False,
)

machine_meta_match = reader.loc[
    reader["machine_id"].str.strip().eq(UNMATCHED_MACHINE_ID)
]

if len(machine_meta_match) == 0:
    print("Machine 2739 NOT FOUND in machine_meta.")
else:
    machine_meta_found = True
    print("Machine 2739 FOUND in machine_meta:")
    print(machine_meta_match.to_string(index=False))



# 5. Check unmatched task names in batch_task


print()
print("FINAL CHECK: UNMATCHED TASK NAMES IN BATCH_TASK")

batch_task_found = set()

reader = pd.read_csv(
    BATCH_TASK,
    usecols=["task_name"],
    dtype={"task_name": "string"},
    chunksize=CHUNK_SIZE,
    low_memory=False,
)

for chunk in reader:

    found = set(
        chunk.loc[
            chunk["task_name"].isin(UNMATCHED_TASK_NAMES),
            "task_name"
        ].dropna()
    )

    batch_task_found.update(found)

for task_name in sorted(UNMATCHED_TASK_NAMES):

    if task_name in batch_task_found:
        print("FOUND in batch_task:", task_name)
    else:
        print("NOT FOUND in batch_task:", task_name)


print()
print("INSPECTION COMPLETED")
print()
print("Task records:")
print(task_output)
print()
print("Machine 2739 records:")
print(machine_output)