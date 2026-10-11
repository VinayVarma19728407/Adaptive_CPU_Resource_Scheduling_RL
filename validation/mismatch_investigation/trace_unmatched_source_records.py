"""
Trace Unmatched Source Records in Raw Traces.

This script traces original raw dataset archives to identify the source origin of unmatched instance records.
It reads raw tar/csv files and checks line numbers and raw string formats for corrupted or malformed entries.
"""

import os
import pandas as pd

PROJECT_PATH = (
    "C:/Users/gvkva/Desktop/My ongoing projects/"
    "Machine Learning Project/Adaptive-Multi-Core-CPU-Scheduler-RL"
)

RAW_BATCH_TASK = os.path.join(
    PROJECT_PATH, "dataset", "raw", "batch_task.csv"
)

CLEAN_BATCH_TASK = os.path.join(
    PROJECT_PATH, "dataset", "cleaned", "batch_task.csv"
)

RAW_MACHINE_META = os.path.join(
    PROJECT_PATH, "dataset", "raw", "machine_meta.csv"
)

CLEAN_MACHINE_META = os.path.join(
    PROJECT_PATH, "dataset", "cleaned", "machine_meta.csv"
)

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

CHUNK_SIZE = 250000



# BATCH TASK SOURCE TRACE


print()
print("TRACING UNMATCHED TASK NAMES THROUGH BATCH_TASK")


columns = [
    "task_name",
    "instance_num",
    "job_name",
    "task_type",
    "status",
    "start_time",
    "end_time",
    "plan_cpu",
    "plan_mem",
]

for label, path, header in [
    ("RAW", RAW_BATCH_TASK, None),
    ("CLEANED", CLEAN_BATCH_TASK, 0),
]:

    print()
    print(label)
    print(path)


    counts = {
        task: 0
        for task in UNMATCHED_TASK_NAMES
    }

    invalid_duration = {
        task: 0
        for task in UNMATCHED_TASK_NAMES
    }

    if label == "RAW":
        reader = pd.read_csv(
            path,
            header=None,
            names=columns,
            usecols=columns,
            dtype="string",
            chunksize=CHUNK_SIZE,
            low_memory=False,
        )
    else:
        reader = pd.read_csv(
            path,
            header=0,
            usecols=columns,
            dtype="string",
            chunksize=CHUNK_SIZE,
            low_memory=False,
        )

    for chunk in reader:

        chunk["task_name"] = chunk["task_name"].str.strip()

        matches = chunk[
            chunk["task_name"].isin(UNMATCHED_TASK_NAMES)
        ].copy()

        if len(matches) == 0:
            continue

        for task_name, group in matches.groupby("task_name"):

            counts[task_name] += len(group)

            start = pd.to_numeric(
                group["start_time"],
                errors="coerce"
            )

            end = pd.to_numeric(
                group["end_time"],
                errors="coerce"
            )

            invalid_duration[task_name] += int(
                (end < start).sum()
            )

    print()
    print("RESULTS")

    for task_name in sorted(UNMATCHED_TASK_NAMES):

        print()
        print(task_name)
        print("Occurrences:", counts[task_name])

        if label == "RAW":
            print(
                "Rows with end_time < start_time:",
                invalid_duration[task_name]
            )


# MACHINE META SOURCE TRACE


print()

print("TRACING MACHINE 2739 THROUGH MACHINE_META")


for label, path, header in [
    ("RAW", RAW_MACHINE_META, 0),
    ("CLEANED", CLEAN_MACHINE_META, 0),
]:

    print()
    print(label)
    print(path)

    machine_count = 0

    reader = pd.read_csv(
        path,
        header=header,
        dtype="string",
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for chunk in reader:

        if "machine_id" not in chunk.columns:
            raise ValueError(
                "machine_id column not found in " + label
            )

        mask = (
            chunk["machine_id"]
            .astype("string")
            .str.strip()
            .eq("2739")
        )

        machine_count += int(mask.sum())

    print(
        "Machine 2739 records:",
        machine_count
    )


print()
print("SOURCE TRACE COMPLETED")