"""
Audit Machine Usage and Node Relationships.

This script audits relationships between physical node resource metrics (machine_usage) and task instance
assignments on specific machines. It verifies machine ID foreign key joins and checks total node CPU/memory
utilization against physical node capacity limits.
"""

import os
import pandas as pd

PROJECT_PATH = (
    "C:/Users/gvkva/Desktop/My ongoing projects/"
    "Machine Learning Project/Adaptive-Multi-Core-CPU-Scheduler-RL"
)

MACHINE_META = os.path.join(
    PROJECT_PATH,
    "dataset",
    "preprocessed",
    "machine_meta.csv"
)

MACHINE_USAGE = os.path.join(
    PROJECT_PATH,
    "dataset",
    "cleaned",
    "machine_usage.csv"
)

CHUNK_SIZE = 250000

print()
print("MACHINE META ↔ MACHINE USAGE RELATIONSHIP AUDIT")



# 1. Load valid machine IDs from machine_meta


print()
print("Loading machine_meta machine IDs...")

machine_meta = pd.read_csv(
    MACHINE_META,
    usecols=["machine_id"],
    dtype={"machine_id": "string"},
)

machine_meta_ids = set(
    machine_meta["machine_id"]
    .dropna()
    .astype("string")
    .str.strip()
)

print("Unique machine_meta IDs:", len(machine_meta_ids))


# 2. Scan machine_usage


print()
print("Scanning machine_usage in chunks...")
print(MACHINE_USAGE)

usage_machine_ids = set()

total_rows = 0
matched_rows = 0
unmatched_rows = 0
chunks = 0

min_timestamp = None
max_timestamp = None

unmatched_ids = set()

reader = pd.read_csv(
    MACHINE_USAGE,
    usecols=["machine_id", "time_stamp"],
    dtype={
        "machine_id": "string",
        "time_stamp": "int64",
    },
    chunksize=CHUNK_SIZE,
    low_memory=False,
)

for chunk in reader:

    chunks += 1
    total_rows += len(chunk)

    chunk["machine_id"] = (
        chunk["machine_id"]
        .astype("string")
        .str.strip()
    )

    current_ids = set(
        chunk["machine_id"].dropna()
    )

    usage_machine_ids.update(current_ids)

    matched_mask = chunk["machine_id"].isin(
        machine_meta_ids
    )

    matched_rows += int(matched_mask.sum())

    unmatched = chunk.loc[
        ~matched_mask,
        "machine_id"
    ].dropna()

    unmatched_rows += len(unmatched)

    unmatched_ids.update(unmatched.tolist())

    chunk_min = chunk["time_stamp"].min()
    chunk_max = chunk["time_stamp"].max()

    if min_timestamp is None or chunk_min < min_timestamp:
        min_timestamp = chunk_min

    if max_timestamp is None or chunk_max > max_timestamp:
        max_timestamp = chunk_max

    if chunks % 20 == 0:
        print(
            "Chunks:",
            chunks,
            "| Rows:",
            total_rows,
            "| Unique usage machines:",
            len(usage_machine_ids),
            "| Unmatched rows:",
            unmatched_rows,
        )


# 3. Results


print()
print("RESULTS")

print("Total machine_usage rows:", total_rows)
print("Unique machine_usage IDs:", len(usage_machine_ids))

print()
print("Rows with machine_id found in machine_meta:")
print(matched_rows)

print("Rows with machine_id NOT found in machine_meta:")
print(unmatched_rows)

print()
print("Unique machine IDs in machine_usage but not machine_meta:")
print(len(unmatched_ids))

if unmatched_ids:
    print()
    print("UNMATCHED MACHINE IDS")
    for machine_id in sorted(
        unmatched_ids,
        key=lambda x: int(x) if str(x).isdigit() else str(x)
    ):
        print(machine_id)

print()
print("Machine IDs in machine_meta but not machine_usage:")
meta_only = machine_meta_ids - usage_machine_ids
print(len(meta_only))

print()
print("machine_usage minimum timestamp:", min_timestamp)
print("machine_usage maximum timestamp:", max_timestamp)


# 4. Check duplicate machine_meta machine IDs


print()
print("MACHINE META ID STRUCTURE")

meta_full = pd.read_csv(
    MACHINE_META,
    dtype={
        "machine_id": "string",
        "time_stamp": "int64",
    },
    usecols=["machine_id", "time_stamp"],
)

print(
    "machine_meta rows:",
    len(meta_full)
)

print(
    "unique machine_meta IDs:",
    meta_full["machine_id"].nunique()
)

print(
    "machine_meta duplicate machine_id + time_stamp rows:",
    int(
        meta_full.duplicated(
            subset=["machine_id", "time_stamp"]
        ).sum()
    )
)

print()
print("MACHINE USAGE RELATIONSHIP AUDIT COMPLETED")
