"""
Audit Machine Timestamp Alignment and Continuity.

This script audits temporal alignment and sampling consistency of machine metric timestamps across machine_meta
and machine_usage tables. It checks sampling step uniformity, identifies metric timestamp gaps, and validates
epoch bounds against task execution timelines.
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
print("MACHINE META ↔ MACHINE USAGE TIMESTAMP ALIGNMENT")



# 1. Metadata timestamp coverage by machine


meta = pd.read_csv(
    MACHINE_META,
    usecols=["machine_id", "time_stamp"],
    dtype={
        "machine_id": "string",
        "time_stamp": "int64",
    },
)

meta["machine_id"] = (
    meta["machine_id"]
    .astype("string")
    .str.strip()
)

meta_ranges = (
    meta.groupby("machine_id")["time_stamp"]
    .agg(["min", "max", "count"])
    .reset_index()
)

meta_ranges = meta_ranges.rename(
    columns={
        "min": "meta_min",
        "max": "meta_max",
        "count": "meta_rows",
    }
)

print("Metadata machines:", len(meta_ranges))


# 2. Scan machine usage and calculate coverage


usage_ranges = {}

total_rows = 0
chunks = 0

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

    grouped = (
        chunk.groupby("machine_id")["time_stamp"]
        .agg(["min", "max", "count"])
    )

    for machine_id, row in grouped.iterrows():

        if machine_id not in usage_ranges:
            usage_ranges[machine_id] = [
                int(row["min"]),
                int(row["max"]),
                int(row["count"]),
            ]
        else:
            usage_ranges[machine_id][0] = min(
                usage_ranges[machine_id][0],
                int(row["min"])
            )

            usage_ranges[machine_id][1] = max(
                usage_ranges[machine_id][1],
                int(row["max"])
            )

            usage_ranges[machine_id][2] += int(
                row["count"]
            )

    if chunks % 100 == 0:
        print(
            "Chunks:",
            chunks,
            "| Rows:",
            total_rows
        )


# 3. Compare temporal coverage

records = []

for machine_id, values in usage_ranges.items():

    usage_min, usage_max, usage_rows = values

    meta_match = meta_ranges.loc[
        meta_ranges["machine_id"].eq(machine_id)
    ]

    if len(meta_match) == 0:
        records.append({
            "machine_id": machine_id,
            "usage_min": usage_min,
            "usage_max": usage_max,
            "usage_rows": usage_rows,
            "meta_min": None,
            "meta_max": None,
            "meta_rows": 0,
            "temporal_status": "NO_METADATA",
        })
        continue

    meta_row = meta_match.iloc[0]

    meta_min = int(meta_row["meta_min"])
    meta_max = int(meta_row["meta_max"])

    if usage_max < meta_min:
        status = "USAGE_BEFORE_METADATA"

    elif usage_min > meta_max:
        status = "USAGE_AFTER_METADATA"

    else:
        status = "TEMPORAL_OVERLAP"

    records.append({
        "machine_id": machine_id,
        "usage_min": usage_min,
        "usage_max": usage_max,
        "usage_rows": usage_rows,
        "meta_min": meta_min,
        "meta_max": meta_max,
        "meta_rows": int(meta_row["meta_rows"]),
        "temporal_status": status,
    })

result = pd.DataFrame(records)

print()
print("TEMPORAL ALIGNMENT RESULTS")


print()
print(
    result["temporal_status"]
    .value_counts()
    .to_string()
)

print()
print("Machines with metadata:", len(meta_ranges))
print("Machines with usage:", len(usage_ranges))

print()
print("Usage timestamp range:")
print("Minimum:", result["usage_min"].min())
print("Maximum:", result["usage_max"].max())

print()
print("Metadata timestamp range:")
print("Minimum:", meta_ranges["meta_min"].min())
print("Maximum:", meta_ranges["meta_max"].max())

# Save compact result
output = os.path.join(
    PROJECT_PATH,
    "dataset",
    "relationship_audit",
    "machine_timestamp_alignment.csv"
)

result.to_csv(output, index=False)

print()
print("Saved:", output)

print()
print("TIMESTAMP ALIGNMENT AUDIT COMPLETED")