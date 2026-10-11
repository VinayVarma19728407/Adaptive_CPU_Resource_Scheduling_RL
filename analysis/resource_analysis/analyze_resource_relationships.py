"""
Resource Relationship Analysis.

This script examines inter-resource relationships within Alibaba batch workloads. It calculates
cross-column correlation matrices, scatter metrics, and resource scaling behaviors between CPU
and memory allocations, task execution durations, and instance-level workload metrics across cluster nodes.
"""

import pandas as pd
import numpy as np

BATCH_TASK = "dataset/preprocessed/batch_task.csv"
BATCH_INSTANCE = "dataset/preprocessed/batch_instance_training.csv"

CHUNK_SIZE = 250_000


# ---------------------------------------------------------------------------
# 1. BATCH TASK RESOURCE INFORMATION
# ---------------------------------------------------------------------------

print("LOADING BATCH TASK RESOURCE INFORMATION")

batch_task = pd.read_csv(
    BATCH_TASK,
    usecols=[
        "task_name",
        "job_name",
        "plan_cpu",
        "plan_mem",
    ],
)

print("batch_task rows:", len(batch_task))

print("\nBATCH TASK RESOURCE SUMMARY")
print(
    batch_task[
        [
            "plan_cpu",
            "plan_mem",
        ]
    ].describe()
)


# ---------------------------------------------------------------------------
# 2. BATCH INSTANCE RESOURCE INFORMATION
# ---------------------------------------------------------------------------

print("\nPROCESSING BATCH INSTANCE TRAINING DATA IN CHUNKS")

usecols = [
    "instance_name",
    "task_name",
    "job_name",
    "duration",
    "cpu_avg",
    "cpu_max",
    "mem_avg",
    "mem_max",
]

resource_columns = [
    "cpu_avg",
    "cpu_max",
    "mem_avg",
    "mem_max",
]

analysis_columns = [
    "duration",
    "cpu_avg",
    "cpu_max",
    "mem_avg",
    "mem_max",
]


# Running statistics
count = 0
missing_counts = {column: 0 for column in analysis_columns}

sums = {column: 0.0 for column in analysis_columns}
sums_squared = {column: 0.0 for column in analysis_columns}

mins = {column: np.inf for column in analysis_columns}
maxs = {column: -np.inf for column in analysis_columns}


# For correlation calculations
sum_xy = {
    ("duration", column): 0.0
    for column in resource_columns
}

sum_x = {
    "duration": 0.0,
    **{column: 0.0 for column in resource_columns},
}

sum_x_squared = {
    "duration": 0.0,
    **{column: 0.0 for column in resource_columns},
}

pair_counts = {
    ("duration", column): 0
    for column in resource_columns
}


chunk_number = 0

for chunk in pd.read_csv(
    BATCH_INSTANCE,
    usecols=usecols,
    chunksize=CHUNK_SIZE,
):

    chunk_number += 1

    numeric = chunk[analysis_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    for column in analysis_columns:

        values = numeric[column].dropna()

        if len(values) == 0:
            continue

        missing_counts[column] += numeric[column].isna().sum()

        sums[column] += values.sum()
        sums_squared[column] += (values ** 2).sum()

        mins[column] = min(mins[column], values.min())
        maxs[column] = max(maxs[column], values.max())

    # ---------------------------------------------------------------
    # Duration ↔ resource correlations
    # ---------------------------------------------------------------

    duration = numeric["duration"]

    for resource in resource_columns:

        valid = duration.notna() & numeric[resource].notna()

        d = duration[valid].to_numpy(dtype=float)
        r = numeric.loc[valid, resource].to_numpy(dtype=float)

        n = len(d)

        if n == 0:
            continue

        pair_counts[("duration", resource)] += n

        sum_x["duration"] += d.sum()
        sum_x_squared["duration"] += (d ** 2).sum()

        sum_x[resource] += r.sum()
        sum_x_squared[resource] += (r ** 2).sum()

        sum_xy[("duration", resource)] += (d * r).sum()

    count += len(chunk)

    if chunk_number % 100 == 0:
        print(
            f"Processed {count:,} batch_instance rows..."
        )


print("\nBATCH INSTANCE PROCESSING COMPLETE")
print("batch_instance rows processed:", f"{count:,}")


# ---------------------------------------------------------------------------
# 3. BATCH INSTANCE SUMMARY
# ---------------------------------------------------------------------------

print("\nBATCH INSTANCE RESOURCE SUMMARY")

for column in analysis_columns:

    n = count - missing_counts[column]

    if n == 0:
        print(f"\n{column}: no valid values")
        continue

    mean = sums[column] / n

    variance = (
        sums_squared[column] / n
        - mean ** 2
    )

    variance = max(variance, 0.0)

    std = np.sqrt(variance)

    print(f"\n{column}")
    print(f"count : {n:,}")
    print(f"mean  : {mean}")
    print(f"std   : {std}")
    print(f"min   : {mins[column]}")
    print(f"max   : {maxs[column]}")
    print(f"missing: {missing_counts[column]:,}")


# ---------------------------------------------------------------------------
# 4. DURATION ↔ RESOURCE CORRELATIONS
# ---------------------------------------------------------------------------

print("\nDURATION ↔ RESOURCE CORRELATIONS")

for resource in resource_columns:

    key = ("duration", resource)

    n = pair_counts[key]

    if n < 2:
        print(f"{resource}: insufficient data")
        continue

    sx = sum_x["duration"]
    sy = sum_x[resource]

    sxx = sum_x_squared["duration"]
    syy = sum_x_squared[resource]

    sxy = sum_xy[key]

    numerator = (
        n * sxy
        - sx * sy
    )

    denominator = np.sqrt(
        (n * sxx - sx ** 2)
        *
        (n * syy - sy ** 2)
    )

    if denominator == 0:
        correlation = np.nan
    else:
        correlation = numerator / denominator

    print(
        f"duration ↔ {resource}: "
        f"{correlation}"
    )


print("\nANALYSIS COMPLETE")