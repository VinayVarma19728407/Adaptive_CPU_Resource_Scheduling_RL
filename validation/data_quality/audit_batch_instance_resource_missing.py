"""
Audit Missing Resource Values in Batch Instance Traces.

This script inspects missing/NaN value patterns across CPU and memory columns in batch_instance trace chunks.
It quantifies missing record counts for planned, mean, and max resource fields, generating audit reports
to evaluate missing data handling and imputation strategies.
"""

import gc
import pandas as pd
from pathlib import Path

BASE = Path("dataset")

SPLITS = {
    "training": BASE / "preprocessed" / "batch_instance_training.csv",
    "validation": BASE / "preprocessed" / "batch_instance_validation.csv",
    "testing": BASE / "preprocessed" / "batch_instance_testing.csv"
}

CHUNK_SIZE = 250_000

COLUMNS = [
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
    "duration"
]

RESOURCE_COLUMNS = [
    "cpu_avg",
    "cpu_max",
    "mem_avg",
    "mem_max"
]

results = []


for split_name, split_path in SPLITS.items():

    print()
    print(f"Checking {split_name}")

    total_rows = 0

    missing_counts = {
        column: 0
        for column in COLUMNS
    }

    minimums = {
        column: None
        for column in RESOURCE_COLUMNS
    }

    maximums = {
        column: None
        for column in RESOURCE_COLUMNS
    }

    below_zero = {
        column: 0
        for column in RESOURCE_COLUMNS
    }

    above_100 = {
        column: 0
        for column in RESOURCE_COLUMNS
    }

    chunk_number = 0

    for chunk in pd.read_csv(
        split_path,
        usecols=COLUMNS,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        chunk_number += 1
        total_rows += len(chunk)

        for column in COLUMNS:

            missing_counts[column] += int(
                chunk[column].isna().sum()
            )

        for column in RESOURCE_COLUMNS:

            series = chunk[column]

            current_min = series.min(
                skipna=True
            )

            current_max = series.max(
                skipna=True
            )

            if pd.notna(current_min):

                if minimums[column] is None:
                    minimums[column] = current_min

                elif current_min < minimums[column]:
                    minimums[column] = current_min

            if pd.notna(current_max):

                if maximums[column] is None:
                    maximums[column] = current_max

                elif current_max > maximums[column]:
                    maximums[column] = current_max

            below_zero[column] += int(
                (series < 0).sum()
            )

            above_100[column] += int(
                (series > 100).sum()
            )

        if chunk_number % 100 == 0:

            print(
                f"Chunks checked: {chunk_number:,} | "
                f"Rows checked: {total_rows:,}"
            )

        del chunk
        gc.collect()

    print()
    print(f"{split_name} missing values")

    for column in COLUMNS:

        print(
            f"{column}: "
            f"{missing_counts[column]:,}"
        )

    print()
    print(f"{split_name} resource ranges")

    for column in RESOURCE_COLUMNS:

        print(
            f"{column}: "
            f"min {minimums[column]} | "
            f"max {maximums[column]}"
        )

    print()
    print(f"{split_name} resource range counts")

    for column in RESOURCE_COLUMNS:

        print(
            f"{column} below zero: "
            f"{below_zero[column]:,}"
        )

        print(
            f"{column} above 100: "
            f"{above_100[column]:,}"
        )

    results.append({
        "split": split_name,
        "rows": total_rows,
        "cpu_avg_missing": missing_counts["cpu_avg"],
        "cpu_max_missing": missing_counts["cpu_max"],
        "mem_avg_missing": missing_counts["mem_avg"],
        "mem_max_missing": missing_counts["mem_max"],
        "cpu_avg_min": minimums["cpu_avg"],
        "cpu_avg_max": maximums["cpu_avg"],
        "cpu_max_min": minimums["cpu_max"],
        "cpu_max_max": maximums["cpu_max"],
        "mem_avg_min": minimums["mem_avg"],
        "mem_avg_max": maximums["mem_avg"],
        "mem_max_min": minimums["mem_max"],
        "cpu_avg_below_zero": below_zero["cpu_avg"],
        "cpu_avg_above_100": above_100["cpu_avg"],
        "cpu_max_below_zero": below_zero["cpu_max"],
        "cpu_max_above_100": above_100["cpu_max"],
        "mem_avg_below_zero": below_zero["mem_avg"],
        "mem_avg_above_100": above_100["mem_avg"],
        "mem_max_below_zero": below_zero["mem_max"],
        "mem_max_above_100": above_100["mem_max"]
    })

    del missing_counts
    del minimums
    del maximums
    del below_zero
    del above_100

    gc.collect()


output = pd.DataFrame(results)

output_file = (
    BASE /
    "relationship_audit" /
    "batch_instance_resource_missing_audit.csv"
)

output.to_csv(
    output_file,
    index=False
)

del output
del results

gc.collect()

print()
print("Resource and missing value audit complete")

print(
    f"Output file: {output_file}"
)