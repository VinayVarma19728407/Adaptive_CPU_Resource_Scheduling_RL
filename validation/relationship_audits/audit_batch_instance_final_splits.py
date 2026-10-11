"""
Audit Batch Instance Final Dataset Splits.

This script audits the integrity, record counts, temporal bounds, and schema consistency of the finalized
train, validation, and test split files for batch_instance. It verifies non-overlapping temporal windows,
checks for missing values, and validates split sizes for dataset validation.
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

OUTPUT_DIR = BASE / "relationship_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 250_000

EXPECTED_COLUMNS = [
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
    "duration"
]

EXPECTED_ROWS = {
    "training": 808_882_311,
    "validation": 269_627_437,
    "testing": 269_627_438
}


results = []

split_start_times = {}
split_end_times = {}


for split_name, split_path in SPLITS.items():

    print()
    print(f"Checking {split_name}")

    header = pd.read_csv(
        split_path,
        nrows=0,
        low_memory=False
    )

    actual_columns = list(header.columns)

    schema_ok = actual_columns == EXPECTED_COLUMNS

    print(
        f"Schema correct: {schema_ok}"
    )

    if not schema_ok:

        print(
            f"Expected columns: {EXPECTED_COLUMNS}"
        )

        print(
            f"Actual columns: {actual_columns}"
        )

        raise ValueError(
            f"Column schema mismatch in {split_name}"
        )

    del header
    gc.collect()

    total_rows = 0

    missing_values = {
        column: 0
        for column in EXPECTED_COLUMNS
    }

    status_counts = {}
    task_type_counts = {}

    negative_duration = 0
    invalid_time_order = 0
    invalid_sequence = 0

    zero_duration = 0
    positive_duration = 0

    resource_invalid = {
        "cpu_avg_below_zero": 0,
        "cpu_avg_above_100": 0,
        "cpu_max_below_zero": 0,
        "cpu_max_above_100": 0,
        "mem_avg_below_zero": 0,
        "mem_max_above_100": 0
    }

    split_min_start = None
    split_max_start = None
    split_min_end = None
    split_max_end = None

    chunk_number = 0

    for chunk in pd.read_csv(
        split_path,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        chunk_number += 1
        total_rows += len(chunk)

        for column in EXPECTED_COLUMNS:

            missing_values[column] += int(
                chunk[column].isna().sum()
            )

        status_values = chunk["status"].value_counts()

        for value, count in status_values.items():

            key = str(value)

            status_counts[key] = (
                status_counts.get(key, 0) + int(count)
            )

        task_type_values = chunk["task_type"].value_counts()

        for value, count in task_type_values.items():

            key = str(value)

            task_type_counts[key] = (
                task_type_counts.get(key, 0) + int(count)
            )

        negative_duration += int(
            (chunk["duration"] < 0).sum()
        )

        invalid_time_order += int(
            (chunk["start_time"] > chunk["end_time"]).sum()
        )

        invalid_sequence += int(
            (chunk["seq_no"] > chunk["total_seq_no"]).sum()
        )

        zero_duration += int(
            (chunk["duration"] == 0).sum()
        )

        positive_duration += int(
            (chunk["duration"] > 0).sum()
        )

        resource_invalid["cpu_avg_below_zero"] += int(
            (chunk["cpu_avg"] < 0).sum()
        )

        resource_invalid["cpu_avg_above_100"] += int(
            (chunk["cpu_avg"] > 100).sum()
        )

        resource_invalid["cpu_max_below_zero"] += int(
            (chunk["cpu_max"] < 0).sum()
        )

        resource_invalid["cpu_max_above_100"] += int(
            (chunk["cpu_max"] > 100).sum()
        )

        resource_invalid["mem_avg_below_zero"] += int(
            (chunk["mem_avg"] < 0).sum()
        )

        resource_invalid["mem_max_above_100"] += int(
            (chunk["mem_max"] > 100).sum()
        )

        current_min_start = chunk["start_time"].min()
        current_max_start = chunk["start_time"].max()

        current_min_end = chunk["end_time"].min()
        current_max_end = chunk["end_time"].max()

        if split_min_start is None:
            split_min_start = current_min_start
        elif current_min_start < split_min_start:
            split_min_start = current_min_start

        if split_max_start is None:
            split_max_start = current_max_start
        elif current_max_start > split_max_start:
            split_max_start = current_max_start

        if split_min_end is None:
            split_min_end = current_min_end
        elif current_min_end < split_min_end:
            split_min_end = current_min_end

        if split_max_end is None:
            split_max_end = current_max_end
        elif current_max_end > split_max_end:
            split_max_end = current_max_end

        if chunk_number % 100 == 0:

            print(
                f"{split_name} chunks checked: "
                f"{chunk_number:,} | "
                f"rows checked: {total_rows:,}"
            )

        del status_values
        del task_type_values
        del chunk

        gc.collect()

    expected_rows = EXPECTED_ROWS[split_name]

    row_count_ok = (
        total_rows == expected_rows
    )

    missing_total = sum(
        missing_values.values()
    )

    resource_violation_total = sum(
        resource_invalid.values()
    )

    print()
    print(f"{split_name} results")

    print(
        f"Rows checked: {total_rows:,}"
    )

    print(
        f"Expected rows: {expected_rows:,}"
    )

    print(
        f"Row count correct: {row_count_ok}"
    )

    print(
        f"Missing values: {missing_total:,}"
    )

    print(
        f"Negative durations: {negative_duration:,}"
    )

    print(
        f"Invalid time order: {invalid_time_order:,}"
    )

    print(
        f"Invalid sequence values: {invalid_sequence:,}"
    )

    print(
        f"Zero duration rows: {zero_duration:,}"
    )

    print(
        f"Positive duration rows: {positive_duration:,}"
    )

    print(
        f"Resource range violations: "
        f"{resource_violation_total:,}"
    )

    print(
        f"Start time range: "
        f"{split_min_start} to {split_max_start}"
    )

    print(
        f"End time range: "
        f"{split_min_end} to {split_max_end}"
    )

    results.append({
        "split": split_name,
        "rows": total_rows,
        "expected_rows": expected_rows,
        "row_count_ok": row_count_ok,
        "missing_values": missing_total,
        "negative_duration": negative_duration,
        "invalid_time_order": invalid_time_order,
        "invalid_sequence": invalid_sequence,
        "zero_duration": zero_duration,
        "positive_duration": positive_duration,
        "resource_range_violations": resource_violation_total,
        "min_start_time": split_min_start,
        "max_start_time": split_max_start,
        "min_end_time": split_min_end,
        "max_end_time": split_max_end
    })

    split_start_times[split_name] = split_min_start
    split_end_times[split_name] = split_max_end

    del missing_values
    del status_counts
    del task_type_counts
    del resource_invalid

    gc.collect()


print()
print("Checking chronological split boundaries")

training_end = split_end_times["training"]
validation_start = split_start_times["validation"]

validation_end = split_end_times["validation"]
testing_start = split_start_times["testing"]

training_before_validation = (
    training_end <= validation_start
)

validation_before_testing = (
    validation_end <= testing_start
)

print(
    f"Training ends before validation starts: "
    f"{training_before_validation}"
)

print(
    f"Validation ends before testing starts: "
    f"{validation_before_testing}"
)


summary = pd.DataFrame(results)

summary["training_before_validation"] = (
    training_before_validation
)

summary["validation_before_testing"] = (
    validation_before_testing
)

output_file = (
    OUTPUT_DIR /
    "batch_instance_final_split_validation.csv"
)

summary.to_csv(
    output_file,
    index=False
)

del summary
del results
del split_start_times
del split_end_times

gc.collect()

print()
print("Final split validation complete")

print(
    f"Output file: {output_file}"
)