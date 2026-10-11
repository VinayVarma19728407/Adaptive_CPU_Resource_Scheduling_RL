"""
Integrated Data Validation Pipeline.

This script runs an end-to-end data validation pipeline verifying schema formats, data types, value ranges,
and integrity constraints across preprocessed batch trace tables (batch_task, batch_instance, machine_meta, machine_usage).
"""

import os
import gc
import pandas as pd


BASE_DIR = "dataset"

BATCH_TASK_FILE = os.path.join(
    BASE_DIR,
    "preprocessed",
    "batch_task.csv"
)

MACHINE_META_FILE = os.path.join(
    BASE_DIR,
    "preprocessed",
    "machine_meta.csv"
)

MACHINE_USAGE_FILE = os.path.join(
    BASE_DIR,
    "preprocessed",
    "machine_usage.csv"
)

TRAIN_FILE = os.path.join(
    BASE_DIR,
    "splits",
    "batch_instance_train.csv"
)

VALIDATION_FILE = os.path.join(
    BASE_DIR,
    "splits",
    "batch_instance_validation.csv"
)

TEST_FILE = os.path.join(
    BASE_DIR,
    "splits",
    "batch_instance_test.csv"
)

DATA_DICTIONARY_FILE = os.path.join(
    BASE_DIR,
    "data_dictionary.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "integrated_validation"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "integrated_data_validation.csv"
)

CHUNK_SIZE = 250000

os.makedirs(OUTPUT_DIR, exist_ok=True)


EXPECTED_BATCH_INSTANCE_COLUMNS = [
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


EXPECTED_BATCH_TASK_COLUMNS = [
    "task_name",
    "instance_num",
    "job_name",
    "task_type",
    "status",
    "start_time",
    "end_time",
    "plan_cpu",
    "plan_mem",
    "task_name_id",
    "job_name_id",
    "task_type_id",
    "status_id",
    "duration"
]


EXPECTED_MACHINE_META_COLUMNS = [
    "machine_id",
    "time_stamp",
    "failure_domain_1",
    "failure_domain_2",
    "cpu_num",
    "mem_size",
    "status"
]


EXPECTED_MACHINE_USAGE_COLUMNS = [
    "machine_id",
    "time_stamp",
    "cpu_util_percent",
    "mem_util_percent",
    "net_in",
    "net_out",
    "disk_io_percent",
    "machine_id_id"
]


results = []


def add_result(check, dataset, status, details):
    results.append({
        "check": check,
        "dataset": dataset,
        "status": status,
        "details": details
    })


def check_file_schema(file_path, dataset, expected_columns):
    if not os.path.exists(file_path):
        add_result(
            "File existence",
            dataset,
            "FAIL",
            f"File not found: {file_path}"
        )
        return

    header = pd.read_csv(
        file_path,
        nrows=0
    )

    actual_columns = list(header.columns)

    if actual_columns == expected_columns:
        add_result(
            "Schema",
            dataset,
            "PASS",
            f"{len(actual_columns)} columns match expected schema."
        )
    else:
        missing = [
            col for col in expected_columns
            if col not in actual_columns
        ]

        extra = [
            col for col in actual_columns
            if col not in expected_columns
        ]

        add_result(
            "Schema",
            dataset,
            "FAIL",
            f"Missing columns: {missing}; Extra columns: {extra}"
        )


def validate_dictionary():
    if not os.path.exists(DATA_DICTIONARY_FILE):
        add_result(
            "Data dictionary existence",
            "data_dictionary",
            "FAIL",
            "Data dictionary file not found."
        )
        return

    dictionary = pd.read_csv(DATA_DICTIONARY_FILE)

    required_dictionary_columns = [
        "dataset",
        "column_name",
        "data_type",
        "role",
        "description",
        "preprocessing_status",
        "missing_allowed",
        "downstream_use",
        "notes"
    ]

    missing = [
        col
        for col in required_dictionary_columns
        if col not in dictionary.columns
    ]

    if len(missing) == 0:
        add_result(
            "Data dictionary schema",
            "data_dictionary",
            "PASS",
            f"{len(dictionary)} documented columns with all required fields."
        )
    else:
        add_result(
            "Data dictionary schema",
            "data_dictionary",
            "FAIL",
            f"Missing columns: {missing}"
        )

    expected_dictionary_counts = {
        "batch_task": 14,
        "machine_meta": 7,
        "machine_usage": 8,
        "batch_instance": 15
    }

    actual_counts = dictionary.groupby("dataset").size().to_dict()

    if actual_counts == expected_dictionary_counts:
        add_result(
            "Data dictionary coverage",
            "data_dictionary",
            "PASS",
            str(actual_counts)
        )
    else:
        add_result(
            "Data dictionary coverage",
            "data_dictionary",
            "FAIL",
            f"Expected {expected_dictionary_counts}; got {actual_counts}"
        )


def validate_basic_file(file_path, dataset):
    if not os.path.exists(file_path):
        return

    header = pd.read_csv(
        file_path,
        nrows=0
    )

    rows = 0

    for chunk in pd.read_csv(
        file_path,
        chunksize=CHUNK_SIZE,
        usecols=list(header.columns)
    ):
        rows += len(chunk)
        del chunk
        gc.collect()

    add_result(
        "Row count available",
        dataset,
        "PASS",
        f"Rows: {rows:,}"
    )


def validate_batch_instance(file_path, dataset):
    if not os.path.exists(file_path):
        return

    usecols = [
        "task_name",
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

    rows = 0
    missing_critical = 0
    invalid_time = 0
    negative_duration = 0
    invalid_sequence = 0
    duration_mismatch = 0
    resource_negative = 0

    for chunk in pd.read_csv(
        file_path,
        usecols=usecols,
        chunksize=CHUNK_SIZE
    ):
        rows += len(chunk)

        critical_columns = [
            "task_name",
            "status",
            "start_time",
            "end_time",
            "machine_id",
            "seq_no",
            "total_seq_no",
            "duration"
        ]

        missing_critical += int(
            chunk[critical_columns].isna().sum().sum()
        )

        invalid_time += int(
            (chunk["end_time"] < chunk["start_time"]).sum()
        )

        negative_duration += int(
            (chunk["duration"] < 0).sum()
        )

        invalid_sequence += int(
            (chunk["seq_no"] > chunk["total_seq_no"]).sum()
        )

        calculated_duration = (
            chunk["end_time"] - chunk["start_time"]
        )

        duration_mismatch += int(
            (
                calculated_duration != chunk["duration"]
            ).sum()
        )

        resource_columns = [
            "cpu_avg",
            "cpu_max",
            "mem_avg",
            "mem_max"
        ]

        resource_negative += int(
            (
                chunk[resource_columns] < 0
            ).sum().sum()
        )

        del calculated_duration
        del chunk
        gc.collect()

    if missing_critical == 0:
        status = "PASS"
    else:
        status = "FAIL"

    add_result(
        "Critical field completeness",
        dataset,
        status,
        f"Missing critical values: {missing_critical:,}"
    )

    if invalid_time == 0:
        status = "PASS"
    else:
        status = "FAIL"

    add_result(
        "Time ordering",
        dataset,
        status,
        f"Invalid end_time < start_time rows: {invalid_time:,}"
    )

    if negative_duration == 0:
        status = "PASS"
    else:
        status = "FAIL"

    add_result(
        "Negative duration",
        dataset,
        status,
        f"Negative duration rows: {negative_duration:,}"
    )

    if invalid_sequence == 0:
        status = "PASS"
    else:
        status = "FAIL"

    add_result(
        "Sequence consistency",
        dataset,
        status,
        f"seq_no > total_seq_no rows: {invalid_sequence:,}"
    )

    if duration_mismatch == 0:
        status = "PASS"
    else:
        status = "FAIL"

    add_result(
        "Duration consistency",
        dataset,
        status,
        f"Duration mismatches: {duration_mismatch:,}"
    )

    if resource_negative == 0:
        status = "PASS"
    else:
        status = "FAIL"

    add_result(
        "Negative resource values",
        dataset,
        status,
        f"Negative resource values: {resource_negative:,}"
    )

    add_result(
        "Batch instance row count",
        dataset,
        "PASS",
        f"Rows checked: {rows:,}"
    )


def validate_resource_missingness(file_path, dataset):
    if not os.path.exists(file_path):
        return

    resource_columns = [
        "cpu_avg",
        "cpu_max",
        "mem_avg",
        "mem_max"
    ]

    missing_counts = {
        column: 0
        for column in resource_columns
    }

    for chunk in pd.read_csv(
        file_path,
        usecols=resource_columns,
        chunksize=CHUNK_SIZE
    ):
        for column in resource_columns:
            missing_counts[column] += int(
                chunk[column].isna().sum()
            )

        del chunk
        gc.collect()

    add_result(
        "Resource missingness preservation",
        dataset,
        "PASS",
        str(missing_counts)
    )


def validate_status_values():
    expected_status_values = {
        "batch_instance": {0, 1, 2, 3, 4, 5}
    }

    for dataset, allowed in expected_status_values.items():

        file_path = {
            "batch_instance": TRAIN_FILE
        }[dataset]

        if not os.path.exists(file_path):
            continue

        observed = set()

        for chunk in pd.read_csv(
            file_path,
            usecols=["status"],
            chunksize=CHUNK_SIZE
        ):
            observed.update(
                chunk["status"].dropna().unique().tolist()
            )

            del chunk
            gc.collect()

        invalid = observed - allowed

        if len(invalid) == 0:
            add_result(
                "Status value validation",
                dataset,
                "PASS",
                f"Observed values: {sorted(observed)}"
            )
        else:
            add_result(
                "Status value validation",
                dataset,
                "FAIL",
                f"Invalid values: {sorted(invalid)}"
            )


def validate_split_counts():
    expected = {
        "training": 808882311,
        "validation": 269627437,
        "testing": 269627438
    }

    files = {
        "training": TRAIN_FILE,
        "validation": VALIDATION_FILE,
        "testing": TEST_FILE
    }

    for name, expected_rows in expected.items():

        file_path = files[name]

        if not os.path.exists(file_path):
            add_result(
                "Split row count",
                name,
                "FAIL",
                "Split file not found."
            )
            continue

        rows = 0

        header = pd.read_csv(
            file_path,
            nrows=0
        )

        for chunk in pd.read_csv(
            file_path,
            usecols=[header.columns[0]],
            chunksize=CHUNK_SIZE
        ):
            rows += len(chunk)

            del chunk
            gc.collect()

        if rows == expected_rows:
            status = "PASS"
        else:
            status = "FAIL"

        add_result(
            "Split row count",
            name,
            status,
            f"Expected {expected_rows:,}; observed {rows:,}"
        )


def validate_no_obvious_leakage():
    add_result(
        "Preprocessing leakage",
        "all datasets",
        "DOCUMENTED",
        "No new normalization, clipping, or resource imputation is performed by this validation script."
    )

    add_result(
        "Preprocessing leakage",
        "batch_instance",
        "DOCUMENTED",
        "Resource values are preserved as prepared. Normalization remains a later RL state-construction operation."
    )

    add_result(
        "Relationship exceptions",
        "all datasets",
        "DOCUMENTED",
        "Previously explained task-name and machine 2739 exceptions are not reclassified as data corruption."
    )

    add_result(
        "Task to job relationship",
        "all datasets",
        "DOCUMENTED",
        "Task-name to job-name is not used as a strict one-to-one integrity constraint."
    )


print("Starting integrated data validation")


check_file_schema(
    BATCH_TASK_FILE,
    "batch_task",
    EXPECTED_BATCH_TASK_COLUMNS
)

check_file_schema(
    MACHINE_META_FILE,
    "machine_meta",
    EXPECTED_MACHINE_META_COLUMNS
)

check_file_schema(
    MACHINE_USAGE_FILE,
    "machine_usage",
    EXPECTED_MACHINE_USAGE_COLUMNS
)

check_file_schema(
    TRAIN_FILE,
    "batch_instance_training",
    EXPECTED_BATCH_INSTANCE_COLUMNS
)

check_file_schema(
    VALIDATION_FILE,
    "batch_instance_validation",
    EXPECTED_BATCH_INSTANCE_COLUMNS
)

check_file_schema(
    TEST_FILE,
    "batch_instance_testing",
    EXPECTED_BATCH_INSTANCE_COLUMNS
)


validate_dictionary()


validate_split_counts()


validate_batch_instance(
    TRAIN_FILE,
    "batch_instance_training"
)

validate_batch_instance(
    VALIDATION_FILE,
    "batch_instance_validation"
)

validate_batch_instance(
    TEST_FILE,
    "batch_instance_testing"
)


validate_resource_missingness(
    TRAIN_FILE,
    "batch_instance_training"
)

validate_resource_missingness(
    VALIDATION_FILE,
    "batch_instance_validation"
)

validate_resource_missingness(
    TEST_FILE,
    "batch_instance_testing"
)


validate_status_values()


validate_no_obvious_leakage()


results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)


print("Integrated data validation completed")
print("Output file:", OUTPUT_FILE)
print("Total checks:", len(results_df))
print(
    "PASS:",
    int((results_df["status"] == "PASS").sum())
)
print(
    "FAIL:",
    int((results_df["status"] == "FAIL").sum())
)
print(
    "DOCUMENTED:",
    int((results_df["status"] == "DOCUMENTED").sum())
)

if (results_df["status"] == "FAIL").any():
    print("Validation contains failures")
else:
    print("No validation failures detected")