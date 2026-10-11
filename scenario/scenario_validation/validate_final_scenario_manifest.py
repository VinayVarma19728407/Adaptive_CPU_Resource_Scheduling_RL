"""
Validate Final Workload Scenario Manifest.

This script validates generated scenario manifest files for RL scheduling environment evaluation. It checks
scenario timestamp boundaries, task counts, machine resource availability, and contract compliance for benchmark scenarios.
"""

import os
import pandas as pd


BASE_DIR = "dataset"

WORKLOAD_DIR = os.path.join(
    BASE_DIR,
    "workload_analysis"
)

MANIFEST_FILE = os.path.join(
    WORKLOAD_DIR,
    "workload_scenario_manifest_final.csv"
)

OUTPUT_FILE = os.path.join(
    WORKLOAD_DIR,
    "workload_scenario_manifest_validation.csv"
)

SPLITS = [
    "training",
    "validation",
    "testing"
]

REQUIRED_COLUMNS = [
    "scenario_id",
    "split",
    "scenario_type",
    "scenario_axis",
    "regime",
    "time_bin",
    "time_start",
    "time_end",
    "workload_intensity",
    "resource_dominance",
    "variability_score",
    "intensity_regime",
    "resource_regime",
    "variability_regime",
    "selection_seed"
]


manifest = pd.read_csv(
    MANIFEST_FILE
)

results = []


def add_result(
    check,
    split,
    value,
    status,
    interpretation
):

    results.append({
        "check": check,
        "split": split,
        "value": value,
        "status": status,
        "interpretation": interpretation
    })


missing_columns = [
    column
    for column in REQUIRED_COLUMNS
    if column not in manifest.columns
]

add_result(
    "Required columns",
    "all",
    len(missing_columns),
    "PASS" if len(missing_columns) == 0 else "FAIL",
    "All required scenario manifest columns are present"
)


duplicate_ids = int(
    manifest[
        "scenario_id"
    ].duplicated().sum()
)

add_result(
    "Duplicate scenario IDs",
    "all",
    duplicate_ids,
    "PASS" if duplicate_ids == 0 else "FAIL",
    "Scenario IDs must be unique"
)


null_ids = int(
    manifest[
        "scenario_id"
    ].isna().sum()
)

add_result(
    "Missing scenario IDs",
    "all",
    null_ids,
    "PASS" if null_ids == 0 else "FAIL",
    "Scenario IDs must not be missing"
)


for split in SPLITS:

    data = manifest[
        manifest["split"] == split
    ].copy()

    add_result(
        "Scenario row count",
        split,
        len(data),
        "PASS" if len(data) > 0 else "FAIL",
        "Each split must contain scenarios"
    )

    duplicate_time_bins = int(
        data[
            [
                "scenario_type",
                "scenario_axis",
                "regime",
                "time_bin"
            ]
        ].duplicated().sum()
    )

    add_result(
        "Duplicate scenario selections",
        split,
        duplicate_time_bins,
        "PASS" if duplicate_time_bins == 0 else "FAIL",
        "Identical scenario selections should not be duplicated"
    )

    invalid_time_order = int(
        (
            data["time_end"]
            <= data["time_start"]
        ).sum()
    )

    add_result(
        "Invalid time interval",
        split,
        invalid_time_order,
        "PASS" if invalid_time_order == 0 else "FAIL",
        "Every scenario window must have positive duration"
    )

    invalid_time_bins = int(
        (
            data["time_bin"] < 0
        ).sum()
    )

    add_result(
        "Invalid time bin",
        split,
        invalid_time_bins,
        "PASS" if invalid_time_bins == 0 else "FAIL",
        "Time bins must be nonnegative"
    )

    natural = data[
        data["scenario_type"]
        == "natural-frequency"
    ]

    natural_count = len(
        natural
    )

    add_result(
        "Natural-frequency coverage",
        split,
        natural_count,
        "PASS" if natural_count == 1000 else "FAIL",
        "Natural-frequency scenarios should contain all 1000 temporal windows"
    )

    natural_duplicate_bins = int(
        natural[
            "time_bin"
        ].duplicated().sum()
    )

    add_result(
        "Natural-frequency duplicate bins",
        split,
        natural_duplicate_bins,
        "PASS"
        if natural_duplicate_bins == 0
        else "FAIL",
        "Each natural-frequency time window should appear once"
    )

    balanced = data[
        data["scenario_type"]
        == "balanced"
    ]

    add_result(
        "Balanced scenario coverage",
        split,
        len(balanced),
        "PASS" if len(balanced) > 0 else "FAIL",
        "Balanced scenarios must exist"
    )

    invalid_seed = int(
        (
            data["selection_seed"]
            != 20261004
        ).sum()
    )

    add_result(
        "Fixed selection seed",
        split,
        invalid_seed,
        "PASS" if invalid_seed == 0 else "FAIL",
        "All scenario selections use the fixed reproducibility seed"
    )


scenario_axes = set(
    manifest[
        "scenario_axis"
    ].dropna()
    .unique()
)

allowed_axes = {
    "all",
    "intensity",
    "resource",
    "variability"
}

unexpected_axes = (
    scenario_axes
    - allowed_axes
)

add_result(
    "Scenario axis validity",
    "all",
    len(unexpected_axes),
    "PASS" if len(unexpected_axes) == 0 else "FAIL",
    "Scenario axes use the defined categories"
)


scenario_types = set(
    manifest[
        "scenario_type"
    ].dropna()
    .unique()
)

allowed_types = {
    "natural-frequency",
    "balanced"
}

unexpected_types = (
    scenario_types
    - allowed_types
)

add_result(
    "Scenario type validity",
    "all",
    len(unexpected_types),
    "PASS" if len(unexpected_types) == 0 else "FAIL",
    "Scenario types use the defined categories"
)


validation_df = pd.DataFrame(
    results
)

validation_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    "Final scenario manifest validation completed"
)

print(
    "Output file:",
    OUTPUT_FILE
)

print(
    "Total checks:",
    len(validation_df)
)

print(
    "PASS:",
    int(
        (
            validation_df["status"]
            == "PASS"
        ).sum()
    )
)

print(
    "FAIL:",
    int(
        (
            validation_df["status"]
            == "FAIL"
        ).sum()
    )
)

print()

print(
    validation_df[
        validation_df["status"] == "FAIL"
    ].to_string(
        index=False
    )
)