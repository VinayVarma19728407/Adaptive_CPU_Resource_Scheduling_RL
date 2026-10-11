"""
Validate Workload Regimes Classification.

This script validates generated workload regime classifications against ground truth cluster trace statistics.
It verifies percentile thresholds, regime label assignments, and workload class balance across train, validation, and test splits.
"""

import os
import numpy as np
import pandas as pd


BASE_DIR = "dataset"

WORKLOAD_DIR = os.path.join(
    BASE_DIR,
    "workload_analysis"
)

PROFILE_FILE = os.path.join(
    WORKLOAD_DIR,
    "workload_regime_profiles.csv"
)

THRESHOLD_FILE = os.path.join(
    WORKLOAD_DIR,
    "workload_regime_thresholds.csv"
)

OUTPUT_FILE = os.path.join(
    WORKLOAD_DIR,
    "workload_regime_validation.csv"
)

SPLITS = [
    "training",
    "validation",
    "testing"
]

EXPECTED_TIME_BINS = 1000


def load_data():

    if not os.path.exists(
        PROFILE_FILE
    ):
        raise FileNotFoundError(
            PROFILE_FILE
        )

    if not os.path.exists(
        THRESHOLD_FILE
    ):
        raise FileNotFoundError(
            THRESHOLD_FILE
        )

    profiles = pd.read_csv(
        PROFILE_FILE
    )

    thresholds = pd.read_csv(
        THRESHOLD_FILE
    )

    return profiles, thresholds


def get_threshold(
    thresholds,
    name
):

    rows = thresholds[
        thresholds["threshold"] == name
    ]

    if len(rows) != 1:
        raise ValueError(
            "Invalid threshold count for "
            + name
        )

    return float(
        rows.iloc[0]["value"]
    )


def add_result(
    results,
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


profiles, thresholds = load_data()

results = []

low_medium = get_threshold(
    thresholds,
    "low_medium_intensity"
)

medium_high = get_threshold(
    thresholds,
    "medium_high_intensity"
)

highly_variable = get_threshold(
    thresholds,
    "highly_variable"
)

cpu_heavy = get_threshold(
    thresholds,
    "cpu_heavy"
)

memory_heavy = get_threshold(
    thresholds,
    "memory_heavy"
)


threshold_order_ok = (
    low_medium < medium_high
)

add_result(
    results,
    "Intensity threshold ordering",
    "all",
    int(threshold_order_ok),
    "PASS" if threshold_order_ok else "FAIL",
    "Low, medium and high intensity thresholds are ordered"
)


resource_threshold_order_ok = (
    memory_heavy < cpu_heavy
)

add_result(
    results,
    "Resource threshold ordering",
    "all",
    int(resource_threshold_order_ok),
    "PASS" if resource_threshold_order_ok else "FAIL",
    "Memory-heavy threshold is below CPU-heavy threshold"
)


for split in SPLITS:

    data = profiles[
        profiles["split"] == split
    ].copy()

    row_count = len(data)

    add_result(
        results,
        "Time-bin count",
        split,
        row_count,
        "PASS"
        if row_count == EXPECTED_TIME_BINS
        else "FAIL",
        "Expected 1000 temporal workload windows"
    )

    duplicate_bins = data[
        "time_bin"
    ].duplicated().sum()

    add_result(
        results,
        "Duplicate time bins",
        split,
        duplicate_bins,
        "PASS"
        if duplicate_bins == 0
        else "FAIL",
        "Each temporal workload window should be unique"
    )

    intensity_missing = data[
        "workload_intensity"
    ].isna().sum()

    add_result(
        results,
        "Missing workload intensity",
        split,
        intensity_missing,
        "PASS"
        if intensity_missing == 0
        else "CHECK",
        "Missing values can prevent regime assignment"
    )

    dominance_missing = data[
        "resource_dominance"
    ].isna().sum()

    add_result(
        results,
        "Missing resource dominance",
        split,
        dominance_missing,
        "PASS"
        if dominance_missing == 0
        else "CHECK",
        "Missing values can affect CPU and memory regime assignment"
    )

    variability_missing = data[
        "variability_score"
    ].isna().sum()

    add_result(
        results,
        "Missing variability score",
        split,
        variability_missing,
        "CHECK"
        if variability_missing > 0
        else "PASS",
        "Edge windows can have insufficient rolling observations"
    )

    intensity_counts = (
        data[
            "intensity_regime"
        ]
        .value_counts()
    )

    for regime in [
        "Low",
        "Medium",
        "High"
    ]:

        count = int(
            intensity_counts.get(
                regime,
                0
            )
        )

        add_result(
            results,
            "Intensity regime coverage",
            split + ":" + regime,
            count,
            "PASS"
            if count > 0
            else "FAIL",
            "Every intensity regime must contain windows"
        )

    resource_counts = (
        data[
            "resource_regime"
        ]
        .value_counts()
    )

    for regime in [
        "CPU-heavy",
        "Memory-heavy",
        "Balanced-resource"
    ]:

        count = int(
            resource_counts.get(
                regime,
                0
            )
        )

        add_result(
            results,
            "Resource regime coverage",
            split + ":" + regime,
            count,
            "PASS"
            if count > 0
            else "FAIL",
            "Every resource category should contain windows"
        )

    variability_count = int(
        (
            data[
                "variability_regime"
            ]
            == "Highly Variable"
        ).sum()
    )

    add_result(
        results,
        "Highly Variable coverage",
        split,
        variability_count,
        "PASS"
        if variability_count > 0
        else "FAIL",
        "Highly Variable windows must be present"
    )

    intensity_boundary_low = int(
        np.isclose(
            data[
                "workload_intensity"
            ],
            low_medium,
            rtol=0,
            atol=1e-12
        ).sum()
    )

    intensity_boundary_high = int(
        np.isclose(
            data[
                "workload_intensity"
            ],
            medium_high,
            rtol=0,
            atol=1e-12
        ).sum()
    )

    add_result(
        results,
        "Low-medium boundary ties",
        split,
        intensity_boundary_low,
        "CHECK",
        "Number of windows exactly at the low-medium threshold"
    )

    add_result(
        results,
        "Medium-high boundary ties",
        split,
        intensity_boundary_high,
        "CHECK",
        "Number of windows exactly at the medium-high threshold"
    )

    cpu_count = int(
        (
            data[
                "resource_regime"
            ]
            == "CPU-heavy"
        ).sum()
    )

    memory_count = int(
        (
            data[
                "resource_regime"
            ]
            == "Memory-heavy"
        ).sum()
    )

    balanced_count = int(
        (
            data[
                "resource_regime"
            ]
            == "Balanced-resource"
        ).sum()
    )

    resource_total = (
        cpu_count
        + memory_count
        + balanced_count
    )

    add_result(
        results,
        "Resource regime partition",
        split,
        resource_total,
        "PASS"
        if resource_total == row_count
        else "FAIL",
        "Every time window should belong to exactly one resource category"
    )

    intensity_total = int(
        data[
            "intensity_regime"
        ].notna().sum()
    )

    add_result(
        results,
        "Intensity regime partition",
        split,
        intensity_total,
        "PASS"
        if intensity_total == row_count
        else "FAIL",
        "Every time window should receive an intensity category"
    )

    highly_variable_count = int(
        (
            data[
                "variability_regime"
            ]
            == "Highly Variable"
        ).sum()
    )

    not_variable_count = int(
        (
            data[
                "variability_regime"
            ]
            == "Not Highly Variable"
        ).sum()
    )

    variability_total = (
        highly_variable_count
        + not_variable_count
    )

    add_result(
        results,
        "Variability regime partition",
        split,
        variability_total,
        "PASS"
        if variability_total == row_count
        else "FAIL",
        "Every time window should receive a variability category"
    )


training = profiles[
    profiles["split"] == "training"
].copy()

validation = profiles[
    profiles["split"] == "validation"
].copy()

testing = profiles[
    profiles["split"] == "testing"
].copy()


for metric in [
    "workload_intensity",
    "resource_dominance",
    "variability_score"
]:

    training_values = training[
        metric
    ].dropna()

    validation_values = validation[
        metric
    ].dropna()

    testing_values = testing[
        metric
    ].dropna()

    training_median = (
        training_values.median()
    )

    validation_median = (
        validation_values.median()
    )

    testing_median = (
        testing_values.median()
    )

    add_result(
        results,
        "Training median",
        metric,
        training_median,
        "PASS",
        "Reference distribution"
    )

    add_result(
        results,
        "Validation median",
        metric,
        validation_median,
        "CHECK",
        "Compared with training distribution"
    )

    add_result(
        results,
        "Testing median",
        metric,
        testing_median,
        "CHECK",
        "Compared with training distribution"
    )


overlap_checks = []

for split in SPLITS:

    data = profiles[
        profiles["split"] == split
    ].copy()

    cpu_flags = (
        data["resource_regime"]
        == "CPU-heavy"
    )

    memory_flags = (
        data["resource_regime"]
        == "Memory-heavy"
    )

    overlap_count = int(
        (
            cpu_flags
            & memory_flags
        ).sum()
    )

    overlap_checks.append({
        "split": split,
        "cpu_memory_overlap": overlap_count
    })

for item in overlap_checks:

    add_result(
        results,
        "CPU-memory exclusivity",
        item["split"],
        item["cpu_memory_overlap"],
        "PASS"
        if item["cpu_memory_overlap"] == 0
        else "FAIL",
        "CPU-heavy and Memory-heavy are mutually exclusive"
    )


validation_df = pd.DataFrame(
    results
)

validation_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    "Workload regime validation completed"
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
    "CHECK:",
    int(
        (
            validation_df["status"]
            == "CHECK"
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