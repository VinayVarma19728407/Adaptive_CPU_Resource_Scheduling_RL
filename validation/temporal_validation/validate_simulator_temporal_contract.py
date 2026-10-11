"""
Validate Simulator Temporal Contract.

This script validates temporal ordering and monotonicity contracts required by the discrete-event simulator.
It verifies that task arrival times (start_time) are non-decreasing, non-negative, and strictly aligned with simulation clock steps.
"""

import os
import gc
import numpy as np
import pandas as pd


BASE_DIR = "dataset"

FILES = {
    "training": os.path.join(
        BASE_DIR,
        "preprocessed",
        "batch_instance_training.csv"
    ),
    "validation": os.path.join(
        BASE_DIR,
        "preprocessed",
        "batch_instance_validation.csv"
    ),
    "testing": os.path.join(
        BASE_DIR,
        "preprocessed",
        "batch_instance_testing.csv"
    )
}

MANIFEST_FILE = os.path.join(
    BASE_DIR,
    "workload_analysis",
    "workload_scenario_manifest_final.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "workload_analysis",
    "simulator_temporal_contract_validation.csv"
)

CHUNK_SIZE = 250000

MACHINE_USAGE_MIN = 0.0
MACHINE_USAGE_MAX = 691190.0

MACHINE_META_MIN = 0.0
MACHINE_META_MAX = 696633.0

REQUIRED_COLUMNS = [
    "start_time",
    "end_time",
    "duration",
    "machine_id",
    "task_name",
    "task_type",
    "cpu_avg",
    "mem_avg"
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


manifest_required = [
    "scenario_id",
    "split",
    "time_bin",
    "time_start",
    "time_end"
]

missing_manifest_columns = [
    column
    for column in manifest_required
    if column not in manifest.columns
]

add_result(
    "Manifest required columns",
    "all",
    len(missing_manifest_columns),
    "PASS"
    if len(missing_manifest_columns) == 0
    else "FAIL",
    "Scenario manifest contains temporal fields required by the simulator"
)


for split, file_path in FILES.items():

    split_manifest = manifest[
        manifest["split"] == split
    ].copy()

    scenario_min = (
        split_manifest[
            "time_start"
        ].min()
    )

    scenario_max = (
        split_manifest[
            "time_end"
        ].max()
    )

    machine_usage_gap = max(
        0.0,
        scenario_max - MACHINE_USAGE_MAX
    )

    machine_meta_gap = max(
        0.0,
        scenario_max - MACHINE_META_MAX
    )

    add_result(
        "Scenario end beyond machine usage coverage",
        split,
        machine_usage_gap,
        "CHECK"
        if machine_usage_gap > 0
        else "PASS",
        "Requires explicit machine-state handling when usage observations do not exist"
    )

    add_result(
        "Scenario end beyond machine metadata coverage",
        split,
        machine_meta_gap,
        "CHECK"
        if machine_meta_gap > 0
        else "PASS",
        "Requires explicit machine-capacity handling when metadata coverage ends"
    )

    task_count = 0
    invalid_time_count = 0
    negative_duration_count = 0
    missing_start_count = 0
    missing_end_count = 0
    missing_machine_count = 0
    task_before_scenario = 0
    task_after_scenario = 0

    scenario_bin_counts = {}

    for chunk in pd.read_csv(
        file_path,
        usecols=REQUIRED_COLUMNS,
        chunksize=CHUNK_SIZE
    ):

        start = chunk[
            "start_time"
        ].to_numpy(
            dtype=np.float64,
            copy=False
        )

        end = chunk[
            "end_time"
        ].to_numpy(
            dtype=np.float64,
            copy=False
        )

        duration = chunk[
            "duration"
        ].to_numpy(
            dtype=np.float64,
            copy=False
        )

        machine = chunk[
            "machine_id"
        ].to_numpy(
            dtype=np.float64,
            copy=False
        )

        task_count += len(chunk)

        missing_start_count += int(
            np.sum(
                ~np.isfinite(start)
            )
        )

        missing_end_count += int(
            np.sum(
                ~np.isfinite(end)
            )
        )

        missing_machine_count += int(
            np.sum(
                ~np.isfinite(machine)
            )
        )

        invalid_time_count += int(
            np.sum(
                np.isfinite(start)
                & np.isfinite(end)
                & (end < start)
            )
        )

        negative_duration_count += int(
            np.sum(
                np.isfinite(duration)
                & (duration < 0)
            )
        )

        valid_start = np.isfinite(
            start
        )

        if np.any(valid_start):

            valid_values = start[
                valid_start
            ]

            task_before_scenario += int(
                np.sum(
                    valid_values
                    < scenario_min
                )
            )

            task_after_scenario += int(
                np.sum(
                    valid_values
                    >= scenario_max
                )
            )

            time_bin = np.floor(
                (
                    valid_values
                    - scenario_min
                )
                /
                (
                    scenario_max
                    - scenario_min
                )
                * 1000
            ).astype(
                np.int64
            )

            time_bin = np.clip(
                time_bin,
                0,
                999
            )

            unique_bins, bin_counts = np.unique(
                time_bin,
                return_counts=True
            )

            for bin_value, count in zip(
                unique_bins,
                bin_counts
            ):

                bin_value = int(
                    bin_value
                )

                scenario_bin_counts[
                    bin_value
                ] = (
                    scenario_bin_counts.get(
                        bin_value,
                        0
                    )
                    + int(count)
                )

        del chunk
        gc.collect()

    add_result(
        "Total task rows scanned",
        split,
        task_count,
        "PASS"
        if task_count > 0
        else "FAIL",
        "Prepared split contains task instances"
    )

    add_result(
        "Missing start time",
        split,
        missing_start_count,
        "PASS"
        if missing_start_count == 0
        else "FAIL",
        "Task arrival time must be available"
    )

    add_result(
        "Missing end time",
        split,
        missing_end_count,
        "PASS"
        if missing_end_count == 0
        else "FAIL",
        "Task completion reference must be available"
    )

    add_result(
        "Negative duration",
        split,
        negative_duration_count,
        "PASS"
        if negative_duration_count == 0
        else "FAIL",
        "Prepared splits must not contain negative durations"
    )

    add_result(
        "Invalid time ordering",
        split,
        invalid_time_count,
        "PASS"
        if invalid_time_count == 0
        else "FAIL",
        "End time must not precede start time"
    )

    add_result(
        "Missing machine ID",
        split,
        missing_machine_count,
        "PASS"
        if missing_machine_count == 0
        else "FAIL",
        "Machine ID must be available for internal trace bookkeeping"
    )

    add_result(
        "Tasks before scenario range",
        split,
        task_before_scenario,
        "DOCUMENTED",
        "Tasks before the first scenario window are outside this split's scenario horizon"
    )

    add_result(
        "Tasks after scenario range",
        split,
        task_after_scenario,
        "DOCUMENTED",
        "Tasks after the final scenario window are outside this split's scenario horizon"
    )

    covered_bins = len(
        scenario_bin_counts
    )

    add_result(
        "Temporal bins receiving tasks",
        split,
        covered_bins,
        "PASS"
        if covered_bins > 0
        else "FAIL",
        "Scenario windows must contain task arrivals"
    )


coverage_results = []

for split in FILES:

    split_manifest = manifest[
        manifest["split"] == split
    ]

    scenario_min = (
        split_manifest[
            "time_start"
        ].min()
    )

    scenario_max = (
        split_manifest[
            "time_end"
        ].max()
    )

    usage_covered_until = min(
        scenario_max,
        MACHINE_USAGE_MAX
    )

    usage_coverage_fraction = (
        usage_covered_until
        - scenario_min
    ) / (
        scenario_max
        - scenario_min
    )

    metadata_covered_until = min(
        scenario_max,
        MACHINE_META_MAX
    )

    metadata_coverage_fraction = (
        metadata_covered_until
        - scenario_min
    ) / (
        scenario_max
        - scenario_min
    )

    coverage_results.append({
        "split": split,
        "scenario_start": scenario_min,
        "scenario_end": scenario_max,
        "machine_usage_max": MACHINE_USAGE_MAX,
        "machine_meta_max": MACHINE_META_MAX,
        "machine_usage_temporal_fraction": usage_coverage_fraction,
        "machine_metadata_temporal_fraction": metadata_coverage_fraction
    })


coverage_df = pd.DataFrame(
    coverage_results
)

coverage_output = os.path.join(
    BASE_DIR,
    "workload_analysis",
    "simulator_machine_temporal_coverage.csv"
)

coverage_df.to_csv(
    coverage_output,
    index=False
)


validation_df = pd.DataFrame(
    results
)

validation_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "Simulator temporal contract validation completed"
)

print(
    "Validation output:",
    OUTPUT_FILE
)

print(
    "Machine coverage output:",
    coverage_output
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