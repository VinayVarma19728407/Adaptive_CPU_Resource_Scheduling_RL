"""
Create Workload Scenario Manifest.

This script creates a workload scenario manifest file defining standardized temporal evaluation scenarios for
RL scheduler benchmarks. It extracts representative temporal windows (low load, peak load, mixed load),
populating scenario metadata, task density, and timestamp boundaries.
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

OUTPUT_FILE = os.path.join(
    WORKLOAD_DIR,
    "workload_scenario_manifest_final.csv"
)

SEED = 20261004

SPLITS = [
    "training",
    "validation",
    "testing"
]


def select_rows(
    data,
    condition,
    count,
    rng
):

    eligible = data[
        condition
    ].copy()

    if count <= 0:
        return eligible.iloc[0:0].copy()

    if len(eligible) <= count:
        return eligible.copy()

    selected_indices = rng.choice(
        len(eligible),
        size=count,
        replace=False
    )

    return eligible.iloc[
        np.sort(selected_indices)
    ].copy()


profiles = pd.read_csv(
    PROFILE_FILE
)

required_columns = [
    "split",
    "time_bin",
    "time_start",
    "time_end",
    "intensity_regime",
    "resource_regime",
    "variability_regime",
    "workload_intensity",
    "resource_dominance",
    "variability_score"
]

missing_columns = [
    column
    for column in required_columns
    if column not in profiles.columns
]

if missing_columns:
    raise ValueError(
        "Missing columns: "
        + str(missing_columns)
    )


manifest_rows = []


for split in SPLITS:

    split_data = profiles[
        profiles["split"] == split
    ].copy()

    split_data = split_data.sort_values(
        "time_bin"
    ).reset_index(
        drop=True
    )

    rng = np.random.default_rng(
        SEED
    )

    for _, row in split_data.iterrows():

        manifest_rows.append({
            "scenario_id":
                split.upper()
                + "_NATURAL_"
                + str(int(row["time_bin"])),
            "split":
                split,
            "scenario_type":
                "natural-frequency",
            "scenario_axis":
                "all",
            "regime":
                "natural",
            "time_bin":
                int(row["time_bin"]),
            "time_start":
                row["time_start"],
            "time_end":
                row["time_end"],
            "workload_intensity":
                row["workload_intensity"],
            "resource_dominance":
                row["resource_dominance"],
            "variability_score":
                row["variability_score"],
            "intensity_regime":
                row["intensity_regime"],
            "resource_regime":
                row["resource_regime"],
            "variability_regime":
                row["variability_regime"],
            "selection_seed":
                SEED
        })


for split in SPLITS:

    split_data = profiles[
        profiles["split"] == split
    ].copy()

    rng = np.random.default_rng(
        SEED
    )


    intensity_counts = {
        "Low": int(
            (
                split_data[
                    "intensity_regime"
                ]
                == "Low"
            ).sum()
        ),
        "Medium": int(
            (
                split_data[
                    "intensity_regime"
                ]
                == "Medium"
            ).sum()
        ),
        "High": int(
            (
                split_data[
                    "intensity_regime"
                ]
                == "High"
            ).sum()
        )
    }

    intensity_balanced_count = min(
        intensity_counts.values()
    )

    for regime in [
        "Low",
        "Medium",
        "High"
    ]:

        condition = (
            split_data[
                "intensity_regime"
            ]
            == regime
        )

        selected = select_rows(
            split_data,
            condition,
            intensity_balanced_count,
            rng
        )

        for _, row in selected.iterrows():

            manifest_rows.append({
                "scenario_id":
                    split.upper()
                    + "_BALANCED_INTENSITY_"
                    + regime.upper()
                    + "_"
                    + str(int(row["time_bin"])),
                "split":
                    split,
                "scenario_type":
                    "balanced",
                "scenario_axis":
                    "intensity",
                "regime":
                    regime,
                "time_bin":
                    int(row["time_bin"]),
                "time_start":
                    row["time_start"],
                "time_end":
                    row["time_end"],
                "workload_intensity":
                    row["workload_intensity"],
                "resource_dominance":
                    row["resource_dominance"],
                "variability_score":
                    row["variability_score"],
                "intensity_regime":
                    row["intensity_regime"],
                "resource_regime":
                    row["resource_regime"],
                "variability_regime":
                    row["variability_regime"],
                "selection_seed":
                    SEED
            })


    resource_counts = {
        "CPU-heavy": int(
            (
                split_data[
                    "resource_regime"
                ]
                == "CPU-heavy"
            ).sum()
        ),
        "Memory-heavy": int(
            (
                split_data[
                    "resource_regime"
                ]
                == "Memory-heavy"
            ).sum()
        )
    }

    resource_balanced_count = min(
        resource_counts.values()
    )

    for regime in [
        "CPU-heavy",
        "Memory-heavy"
    ]:

        condition = (
            split_data[
                "resource_regime"
            ]
            == regime
        )

        selected = select_rows(
            split_data,
            condition,
            resource_balanced_count,
            rng
        )

        for _, row in selected.iterrows():

            manifest_rows.append({
                "scenario_id":
                    split.upper()
                    + "_BALANCED_RESOURCE_"
                    + regime.upper().replace(
                        "-",
                        "_"
                    )
                    + "_"
                    + str(int(row["time_bin"])),
                "split":
                    split,
                "scenario_type":
                    "balanced",
                "scenario_axis":
                    "resource",
                "regime":
                    regime,
                "time_bin":
                    int(row["time_bin"]),
                "time_start":
                    row["time_start"],
                "time_end":
                    row["time_end"],
                "workload_intensity":
                    row["workload_intensity"],
                "resource_dominance":
                    row["resource_dominance"],
                "variability_score":
                    row["variability_score"],
                "intensity_regime":
                    row["intensity_regime"],
                "resource_regime":
                    row["resource_regime"],
                "variability_regime":
                    row["variability_regime"],
                "selection_seed":
                    SEED
            })


    variable_count = int(
        (
            split_data[
                "variability_regime"
            ]
            == "Highly Variable"
        ).sum()
    )

    stable_count = int(
        (
            split_data[
                "variability_regime"
            ]
            == "Not Highly Variable"
        ).sum()
    )

    variability_balanced_count = min(
        variable_count,
        stable_count
    )

    for regime in [
        "Highly Variable",
        "Not Highly Variable"
    ]:

        condition = (
            split_data[
                "variability_regime"
            ]
            == regime
        )

        selected = select_rows(
            split_data,
            condition,
            variability_balanced_count,
            rng
        )

        for _, row in selected.iterrows():

            manifest_rows.append({
                "scenario_id":
                    split.upper()
                    + "_BALANCED_VARIABILITY_"
                    + regime.upper().replace(
                        " ",
                        "_"
                    )
                    + "_"
                    + str(int(row["time_bin"])),
                "split":
                    split,
                "scenario_type":
                    "balanced",
                "scenario_axis":
                    "variability",
                "regime":
                    regime,
                "time_bin":
                    int(row["time_bin"]),
                "time_start":
                    row["time_start"],
                "time_end":
                    row["time_end"],
                "workload_intensity":
                    row["workload_intensity"],
                "resource_dominance":
                    row["resource_dominance"],
                "variability_score":
                    row["variability_score"],
                "intensity_regime":
                    row["intensity_regime"],
                "resource_regime":
                    row["resource_regime"],
                "variability_regime":
                    row["variability_regime"],
                "selection_seed":
                    SEED
            })


manifest = pd.DataFrame(
    manifest_rows
)

manifest = manifest.sort_values(
    [
        "split",
        "scenario_type",
        "scenario_axis",
        "time_bin",
        "regime"
    ]
).reset_index(
    drop=True
)

duplicate_ids = manifest[
    "scenario_id"
].duplicated().sum()

if duplicate_ids != 0:
    raise ValueError(
        "Duplicate scenario IDs found: "
        + str(duplicate_ids)
    )

manifest.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    "Final workload scenario manifest created"
)

print(
    "Output file:",
    OUTPUT_FILE
)

print(
    "Total scenario rows:",
    len(manifest)
)

print(
    "Natural frequency rows:",
    int(
        (
            manifest["scenario_type"]
            == "natural-frequency"
        ).sum()
    )
)

print(
    "Balanced rows:",
    int(
        (
            manifest["scenario_type"]
            == "balanced"
        ).sum()
    )
)

print(
    "Unique scenario IDs:",
    manifest[
        "scenario_id"
    ].nunique()
)

print(
    "Duplicate scenario IDs:",
    duplicate_ids
)