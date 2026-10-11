"""
Workload Regime Classification & Analysis.

This script identifies and categorizes operational workload regimes (bursty, steady-state, high-memory, high-cpu)
from batch instance traces. It aggregates task instances into temporal windows, computes resource intensity
metrics, and exports summary statistics used to define simulator evaluation scenarios.
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

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "workload_analysis"
)

TEMPORAL_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "temporal_workload_statistics.csv"
)

REGIME_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "workload_regime_profiles.csv"
)

THRESHOLD_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "workload_regime_thresholds.csv"
)

SCENARIO_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "workload_scenario_manifest.csv"
)

CHUNK_SIZE = 250000

TIME_BINS = 1000

ROLLING_WINDOW = 21

FILES_COLUMNS = [
    "start_time",
    "duration",
    "cpu_avg",
    "mem_avg"
]

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


TIME_RANGES = {
    "training": (1343.0, 729925.0),
    "validation": (442534.0, 727465.0),
    "testing": (578482.0, 772736.0)
}


def create_empty_statistics():

    return {
        "arrival_count": np.zeros(
            TIME_BINS,
            dtype=np.int64
        ),
        "duration_sum": np.zeros(
            TIME_BINS,
            dtype=np.float64
        ),
        "cpu_work_sum": np.zeros(
            TIME_BINS,
            dtype=np.float64
        ),
        "mem_work_sum": np.zeros(
            TIME_BINS,
            dtype=np.float64
        ),
        "cpu_known_count": np.zeros(
            TIME_BINS,
            dtype=np.int64
        ),
        "mem_known_count": np.zeros(
            TIME_BINS,
            dtype=np.int64
        )
    }


def process_split(
    split_name,
    file_path
):

    print(
        "Analyzing temporal workload:",
        split_name
    )

    time_min, time_max = TIME_RANGES[
        split_name
    ]

    statistics = create_empty_statistics()

    time_width = (
        time_max - time_min
    ) / TIME_BINS

    chunk_number = 0

    for chunk in pd.read_csv(
        file_path,
        usecols=FILES_COLUMNS,
        chunksize=CHUNK_SIZE
    ):

        chunk_number += 1

        start = chunk[
            "start_time"
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

        cpu = chunk[
            "cpu_avg"
        ].to_numpy(
            dtype=np.float64,
            copy=False
        )

        memory = chunk[
            "mem_avg"
        ].to_numpy(
            dtype=np.float64,
            copy=False
        )

        valid_start = np.isfinite(
            start
        )

        if not np.any(valid_start):

            del chunk
            gc.collect()
            continue

        start = start[
            valid_start
        ]

        duration = duration[
            valid_start
        ]

        cpu = cpu[
            valid_start
        ]

        memory = memory[
            valid_start
        ]

        bin_index = np.floor(
            (
                start - time_min
            ) / time_width
        ).astype(
            np.int64
        )

        bin_index = np.clip(
            bin_index,
            0,
            TIME_BINS - 1
        )

        arrival_counts = np.bincount(
            bin_index,
            minlength=TIME_BINS
        )

        statistics[
            "arrival_count"
        ] += arrival_counts

        valid_duration = np.isfinite(
            duration
        )

        if np.any(valid_duration):

            duration_counts = np.bincount(
                bin_index[valid_duration],
                weights=duration[valid_duration],
                minlength=TIME_BINS
            )

            statistics[
                "duration_sum"
            ] += duration_counts

        valid_cpu = (
            np.isfinite(cpu)
            & np.isfinite(duration)
        )

        if np.any(valid_cpu):

            cpu_work = (
                cpu[valid_cpu]
                * duration[valid_cpu]
            )

            cpu_counts = np.bincount(
                bin_index[valid_cpu],
                weights=cpu_work,
                minlength=TIME_BINS
            )

            cpu_known = np.bincount(
                bin_index[valid_cpu],
                minlength=TIME_BINS
            )

            statistics[
                "cpu_work_sum"
            ] += cpu_counts

            statistics[
                "cpu_known_count"
            ] += cpu_known

        valid_memory = (
            np.isfinite(memory)
            & np.isfinite(duration)
        )

        if np.any(valid_memory):

            memory_work = (
                memory[valid_memory]
                * duration[valid_memory]
            )

            memory_counts = np.bincount(
                bin_index[valid_memory],
                weights=memory_work,
                minlength=TIME_BINS
            )

            memory_known = np.bincount(
                bin_index[valid_memory],
                minlength=TIME_BINS
            )

            statistics[
                "mem_work_sum"
            ] += memory_counts

            statistics[
                "mem_known_count"
            ] += memory_known

        if chunk_number % 100 == 0:

            print(
                "Chunks processed:",
                chunk_number,
                flush=True
            )

        del chunk
        gc.collect()

    bin_starts = (
        time_min
        + np.arange(TIME_BINS)
        * time_width
    )

    bin_ends = (
        bin_starts
        + time_width
    )

    result = pd.DataFrame({
        "split": split_name,
        "time_bin": np.arange(
            TIME_BINS
        ),
        "time_start": bin_starts,
        "time_end": bin_ends,
        "arrival_count":
            statistics["arrival_count"],
        "arrival_rate":
            statistics["arrival_count"]
            / time_width,
        "duration_sum":
            statistics["duration_sum"],
        "duration_rate":
            statistics["duration_sum"]
            / time_width,
        "cpu_work_sum":
            statistics["cpu_work_sum"],
        "cpu_work_rate":
            statistics["cpu_work_sum"]
            / time_width,
        "memory_work_sum":
            statistics["mem_work_sum"],
        "memory_work_rate":
            statistics["mem_work_sum"]
            / time_width,
        "cpu_known_count":
            statistics["cpu_known_count"],
        "memory_known_count":
            statistics["mem_known_count"]
    })

    return result


all_profiles = []


for split_name, file_path in FILES.items():

    if not os.path.exists(
        file_path
    ):

        print(
            "File not found:",
            file_path
        )

        continue

    profile = process_split(
        split_name,
        file_path
    )

    all_profiles.append(
        profile
    )

    gc.collect()


temporal_df = pd.concat(
    all_profiles,
    ignore_index=True
)

temporal_df.to_csv(
    TEMPORAL_OUTPUT,
    index=False
)

print(
    "Temporal workload statistics saved:",
    TEMPORAL_OUTPUT
)


training = temporal_df[
    temporal_df["split"] == "training"
].copy()


def percentile_rank(
    reference,
    values
):

    reference = np.asarray(
        reference,
        dtype=np.float64
    )

    values = np.asarray(
        values,
        dtype=np.float64
    )

    reference = reference[
        np.isfinite(reference)
    ]

    sorted_reference = np.sort(
        reference
    )

    positions = np.searchsorted(
        sorted_reference,
        values,
        side="right"
    )

    ranks = positions / len(
        sorted_reference
    )

    return ranks


arrival_training = training[
    "arrival_rate"
].to_numpy()

cpu_training = training[
    "cpu_work_rate"
].to_numpy()

memory_training = training[
    "memory_work_rate"
].to_numpy()


def valid_values(values):

    return values[
        np.isfinite(values)
        & (values >= 0)
    ]


arrival_training = valid_values(
    arrival_training
)

cpu_training = valid_values(
    cpu_training
)

memory_training = valid_values(
    memory_training
)


training_arrival_thresholds = np.percentile(
    arrival_training,
    [33.333333, 66.666667]
)

training_cpu_thresholds = np.percentile(
    cpu_training,
    [20, 80]
)

training_memory_thresholds = np.percentile(
    memory_training,
    [20, 80]
)


training_profiles = training.copy()

training_profiles[
    "arrival_score"
] = percentile_rank(
    arrival_training,
    training_profiles[
        "arrival_rate"
    ].to_numpy()
)

training_profiles[
    "cpu_score"
] = percentile_rank(
    cpu_training,
    training_profiles[
        "cpu_work_rate"
    ].to_numpy()
)

training_profiles[
    "memory_score"
] = percentile_rank(
    memory_training,
    training_profiles[
        "memory_work_rate"
    ].to_numpy()
)

training_profiles[
    "workload_intensity"
] = (
    training_profiles[
        [
            "arrival_score",
            "cpu_score",
            "memory_score"
        ]
    ].mean(
        axis=1
    )
)

training_profiles[
    "resource_dominance"
] = (
    training_profiles[
        "cpu_score"
    ]
    -
    training_profiles[
        "memory_score"
    ]
)


rolling_mean = (
    training_profiles[
        "workload_intensity"
    ]
    .rolling(
        ROLLING_WINDOW,
        center=True,
        min_periods=5
    )
    .mean()
)

rolling_std = (
    training_profiles[
        "workload_intensity"
    ]
    .rolling(
        ROLLING_WINDOW,
        center=True,
        min_periods=5
    )
    .std()
)

training_profiles[
    "variability_score"
] = (
    rolling_std
    /
    rolling_mean.replace(
        0,
        np.nan
    )
)

variability_training = valid_values(
    training_profiles[
        "variability_score"
    ].to_numpy()
)

variability_threshold = np.percentile(
    variability_training,
    80
)

intensity_thresholds = np.percentile(
    training_profiles[
        "workload_intensity"
    ].dropna(),
    [33.333333, 66.666667]
)

cpu_dominance_threshold = np.percentile(
    training_profiles[
        "resource_dominance"
    ].dropna(),
    80
)

memory_dominance_threshold = np.percentile(
    training_profiles[
        "resource_dominance"
    ].dropna(),
    20
)


threshold_rows = [
    {
        "threshold": "low_medium_intensity",
        "value": intensity_thresholds[0],
        "source": "training"
    },
    {
        "threshold": "medium_high_intensity",
        "value": intensity_thresholds[1],
        "source": "training"
    },
    {
        "threshold": "highly_variable",
        "value": variability_threshold,
        "source": "training"
    },
    {
        "threshold": "cpu_heavy",
        "value": cpu_dominance_threshold,
        "source": "training"
    },
    {
        "threshold": "memory_heavy",
        "value": memory_dominance_threshold,
        "source": "training"
    }
]

threshold_df = pd.DataFrame(
    threshold_rows
)

threshold_df.to_csv(
    THRESHOLD_OUTPUT,
    index=False
)


regime_profiles = []


for split_name in [
    "training",
    "validation",
    "testing"
]:

    split_profile = temporal_df[
        temporal_df["split"] == split_name
    ].copy()

    split_profile[
        "arrival_score"
    ] = percentile_rank(
        arrival_training,
        split_profile[
            "arrival_rate"
        ].to_numpy()
    )

    split_profile[
        "cpu_score"
    ] = percentile_rank(
        cpu_training,
        split_profile[
            "cpu_work_rate"
        ].to_numpy()
    )

    split_profile[
        "memory_score"
    ] = percentile_rank(
        memory_training,
        split_profile[
            "memory_work_rate"
        ].to_numpy()
    )

    split_profile[
        "workload_intensity"
    ] = (
        split_profile[
            [
                "arrival_score",
                "cpu_score",
                "memory_score"
            ]
        ].mean(
            axis=1
        )
    )

    split_profile[
        "resource_dominance"
    ] = (
        split_profile[
            "cpu_score"
        ]
        -
        split_profile[
            "memory_score"
        ]
    )

    split_profile[
        "rolling_mean"
    ] = (
        split_profile[
            "workload_intensity"
        ]
        .rolling(
            ROLLING_WINDOW,
            center=True,
            min_periods=5
        )
        .mean()
    )

    split_profile[
        "rolling_std"
    ] = (
        split_profile[
            "workload_intensity"
        ]
        .rolling(
            ROLLING_WINDOW,
            center=True,
            min_periods=5
        )
        .std()
    )

    split_profile[
        "variability_score"
    ] = (
        split_profile[
            "rolling_std"
        ]
        /
        split_profile[
            "rolling_mean"
        ].replace(
            0,
            np.nan
        )
    )

    split_profile[
        "intensity_regime"
    ] = np.select(
        [
            split_profile[
                "workload_intensity"
            ]
            <= intensity_thresholds[0],

            split_profile[
                "workload_intensity"
            ]
            <= intensity_thresholds[1]
        ],
        [
            "Low",
            "Medium"
        ],
        default="High"
    )

    split_profile[
        "resource_regime"
    ] = np.select(
        [
            split_profile[
                "resource_dominance"
            ]
            >= cpu_dominance_threshold,

            split_profile[
                "resource_dominance"
            ]
            <= memory_dominance_threshold
        ],
        [
            "CPU-heavy",
            "Memory-heavy"
        ],
        default="Balanced-resource"
    )

    split_profile[
        "variability_regime"
    ] = np.where(
        split_profile[
            "variability_score"
        ]
        >= variability_threshold,
        "Highly Variable",
        "Not Highly Variable"
    )

    regime_profiles.append(
        split_profile
    )


regime_df = pd.concat(
    regime_profiles,
    ignore_index=True
)

regime_df.to_csv(
    REGIME_OUTPUT,
    index=False
)


scenario_rows = []


for split_name in [
    "training",
    "validation",
    "testing"
]:

    split_df = regime_df[
        regime_df["split"] == split_name
    ]

    intensity_counts = (
        split_df[
            "intensity_regime"
        ]
        .value_counts()
        .to_dict()
    )

    variability_count = int(
        (
            split_df[
                "variability_regime"
            ]
            == "Highly Variable"
        ).sum()
    )

    cpu_count = int(
        (
            split_df[
                "resource_regime"
            ]
            == "CPU-heavy"
        ).sum()
    )

    memory_count = int(
        (
            split_df[
                "resource_regime"
            ]
            == "Memory-heavy"
        ).sum()
    )

    for regime in [
        "Low",
        "Medium",
        "High"
    ]:

        scenario_rows.append({
            "split": split_name,
            "scenario_type": "natural-frequency",
            "regime": regime,
            "available_time_bins": int(
                intensity_counts.get(
                    regime,
                    0
                )
            )
        })

    scenario_rows.append({
        "split": split_name,
        "scenario_type": "natural-frequency",
        "regime": "Highly Variable",
        "available_time_bins": variability_count
    })

    scenario_rows.append({
        "split": split_name,
        "scenario_type": "natural-frequency",
        "regime": "CPU-heavy",
        "available_time_bins": cpu_count
    })

    scenario_rows.append({
        "split": split_name,
        "scenario_type": "natural-frequency",
        "regime": "Memory-heavy",
        "available_time_bins": memory_count
    })


for regime in [
    "Low",
    "Medium",
    "High"
]:

    scenario_rows.append({
        "split": "all",
        "scenario_type": "balanced",
        "regime": regime,
        "selection_rule": "equal number of eligible time windows"
    })


scenario_rows.append({
    "split": "all",
    "scenario_type": "balanced",
    "regime": "Highly Variable",
    "selection_rule": "equal number of eligible time windows"
})

scenario_rows.append({
    "split": "all",
    "scenario_type": "balanced",
    "regime": "CPU-heavy",
    "selection_rule": "equal number of eligible time windows"
})

scenario_rows.append({
    "split": "all",
    "scenario_type": "balanced",
    "regime": "Memory-heavy",
    "selection_rule": "equal number of eligible time windows"
})


scenario_df = pd.DataFrame(
    scenario_rows
)

scenario_df.to_csv(
    SCENARIO_OUTPUT,
    index=False
)

print(
    "Workload regime analysis completed"
)

print(
    "Temporal output:",
    TEMPORAL_OUTPUT
)

print(
    "Regime output:",
    REGIME_OUTPUT
)

print(
    "Threshold output:",
    THRESHOLD_OUTPUT
)

print(
    "Scenario manifest:",
    SCENARIO_OUTPUT
)