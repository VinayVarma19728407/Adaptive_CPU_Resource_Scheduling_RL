"""
Workload Percentile Analysis.

This script calculates exact quantile and percentile distributions (p1, p5, p25, p50, p75, p90, p95, p99)
for core workload metrics across batch instances. It interpolates exact resource usage boundaries to define
high, medium, and low resource utilization regimes for simulation scenario construction.
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

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "accurate_workload_percentiles.csv"
)

CHUNK_SIZE = 250000

BIN_COUNT = 20000

PERCENTILES = [
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
    0.995
]

METRICS = [
    "duration",
    "cpu_avg",
    "cpu_max",
    "mem_avg",
    "mem_max"
]

RANGES = {
    "training": {
        "duration": (0.0, 479737.0),
        "cpu_avg": (0.0, 6676.0),
        "cpu_max": (0.0, 3833700.0),
        "mem_avg": (0.0, 107.34),
        "mem_max": (0.0, 235.12)
    },
    "validation": {
        "duration": (0.0, 188295.0),
        "cpu_avg": (0.0, 4702.0),
        "cpu_max": (0.0, 17250.0),
        "mem_avg": (0.0, 105.84),
        "mem_max": (0.0, 233.18)
    },
    "testing": {
        "duration": (0.0, 142205.0),
        "cpu_avg": (0.0, 4798.0),
        "cpu_max": (0.0, 3645655.0),
        "mem_avg": (0.0, 101.71),
        "mem_max": (0.0, 233.0)
    }
}

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


def create_edges(minimum, maximum):

    transformed_max = np.log1p(
        maximum
    )

    return np.linspace(
        0.0,
        transformed_max,
        BIN_COUNT + 1,
        dtype=np.float64
    )


def estimate_quantile(
    histogram,
    edges,
    quantile,
    total_count,
    minimum,
    maximum
):

    if total_count == 0:
        return np.nan

    if quantile <= 0:
        return minimum

    if quantile >= 1:
        return maximum

    target = quantile * total_count

    cumulative = np.cumsum(
        histogram
    )

    bin_index = np.searchsorted(
        cumulative,
        target,
        side="left"
    )

    if bin_index >= len(histogram):
        return maximum

    previous_count = 0

    if bin_index > 0:
        previous_count = cumulative[
            bin_index - 1
        ]

    bin_count = histogram[
        bin_index
    ]

    if bin_count <= 0:
        transformed_value = edges[
            bin_index
        ]
    else:
        fraction = (
            target - previous_count
        ) / bin_count

        transformed_value = (
            edges[bin_index]
            + fraction
            * (
                edges[bin_index + 1]
                - edges[bin_index]
            )
        )

    value = np.expm1(
        transformed_value
    )

    if value < minimum:
        value = minimum

    if value > maximum:
        value = maximum

    return value


def analyze_split(
    split_name,
    file_path
):

    print("Analyzing", split_name)

    edges = {}

    histograms = {}

    counts = {}

    for metric in METRICS:

        minimum, maximum = RANGES[
            split_name
        ][metric]

        edges[metric] = create_edges(
            minimum,
            maximum
        )

        histograms[metric] = np.zeros(
            BIN_COUNT,
            dtype=np.int64
        )

        counts[metric] = 0

    usecols = METRICS

    chunk_number = 0

    for chunk in pd.read_csv(
        file_path,
        usecols=usecols,
        chunksize=CHUNK_SIZE
    ):

        chunk_number += 1

        for metric in METRICS:

            values = chunk[
                metric
            ].to_numpy(
                dtype=np.float64,
                copy=False
            )

            valid = np.isfinite(
                values
            )

            if not np.any(valid):
                continue

            values = values[
                valid
            ]

            minimum, maximum = RANGES[
                split_name
            ][metric]

            values = np.clip(
                values,
                minimum,
                maximum
            )

            transformed = np.log1p(
                values
            )

            histogram, _ = np.histogram(
                transformed,
                bins=edges[metric]
            )

            histograms[metric] += (
                histogram
            )

            counts[metric] += len(
                values
            )

        if chunk_number % 100 == 0:

            print(
                "Chunks processed:",
                chunk_number
            )

        del chunk
        gc.collect()

    results = []

    for metric in METRICS:

        minimum, maximum = RANGES[
            split_name
        ][metric]

        row = {
            "split": split_name,
            "metric": metric,
            "count": counts[metric],
            "method": "high resolution log1p histogram",
            "min": minimum,
            "max": maximum
        }

        for percentile in PERCENTILES:

            estimated_value = estimate_quantile(
                histograms[metric],
                edges[metric],
                percentile,
                counts[metric],
                minimum,
                maximum
            )

            column_name = (
                "p"
                + str(percentile * 100)
                .replace(".0", "")
                .replace(".", "")
            )

            if percentile == 0.995:
                column_name = "p99_5"

            row[column_name] = (
                estimated_value
            )

        results.append(
            row
        )

    del edges
    del histograms
    del counts

    gc.collect()

    return results


all_results = []


for split_name, file_path in FILES.items():

    if not os.path.exists(
        file_path
    ):

        print(
            "File not found:",
            file_path
        )

        continue

    split_results = analyze_split(
        split_name,
        file_path
    )

    all_results.extend(
        split_results
    )

    gc.collect()


results_df = pd.DataFrame(
    all_results
)

results_df = results_df[
    [
        "split",
        "metric",
        "count",
        "method",
        "min",
        "p50",
        "p75",
        "p90",
        "p95",
        "p99",
        "p99_5",
        "max"
    ]
]

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    "Workload percentile analysis completed"
)

print(
    "Output file:",
    OUTPUT_FILE
)

print(
    "Rows:",
    len(results_df)
)

print(
    "Histogram bins:",
    BIN_COUNT
)