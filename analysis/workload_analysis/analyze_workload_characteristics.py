"""
Workload Characteristic Analysis.

This script computes comprehensive workload characterization statistics across Alibaba trace data splits.
It calculates histogram distributions, summary percentiles, and mean/std stats for CPU/memory consumption,
task arrival frequencies, and execution durations to characterize cluster workload dynamics.
"""

import os
import gc
import numpy as np
import pandas as pd


BASE_DIR = "dataset"

TRAIN_FILE = os.path.join(
    BASE_DIR,
    "preprocessed",
    "batch_instance_training.csv"
)

VALIDATION_FILE = os.path.join(
    BASE_DIR,
    "preprocessed",
    "batch_instance_validation.csv"
)

TEST_FILE = os.path.join(
    BASE_DIR,
    "preprocessed",
    "batch_instance_testing.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "workload_analysis"
)

CHUNK_SIZE = 250000

os.makedirs(OUTPUT_DIR, exist_ok=True)


FILES = {
    "training": TRAIN_FILE,
    "validation": VALIDATION_FILE,
    "testing": TEST_FILE
}


USECOLS = [
    "task_type",
    "status",
    "start_time",
    "end_time",
    "cpu_avg",
    "cpu_max",
    "mem_avg",
    "mem_max",
    "duration"
]


def percentile_from_histogram(histogram, edges, total):
    if total == 0:
        return np.nan

    cumulative = np.cumsum(histogram)

    targets = [
        0.50,
        0.75,
        0.90,
        0.95,
        0.99
    ]

    values = []

    for target in targets:
        position = target * total
        index = np.searchsorted(cumulative, position)

        if index >= len(edges) - 1:
            index = len(edges) - 2

        values.append(
            (edges[index] + edges[index + 1]) / 2
        )

    return values


def create_histogram_stats(file_path):

    duration_min = np.inf
    duration_max = -np.inf

    cpu_avg_min = np.inf
    cpu_avg_max = -np.inf

    cpu_max_min = np.inf
    cpu_max_max = -np.inf

    mem_avg_min = np.inf
    mem_avg_max = -np.inf

    mem_max_min = np.inf
    mem_max_max = -np.inf

    for chunk in pd.read_csv(
        file_path,
        usecols=USECOLS,
        chunksize=CHUNK_SIZE
    ):

        duration_values = chunk["duration"].dropna()

        if len(duration_values) > 0:
            duration_min = min(
                duration_min,
                duration_values.min()
            )
            duration_max = max(
                duration_max,
                duration_values.max()
            )

        for column, current_min, current_max in [
            ("cpu_avg", cpu_avg_min, cpu_avg_max),
            ("cpu_max", cpu_max_min, cpu_max_max),
            ("mem_avg", mem_avg_min, mem_avg_max),
            ("mem_max", mem_max_min, mem_max_max)
        ]:
            values = chunk[column].dropna()

            if len(values) > 0:
                if column == "cpu_avg":
                    cpu_avg_min = min(cpu_avg_min, values.min())
                    cpu_avg_max = max(cpu_avg_max, values.max())

                elif column == "cpu_max":
                    cpu_max_min = min(cpu_max_min, values.min())
                    cpu_max_max = max(cpu_max_max, values.max())

                elif column == "mem_avg":
                    mem_avg_min = min(mem_avg_min, values.min())
                    mem_avg_max = max(mem_avg_max, values.max())

                elif column == "mem_max":
                    mem_max_min = min(mem_max_min, values.min())
                    mem_max_max = max(mem_max_max, values.max())

        del chunk
        gc.collect()

    return {
        "duration": (duration_min, duration_max),
        "cpu_avg": (cpu_avg_min, cpu_avg_max),
        "cpu_max": (cpu_max_min, cpu_max_max),
        "mem_avg": (mem_avg_min, mem_avg_max),
        "mem_max": (mem_max_min, mem_max_max)
    }


def analyze_split(split_name, file_path):

    print("Analyzing", split_name)

    row_count = 0

    missing_counts = {
        column: 0
        for column in USECOLS
    }

    zero_duration = 0
    positive_duration = 0

    negative_duration = 0

    duration_sum = 0.0
    duration_sum_sq = 0.0

    cpu_avg_sum = 0.0
    cpu_avg_sum_sq = 0.0
    cpu_avg_count = 0

    cpu_max_sum = 0.0
    cpu_max_sum_sq = 0.0
    cpu_max_count = 0

    mem_avg_sum = 0.0
    mem_avg_sum_sq = 0.0
    mem_avg_count = 0

    mem_max_sum = 0.0
    mem_max_sum_sq = 0.0
    mem_max_count = 0

    task_type_counts = {}
    status_counts = {}

    cpu_heavy_count = 0
    memory_heavy_count = 0
    balanced_count = 0

    time_min = np.inf
    time_max = -np.inf

    cpu_hist = None
    cpu_edges = None

    mem_hist = None
    mem_edges = None

    duration_hist = None
    duration_edges = None

    first_pass_ranges = create_histogram_stats(
        file_path
    )

    duration_range = first_pass_ranges["duration"]
    cpu_range = first_pass_ranges["cpu_avg"]
    mem_range = first_pass_ranges["mem_avg"]

    if np.isfinite(duration_range[0]) and np.isfinite(duration_range[1]):
        if duration_range[0] == duration_range[1]:
            duration_range = (
                duration_range[0],
                duration_range[0] + 1
            )

        duration_edges = np.linspace(
            duration_range[0],
            duration_range[1],
            1001
        )

        duration_hist = np.zeros(
            1000,
            dtype=np.int64
        )

    if np.isfinite(cpu_range[0]) and np.isfinite(cpu_range[1]):
        if cpu_range[0] == cpu_range[1]:
            cpu_range = (
                cpu_range[0],
                cpu_range[0] + 1
            )

        cpu_edges = np.linspace(
            cpu_range[0],
            cpu_range[1],
            1001
        )

        cpu_hist = np.zeros(
            1000,
            dtype=np.int64
        )

    if np.isfinite(mem_range[0]) and np.isfinite(mem_range[1]):
        if mem_range[0] == mem_range[1]:
            mem_range = (
                mem_range[0],
                mem_range[0] + 1
            )

        mem_edges = np.linspace(
            mem_range[0],
            mem_range[1],
            1001
        )

        mem_hist = np.zeros(
            1000,
            dtype=np.int64
        )

    for chunk in pd.read_csv(
        file_path,
        usecols=USECOLS,
        chunksize=CHUNK_SIZE
    ):

        row_count += len(chunk)

        for column in USECOLS:
            missing_counts[column] += int(
                chunk[column].isna().sum()
            )

        duration = chunk["duration"].dropna()

        zero_duration += int(
            (duration == 0).sum()
        )

        positive_duration += int(
            (duration > 0).sum()
        )

        negative_duration += int(
            (duration < 0).sum()
        )

        if len(duration) > 0:

            duration_sum += duration.sum()
            duration_sum_sq += (
                (duration * duration).sum()
            )

            if duration_hist is not None:
                values = duration.to_numpy()

                hist, _ = np.histogram(
                    values,
                    bins=duration_edges
                )

                duration_hist += hist

        cpu_avg = chunk["cpu_avg"].dropna()

        if len(cpu_avg) > 0:

            cpu_avg_sum += cpu_avg.sum()
            cpu_avg_sum_sq += (
                (cpu_avg * cpu_avg).sum()
            )

            cpu_avg_count += len(cpu_avg)

            if cpu_hist is not None:
                values = cpu_avg.to_numpy()

                hist, _ = np.histogram(
                    values,
                    bins=cpu_edges
                )

                cpu_hist += hist

        cpu_max = chunk["cpu_max"].dropna()

        if len(cpu_max) > 0:

            cpu_max_sum += cpu_max.sum()
            cpu_max_sum_sq += (
                (cpu_max * cpu_max).sum()
            )

            cpu_max_count += len(cpu_max)

        mem_avg = chunk["mem_avg"].dropna()

        if len(mem_avg) > 0:

            mem_avg_sum += mem_avg.sum()
            mem_avg_sum_sq += (
                (mem_avg * mem_avg).sum()
            )

            mem_avg_count += len(mem_avg)

            if mem_hist is not None:
                values = mem_avg.to_numpy()

                hist, _ = np.histogram(
                    values,
                    bins=mem_edges
                )

                mem_hist += hist

        mem_max = chunk["mem_max"].dropna()

        if len(mem_max) > 0:

            mem_max_sum += mem_max.sum()
            mem_max_sum_sq += (
                (mem_max * mem_max).sum()
            )

            mem_max_count += len(mem_max)

        valid_start = chunk["start_time"].dropna()

        valid_end = chunk["end_time"].dropna()

        if len(valid_start) > 0:
            time_min = min(
                time_min,
                valid_start.min()
            )

        if len(valid_end) > 0:
            time_max = max(
                time_max,
                valid_end.max()
            )

        task_values = chunk["task_type"].dropna()

        for value, count in task_values.value_counts().items():
            key = str(value)
            task_type_counts[key] = (
                task_type_counts.get(key, 0) + int(count)
            )

        status_values = chunk["status"].dropna()

        for value, count in status_values.value_counts().items():
            key = str(value)
            status_counts[key] = (
                status_counts.get(key, 0) + int(count)
            )

        cpu_values = chunk["cpu_avg"].to_numpy()
        mem_values = chunk["mem_avg"].to_numpy()

        valid_mask = (
            ~pd.isna(cpu_values)
            & ~pd.isna(mem_values)
        )

        valid_cpu = cpu_values[valid_mask]
        valid_mem = mem_values[valid_mask]

        cpu_heavy_count += int(
            (valid_cpu > valid_mem).sum()
        )

        memory_heavy_count += int(
            (valid_mem > valid_cpu).sum()
        )

        balanced_count += int(
            (valid_cpu == valid_mem).sum()
        )

        del chunk
        gc.collect()

    def calculate_mean(total, count):
        if count == 0:
            return np.nan
        return total / count

    def calculate_std(total, total_sq, count):
        if count <= 1:
            return np.nan

        mean = total / count

        variance = (
            total_sq / count
            - mean * mean
        )

        variance = max(variance, 0)

        return np.sqrt(variance)

    duration_count = (
        zero_duration
        + positive_duration
        + negative_duration
    )

    duration_mean = calculate_mean(
        duration_sum,
        duration_count
    )

    duration_std = calculate_std(
        duration_sum,
        duration_sum_sq,
        duration_count
    )

    cpu_avg_mean = calculate_mean(
        cpu_avg_sum,
        cpu_avg_count
    )

    cpu_avg_std = calculate_std(
        cpu_avg_sum,
        cpu_avg_sum_sq,
        cpu_avg_count
    )

    cpu_max_mean = calculate_mean(
        cpu_max_sum,
        cpu_max_count
    )

    cpu_max_std = calculate_std(
        cpu_max_sum,
        cpu_max_sum_sq,
        cpu_max_count
    )

    mem_avg_mean = calculate_mean(
        mem_avg_sum,
        mem_avg_count
    )

    mem_avg_std = calculate_std(
        mem_avg_sum,
        mem_avg_sum_sq,
        mem_avg_count
    )

    mem_max_mean = calculate_mean(
        mem_max_sum,
        mem_max_count
    )

    mem_max_std = calculate_std(
        mem_max_sum,
        mem_max_sum_sq,
        mem_max_count
    )

    duration_percentiles = percentile_from_histogram(
        duration_hist,
        duration_edges,
        duration_count
    )

    cpu_percentiles = percentile_from_histogram(
        cpu_hist,
        cpu_edges,
        cpu_avg_count
    )

    mem_percentiles = percentile_from_histogram(
        mem_hist,
        mem_edges,
        mem_avg_count
    )

    summary = {
        "split": split_name,
        "row_count": row_count,
        "time_min": time_min,
        "time_max": time_max,
        "duration_mean": duration_mean,
        "duration_std": duration_std,
        "duration_min": duration_range[0],
        "duration_max": duration_range[1],
        "duration_p50": duration_percentiles[0],
        "duration_p75": duration_percentiles[1],
        "duration_p90": duration_percentiles[2],
        "duration_p95": duration_percentiles[3],
        "duration_p99": duration_percentiles[4],
        "cpu_avg_mean": cpu_avg_mean,
        "cpu_avg_std": cpu_avg_std,
        "cpu_avg_min": cpu_range[0],
        "cpu_avg_max": cpu_range[1],
        "cpu_avg_p50": cpu_percentiles[0],
        "cpu_avg_p75": cpu_percentiles[1],
        "cpu_avg_p90": cpu_percentiles[2],
        "cpu_avg_p95": cpu_percentiles[3],
        "cpu_avg_p99": cpu_percentiles[4],
        "cpu_max_mean": cpu_max_mean,
        "cpu_max_std": cpu_max_std,
        "cpu_max_min": cpu_range[0],
        "cpu_max_max": cpu_range[1],
        "mem_avg_mean": mem_avg_mean,
        "mem_avg_std": mem_avg_std,
        "mem_avg_min": mem_range[0],
        "mem_avg_max": mem_range[1],
        "mem_avg_p50": mem_percentiles[0],
        "mem_avg_p75": mem_percentiles[1],
        "mem_avg_p90": mem_percentiles[2],
        "mem_avg_p95": mem_percentiles[3],
        "mem_avg_p99": mem_percentiles[4],
        "mem_max_mean": mem_max_mean,
        "mem_max_std": mem_max_std,
        "mem_max_min": mem_range[0],
        "mem_max_max": mem_range[1],
        "zero_duration": zero_duration,
        "positive_duration": positive_duration,
        "negative_duration": negative_duration,
        "cpu_heavy_count": cpu_heavy_count,
        "memory_heavy_count": memory_heavy_count,
        "balanced_count": balanced_count
    }

    return (
        summary,
        missing_counts,
        task_type_counts,
        status_counts
    )


all_summaries = []
all_missing = []
all_task_types = []
all_statuses = []


for split_name, file_path in FILES.items():

    if not os.path.exists(file_path):
        print("File not found:", file_path)
        continue

    summary, missing, task_types, statuses = analyze_split(
        split_name,
        file_path
    )

    all_summaries.append(summary)

    for column, count in missing.items():
        all_missing.append({
            "split": split_name,
            "column": column,
            "missing_count": count,
            "missing_percent": (
                count / summary["row_count"] * 100
            )
        })

    for task_type, count in task_types.items():
        all_task_types.append({
            "split": split_name,
            "task_type": task_type,
            "count": count
        })

    for status, count in statuses.items():
        all_statuses.append({
            "split": split_name,
            "status": status,
            "count": count
        })


summary_df = pd.DataFrame(
    all_summaries
)

missing_df = pd.DataFrame(
    all_missing
)

task_type_df = pd.DataFrame(
    all_task_types
)

status_df = pd.DataFrame(
    all_statuses
)


summary_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "workload_summary.csv"
    ),
    index=False
)

missing_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "workload_missingness.csv"
    ),
    index=False
)

task_type_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "task_type_distribution.csv"
    ),
    index=False
)

status_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "status_distribution.csv"
    ),
    index=False
)


print("Workload analysis completed")
print("Output directory:", OUTPUT_DIR)
print("Splits analyzed:", len(summary_df))
print("Summary rows:", len(summary_df))
print("Task type records:", len(task_type_df))
print("Status records:", len(status_df))
print("Missingness records:", len(missing_df))