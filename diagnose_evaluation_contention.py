# =============================================================================
# FILE: diagnose_evaluation_contention.py
# ROLE IN PROJECT:
#   A thorough contention-verification script run after evaluation experiments
#   to confirm that no policy exceeded machine resource capacity at any point
#   during the simulation.
#
#   For each (policy, seed, machine_id) group in the evaluation results it:
#     - Replays task start/end events in timestamp order.
#     - Tracks cumulative active CPU and active memory at every event.
#     - Records peak_cpu, peak_memory, and peak_active_tasks.
#     - Flags any machine where peak_cpu > CPU_CAPACITY or
#       peak_memory > MEMORY_CAPACITY (beyond floating-point tolerance).
#
#   Also verifies temporal consistency: completion_time - start_time must
#   match baseline_duration within tolerance.
#
#   Input:  experiments/results/.../task_results.csv (from evaluation runs)
#   Output: printed diagnostic tables (no files written)
# =============================================================================
from pathlib import Path

import pandas as pd

RESULTS_FILE = Path(
    r".\experiments\results\final_dataset_ppo_v1_20261011_a_evaluation_retry_20261012\task_results.csv"
)

CPU_CAPACITY = 96.0
MEMORY_CAPACITY = 100.0
TOLERANCE = 1e-7

df = pd.read_csv(RESULTS_FILE)

required = [
    "policy",
    "evaluation_seed",
    "scenario_task_id",
    "start_time",
    "completion_time",
    "allocated_cpu",
    "allocated_memory",
    "baseline_duration",
]

missing = [column for column in required if column not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

for column in [
    "start_time",
    "completion_time",
    "allocated_cpu",
    "allocated_memory",
    "baseline_duration",
]:
    df[column] = pd.to_numeric(df[column], errors="raise")

print("\nTask timing consistency")
df["simulated_duration"] = (
    df["completion_time"] - df["start_time"]
)
df["duration_difference"] = (
    df["simulated_duration"] - df["baseline_duration"]
)

print(
    df.groupby("policy").agg(
        task_count=("scenario_task_id", "count"),
        max_duration_difference=(
            "duration_difference",
            lambda s: s.abs().max(),
        ),
        negative_duration_count=(
            "simulated_duration",
            lambda s: (s < 0).sum(),
        ),
    ).to_string()
)

print("\nPeak active resource usage by policy and machine")

all_results = []

for (policy, seed), group in df.groupby(
    ["policy", "evaluation_seed"]
):
    for machine_id, machine_tasks in group.groupby("machine_id"):
        events = []

        for row in machine_tasks.itertuples(index=False):
            start = float(row.start_time)
            end = float(row.completion_time)
            cpu = float(row.allocated_cpu)
            memory = float(row.allocated_memory)

            # End events sort before start events at equal timestamps.
            events.append((end, 0, -cpu, -memory))
            events.append((start, 1, cpu, memory))

        events.sort()

        active_cpu = 0.0
        active_memory = 0.0
        peak_cpu = 0.0
        peak_memory = 0.0
        peak_active_tasks = 0
        active_tasks = 0

        for _, event_type, cpu_delta, memory_delta in events:
            active_cpu += cpu_delta
            active_memory += memory_delta

            if event_type == 0:
                active_tasks -= 1
            else:
                active_tasks += 1

            peak_cpu = max(peak_cpu, active_cpu)
            peak_memory = max(peak_memory, active_memory)
            peak_active_tasks = max(peak_active_tasks, active_tasks)

        all_results.append(
            {
                "policy": policy,
                "seed": seed,
                "machine_id": machine_id,
                "tasks": len(machine_tasks),
                "peak_cpu": peak_cpu,
                "peak_memory": peak_memory,
                "peak_active_tasks": peak_active_tasks,
                "cpu_over_capacity": peak_cpu > CPU_CAPACITY + TOLERANCE,
                "memory_over_capacity": (
                    peak_memory > MEMORY_CAPACITY + TOLERANCE
                ),
            }
        )

peaks = pd.DataFrame(all_results)

print(
    peaks.groupby("policy").agg(
        machine_seed_pairs=("machine_id", "count"),
        maximum_peak_cpu=("peak_cpu", "max"),
        maximum_peak_memory=("peak_memory", "max"),
        maximum_active_tasks=("peak_active_tasks", "max"),
        cpu_capacity_violations=("cpu_over_capacity", "sum"),
        memory_capacity_violations=("memory_over_capacity", "sum"),
    ).to_string()
)

print("\nHighest CPU usage cases")
print(
    peaks.sort_values("peak_cpu", ascending=False)
    .groupby("policy")
    .head(3)[
        [
            "policy",
            "seed",
            "machine_id",
            "tasks",
            "peak_cpu",
            "peak_memory",
            "peak_active_tasks",
        ]
    ]
    .to_string(index=False)
)

print("\nHighest memory usage cases")
print(
    peaks.sort_values("peak_memory", ascending=False)
    .groupby("policy")
    .head(3)[
        [
            "policy",
            "seed",
            "machine_id",
            "tasks",
            "peak_cpu",
            "peak_memory",
            "peak_active_tasks",
        ]
    ]
    .to_string(index=False)
)