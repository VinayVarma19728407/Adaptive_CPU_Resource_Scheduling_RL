
from pathlib import Path

import pandas as pd

from simulator.config import (
    PROJECT_ROOT,
    RANDOM_SEED,
    REQUIRE_CPU_DEMAND,
    REQUIRE_MEMORY_DEMAND,
    REQUIRE_BASELINE_DURATION,
)
from simulator.task_data import sample_task_records


# Cache the sampled task pool so batch_task.csv is scanned only once
# during initial pool creation.
CACHE_DIR = PROJECT_ROOT / "experiments" / "data"
TASK_POOL_CACHE = CACHE_DIR / "task_pool_seed42.csv"

REQUIRED_NUMERIC_FIELDS = [
    "plan_cpu",
    "plan_mem",
    "duration",
]


def load_or_create_task_pool(
    sample_size: int = 10_000,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Load a cached task sample or create it once.

    This is a candidate pool, not yet a complete workload scenario.
    Arrival times and scenario-specific task selection are handled later.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if TASK_POOL_CACHE.exists():
        print(f"Loading cached task pool: {TASK_POOL_CACHE}")
        pool = pd.read_csv(TASK_POOL_CACHE)
    else:
        print("Creating task pool from batch_task.csv.")
        print("The first run scans the source file in chunks.")

        pool = sample_task_records(
            sample_size=sample_size,
            seed=seed,
        )

        pool.to_csv(TASK_POOL_CACHE, index=False)
        print(f"Cached task pool: {TASK_POOL_CACHE}")

    return pool


def validate_task_pool(pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (eligible_tasks, exclusions).

    Missing or invalid demand/duration is not silently replaced.
    Negative demand or negative duration is rejected.
    Zero duration is preserved as potentially valid.
    """
    required_columns = [
        "task_name",
        "job_name",
        "plan_cpu",
        "plan_mem",
        "duration",
    ]

    missing_columns = set(required_columns) - set(pool.columns)
    if missing_columns:
        raise ValueError(
            f"Task pool is missing required columns: {sorted(missing_columns)}"
        )

    data = pool.copy()

    for col in REQUIRED_NUMERIC_FIELDS:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    reasons = []

    for idx, row in data.iterrows():
        row_reasons = []

        if REQUIRE_CPU_DEMAND:
            if pd.isna(row["plan_cpu"]):
                row_reasons.append("missing_cpu_demand")
            elif row["plan_cpu"] <= 0:
                row_reasons.append("nonpositive_cpu_demand")

        if REQUIRE_MEMORY_DEMAND:
            if pd.isna(row["plan_mem"]):
                row_reasons.append("missing_memory_demand")
            elif row["plan_mem"] <= 0:
                row_reasons.append("nonpositive_memory_demand")

        if REQUIRE_BASELINE_DURATION:
            if pd.isna(row["duration"]):
                row_reasons.append("missing_duration")
            elif row["duration"] < 0:
                row_reasons.append("negative_duration")

        reasons.append((idx, row_reasons))

    reason_map = dict(reasons)
    data["_exclusion_reasons"] = [
        ";".join(reason_map[idx]) for idx in data.index
    ]

    eligible = data[data["_exclusion_reasons"] == ""].copy()
    exclusions = data[data["_exclusion_reasons"] != ""].copy()

    eligible = eligible.drop(columns=["_exclusion_reasons"])
    exclusions = exclusions.rename(
        columns={"_exclusion_reasons": "exclusion_reason"}
    )

    return eligible.reset_index(drop=True), exclusions.reset_index(drop=True)



def generate_initial_scenarios(
    eligible_pool: pd.DataFrame,
    tasks_per_scenario: int = 500,
    seed: int = RANDOM_SEED,
) -> None:
    """
    Generate reproducible initial workloads with controlled arrivals.

    Arrival intervals are relative to the median baseline duration
    in the eligible task pool. They are experimental settings, not
    estimates of real-world arrival behavior.
    """
    import numpy as np

    if eligible_pool.empty:
        raise ValueError("Cannot generate scenarios from an empty task pool.")

    if tasks_per_scenario <= 0:
        raise ValueError("tasks_per_scenario must be positive.")

    if len(eligible_pool) < tasks_per_scenario:
        raise ValueError(
            f"Requested {tasks_per_scenario} tasks, but only "
            f"{len(eligible_pool)} eligible records are available."
        )

    durations = pd.to_numeric(
        eligible_pool["duration"], errors="coerce"
    ).dropna()

    durations = durations[durations >= 0]

    if durations.empty:
        raise ValueError("No valid baseline durations are available.")

    median_duration = float(durations.median())

    # Use at least one simulation-time unit between arrivals.
    reference_interval = max(1, int(round(median_duration)))

    intervals = {
        "light": max(1, 2 * reference_interval),
        "moderate": reference_interval,
        "heavy": max(1, reference_interval // 2),
    }

    output_dir = PROJECT_ROOT / "experiments" / "scenarios"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nSCENARIO GENERATION")
    print("-" * 60)
    print(f"Eligible source pool: {len(eligible_pool):,}")
    print(f"Tasks per scenario: {tasks_per_scenario:,}")
    print(f"Median baseline duration: {median_duration:g}")
    print(f"Reference inter-arrival interval: {reference_interval}")

    # The same selected task records are used across intensity levels,
    # enabling fair comparisons of the arrival-rate effect.
    selected = eligible_pool.sample(
        n=tasks_per_scenario,
        replace=False,
        random_state=seed,
    ).reset_index(drop=True)

    for scenario_index, (name, interval) in enumerate(intervals.items()):
        scenario = selected.copy()

        # Give every scenario row its own identity. task_name alone
        # is not assumed to be unique.
        scenario.insert(
            0,
            "scenario_task_id",
            [f"{name}_{i:06d}" for i in range(len(scenario))],
        )

        scenario["arrival_time"] = (
            np.arange(len(scenario), dtype="int64") * interval
        )

        scenario["scenario_name"] = name
        scenario["scenario_seed"] = seed
        scenario["inter_arrival_interval"] = interval

        path = output_dir / f"{name}_scenario_seed{seed}.csv"
        scenario.to_csv(path, index=False)

        print(f"\n{name.upper()} SCENARIO")
        print(f"Inter-arrival interval: {interval}")
        print(f"Tasks: {len(scenario):,}")
        print(f"Final arrival time: {scenario['arrival_time'].iloc[-1]}")
        print(f"Saved: {path}")

    print("\nAll scenarios generated successfully.")
    print("Arrival times are synthetic simulation-time values.")
    print("No source dataset was modified.")


def main():
    pool = load_or_create_task_pool()

    eligible, exclusions = validate_task_pool(pool)

    print("\nTASK POOL VALIDATION")
    print("-" * 60)
    print(f"Candidate task records: {len(pool):,}")
    print(f"Eligible task records: {len(eligible):,}")
    print(f"Excluded task records: {len(exclusions):,}")

    if not exclusions.empty:
        print("\nExclusion reasons:")
        print(
            exclusions["exclusion_reason"]
            .str.split(";")
            .explode()
            .value_counts()
            .to_string()
        )

    eligible_path = CACHE_DIR / "eligible_task_pool_seed42.csv"
    exclusions_path = CACHE_DIR / "excluded_task_pool_seed42.csv"

    eligible.to_csv(eligible_path, index=False)
    exclusions.to_csv(exclusions_path, index=False)

    generate_initial_scenarios(
        eligible_pool=eligible,
        tasks_per_scenario=500,
        seed=RANDOM_SEED,
    )


if __name__ == "__main__":
    main()
