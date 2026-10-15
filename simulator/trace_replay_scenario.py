"""Build a chronological historical-start-time replay scenario.

Caution:
batch_task.start_time is a recorded task start, not a verified
submission timestamp. arrival_time is a historical-start-time proxy.
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "dataset" / "cleaned" / "batch_task.csv"
OUTPUT_DIR = ROOT / "experiments" / "data" / "trace_replay"
OUTPUT = OUTPUT_DIR / "historical_start_proxy_scenario_v2.csv"

TASK_LIMIT = 512
CHUNK_SIZE = 250_000
WINDOW_FRACTION = 0.50  # Begin around the middle of the valid timeline.


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Source dataset not found: {SOURCE}")

    if OUTPUT.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT}\n"
            "Choose a new output filename rather than overwriting it."
        )

    # Pass 1: determine the timestamp range among positive-duration rows.
    min_time = None
    max_time = None
    valid_count = 0

    for chunk in pd.read_csv(
        SOURCE,
        usecols=[
            "task_name", "job_name", "task_type", "status",
            "start_time", "end_time", "plan_cpu", "plan_mem",
        ],
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        start = pd.to_numeric(chunk["start_time"], errors="coerce")
        end = pd.to_numeric(chunk["end_time"], errors="coerce")
        cpu = pd.to_numeric(chunk["plan_cpu"], errors="coerce")
        mem = pd.to_numeric(chunk["plan_mem"], errors="coerce")

        valid = (
            start.notna()
            & end.notna()
            & cpu.notna()
            & mem.notna()
            & (end > start)
            & (cpu >= 0)
            & (mem >= 0)
        )

        if not valid.any():
            continue

        starts = start.loc[valid]
        valid_count += len(starts)

        chunk_min = float(starts.min())
        chunk_max = float(starts.max())

        min_time = chunk_min if min_time is None else min(min_time, chunk_min)
        max_time = chunk_max if max_time is None else max(max_time, chunk_max)

    if valid_count < TASK_LIMIT:
        raise ValueError(
            f"Only {valid_count} positive-duration valid records found."
        )

    target_time = min_time + WINDOW_FRACTION * (max_time - min_time)
    print("Positive-duration valid records:", f"{valid_count:,}")
    print("Valid start-time range:", min_time, "to", max_time)
    print("Target middle timestamp:", target_time)

    # Pass 2: collect valid rows in a bounded window near the target.
    # A window is expanded only if the first pass finds too few records.
    window_width = max((max_time - min_time) * 0.01, 1.0)
    selected = None

    for attempt in range(8):
        lower = max(min_time, target_time - window_width / 2)
        upper = min(max_time, target_time + window_width / 2)
        parts = []

        for chunk in pd.read_csv(
            SOURCE,
            usecols=[
                "task_name", "job_name", "task_type", "status",
                "start_time", "end_time", "plan_cpu", "plan_mem",
            ],
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ):
            start = pd.to_numeric(chunk["start_time"], errors="coerce")
            end = pd.to_numeric(chunk["end_time"], errors="coerce")
            cpu = pd.to_numeric(chunk["plan_cpu"], errors="coerce")
            mem = pd.to_numeric(chunk["plan_mem"], errors="coerce")

            valid = (
                start.notna()
                & end.notna()
                & cpu.notna()
                & mem.notna()
                & (end > start)
                & (cpu >= 0)
                & (mem >= 0)
                & (start >= lower)
                & (start <= upper)
            )

            if valid.any():
                part = chunk.loc[valid].copy()
                part["start_time"] = start.loc[valid].astype(float)
                part["end_time"] = end.loc[valid].astype(float)
                part["plan_cpu"] = cpu.loc[valid].astype(float)
                part["plan_mem"] = mem.loc[valid].astype(float)
                parts.append(part)

        if parts:
            candidate = pd.concat(parts, ignore_index=True)
            candidate = candidate.sort_values(
                ["start_time", "end_time", "job_name", "task_name"],
                kind="mergesort",
            )

            if len(candidate) >= TASK_LIMIT:
                selected = candidate
                break

        window_width *= 2

    if selected is None:
        raise ValueError(
            "Could not find enough valid positive-duration records "
            "near the selected middle timeline. No output was written."
        )

    # Select a contiguous chronological slice, not arbitrary scattered rows.
    center = len(selected) // 2
    begin = max(0, center - TASK_LIMIT // 2)
    scenario = selected.iloc[begin:begin + TASK_LIMIT].copy()

    origin = float(scenario["start_time"].min())
    scenario["arrival_time"] = scenario["start_time"] - origin
    scenario["duration"] = scenario["end_time"] - scenario["start_time"]
    scenario["historical_start_time"] = scenario["start_time"]
    scenario["historical_end_time"] = scenario["end_time"]
    scenario["arrival_time_semantics"] = "historical_start_time_proxy"

    columns = [
        "task_name", "job_name", "task_type", "status",
        "plan_cpu", "plan_mem", "duration", "arrival_time",
        "historical_start_time", "historical_end_time",
        "arrival_time_semantics",
    ]
    scenario = scenario[columns].reset_index(drop=True)

    if len(scenario) != TASK_LIMIT:
        raise AssertionError("Scenario task count differs from TASK_LIMIT.")
    if (scenario["duration"] <= 0).any():
        raise AssertionError("Scenario contains non-positive task durations.")
    if not scenario["arrival_time"].is_monotonic_increasing:
        raise AssertionError("Scenario is not chronologically ordered.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scenario.to_csv(OUTPUT, index=False)

    gaps = scenario["arrival_time"].diff().dropna()

    print("\nSCENARIO BUILD SUCCESSFUL")
    print("Tasks:", len(scenario))
    print("Source start range:", scenario["historical_start_time"].min(),
          "to", scenario["historical_start_time"].max())
    print("Relative time range:", scenario["arrival_time"].min(),
          "to", scenario["arrival_time"].max())
    print("Minimum duration:", scenario["duration"].min())
    print("Median duration:", scenario["duration"].median())
    print("Maximum duration:", scenario["duration"].max())
    print("Simultaneous-start gaps:", int((gaps == 0).sum()))
    print("Positive inter-start gaps:", int((gaps > 0).sum()))
    print("Arrival interpretation: historical start-time proxy only")
    print("Output:", OUTPUT)
    print("Source dataset was not modified.")


if __name__ == "__main__":
    main()