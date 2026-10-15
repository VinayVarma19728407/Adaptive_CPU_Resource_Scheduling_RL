from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "experiments" / "data" / "trace_replay" / "historical_start_proxy_scenario_v2.csv"


def main():
    df = pd.read_csv(INPUT)

    required = {"arrival_time", "duration", "plan_cpu", "plan_mem"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df[
        (df["duration"] > 0)
        & df[["arrival_time", "duration", "plan_cpu", "plan_mem"]]
        .notna()
        .all(axis=1)
    ].copy()

    df["cpu_units"] = df["plan_cpu"] / 100.0
    events = []

    for row in df.itertuples(index=False):
        start = float(row.arrival_time)
        end = start + float(row.duration)

        # End events sort before start events at equal timestamps.
        events.append((end, 0, -1, -row.cpu_units, -row.plan_mem))
        events.append((start, 1, 1, row.cpu_units, row.plan_mem))

    events.sort()

    active = 0
    cpu = 0.0
    memory = 0.0
    peak_tasks = 0
    peak_cpu = 0.0
    peak_memory = 0.0
    peak_time = None
    snapshots = []

    i = 0
    while i < len(events):
        time = events[i][0]

        while i < len(events) and events[i][0] == time:
            _, _, task_delta, cpu_delta, mem_delta = events[i]
            active += task_delta
            cpu += cpu_delta
            memory += mem_delta
            i += 1

        snapshots.append((time, active, cpu, memory))

        if active > peak_tasks:
            peak_tasks = active
            peak_time = time

        peak_cpu = max(peak_cpu, cpu)
        peak_memory = max(peak_memory, memory)

    result = pd.DataFrame(
        snapshots,
        columns=["time", "concurrent_tasks", "cpu_units", "memory_units"],
    )

    print("Scenario:", INPUT)
    print("Valid positive-duration tasks:", len(df))
    print("Peak concurrent tasks:", peak_tasks, "at time", peak_time)
    print("Peak summed CPU demand:", round(peak_cpu, 3), "of 384000 fleet units")
    print("Peak summed memory demand:", round(peak_memory, 3), "of 400000 fleet units")
    print("Concurrency quantiles:")
    print(result["concurrent_tasks"].quantile([0.50, 0.90, 0.95, 0.99]).to_string())
    print("\nTop 10 timestamps by concurrent task count:")
    print(
        result.sort_values("concurrent_tasks", ascending=False)
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()