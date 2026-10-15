
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"

CPU_CAPACITY = 96.0       # cores per eligible machine
MEMORY_CAPACITY = 100.0   # normalized memory units per machine
TOLERANCE = 1e-8

REQUIRED_COLUMNS = {
    "scenario_task_id",
    "start_time",
    "completion_time",
    "machine_id",
    "allocated_cpu",
    "allocated_memory",
    "status",
}


def audit_policy(policy: str) -> None:
    """Audit completed-task resource usage for one placement policy."""

    path = RESULTS / f"heavy_{policy}_seed42_results.csv"

    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns in {path.name}: {sorted(missing)}"
        )

    print(f"\n{'=' * 60}")
    print(f"POLICY: {policy}")
    print(f"{'=' * 60}")

    print("Result rows:", len(df))
    print("Unique scenario tasks:", df["scenario_task_id"].nunique())
    print("Machines used:", df["machine_id"].nunique())

    print("\nStatus counts:")
    print(df["status"].value_counts(dropna=False).to_string())

    completed = df[
        df["status"].astype(str).str.strip().str.lower() == "completed"
    ].copy()

    if completed.empty:
        print("\nNo completed tasks available for resource auditing.")
        return

    numeric_columns = [
        "start_time",
        "completion_time",
        "allocated_cpu",
        "allocated_memory",
    ]

    for column in numeric_columns:
        completed[column] = pd.to_numeric(
            completed[column], errors="raise"
        )

    if completed[numeric_columns].isna().any().any():
        raise ValueError(
            "Missing numeric values in completed-task records."
        )

    if completed["machine_id"].isna().any():
        raise ValueError("Completed tasks contain missing machine IDs.")

    if completed["scenario_task_id"].duplicated().any():
        raise ValueError(
            "Duplicate scenario_task_id values found among completed tasks."
        )

    if (
        completed["completion_time"] < completed["start_time"]
    ).any():
        raise ValueError("Found a task completing before it starts.")

    if (completed["allocated_cpu"] < 0).any():
        raise ValueError("Found negative CPU allocation.")

    if (completed["allocated_memory"] < 0).any():
        raise ValueError("Found negative memory allocation.")

    # Zero-duration tasks are counted separately. They contribute no
    # positive-duration resource interval to this time-based audit.
    zero_duration_count = int(
        (completed["completion_time"] == completed["start_time"]).sum()
    )

    machine_summaries = []

    for machine_id, tasks in completed.groupby("machine_id"):
        events = []

        for row in tasks.itertuples(index=False):
            start = float(row.start_time)
            end = float(row.completion_time)
            cpu = float(row.allocated_cpu)
            memory = float(row.allocated_memory)

            if end == start:
                continue

            # Event format:
            # (timestamp, order, CPU delta, memory delta, task delta)
            #
            # At identical timestamps, completions (order 0) are processed
            # before starts (order 1). This models half-open intervals:
            # [start_time, completion_time).
            events.append((start, 1, cpu, memory, 1))
            events.append((end, 0, -cpu, -memory, -1))

        events.sort(key=lambda event: (event[0], event[1]))

        cpu_used = 0.0
        memory_used = 0.0
        running_tasks = 0

        peak_cpu = 0.0
        peak_memory = 0.0
        peak_concurrent = 0

        cpu_violation_events = 0
        memory_violation_events = 0

        for timestamp, order, cpu_delta, memory_delta, task_delta in events:
            cpu_used += cpu_delta
            memory_used += memory_delta
            running_tasks += task_delta

            if cpu_used < -TOLERANCE or memory_used < -TOLERANCE:
                raise ValueError(
                    f"Resource accounting became negative on machine "
                    f"{machine_id} at time {timestamp}. "
                    f"CPU used={cpu_used}, memory used={memory_used}. "
                    "Check result timestamps and task allocations."
                )

            if running_tasks < 0:
                raise ValueError(
                    f"Running-task count became negative on machine "
                    f"{machine_id} at time {timestamp}."
                )

            if cpu_used > CPU_CAPACITY + TOLERANCE:
                cpu_violation_events += 1

            if memory_used > MEMORY_CAPACITY + TOLERANCE:
                memory_violation_events += 1

            peak_cpu = max(peak_cpu, cpu_used)
            peak_memory = max(peak_memory, memory_used)
            peak_concurrent = max(peak_concurrent, running_tasks)

        machine_summaries.append({
            "machine_id": machine_id,
            "tasks_assigned": len(tasks),
            "positive_duration_tasks": int(
                (tasks["completion_time"] > tasks["start_time"]).sum()
            ),
            "zero_duration_tasks": int(
                (tasks["completion_time"] == tasks["start_time"]).sum()
            ),
            "peak_concurrent_tasks": peak_concurrent,
            "peak_cpu_cores": peak_cpu,
            "peak_cpu_utilization_pct": (
                peak_cpu / CPU_CAPACITY * 100.0
            ),
            "peak_memory_units": peak_memory,
            "peak_memory_utilization_pct": (
                peak_memory / MEMORY_CAPACITY * 100.0
            ),
            "cpu_violation_events": cpu_violation_events,
            "memory_violation_events": memory_violation_events,
        })

    summary = pd.DataFrame(machine_summaries)

    if summary.empty:
        print("\nNo machine resource intervals available to audit.")
        return

    summary = summary.sort_values("machine_id")

    print("\nZero-duration completed tasks:", zero_duration_count)

    print("\nPer-machine peak resource usage:")
    print(summary.round(3).to_string(index=False))

    print("\nOverall audit:")
    print(
        "Maximum peak CPU utilization (%):",
        round(summary["peak_cpu_utilization_pct"].max(), 3),
    )
    print(
        "Maximum peak memory utilization (%):",
        round(summary["peak_memory_utilization_pct"].max(), 3),
    )
    print(
        "Maximum concurrent tasks on any machine:",
        int(summary["peak_concurrent_tasks"].max()),
    )
    print(
        "CPU violation events:",
        int(summary["cpu_violation_events"].sum()),
    )
    print(
        "Memory violation events:",
        int(summary["memory_violation_events"].sum()),
    )

    output_path = RESULTS / f"heavy_{policy}_resource_audit.csv"
    summary.to_csv(output_path, index=False)

    print("Audit saved to:", output_path)


if __name__ == "__main__":
    for placement_policy in ["first_fit", "least_loaded"]:
        audit_policy(placement_policy)
