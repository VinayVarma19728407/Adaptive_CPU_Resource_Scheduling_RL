from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = (
    ROOT / "experiments" / "results"
    / "trace_replay_2machine_diagnostic_20261013"
    / "task_results.csv"
)

df = pd.read_csv(INPUT)

for col in [
    "start_time", "completion_time",
    "allocated_cpu", "allocated_memory"
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df[
    df["status"].eq("completed")
    & df["start_time"].notna()
    & df["completion_time"].notna()
    & (df["completion_time"] > df["start_time"])
].copy()

for policy, group in df.groupby("policy"):
    events = []

    for row in group.itertuples(index=False):
        # Half-open intervals: resources are released at completion.
        events.append((
            row.start_time, 1,
            row.machine_id, row.allocated_cpu,
            row.allocated_memory
        ))
        events.append((
            row.completion_time, 0,
            row.machine_id, row.allocated_cpu,
            row.allocated_memory
        ))

    # Release resources before allocating at equal timestamps.
    events.sort(key=lambda x: (x[0], x[1]))

    usage = {}
    peak_cpu = 0.0
    peak_memory = 0.0
    peak_cpu_machine = None
    peak_memory_machine = None
    peak_cpu_time = None
    peak_memory_time = None

    for time, kind, machine, cpu, memory in events:
        state = usage.setdefault(machine, [0.0, 0.0])

        if kind == 0:
            state[0] -= cpu
            state[1] -= memory
        else:
            state[0] += cpu
            state[1] += memory

        if state[0] > peak_cpu:
            peak_cpu = state[0]
            peak_cpu_machine = machine
            peak_cpu_time = time

        if state[1] > peak_memory:
            peak_memory = state[1]
            peak_memory_machine = machine
            peak_memory_time = time

    print(f"\nPolicy: {policy}")
    print(f"Peak CPU allocation on one machine: {peak_cpu:.4f}")
    print(f"CPU peak machine/time: {peak_cpu_machine} / {peak_cpu_time}")
    print(f"Peak memory allocation on one machine: {peak_memory:.4f}")
    print(
        "Memory peak machine/time: "
        f"{peak_memory_machine} / {peak_memory_time}"
    )
    print("Machine capacity: CPU=96, memory=100")