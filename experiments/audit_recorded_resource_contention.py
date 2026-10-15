import pandas as pd
from pathlib import Path

p = Path(r"experiments\results\stress_4000_machine_high_contention_1000_tasks_20261013\task_results.csv")
df = pd.read_csv(p)
df = df[df["status"].eq("completed")].copy()

print("RESOURCE CONTENTION AUDIT")
print("=" * 70)

for policy, group in df.groupby("policy"):
    print(f"\nPolicy: {policy}")
    print(f"Completed tasks: {len(group)}")
    print(f"Tasks with positive recorded wait: {(group['wait_time'] > 1e-9).sum()}")
    print(f"Maximum recorded wait: {group['wait_time'].max():.6f}")

    peak_tasks = 0
    peak_cpu = 0.0
    peak_mem = 0.0
    machines_with_overlap = 0

    for machine_id, mg in group.groupby("machine_id"):
        events = []
        for r in mg.itertuples(index=False):
            events.append((r.start_time, 1, r.allocated_cpu, r.allocated_memory))
            events.append((r.completion_time, -1, r.allocated_cpu, r.allocated_memory))

        # Process completions before starts at identical timestamps.
        events.sort(key=lambda e: (e[0], e[1]))

        active = 0
        cpu = 0.0
        mem = 0.0
        machine_peak_tasks = 0
        machine_peak_cpu = 0.0
        machine_peak_mem = 0.0

        for _, kind, c, m in events:
            active += kind
            cpu += kind * c
            mem += kind * m
            machine_peak_tasks = max(machine_peak_tasks, active)
            machine_peak_cpu = max(machine_peak_cpu, cpu)
            machine_peak_mem = max(machine_peak_mem, mem)

        peak_tasks = max(peak_tasks, machine_peak_tasks)
        peak_cpu = max(peak_cpu, machine_peak_cpu)
        peak_mem = max(peak_mem, machine_peak_mem)
        if machine_peak_tasks > 1:
            machines_with_overlap += 1

    print(f"Machines used: {group['machine_id'].nunique()}")
    print(f"Machines with overlapping tasks: {machines_with_overlap}")
    print(f"Peak concurrent tasks on one machine: {peak_tasks}")
    print(f"Peak per-machine CPU demand: {peak_cpu:.6f}")
    print(f"Peak per-machine memory demand: {peak_mem:.6f}")
    print(f"Maximum start-arrival difference: {(group['start_time'] - group['arrival_time']).max():.6f}")
