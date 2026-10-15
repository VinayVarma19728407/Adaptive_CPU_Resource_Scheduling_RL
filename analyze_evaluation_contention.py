# =============================================================================
# FILE: analyze_evaluation_contention.py
# ROLE IN PROJECT:
#   A quick post-evaluation diagnostic script. After running an evaluation
#   experiment (which saves task_results.csv), this script reads that CSV and
#   reports:
#     1. CONCURRENT TASK PEAKS — for each (policy, machine) pair, it replays
#        the start/completion events and computes the peak number of tasks that
#        were simultaneously active on that machine.
#     2. TASK STATUS BY POLICY — how many tasks completed, were unschedulable,
#        etc., broken down by policy.
#     3. WAITING TIME AND DURATION CHECK — mean/max wait time, flow time, and
#        mean execution duration per policy.
#
#   Purpose: detect if any policy causes unexpected resource contention (more
#   concurrent tasks than a single machine can hold) or abnormal wait times.
#
#   Input:  experiments/results/ppo_v1_vs_v2_extended_20261012/task_results.csv
#   Output: printed tables to stdout (no files written)
# =============================================================================
import pandas as pd

path = r"experiments\results\ppo_v1_vs_v2_extended_20261012\task_results.csv"
df = pd.read_csv(path)

for col in ["start_time", "completion_time"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("=" * 75)
print("CONCURRENT TASK PEAKS PER POLICY AND MACHINE")
print("=" * 75)

results = []

for (policy, machine), group in df.groupby(["policy", "machine_id"]):
    events = pd.concat([
        pd.DataFrame({
            "time": group["start_time"],
            "delta": 1
        }),
        pd.DataFrame({
            "time": group["completion_time"],
            "delta": -1
        })
    ]).dropna()

    # Process completions before starts at equal timestamps.
    events = events.sort_values(
        ["time", "delta"], kind="stable"
    )
    active = events["delta"].cumsum()

    results.append({
        "policy": policy,
        "machine_id": machine,
        "tasks": len(group),
        "peak_concurrent_tasks": int(active.max()) if len(active) else 0,
        "mean_duration": round(
            (group["completion_time"] - group["start_time"]).mean(), 3
        )
    })

print(pd.DataFrame(results).sort_values(
    ["policy", "machine_id"]
).to_string(index=False))

print("\n" + "=" * 75)
print("TASK STATUS BY POLICY")
print("=" * 75)
print(pd.crosstab(df["policy"], df["status"]).to_string())

print("\n" + "=" * 75)
print("WAITING TIME AND DURATION CHECK")
print("=" * 75)
print(df.groupby("policy").agg(
    task_count=("scenario_task_id", "count"),
    mean_wait=("wait_time", "mean"),
    max_wait=("wait_time", "max"),
    mean_flow=("flow_time", "mean"),
    mean_execution=("baseline_duration", "mean")
).round(3).to_string())