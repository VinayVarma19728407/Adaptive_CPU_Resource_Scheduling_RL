"""
============================================================
File: audit_light_results.py
Module: tests (Data Integrity & Audit Scripts)
============================================================
Standalone audit script that cross-validates the baseline simulation
results for the "light" scenario (seed 42) against its originating
scenario file. It ensures that every task present in the scenario CSV
has a corresponding result row and vice versa, verifies that per-task
resource demands are internally consistent, and checks whether tasks
that were marked "unschedulable" actually exceed the machine capacity
limits established by the machine metadata audit (96 CPU cores,
100 memory units).

This script is intended to be run directly (not via a test framework)
to produce a human-readable audit report. It raises ValueError early
if any data-integrity problem is detected so that the audit cannot
silently pass on bad data.

Key Components:
  - Merge validation: outer-joins results with scenario on
    `scenario_task_id` and asserts a perfect one-to-one match.
  - Capacity feasibility check: flags tasks whose CPU demand (converted
    from hundredths of a core) exceeds 96 cores or whose memory demand
    exceeds 100 normalized units.
  - Rejected-task verification: reports how many "unschedulable" tasks
    genuinely exceed capacity vs. those that should have been feasible.
  - Longest-duration summary: prints the 10 longest tasks by duration.

Inputs:
  - experiments/results/light_baseline_seed42_results.csv
      Simulation output CSV with columns including `scenario_task_id`,
      `status`, and `reason`.
  - experiments/scenarios/light_scenario_seed42.csv
      Scenario definition CSV with columns `scenario_task_id`,
      `plan_cpu`, `plan_mem`, and `duration`.

Outputs:
  - Human-readable audit report printed to stdout.
  - Raises ValueError on any detected data inconsistency.
  - Does NOT modify any source data or simulation result files.
============================================================
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

results_path = (
    ROOT / "experiments" / "results"
    / "light_baseline_seed42_results.csv"
)
scenario_path = (
    ROOT / "experiments" / "scenarios"
    / "light_scenario_seed42.csv"
)

results = pd.read_csv(results_path)
scenario = pd.read_csv(scenario_path)

print("LIGHT SCENARIO DEMAND AUDIT")
print("-" * 60)

required = {
    "scenario_task_id", "plan_cpu", "plan_mem", "duration"
}
missing = required - set(scenario.columns)
if missing:
    raise ValueError(f"Scenario file is missing columns: {sorted(missing)}")

if not results["scenario_task_id"].is_unique:
    raise ValueError("Duplicate task IDs in results.")

if not scenario["scenario_task_id"].is_unique:
    raise ValueError("Duplicate task IDs in scenario.")

audit = results.merge(
    scenario[
        ["scenario_task_id", "plan_cpu", "plan_mem", "duration"]
    ],
    on="scenario_task_id",
    how="outer",
    validate="one_to_one",
    indicator=True,
)

print("Merge status:")
print(audit["_merge"].value_counts().to_string())

if not (audit["_merge"] == "both").all():
    raise ValueError("Scenario and result task IDs do not match.")

for column in ["plan_cpu", "plan_mem", "duration"]:
    audit[column] = pd.to_numeric(audit[column], errors="coerce")

if audit[["plan_cpu", "plan_mem", "duration"]].isna().any().any():
    raise ValueError("Missing or invalid task demand/duration values.")

# All eligible machines currently have capacities of 96 CPU and
# 100 memory units, according to the machine metadata audit.
# Alibaba trace plan_cpu uses hundredths of a CPU core.
audit["plan_cpu_cores"] = audit["plan_cpu"] / 100.0
# Machine CPU capacity is expressed in cores.
audit["cpu_exceeds_capacity"] = audit["plan_cpu_cores"] > 96
# Memory remains on the trace's normalized scale.
audit["memory_exceeds_capacity"] = audit["plan_mem"] > 100
audit["individually_infeasible"] = (
    audit["cpu_exceeds_capacity"] | audit["memory_exceeds_capacity"]
)

rejected = audit[audit["status"] == "unschedulable"]
completed = audit[audit["status"] == "completed"]

print("\nFull scenario demand summary:")
print(audit[["plan_cpu", "plan_mem", "duration"]].describe().to_string())

print("\nCapacity comparison:")
print(
    "Tasks with CPU demand > 96 cores:",
    int(audit["cpu_exceeds_capacity"].sum()),
)
print(f"Tasks with memory demand > 100: {audit['memory_exceeds_capacity'].sum()}")
print(f"Tasks exceeding at least one capacity: {audit['individually_infeasible'].sum()}")

print("\nRejected-task verification:")
print(f"Rejected tasks: {len(rejected)}")
print(
    "Rejected tasks exceeding at least one capacity:",
    int(rejected["individually_infeasible"].sum()),
)
print(
    "Rejected tasks that fit both capacities:",
    int((~rejected["individually_infeasible"]).sum()),
)

print("\nCompleted-task verification:")
print(f"Completed tasks: {len(completed)}")
print(
    "Completed tasks exceeding at least one capacity:",
    int(completed["individually_infeasible"].sum()),
)

print("\nRejected task demand summary:")
if rejected.empty:
    print("No rejected tasks.")
else:
    print(
        rejected[
            [
                "scenario_task_id", "plan_cpu", "plan_mem",
                "duration", "reason"
            ]
        ].head(20).to_string(index=False)
    )

print("\nLongest-duration tasks:")
print(
    audit[
        [
            "scenario_task_id", "plan_cpu", "plan_mem",
            "duration", "status"
        ]
    ]
    .sort_values("duration", ascending=False)
    .head(10)
    .to_string(index=False)
)

print("\nAudit complete. No source data or simulation results were modified.")
