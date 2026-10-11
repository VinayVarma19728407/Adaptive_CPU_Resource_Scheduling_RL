
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
audit["cpu_exceeds_capacity"] = audit["plan_cpu"] > 96
audit["memory_exceeds_capacity"] = audit["plan_mem"] > 100
audit["individually_infeasible"] = (
    audit["cpu_exceeds_capacity"] | audit["memory_exceeds_capacity"]
)

rejected = audit[audit["status"] == "unschedulable"]
completed = audit[audit["status"] == "completed"]

print("\nFull scenario demand summary:")
print(audit[["plan_cpu", "plan_mem", "duration"]].describe().to_string())

print("\nCapacity comparison:")
print(f"Tasks with CPU demand > 96: {audit['cpu_exceeds_capacity'].sum()}")
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
