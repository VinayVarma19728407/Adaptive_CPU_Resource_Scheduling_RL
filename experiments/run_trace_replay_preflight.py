"""Preflight the historical-start-time proxy workload.

Uses the existing deterministic simulator. Does not train PPO or modify
existing model/evaluation artifacts.
"""

from pathlib import Path
import json

import sys
from pathlib import Path
import json

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulator.environment import SchedulingEnvironment


SCENARIO_FILE = (
    PROJECT_ROOT
    / "experiments"
    / "data"
    / "trace_replay"
    / "historical_start_proxy_scenario_v2.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "trace_replay_preflight_20261012"
)

TASK_LIMIT = 512


def main():
    if not SCENARIO_FILE.is_file():
        raise FileNotFoundError(f"Scenario not found: {SCENARIO_FILE}")

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output directory already exists: {OUTPUT_DIR}. "
            "Choose a new output directory rather than overwriting results."
        )

    scenario = pd.read_csv(SCENARIO_FILE)

    required = {
        "task_name",
        "job_name",
        "plan_cpu",
        "plan_mem",
        "duration",
        "arrival_time",
    }
    missing = required - set(scenario.columns)
    if missing:
        raise ValueError(f"Scenario is missing columns: {sorted(missing)}")

    if len(scenario) != TASK_LIMIT:
        raise ValueError(
            f"Expected {TASK_LIMIT} tasks, found {len(scenario)}."
        )

    # The simulator requires unique task IDs even if source task names repeat.
    scenario.insert(
        0,
        "scenario_task_id",
        [f"trace_task_{i:06d}" for i in range(len(scenario))],
    )

    # Keep only the columns required by SchedulingEnvironment.
    simulator_scenario = scenario[
        [
            "scenario_task_id",
            "task_name",
            "job_name",
            "plan_cpu",
            "plan_mem",
            "duration",
            "arrival_time",
        ]
    ].copy()

    print("=" * 70)
    print("TRACE-REPLAY PREFLIGHT")
    print("=" * 70)
    print("Tasks:", len(simulator_scenario))
    print("Arrival interpretation: historical start-time proxy")
    print("CPU conversion in simulator: plan_cpu / 100")
    print("Memory conversion in simulator: plan_mem")

    # Use the simulator's own machine-loading behavior and placement logic.
    # Separate fresh environments ensure policies start from identical state.
    policies = ["first_fit", "least_loaded"]
    metrics_rows = []
    result_frames = []

    for policy in policies:
        print(f"\nRunning policy: {policy}")

        env = SchedulingEnvironment(
            scenario=simulator_scenario,
            placement_policy=policy,
        )
        outcome = env.run()

        metrics = dict(outcome["metrics"])
        metrics["policy"] = policy
        metrics["final_time"] = outcome["final_time"]
        metrics_rows.append(metrics)

        results = outcome["results"].copy()
        results["policy"] = policy
        result_frames.append(results)

        print("Completed:", metrics["tasks_completed"])
        print("Unschedulable:", metrics["tasks_unschedulable"])
        print("Mean wait:", round(metrics["mean_wait_time"], 6))
        print("Total wait:", round(metrics["total_wait_time"], 6))
        print("Mean flow:", round(metrics["mean_flow_time"], 6))
        print("Makespan:", round(metrics["makespan"], 6))
        print("Completion rate:", round(metrics["completion_rate"], 6))

    metrics_df = pd.DataFrame(metrics_rows)
    results_df = pd.concat(result_frames, ignore_index=True)

    # Create outputs only after both policy runs succeed.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    metrics_df.to_csv(OUTPUT_DIR / "policy_metrics.csv", index=False)
    results_df.to_csv(OUTPUT_DIR / "task_results.csv", index=False)

    metadata = {
        "scenario_file": str(SCENARIO_FILE),
        "task_count": len(simulator_scenario),
        "arrival_semantics": "historical_start_time_proxy",
        "policies": policies,
        "ppo_trained": False,
        "source_dataset_modified": False,
        "existing_ppo_artifacts_modified": False,
    }

    with open(
        OUTPUT_DIR / "preflight_config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2)

    print("\nPREFLIGHT RESULTS SAVED")
    print(OUTPUT_DIR)
    print("\nPolicy comparison:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()