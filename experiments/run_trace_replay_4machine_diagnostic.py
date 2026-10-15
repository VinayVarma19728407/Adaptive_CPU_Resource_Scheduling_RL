"""Compare deterministic schedulers on the historical-start-time proxy
using the same seeded eight-machine selection as the existing PPO setup.
Does not modify existing model or evaluation artifacts.
"""

import json
from pathlib import Path

import pandas as pd

from simulator.config import PROJECT_ROOT, RANDOM_SEED
from simulator.machine_state import MachineStateManager
from simulator.task_data import load_machine_metadata
from simulator.environment import SchedulingEnvironment

ROOT = Path(__file__).resolve().parents[1]

SCENARIO_FILE = (
    PROJECT_ROOT / "experiments" / "data" / "trace_replay"
    / "historical_start_proxy_scenario_v2.csv"
)

OUTPUT_DIR = ROOT / "experiments" / "results" / "trace_replay_4machine_diagnostic_20261013"

MACHINE_COUNT = 4
TASK_LIMIT = 512


def main():
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output exists; choose a new directory: {OUTPUT_DIR}"
        )

    scenario = pd.read_csv(SCENARIO_FILE)

    required = {
        "task_name", "job_name", "plan_cpu",
        "plan_mem", "duration", "arrival_time",
    }
    missing = required - set(scenario.columns)
    if missing:
        raise ValueError(f"Missing scenario columns: {sorted(missing)}")
    if len(scenario) != TASK_LIMIT:
        raise ValueError(f"Expected {TASK_LIMIT} tasks, got {len(scenario)}")

    scenario.insert(
        0, "scenario_task_id",
        [f"trace_task_{i:06d}" for i in range(len(scenario))]
    )

    all_machines = load_machine_metadata()
    if len(all_machines) < MACHINE_COUNT:
        raise ValueError("Fewer than eight eligible machines are available.")

    # Match the existing training script's selection method and seed.
    machines = (
        all_machines.sample(n=MACHINE_COUNT, random_state=RANDOM_SEED)
        .sort_values("machine_id")
        .reset_index(drop=True)
    )

    manager = MachineStateManager(machines)
    print("Eight-machine historical-workload preflight")
    print("Tasks:", len(scenario))
    print("Machines:", len(machines))
    print("Seed:", RANDOM_SEED)
    print("Arrival semantics: historical start-time proxy")

    policies = ["first_fit", "least_loaded"]
    metric_rows = []
    result_frames = []

    for policy in policies:
        # Each policy must receive a fresh machine state.
        fresh_manager = MachineStateManager(machines)
        env = SchedulingEnvironment(
            scenario=scenario,
            machine_state=fresh_manager,
            placement_policy=policy,
        )
        outcome = env.run()

        metrics = dict(outcome["metrics"])
        metrics["policy"] = policy
        metrics["final_time"] = outcome["final_time"]
        metrics["machine_count"] = len(machines)
        metric_rows.append(metrics)

        frame = outcome["results"].copy()
        frame["policy"] = policy
        result_frames.append(frame)

        print(f"\nPolicy: {policy}")
        for key in (
            "tasks_completed", "tasks_unschedulable",
            "total_wait_time", "mean_wait_time",
            "mean_flow_time", "makespan", "completion_rate",
        ):
            print(f"{key}: {metrics.get(key)}")

        fresh_manager.validate_invariants()
        if fresh_manager.active_allocation_count != 0:
            raise AssertionError("Resources remain allocated after simulation.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(metric_rows).to_csv(
        OUTPUT_DIR / "policy_metrics.csv", index=False
    )
    pd.concat(result_frames, ignore_index=True).to_csv(
        OUTPUT_DIR / "task_results.csv", index=False
    )

    config = {
        "scenario_file": str(SCENARIO_FILE),
        "task_count": len(scenario),
        "machine_count": len(machines),
        "machine_selection": "sample without replacement",
        "seed": RANDOM_SEED,
        "arrival_semantics": "historical_start_time_proxy",
        "policies": policies,
        "ppo_trained": False,
        "existing_artifacts_modified": False,
    }
    with open(OUTPUT_DIR / "preflight_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("\nResults saved to:", OUTPUT_DIR)
    print(pd.DataFrame(metric_rows).to_string(index=False))


if __name__ == "__main__":
    main()