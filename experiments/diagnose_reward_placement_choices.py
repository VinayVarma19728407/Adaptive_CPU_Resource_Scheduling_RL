
import json
from pathlib import Path

import pandas as pd

from simulator.rl_environment import RLSchedulingEnvironment


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "experiments" / "evaluation" / (
    "reward_placement_choices_diagnostic_20261014"
)


def build_scenario():
    machines = pd.DataFrame([
        {
            "machine_id": "small",
            "cpu_num": 10,
            "mem_size": 10,
            "status": 1,
        },
        {
            "machine_id": "large",
            "cpu_num": 20,
            "mem_size": 20,
            "status": 1,
        },
    ])

    # plan_cpu is converted to CPU cores by the environment
    # using plan_cpu / 100.0.
    tasks = pd.DataFrame([
        {
            "scenario_task_id": "t0",
            "task_name": "large_task",
            "job_name": "job0",
            "plan_cpu": 800,
            "plan_mem": 8,
            "duration": 20,
            "arrival_time": 0,
        },
        {
            "scenario_task_id": "t1",
            "task_name": "small_task_1",
            "job_name": "job1",
            "plan_cpu": 300,
            "plan_mem": 3,
            "duration": 10,
            "arrival_time": 0,
        },
        {
            "scenario_task_id": "t2",
            "task_name": "small_task_2",
            "job_name": "job2",
            "plan_cpu": 300,
            "plan_mem": 3,
            "duration": 10,
            "arrival_time": 0,
        },
    ])

    return machines, tasks


def run_episode(first_action):
    machines, tasks = build_scenario()

    env = RLSchedulingEnvironment(
        scenario=tasks,
        machine_metadata=machines,
        candidate_count=2,
        reward_mode="extended_penalty",
    )

    records = []

    try:
        observation, info = env.reset(seed=42)
        terminated = False
        truncated = False
        decision = 0

        while not (terminated or truncated):
            masks = env.action_masks()
            valid_actions = [
                i for i, valid in enumerate(masks) if valid
            ]

            if not valid_actions:
                raise RuntimeError(
                    "No valid action available before episode end."
                )

            # Change only the first action. Subsequent decisions
            # use the first valid candidate.
            if decision == 0 and first_action in valid_actions:
                action = first_action
            else:
                action = valid_actions[0]

            observation, reward, terminated, truncated, info = (
                env.step(action)
            )

            components = info.get("penalty_components", {})

            records.append({
                "decision": decision,
                "action_index": action,
                "task_id": info.get("last_task_id"),
                "machine_id": info.get("last_machine_id"),
                "wait_time": info.get("last_wait_time"),
                "reward": reward,
                **{
                    f"penalty_{name}": value
                    for name, value in components.items()
                },
            })

            decision += 1

        results = pd.DataFrame(env.sim.results.values())
        env.sim.machine_state.validate_invariants()

        metrics = {
            "first_action": first_action,
            "decisions": decision,
            "completed_tasks": int(
                (results["status"] == "completed").sum()
            ),
            "unschedulable_tasks": int(
                (results["status"] == "unschedulable").sum()
            ),
            "total_wait_time": float(results["wait_time"].sum()),
            "mean_wait_time": float(results["wait_time"].mean()),
            "makespan": float(env.sim.current_time),
            "episode_reward": float(
                sum(row["reward"] for row in records)
            ),
            "resource_invariants_passed": True,
            "active_allocations_at_end": (
                env.sim.machine_state.active_allocation_count
            ),
        }

        return metrics, records, results

    finally:
        env.close()


def main():
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {OUTPUT_DIR}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    summaries = []
    all_records = []
    all_task_results = []

    for first_action in (0, 1):
        metrics, records, results = run_episode(first_action)
        summaries.append(metrics)

        for row in records:
            all_records.append({
                "first_action": first_action,
                **row,
            })

        for row in results.to_dict(orient="records"):
            all_task_results.append({
                "first_action": first_action,
                **row,
            })

    pd.DataFrame(summaries).to_csv(
        OUTPUT_DIR / "episode_comparison.csv", index=False
    )
    pd.DataFrame(all_records).to_csv(
        OUTPUT_DIR / "step_reward_components.csv", index=False
    )
    pd.DataFrame(all_task_results).to_csv(
        OUTPUT_DIR / "task_results.csv", index=False
    )

    with (OUTPUT_DIR / "diagnostic_config.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "reward_mode": "extended_penalty",
                "seed": 42,
                "first_actions_compared": [0, 1],
                "machine_count": 2,
                "task_count": 3,
                "purpose": (
                    "Compare two initial feasible placement choices "
                    "in a controlled heterogeneous scenario."
                ),
                "limitations": (
                    "Only the first action is varied. This is a small "
                    "diagnostic, not a statistical benchmark or proof "
                    "of PPO performance."
                ),
            },
            f,
            indent=2,
        )

    print("\nREWARD PLACEMENT DIAGNOSTIC")
    print(pd.DataFrame(summaries).to_string(index=False))
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
