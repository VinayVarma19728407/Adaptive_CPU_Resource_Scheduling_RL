from __future__ import annotations

from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO

from simulator.config import PROJECT_ROOT
from simulator.environment import SchedulingEnvironment
from simulator.machine_state import MachineStateManager
from simulator.rl_environment import RLSchedulingEnvironment
from simulator.task_data import load_machine_metadata


def create_controlled_workload():
    """Return a small workload with genuine resource contention."""
    machines = pd.DataFrame([
        {"machine_id": "A", "cpu_num": 10, "mem_size": 10, "status": 1},
        {"machine_id": "B", "cpu_num": 6, "mem_size": 10, "status": 1},
    ])

    scenario = pd.DataFrame([
        {
            "scenario_task_id": "task_1",
            "task_name": "task_1",
            "job_name": "job_1",
            "plan_cpu": 600,
            "plan_mem": 4,
            "duration": 10,
            "arrival_time": 0,
        },
        {
            "scenario_task_id": "task_2",
            "task_name": "task_2",
            "job_name": "job_2",
            "plan_cpu": 500,
            "plan_mem": 4,
            "duration": 1,
            "arrival_time": 0,
        },
        {
            "scenario_task_id": "task_3",
            "task_name": "task_3",
            "job_name": "job_3",
            "plan_cpu": 800,
            "plan_mem": 4,
            "duration": 1,
            "arrival_time": 0,
        },
    ])

    return machines, scenario


def summarize_results(name, outcome, results):
    completed = results[results["status"] == "completed"]
    waits = completed["wait_time"].dropna()
    flows = completed["flow_time"].dropna()

    return {
        "policy": name,
        "tasks": len(results),
        "completed": len(completed),
        "unschedulable": int((results["status"] == "unschedulable").sum()),
        "completion_rate": len(completed) / len(results),
        "mean_wait_time": float(waits.mean()) if len(waits) else 0.0,
        "total_wait_time": float(waits.sum()) if len(waits) else 0.0,
        "mean_flow_time": float(flows.mean()) if len(flows) else 0.0,
        "makespan": float(outcome["final_time"]),
    }


def run_baseline(scenario, machine_metadata, policy):
    manager = MachineStateManager(machine_metadata)
    env = SchedulingEnvironment(
        scenario=scenario,
        machine_state=manager,
        placement_policy=policy,
    )

    outcome = env.run()
    manager.validate_invariants()

    return summarize_results(policy, outcome, outcome["results"])


def run_ppo(scenario, machine_metadata, model):
    env = RLSchedulingEnvironment(
        scenario=scenario,
        machine_metadata=machine_metadata,
        candidate_count=2,
    )

    observation, info = env.reset(seed=42)
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)

    sim = env.sim
    assert sim is not None
    sim.machine_state.validate_invariants()

    results = pd.DataFrame(sim.results.values())
    return summarize_results("PPO", {
        "final_time": sim.current_time,
    }, results)


def main():
    model_path = (
        PROJECT_ROOT / "experiments" / "models" / "ppo_smoke_test.zip"
    )
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained smoke-test model not found: {model_path}"
        )

    model = PPO.load(str(model_path), device="cpu")

    # Controlled test uses two machines and the two-candidate smoke-test model.
    machines, scenario = create_controlled_workload()

    records = [
        run_baseline(scenario, machines, "first_fit"),
        run_baseline(scenario, machines, "least_loaded"),
        run_ppo(scenario, machines, model),
    ]

    report = pd.DataFrame(records)

    output_dir = PROJECT_ROOT / "experiments" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "controlled_policy_benchmark.csv"
    report.to_csv(output_path, index=False)

    print("\nCONTROLLED POLICY BENCHMARK")
    print("=" * 75)
    print(report.to_string(index=False))
    print(f"\nSaved benchmark: {output_path}")


if __name__ == "__main__":
    main()