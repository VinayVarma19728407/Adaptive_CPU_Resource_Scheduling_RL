
from __future__ import annotations

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from simulator.config import PROJECT_ROOT
from simulator.environment import SchedulingEnvironment
from simulator.machine_state import MachineStateManager
from simulator.rl_environment import RLSchedulingEnvironment


SEEDS = list(range(100, 120))
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "results"
MODEL_PATH = PROJECT_ROOT / "experiments" / "models" / "ppo_smoke_test.zip"


def create_machines() -> pd.DataFrame:
    """Create a fresh, identical two-machine configuration."""
    return pd.DataFrame([
        {
            "machine_id": "A",
            "cpu_num": 10,
            "mem_size": 10,
            "status": 1,
        },
        {
            "machine_id": "B",
            "cpu_num": 6,
            "mem_size": 10,
            "status": 1,
        },
    ])


def create_scenario(seed: int) -> pd.DataFrame:
    """Generate a reproducible workload with varied demands and arrivals."""
    rng = np.random.default_rng(seed)

    task_count = int(rng.integers(12, 21))
    interarrival = rng.uniform(0.0, 2.0, size=task_count)
    arrivals = np.cumsum(interarrival)

    # Keep every task individually feasible on machine B.
    cpu_cores = rng.integers(2, 7, size=task_count)
    memory = rng.uniform(1.0, 6.0, size=task_count)
    durations = rng.uniform(1.0, 15.0, size=task_count)

    scenario = pd.DataFrame({
        "scenario_task_id": [
            f"seed{seed}_task{i:03d}"
            for i in range(task_count)
        ],
        "task_name": [
            f"task_{i:03d}" for i in range(task_count)
        ],
        "job_name": [
            f"job_{i:03d}" for i in range(task_count)
        ],
        "plan_cpu": cpu_cores * 100,
        "plan_mem": memory,
        "duration": durations,
        "arrival_time": arrivals,
    })

    return scenario


def summarize(
    seed: int,
    policy: str,
    metrics: dict,
    results: pd.DataFrame,
) -> dict:
    """Create a consistent result row for every policy and scenario."""
    if results.empty:
        completed = pd.DataFrame()
        unschedulable = 0
    else:
        completed = results[results["status"] == "completed"]
        unschedulable = int(
            (results["status"] == "unschedulable").sum()
        )

    task_count = len(results)
    completed_count = len(completed)

    # Prefer simulator metrics; calculate from completed task records
    # as a consistency fallback.
    wait_values = (
        completed["wait_time"].dropna()
        if "wait_time" in completed.columns
        else pd.Series(dtype=float)
    )
    flow_values = (
        completed["flow_time"].dropna()
        if "flow_time" in completed.columns
        else pd.Series(dtype=float)
    )

    total_wait = float(
        metrics.get("total_wait_time", wait_values.sum())
    )
    mean_wait = float(
        metrics.get(
            "mean_wait_time",
            wait_values.mean() if len(wait_values) else 0.0,
        )
    )
    mean_flow = float(
        metrics.get(
            "mean_flow_time",
            flow_values.mean() if len(flow_values) else 0.0,
        )
    )

    return {
        "seed": seed,
        "policy": policy,
        "tasks": task_count,
        "completed": completed_count,
        "unschedulable": unschedulable,
        "completion_rate": (
            completed_count / task_count if task_count else 0.0
        ),
        "total_wait_time": total_wait,
        "mean_wait_time": mean_wait,
        "mean_flow_time": mean_flow,
        "makespan": float(metrics.get("makespan", 0.0)),
    }


def run_baseline(
    seed: int,
    scenario: pd.DataFrame,
    policy: str,
) -> dict:
    machines = create_machines()
    manager = MachineStateManager(machines)

    env = SchedulingEnvironment(
        scenario=scenario.copy(),
        machine_state=manager,
        placement_policy=policy,
    )

    outcome = env.run()
    manager.validate_invariants()

    if manager.active_allocation_count != 0:
        raise RuntimeError(
            f"{policy}, seed {seed}: allocations remain after episode."
        )

    return summarize(
        seed,
        policy,
        outcome["metrics"],
        outcome["results"],
    )

def run_ppo(
    seed: int,
    scenario: pd.DataFrame,
    model: PPO,
) -> dict:
    env = RLSchedulingEnvironment(
        scenario=scenario.copy(),
        machine_metadata=create_machines(),
        candidate_count=2,
    )

    observation, info = env.reset(seed=seed)
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action, _ = model.predict(
            observation,
            deterministic=True,
        )
        observation, reward, terminated, truncated, info = env.step(
            action
        )

    if not terminated:
        raise RuntimeError(
            f"PPO, seed {seed}: episode did not terminate normally."
        )

    sim = env.sim
    if sim is None:
        raise RuntimeError(f"PPO, seed {seed}: simulator missing.")

    sim.machine_state.validate_invariants()

    if sim.machine_state.active_allocation_count != 0:
        raise RuntimeError(
            f"PPO, seed {seed}: allocations remain after episode."
        )

    results = pd.DataFrame(list(sim.results.values()))

    # The RL wrapper does not call SchedulingEnvironment.run(),
    # so set makespan from the simulator's final clock.
    ppo_metrics = dict(sim.metrics)
    ppo_metrics["makespan"] = float(sim.current_time)

    return summarize(
        seed,
        "PPO",
        ppo_metrics,
        results,
    )

def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = PPO.load(str(MODEL_PATH), device="cpu")
    records = []

    print("PPO GENERALIZATION BENCHMARK")
    print("=" * 65)
    print(f"Scenarios: {len(SEEDS)}")
    print("Policies: first_fit, least_loaded, PPO")
    print("Machines: A (10 CPU), B (6 CPU)")
    print("Model: existing two-candidate smoke-test PPO")
    print()

    for index, seed in enumerate(SEEDS, start=1):
        scenario = create_scenario(seed)

        # Every policy gets an identical copy of this scenario.
        for policy in ("first_fit", "least_loaded"):
            records.append(
                run_baseline(seed, scenario, policy)
            )

        records.append(run_ppo(seed, scenario, model))

        print(
            f"Scenario {index:02d}/{len(SEEDS)} | "
            f"seed={seed} | tasks={len(scenario)} | completed"
        )

    detailed = pd.DataFrame(records)

    detailed_path = OUTPUT_DIR / "generalization_per_scenario.csv"
    detailed.to_csv(detailed_path, index=False)

    aggregate = (
        detailed.groupby("policy")
        .agg(
            scenarios=("seed", "nunique"),
            total_tasks=("tasks", "sum"),
            total_completed=("completed", "sum"),
            total_unschedulable=("unschedulable", "sum"),
            mean_completion_rate=("completion_rate", "mean"),
            mean_total_wait=("total_wait_time", "mean"),
            std_total_wait=("total_wait_time", "std"),
            mean_wait=("mean_wait_time", "mean"),
            mean_flow=("mean_flow_time", "mean"),
            mean_makespan=("makespan", "mean"),
            std_makespan=("makespan", "std"),
        )
        .reset_index()
    )

    aggregate_path = OUTPUT_DIR / "generalization_summary.csv"
    aggregate.to_csv(aggregate_path, index=False)

    print("\nAGGREGATE RESULTS")
    print("=" * 100)
    print(aggregate.round(4).to_string(index=False))

    print(f"\nPer-scenario results: {detailed_path}")
    print(f"Aggregate summary:    {aggregate_path}")


if __name__ == "__main__":
    main()
