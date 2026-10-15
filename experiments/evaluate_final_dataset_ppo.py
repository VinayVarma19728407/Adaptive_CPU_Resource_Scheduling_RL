from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from simulator.config import PROJECT_ROOT
from simulator.environment import SchedulingEnvironment
from simulator.machine_state import MachineStateManager
from simulator.rl_environment import RLSchedulingEnvironment
from simulator.task_data import load_machine_metadata
from experiments.train_final_dataset_maskable_ppo import load_eligible_task_pool


RUN_NAME = "final_dataset_ppo_v1_20261011_a"
MODEL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "models"
    / "final_dataset_ppo_runs"
    / RUN_NAME
    / "final_dataset_ppo.zip"
)
CONFIG_PATH = MODEL_PATH.parent / "training_config.json"

EVALUATION_SEEDS = list(range(42000, 42020))
TASKS_PER_EPISODE = 512
MACHINE_COUNT = 8
CANDIDATE_COUNT = 8
DEFAULT_ARRIVAL_FRACTION = 0.02


def load_config() -> dict:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Training config not found: {CONFIG_PATH}")
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def create_machine_pool(seed: int, machine_count: int) -> pd.DataFrame:
    all_machines = load_machine_metadata()
    if machine_count > len(all_machines):
        raise ValueError(
            f"Requested {machine_count} machines but only "
            f"{len(all_machines)} eligible machines are available."
        )
    machines = all_machines.sample(
        n=machine_count, random_state=seed
    ).copy()
    machines = machines.sort_values("machine_id").reset_index(drop=True)
    if machines["machine_id"].duplicated().any():
        raise ValueError("Duplicate machine IDs in evaluation machine pool.")
    return machines


def create_task_pool(machines: pd.DataFrame) -> pd.DataFrame:
    from experiments.train_final_dataset_maskable_ppo import (
        load_eligible_task_pool,
    )

    return load_eligible_task_pool(
        max_cpu_capacity=float(machines["cpu_num"].max()),
        max_memory_capacity=float(machines["mem_size"].max()),
    )


def make_fixed_scenario(
    task_pool: pd.DataFrame,
    tasks_per_episode: int,
    arrival_fraction: float,
    seed: int,
) -> pd.DataFrame:
    """Build one reproducible scenario that every policy receives unchanged."""
    if tasks_per_episode > len(task_pool):
        raise ValueError(
            f"Need {tasks_per_episode} tasks, but task pool has {len(task_pool)}."
        )

    scenario_rng = np.random.default_rng(seed)
    sample_seed = int(scenario_rng.integers(0, 2**31 - 1))
    selected = task_pool.sample(
        n=tasks_per_episode,
        replace=False,
        random_state=sample_seed,
    ).reset_index(drop=True)

    durations = selected["duration"].to_numpy(dtype=float)
    median_duration = max(float(np.median(durations)), 1e-6)
    mean_gap = max(median_duration * arrival_fraction, 0.01)

    arrival_rng = np.random.default_rng(sample_seed)
    gaps = arrival_rng.exponential(scale=mean_gap, size=tasks_per_episode)
    arrivals = np.cumsum(gaps)
    arrivals -= arrivals[0]

    return pd.DataFrame({
        "scenario_task_id": [
            f"eval{seed:06d}_task{i:06d}"
            for i in range(tasks_per_episode)
        ],
        "task_name": selected["task_name"].astype(str).to_numpy(),
        "job_name": selected["job_name"].astype(str).to_numpy(),
        "plan_cpu": pd.to_numeric(
            selected["plan_cpu"], errors="raise"
        ).to_numpy(),
        "plan_mem": pd.to_numeric(
            selected["plan_mem"], errors="raise"
        ).to_numpy(),
        "duration": durations,
        "arrival_time": arrivals.astype(float),
    })


def summarize(
    policy: str,
    seed: int,
    scenario: pd.DataFrame,
    results: pd.DataFrame,
    final_time: float,
    machine_manager: MachineStateManager,
    invalid_actions: int = 0,
) -> dict:
    completed = results[results["status"] == "completed"]
    waits = pd.to_numeric(completed["wait_time"], errors="coerce").dropna()
    flows = pd.to_numeric(completed["flow_time"], errors="coerce").dropna()
    task_count = len(scenario)
    completed_count = len(completed)
    unschedulable_count = int((results["status"] == "unschedulable").sum())

    machine_manager.validate_invariants()
    active_allocations = int(machine_manager.active_allocation_count)
    if active_allocations != 0:
        raise RuntimeError(
            f"{policy}, seed {seed}: {active_allocations} active allocations remain."
        )

    return {
        "policy": policy,
        "seed": seed,
        "tasks": task_count,
        "completed": completed_count,
        "unschedulable": unschedulable_count,
        "completion_rate": completed_count / task_count if task_count else 0.0,
        "mean_wait_time": float(waits.mean()) if len(waits) else 0.0,
        "total_wait_time": float(waits.sum()) if len(waits) else 0.0,
        "mean_flow_time": float(flows.mean()) if len(flows) else 0.0,
        "total_flow_time": float(flows.sum()) if len(flows) else 0.0,
        "makespan": float(final_time),
        "invalid_actions": int(invalid_actions),
        "resource_invariants_passed": True,
        "active_allocations_at_end": active_allocations,
    }


def run_baseline(
    scenario: pd.DataFrame,
    machines: pd.DataFrame,
    policy: str,
    seed: int,
) -> tuple[dict, pd.DataFrame]:
    manager = MachineStateManager(machines.copy())
    env = SchedulingEnvironment(
        scenario=scenario.copy(),
        machine_state=manager,
        placement_policy=policy,
    )
    outcome = env.run()
    record = summarize(
        policy=policy,
        seed=seed,
        scenario=scenario,
        results=outcome["results"],
        final_time=float(outcome["final_time"]),
        machine_manager=manager,
    )
    return record, outcome["results"].copy()


def run_maskable_ppo(
    scenario: pd.DataFrame,
    machines: pd.DataFrame,
    model: MaskablePPO,
    seed: int,
) -> tuple[dict, pd.DataFrame]:
    env = RLSchedulingEnvironment(
        scenario=scenario.copy(),
        machine_metadata=machines.copy(),
        candidate_count=CANDIDATE_COUNT,
        reward_mode="core_penalty",
    )

    observation, _ = env.reset(seed=seed)
    terminated = False
    truncated = False
    invalid_actions = 0
    steps = 0

    while not (terminated or truncated):
        action_masks = np.asarray(env.action_masks(), dtype=bool)
        if action_masks.shape != (CANDIDATE_COUNT,):
            raise RuntimeError(
                f"Unexpected action mask shape: {action_masks.shape}"
            )
        if not action_masks.any():
            raise RuntimeError(
                f"No valid action at a live PPO decision (seed {seed})."
            )

        action, _ = model.predict(
            observation,
            deterministic=True,
            action_masks=action_masks,
        )
        action_index = int(np.asarray(action).item())
        if (
            action_index < 0
            or action_index >= len(action_masks)
            or not action_masks[action_index]
        ):
            invalid_actions += 1

        observation, _, terminated, truncated, _ = env.step(action_index)
        steps += 1
        if steps > len(scenario) * 3:
            raise RuntimeError(
                f"PPO exceeded expected decision count for seed {seed}."
            )

    if env.sim is None:
        raise RuntimeError("PPO environment has no simulator after evaluation.")

    results = pd.DataFrame(env.sim.results.values())
    manager = env.sim.machine_state
    record = summarize(
        policy="maskable_ppo",
        seed=seed,
        scenario=scenario,
        results=results,
        final_time=float(env.sim.current_time),
        machine_manager=manager,
        invalid_actions=invalid_actions,
    )
    env.close()
    return record, results


def main() -> None:
    start = time.perf_counter()

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    config = load_config()
    seed = int(config["seed"])
    machine_count = int(config["machine_count"])
    tasks_per_episode = int(config["tasks_per_episode"])
    candidate_count = int(config["candidate_count"])
    arrival_fraction = float(
        config.get(
            "arrival_fraction_of_median_duration",
            DEFAULT_ARRIVAL_FRACTION,
        )
    )

    if candidate_count != CANDIDATE_COUNT:
        raise ValueError(
            f"This evaluator is configured for {CANDIDATE_COUNT} candidates, "
            f"but the model config specifies {candidate_count}."
        )

    machines = create_machine_pool(seed, machine_count)
    task_pool = create_task_pool(machines)
    model = MaskablePPO.load(str(MODEL_PATH), device="cpu")

    output_dir = (
        PROJECT_ROOT
        / "experiments"
        / "results"
        / "final_dataset_ppo_v1_20261011_a_evaluation_20261012_run2"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    episode_records: list[dict] = []
    task_records: list[pd.DataFrame] = []

    print("FINAL DATASET PPO EVALUATION")
    print(f"Model: {MODEL_PATH}")
    print(f"Machines: {len(machines)}")
    print(f"Eligible task pool: {len(task_pool)}")
    print(f"Tasks per episode: {tasks_per_episode}")
    print(f"Evaluation episodes: {len(EVALUATION_SEEDS)}")
    print("Policies: first_fit, least_loaded, maskable_ppo")
    print("Each policy receives the same scenario for each seed.")

    for episode_seed in EVALUATION_SEEDS:
        scenario = make_fixed_scenario(
            task_pool=task_pool,
            tasks_per_episode=tasks_per_episode,
            arrival_fraction=arrival_fraction,
            seed=episode_seed,
        )

        for policy in ("first_fit", "least_loaded"):
            record, task_results = run_baseline(
                scenario, machines, policy, episode_seed
            )
            episode_records.append(record)
            task_results.insert(0, "policy", policy)
            task_results.insert(1, "evaluation_seed", episode_seed)
            task_records.append(task_results)

        record, task_results = run_maskable_ppo(
            scenario, machines, model, episode_seed
        )
        episode_records.append(record)
        task_results.insert(0, "policy", "maskable_ppo")
        task_results.insert(1, "evaluation_seed", episode_seed)
        task_records.append(task_results)

        print(
            f"Seed {episode_seed}: completed all three policies "
            f"for {len(scenario)} tasks."
        )

    episode_df = pd.DataFrame(episode_records)
    task_df = pd.concat(task_records, ignore_index=True)

    numeric_columns = [
        "completion_rate",
        "mean_wait_time",
        "total_wait_time",
        "mean_flow_time",
        "total_flow_time",
        "makespan",
        "unschedulable",
        "invalid_actions",
    ]
    aggregate = episode_df.groupby("policy")[numeric_columns].agg(["mean", "std"])
    aggregate = aggregate.reset_index()
    aggregate.columns = [
        "_".join(str(part) for part in column if str(part)).rstrip("_")
        if isinstance(column, tuple) else str(column)
        for column in aggregate.columns
    ]

    episode_path = output_dir / "episode_metrics.csv"
    aggregate_path = output_dir / "aggregate_metrics.csv"
    task_path = output_dir / "task_results.csv"
    config_path = output_dir / "evaluation_config.json"

    episode_df.to_csv(episode_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    task_df.to_csv(task_path, index=False)

    evaluation_config = {
        "model_path": str(MODEL_PATH),
        "training_config_path": str(CONFIG_PATH),
        "evaluation_seeds": EVALUATION_SEEDS,
        "machine_count": machine_count,
        "machine_ids": machines["machine_id"].astype(str).tolist(),
        "tasks_per_episode": tasks_per_episode,
        "candidate_count": candidate_count,
        "arrival_fraction_of_median_duration": arrival_fraction,
        "task_pool_rows": int(len(task_pool)),
        "policies": ["first_fit", "least_loaded", "maskable_ppo"],
        "notes": [
            "Task records and generated arrival times are identical across policies within each evaluation seed.",
            "Machine selection reproduces the training selection method with the training seed.",
            "Task CPU, memory, and duration are dataset-derived; arrivals are simulated.",
            "PPO uses deterministic prediction with action masks enabled.",
            "This is a held-out workload-seed evaluation, not a fully trace-driven replay of real arrivals.",
        ],
        "elapsed_seconds": round(time.perf_counter() - start, 3),
    }
    config_path.write_text(
        json.dumps(evaluation_config, indent=2), encoding="utf-8"
    )

    print("\nAGGREGATE METRICS (mean and standard deviation across episodes)")
    print(aggregate.to_string(index=False))
    print(f"\nEvaluation outputs: {output_dir}")
    print(f"Episode metrics: {episode_path}")
    print(f"Aggregate metrics: {aggregate_path}")
    print(f"Task results: {task_path}")
    print(f"Evaluation config: {config_path}")


if __name__ == "__main__":
    main()

