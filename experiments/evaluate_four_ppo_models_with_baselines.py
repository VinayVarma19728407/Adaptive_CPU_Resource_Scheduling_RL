from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO

from simulator.config import PROJECT_ROOT
from simulator.environment import SchedulingEnvironment
from simulator.machine_state import MachineStateManager
from simulator.rl_environment import RLSchedulingEnvironment
from simulator.task_data import load_machine_metadata


# Model paths are explicit so the script evaluates the four intended checkpoints.
MODEL_SPECS = [
    {
        "name": "ppo_v1",
        "path": PROJECT_ROOT / "experiments/models/final_dataset_ppo_runs/final_dataset_ppo_v1_20261011_a/final_dataset_ppo.zip",
        "config": PROJECT_ROOT / "experiments/models/final_dataset_ppo_runs/final_dataset_ppo_v1_20261011_a/training_config.json",
        "reward_mode": "core_penalty",
    },
    {
        "name": "ppo_v2",
        "path": PROJECT_ROOT / "experiments/models/final_dataset_ppo_runs/final_dataset_ppo_v2_extended_20261012/final_dataset_ppo.zip",
        "config": PROJECT_ROOT / "experiments/models/final_dataset_ppo_runs/final_dataset_ppo_v2_extended_20261012/training_config.json",
        "reward_mode": "extended_penalty",
    },
    {
        "name": "ppo_full_dataset",
        "path": PROJECT_ROOT / "experiments/models/final_dataset_ppo_runs/final_ppo_full_dataset_20261012/final_ppo_full_dataset.zip",
        "config": PROJECT_ROOT / "experiments/models/final_dataset_ppo_runs/final_ppo_full_dataset_20261012/training_config.json",
        "reward_mode": "extended_penalty",
    },
    {
        "name": "ppo_instance_calibrated",
        "path": PROJECT_ROOT / "experiments/models/final_dataset_ppo_runs/final_ppo_instance_calibrated_fixed_20261013/final_ppo_full_dataset.zip",
        "config": PROJECT_ROOT / "experiments/models/final_dataset_ppo_runs/final_ppo_instance_calibrated_fixed_20261013/training_config.json",
        "reward_mode": "extended_penalty",
    },
]

SEEDS = list(range(42000, 42020))
TASKS_PER_EPISODE = 512
MACHINE_COUNT = 8
CANDIDATE_COUNT = 8
DEFAULT_ARRIVAL_FRACTION = 0.02
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "results" / "four_model_baseline_comparison_20261013"
# Created by the instance-calibrated training run: row order is the eligible source
# record order; columns are plan_cpu, plan_mem, duration (duration calibrated where matched).
CALIBRATED_POOL = PROJECT_ROOT / "experiments" / "data" / "eligible_task_features_instance_calibrated.f64"
ELIGIBLE_COUNT = 13_446_871
FEATURE_COUNT = 3


def load_training_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Training config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def create_machine_pool(seed: int, machine_count: int) -> pd.DataFrame:
    all_machines = load_machine_metadata()
    if machine_count > len(all_machines):
        raise ValueError(f"Requested {machine_count} machines but only {len(all_machines)} are available.")
    machines = all_machines.sample(n=machine_count, random_state=seed).copy()
    machines = machines.sort_values("machine_id").reset_index(drop=True)
    if machines["machine_id"].duplicated().any():
        raise ValueError("Duplicate machine IDs in evaluation machine pool.")
    return machines


def load_calibrated_task_pool() -> np.ndarray:
    if not CALIBRATED_POOL.is_file():
        raise FileNotFoundError(
            f"Calibrated numeric task pool not found: {CALIBRATED_POOL}\n"
            "It should have been created by the completed instance-calibrated training run."
        )
    expected_bytes = ELIGIBLE_COUNT * FEATURE_COUNT * np.dtype(np.float64).itemsize
    actual_bytes = CALIBRATED_POOL.stat().st_size
    if actual_bytes % (FEATURE_COUNT * 8) != 0:
        raise ValueError(f"Calibrated pool has unexpected file size: {actual_bytes} bytes.")
    count = actual_bytes // (FEATURE_COUNT * 8)
    if count != ELIGIBLE_COUNT:
        raise ValueError(
            f"Calibrated pool has {count:,} records, expected {ELIGIBLE_COUNT:,}. "
            "Update ELIGIBLE_COUNT only after checking the training exposure report."
        )
    print(f"Opening calibrated task pool as disk-backed memmap: {count:,} records")
    return np.memmap(CALIBRATED_POOL, dtype=np.float64, mode="r", shape=(count, FEATURE_COUNT))


def make_scenario(pool: np.ndarray, seed: int, arrival_fraction: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(pool), size=TASKS_PER_EPISODE, replace=False)
    values = np.asarray(pool[indices], dtype=np.float64)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"Invalid calibrated task features found in evaluation sample for seed {seed}.")
    durations = values[:, 2]
    median_duration = max(float(np.median(durations)), 1e-6)
    mean_gap = max(median_duration * arrival_fraction, 0.01)
    gaps = np.random.default_rng(seed + 1).exponential(scale=mean_gap, size=TASKS_PER_EPISODE)
    arrivals = np.cumsum(gaps)
    arrivals -= arrivals[0]
    return pd.DataFrame({
        "scenario_task_id": [f"eval{seed:06d}_task{i:06d}" for i in range(TASKS_PER_EPISODE)],
        "task_name": [f"eval_task_type_{i % 256}" for i in range(TASKS_PER_EPISODE)],
        "job_name": [f"eval_job_{i:06d}" for i in range(TASKS_PER_EPISODE)],
        "plan_cpu": values[:, 0],
        "plan_mem": values[:, 1],
        "duration": durations,
        "arrival_time": arrivals.astype(float),
    })


def summarize(policy: str, seed: int, scenario: pd.DataFrame, results: pd.DataFrame,
              final_time: float, manager: MachineStateManager, invalid_actions: int = 0) -> dict:
    completed = results[results["status"] == "completed"]
    waits = pd.to_numeric(completed.get("wait_time", pd.Series(dtype=float)), errors="coerce").dropna()
    flows = pd.to_numeric(completed.get("flow_time", pd.Series(dtype=float)), errors="coerce").dropna()
    task_count = len(scenario)
    completed_count = len(completed)
    unschedulable_count = int((results["status"] == "unschedulable").sum())
    manager.validate_invariants()
    active = int(manager.active_allocation_count)
    if active != 0:
        raise RuntimeError(f"{policy}, seed {seed}: {active} active allocations remain.")
    return {
        "policy": policy, "seed": seed, "tasks": task_count, "completed": completed_count,
        "unschedulable": unschedulable_count,
        "completion_rate": completed_count / task_count if task_count else 0.0,
        "mean_wait_time": float(waits.mean()) if len(waits) else 0.0,
        "total_wait_time": float(waits.sum()) if len(waits) else 0.0,
        "mean_flow_time": float(flows.mean()) if len(flows) else 0.0,
        "total_flow_time": float(flows.sum()) if len(flows) else 0.0,
        "makespan": float(final_time), "invalid_actions": int(invalid_actions),
        "resource_invariants_passed": True, "active_allocations_at_end": active,
    }


def run_baseline(scenario: pd.DataFrame, machines: pd.DataFrame, policy: str, seed: int) -> tuple[dict, pd.DataFrame]:
    manager = MachineStateManager(machines.copy())
    env = SchedulingEnvironment(scenario=scenario.copy(), machine_state=manager, placement_policy=policy)
    outcome = env.run()
    record = summarize(policy, seed, scenario, outcome["results"], float(outcome["final_time"]), manager)
    return record, outcome["results"].copy()


def run_ppo(scenario: pd.DataFrame, machines: pd.DataFrame, model: MaskablePPO,
            policy_name: str, reward_mode: str, seed: int) -> tuple[dict, pd.DataFrame]:
    env = RLSchedulingEnvironment(
        scenario=scenario.copy(), machine_metadata=machines.copy(),
        candidate_count=CANDIDATE_COUNT, reward_mode=reward_mode,
    )
    observation, _ = env.reset(seed=seed)
    terminated = truncated = False
    invalid_actions = 0
    steps = 0
    while not (terminated or truncated):
        masks = np.asarray(env.action_masks(), dtype=bool)
        if masks.shape != (CANDIDATE_COUNT,) or not masks.any():
            raise RuntimeError(f"{policy_name}, seed {seed}: invalid/empty action mask {masks}.")
        action, _ = model.predict(observation, deterministic=True, action_masks=masks)
        action_index = int(np.asarray(action).item())
        if action_index < 0 or action_index >= len(masks) or not masks[action_index]:
            invalid_actions += 1
        observation, _, terminated, truncated, _ = env.step(action_index)
        steps += 1
        if steps > len(scenario) * 3:
            raise RuntimeError(f"{policy_name} exceeded expected decision count for seed {seed}.")
    if env.sim is None:
        raise RuntimeError(f"{policy_name}: environment simulator missing after evaluation.")
    results = pd.DataFrame(env.sim.results.values())
    manager = env.sim.machine_state
    record = summarize(policy_name, seed, scenario, results, float(env.sim.current_time), manager, invalid_actions)
    env.close()
    return record, results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20, help="Number of matched evaluation seeds (default 20).")
    parser.add_argument("--tasks-per-episode", type=int, default=TASKS_PER_EPISODE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.episodes < 1 or args.tasks_per_episode < 1:
        raise ValueError("--episodes and --tasks-per-episode must be positive.")
    seeds = SEEDS[:args.episodes] if args.episodes <= len(SEEDS) else list(range(SEEDS[0], SEEDS[0] + args.episodes))

    start = time.perf_counter()
    for spec in MODEL_SPECS:
        if not spec["path"].is_file():
            raise FileNotFoundError(f"{spec['name']} model missing: {spec['path']}")
        config = load_training_config(spec["config"])
        if int(config.get("candidate_count", CANDIDATE_COUNT)) != CANDIDATE_COUNT:
            raise ValueError(f"{spec['name']} candidate_count differs from evaluator setting {CANDIDATE_COUNT}.")
    pool = load_calibrated_task_pool()
    machines = create_machine_pool(seed=42, machine_count=MACHINE_COUNT)
    models = []
    arrival_fraction = DEFAULT_ARRIVAL_FRACTION
    for spec in MODEL_SPECS:
        config = load_training_config(spec["config"])
        model = MaskablePPO.load(str(spec["path"]), device="cpu")
        models.append((spec, model, config))
        arrival_fraction = float(config.get("arrival_fraction", config.get("arrival_fraction_of_median_duration", arrival_fraction)))

    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}. Choose --output-dir with a new name.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    episode_records = []
    task_records = []
    print("FOUR-MODEL PPO + BASELINE COMPARISON")
    print("Evaluation uses one common instance-calibrated task pool for every policy.")
    print(f"Models: {[spec['name'] for spec, _, _ in models]}")
    print("Baselines: first_fit, least_loaded")
    print(f"Seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)} episodes); tasks/episode={args.tasks_per_episode}")
    print(f"Machine IDs: {machines['machine_id'].astype(str).tolist()}")
    print("Policy inference is deterministic; all policies receive the same scenario per seed.")

    # All model/baseline runs use identical selected task rows and arrival times per seed.
    for seed in seeds:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(pool), size=args.tasks_per_episode, replace=False)
        values = np.asarray(pool[indices], dtype=np.float64)
        durations = values[:, 2]
        median_duration = max(float(np.median(durations)), 1e-6)
        mean_gap = max(median_duration * arrival_fraction, 0.01)
        gaps = np.random.default_rng(seed + 1).exponential(scale=mean_gap, size=args.tasks_per_episode)
        arrivals = np.cumsum(gaps)
        arrivals -= arrivals[0]
        scenario = pd.DataFrame({
            "scenario_task_id": [f"eval{seed:06d}_task{i:06d}" for i in range(args.tasks_per_episode)],
            "task_name": [f"eval_task_type_{i % 256}" for i in range(args.tasks_per_episode)],
            "job_name": [f"eval_job_{i:06d}" for i in range(args.tasks_per_episode)],
            "plan_cpu": values[:, 0], "plan_mem": values[:, 1], "duration": durations,
            "arrival_time": arrivals.astype(float),
        })
        for policy in ("first_fit", "least_loaded"):
            record, task_df = run_baseline(scenario, machines, policy, seed)
            episode_records.append(record)
            task_df.insert(0, "policy", policy)
            task_df.insert(1, "evaluation_seed", seed)
            task_records.append(task_df)
        for spec, model, config in models:
            record, task_df = run_ppo(scenario, machines, model, spec["name"], spec["reward_mode"], seed)
            episode_records.append(record)
            task_df.insert(0, "policy", spec["name"])
            task_df.insert(1, "evaluation_seed", seed)
            task_records.append(task_df)
        print(f"Seed {seed}: completed all 6 policies on the same {args.tasks_per_episode}-task scenario.")

    episode_df = pd.DataFrame(episode_records)
    task_df = pd.concat(task_records, ignore_index=True)
    metrics = [
        "completion_rate", "mean_wait_time", "total_wait_time", "mean_flow_time",
        "total_flow_time", "makespan", "unschedulable", "invalid_actions",
    ]
    aggregate = episode_df.groupby("policy")[metrics].agg(["mean", "std"]).reset_index()
    aggregate.columns = [
        "_".join(str(part) for part in col if str(part)).rstrip("_") if isinstance(col, tuple) else str(col)
        for col in aggregate.columns
    ]
    episode_df.to_csv(args.output_dir / "episode_metrics.csv", index=False)
    aggregate.to_csv(args.output_dir / "aggregate_metrics.csv", index=False)
    task_df.to_csv(args.output_dir / "task_results.csv", index=False)
    metadata = {
        "models": [{"name": s["name"], "path": str(s["path"]), "config": str(s["config"])} for s, _, _ in models],
        "baselines": ["first_fit", "least_loaded"], "seeds": seeds,
        "tasks_per_episode": args.tasks_per_episode, "machine_ids": machines["machine_id"].astype(str).tolist(),
        "candidate_count": CANDIDATE_COUNT, "common_duration_source": str(CALIBRATED_POOL),
        "duration_source_note": "All policies are evaluated on the same instance-calibrated numeric task pool generated by the completed calibrated training run.",
        "arrival_fraction_of_median_duration": arrival_fraction,
        "elapsed_seconds": round(time.perf_counter() - start, 3),
    }
    (args.output_dir / "evaluation_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nAGGREGATE METRICS (mean and standard deviation across matched episodes)")
    print(aggregate.to_string(index=False))
    print(f"\nResults saved to: {args.output_dir}")
    print(f"Episode metrics: {args.output_dir / 'episode_metrics.csv'}")
    print(f"Aggregate metrics: {args.output_dir / 'aggregate_metrics.csv'}")
    print(f"Task results: {args.output_dir / 'task_results.csv'}")


if __name__ == "__main__":
    main()
