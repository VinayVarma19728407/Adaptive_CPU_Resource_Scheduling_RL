from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback

from simulator.config import PROJECT_ROOT, RANDOM_SEED
from simulator.rl_environment import RLSchedulingEnvironment
from simulator.scenario_loader import load_or_create_task_pool, validate_task_pool
from simulator.task_data import load_machine_metadata


DEFAULT_TIMESTEPS = 102_400
DEFAULT_MACHINE_COUNT = 8
DEFAULT_TASKS_PER_EPISODE = 512
DEFAULT_CANDIDATE_COUNT = 8
DEFAULT_ARRIVAL_FRACTION = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one larger Maskable PPO scheduler on Alibaba-derived task "
            "resource requests and durations."
        )
    )
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--machine-count", type=int, default=DEFAULT_MACHINE_COUNT)
    parser.add_argument(
        "--tasks-per-episode", type=int, default=DEFAULT_TASKS_PER_EPISODE
    )
    parser.add_argument(
        "--candidate-count", type=int, default=DEFAULT_CANDIDATE_COUNT
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--arrival-fraction",
        type=float,
        default=DEFAULT_ARRIVAL_FRACTION,
        help=(
            "Mean inter-arrival interval as a fraction of median task duration. "
            "Smaller values create denser workloads. Arrival times are simulated."
        ),
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="final_dataset_ppo_v1",
        help="New run folder name. Existing folders are never overwritten.",
    )
    
    parser.add_argument(
            "--reward-mode",
            choices=["wait_only", "core_penalty", "extended_penalty"],
            default="extended_penalty",
            help="Reward function used by the RL environment.",
        )
    return parser.parse_args()


def load_eligible_task_pool(max_cpu_capacity: float, max_memory_capacity: float) -> pd.DataFrame:
    """Load the existing cached task sample, or create it once if absent."""
    cache_dir = PROJECT_ROOT / "experiments" / "data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    eligible_path = cache_dir / "eligible_task_pool_seed42.csv"

    if eligible_path.is_file():
        print(f"Loading cached eligible task pool: {eligible_path}")
        pool = pd.read_csv(eligible_path, low_memory=False)
    else:
        print("Eligible task cache not found; loading/creating the task sample.")
        pool = load_or_create_task_pool(sample_size=50_000, seed=RANDOM_SEED)
        pool, exclusions = validate_task_pool(pool)
        pool.to_csv(eligible_path, index=False)
        print(f"Saved eligible task cache: {eligible_path}")
        print(f"Rows excluded by initial task validation: {len(exclusions):,}")

    required = {"task_name", "job_name", "plan_cpu", "plan_mem", "duration"}
    missing = required - set(pool.columns)
    if missing:
        raise ValueError(f"Task pool is missing required columns: {sorted(missing)}")

    data = pool.copy()
    for column in ("plan_cpu", "plan_mem", "duration"):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # Alibaba schema: plan_cpu is expressed in hundredths of a CPU core;
    # plan_mem and machine mem_size use the same normalized memory scale.
    valid = (
        data["plan_cpu"].notna()
        & data["plan_mem"].notna()
        & data["duration"].notna()
        & np.isfinite(data["plan_cpu"])
        & np.isfinite(data["plan_mem"])
        & np.isfinite(data["duration"])
        & (data["plan_cpu"] > 0)
        & (data["plan_mem"] > 0)
        & (data["duration"] > 0)
        & ((data["plan_cpu"] / 100.0) <= max_cpu_capacity)
        & (data["plan_mem"] <= max_memory_capacity)
    )
    eligible = data.loc[valid].copy().reset_index(drop=True)

    if eligible.empty:
        raise ValueError(
            "No task records remain after resource and duration validation. "
            "Inspect plan_cpu, plan_mem, duration, and machine capacities."
        )

    return eligible


class DatasetRandomizedRLEnvironment(RLSchedulingEnvironment):
    """Create a new workload from cached Alibaba task records each episode.

    Task CPU/memory requests and durations come from the dataset. Arrival times
    are generated for controlled experiments and are not claimed to be exact
    Alibaba event timestamps.
    """

    def __init__(
        self,
        task_pool: pd.DataFrame,
        machine_metadata: pd.DataFrame,
        tasks_per_episode: int,
        candidate_count: int,
        arrival_fraction: float,
        seed: int,
        reward_mode: str = "core_penalty",
    ) -> None:
        if tasks_per_episode <= 0:
            raise ValueError("--tasks-per-episode must be positive.")
        if tasks_per_episode > len(task_pool):
            raise ValueError(
                f"Requested {tasks_per_episode:,} tasks per episode, but only "
                f"{len(task_pool):,} eligible task records are available."
            )
        if not 0 < arrival_fraction <= 1:
            raise ValueError("--arrival-fraction must be in the interval (0, 1].")

        self.task_pool = task_pool.reset_index(drop=True).copy()
        self.tasks_per_episode = int(tasks_per_episode)
        self.arrival_fraction = float(arrival_fraction)
        self.scenario_rng = np.random.default_rng(seed)
        self.episode_number = 0

        # The base class requires a scenario at construction. It is replaced
        # immediately before each reset with a newly sampled scenario.
        initial_scenario = self._make_scenario()
        super().__init__(
            scenario=initial_scenario,
            machine_metadata=machine_metadata,
            candidate_count=candidate_count,
            reward_mode=reward_mode,
        )

    def _make_scenario(self) -> pd.DataFrame:
        sample_seed = int(self.scenario_rng.integers(0, 2**31 - 1))
        rng = np.random.default_rng(sample_seed)

        selected = self.task_pool.sample(
            n=self.tasks_per_episode,
            replace=False,
            random_state=sample_seed,
        ).reset_index(drop=True)

        durations = selected["duration"].to_numpy(dtype=float)
        median_duration = max(float(np.median(durations)), 1e-6)
        mean_gap = max(median_duration * self.arrival_fraction, 0.01)

        gaps = rng.exponential(scale=mean_gap, size=self.tasks_per_episode)
        arrivals = np.cumsum(gaps)
        arrivals -= arrivals[0]

        scenario = pd.DataFrame({
            "scenario_task_id": [
                f"ep{self.episode_number:06d}_task{i:06d}"
                for i in range(self.tasks_per_episode)
            ],
            "task_name": selected["task_name"].astype(str).to_numpy(),
            "job_name": selected["job_name"].astype(str).to_numpy(),
            "plan_cpu": pd.to_numeric(selected["plan_cpu"], errors="raise").to_numpy(),
            "plan_mem": pd.to_numeric(selected["plan_mem"], errors="raise").to_numpy(),
            "duration": durations,
            "arrival_time": arrivals.astype(float),
        })

        self.episode_number += 1
        return scenario

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.scenario_rng = np.random.default_rng(seed)
        self.scenario_source = self._make_scenario()
        return super().reset(seed=seed, options=options)


def select_machine_pool(machine_count: int, seed: int) -> pd.DataFrame:
    all_machines = load_machine_metadata()

    if machine_count <= 0:
        raise ValueError("--machine-count must be positive.")
    if machine_count > len(all_machines):
        raise ValueError(
            f"Requested {machine_count} machines, but only "
            f"{len(all_machines):,} eligible machines are available."
        )

    selected = all_machines.sample(
        n=machine_count, random_state=seed
    ).copy()
    selected = selected.sort_values("machine_id").reset_index(drop=True)

    if selected["machine_id"].duplicated().any():
        raise ValueError("Duplicate machine IDs found in selected machine pool.")

    return selected


def run_preflight(env: DatasetRandomizedRLEnvironment, seed: int) -> None:
    """Exercise masked decisions before starting the expensive training run."""
    observation, _ = env.reset(seed=seed)

    if not env.observation_space.contains(observation):
        raise RuntimeError("Preflight failed: reset observation is outside its space.")

    steps = 0
    terminated = False
    truncated = False

    while steps < min(20, env.tasks_per_episode) and not (terminated or truncated):
        mask = np.asarray(env.action_masks(), dtype=bool)
        if mask.shape != (env.action_space.n,):
            raise RuntimeError(
                f"Preflight failed: action mask shape {mask.shape} does not match "
                f"action space {env.action_space.n}."
            )
        valid_actions = np.flatnonzero(mask)
        if len(valid_actions) == 0:
            raise RuntimeError("Preflight failed: no valid action at a live decision.")

        observation, _, terminated, truncated, _ = env.step(int(valid_actions[0]))
        if not env.observation_space.contains(observation):
            raise RuntimeError("Preflight failed: step observation is outside its space.")
        steps += 1

    if env.sim is not None:
        env.sim.machine_state.validate_invariants()

    print(
        f"Preflight passed: {steps} masked scheduling decisions, "
        f"{len(env.machine_metadata)} machines, "
        f"{env.candidate_count} candidate actions."
    )
    env.close()


def main() -> None:
    args = parse_args()

    if args.timesteps <= 0:
        raise ValueError("--timesteps must be positive.")
    if args.candidate_count <= 0:
        raise ValueError("--candidate-count must be positive.")
    if args.candidate_count > args.machine_count:
        raise ValueError(
            "--candidate-count cannot exceed --machine-count for this experiment."
        )

    run_root = PROJECT_ROOT / "experiments" / "models" / "final_dataset_ppo_runs"
    run_dir = run_root / args.run_name
    checkpoint_dir = run_dir / "checkpoints"

    # Never overwrite prior models or training outputs.
    if run_dir.exists():
        raise FileExistsError(
            f"Run directory already exists: {run_dir}\n"
            "Choose a new --run-name so existing models/results are preserved."
        )

    machines = select_machine_pool(args.machine_count, args.seed)
    task_pool = load_eligible_task_pool(
        max_cpu_capacity=float(machines["cpu_num"].max()),
        max_memory_capacity=float(machines["mem_size"].max()),
    )

    if args.tasks_per_episode > len(task_pool):
        raise ValueError(
            f"Episode requests {args.tasks_per_episode:,} tasks, but only "
            f"{len(task_pool):,} eligible tasks remain."
        )

    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)

    task_profile = task_pool[["plan_cpu", "plan_mem", "duration"]].describe(
        percentiles=[0.50, 0.90, 0.95, 0.99]
    )
    task_profile.to_csv(run_dir / "eligible_task_pool_profile.csv")

    config = {
        "algorithm": "MaskablePPO",
        "seed": args.seed,
        "timesteps_requested": args.timesteps,
        "machine_count": args.machine_count,
        "tasks_per_episode": args.tasks_per_episode,
        "candidate_count": args.candidate_count,
        "reward_mode": args.reward_mode,
        "network_architecture": {"pi": [256, 128], "vf": [256, 128]},
        "n_steps": 512,
        "batch_size": 128,
        "n_epochs": 10,
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "arrival_fraction_of_median_duration": args.arrival_fraction,
        "task_resource_units": {
            "cpu": "plan_cpu / 100.0 CPU cores",
            "memory": "plan_mem directly; same normalized scale as machine mem_size",
        },
        "arrival_time_note": (
            "Arrival times are generated for controlled experiments; task resource "
            "requests and durations are dataset-derived."
        ),
    }
    (run_dir / "training_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print("\nFINAL DATASET-DRIVEN MASKABLE PPO TRAINING")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Selected machines: {len(machines):,}")
    print(f"Eligible task pool: {len(task_pool):,}")
    print(f"Tasks per episode: {args.tasks_per_episode:,}")
    print(f"Candidate actions: {args.candidate_count}")
    print(f"Requested timesteps: {args.timesteps:,}")
    print(f"Reward mode: {args.reward_mode}")
    print("Policy/value network: [256, 128]")
    print(f"Run directory: {run_dir}")

    preflight_env = DatasetRandomizedRLEnvironment(
        task_pool=task_pool,
        machine_metadata=machines,
        tasks_per_episode=min(args.tasks_per_episode, 32),
        candidate_count=args.candidate_count,
        arrival_fraction=args.arrival_fraction,
        seed=args.seed + 1,
        reward_mode=args.reward_mode,
    )
    run_preflight(preflight_env, args.seed + 1)

    train_env = DatasetRandomizedRLEnvironment(
        task_pool=task_pool,
        machine_metadata=machines,
        tasks_per_episode=args.tasks_per_episode,
        candidate_count=args.candidate_count,
        arrival_fraction=args.arrival_fraction,
        seed=args.seed,
        reward_mode=args.reward_mode,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10_240,
        save_path=str(checkpoint_dir),
        name_prefix="final_dataset_ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    model = MaskablePPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs={"net_arch": {"pi": [256, 128], "vf": [256, 128]}},
        seed=args.seed,
        device="cpu",
        verbose=1,
    )

    start = time.perf_counter()
    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        progress_bar=False,
        reset_num_timesteps=True,
    )
    elapsed = time.perf_counter() - start

    final_model_path = run_dir / "final_dataset_ppo"
    model.save(str(final_model_path))

    summary = {
        "status": "training_completed",
        "timesteps_requested": int(args.timesteps),
        "timesteps_completed": int(model.num_timesteps),
        "elapsed_seconds": round(elapsed, 3),
        "elapsed_minutes": round(elapsed / 60.0, 3),
        "model_file": str(final_model_path.with_suffix(".zip")),
        "run_directory": str(run_dir),
    }
    (run_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    train_env.close()
    print("\nTraining completed.")
    print(f"Actual timesteps: {model.num_timesteps:,}")
    print(f"Elapsed time: {elapsed / 60.0:.2f} minutes")
    print(f"Saved model: {final_model_path}.zip")
    print(f"Saved run metadata: {run_dir / 'training_summary.json'}")
    print("This model must still pass held-out evaluation against both baselines.")


if __name__ == "__main__":
    main()
