from __future__ import annotations

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from simulator.config import PROJECT_ROOT
from simulator.rl_environment import RLSchedulingEnvironment


TRAINING_TIMESTEPS = 20_000
BASE_SEED = 42


def create_machines() -> pd.DataFrame:
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
    rng = np.random.default_rng(seed)
    task_count = int(rng.integers(12, 21))

    arrivals = np.cumsum(
        rng.uniform(0.0, 2.0, size=task_count)
    )
    cpu_cores = rng.integers(2, 7, size=task_count)
    memory = rng.uniform(1.0, 6.0, size=task_count)
    durations = rng.uniform(1.0, 15.0, size=task_count)

    return pd.DataFrame({
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


class RandomizedRLEnvironment(RLSchedulingEnvironment):
    """Generate a new synthetic workload for every episode."""

    def __init__(self, reward_mode: str, seed: int):
        self.scenario_rng = np.random.default_rng(seed)

        initial_scenario = create_scenario(
            int(self.scenario_rng.integers(0, 2**31 - 1))
        )

        super().__init__(
            scenario=initial_scenario,
            machine_metadata=create_machines(),
            candidate_count=2,
            reward_mode=reward_mode,
        )

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.scenario_rng = np.random.default_rng(seed)

        scenario_seed = int(
            self.scenario_rng.integers(0, 2**31 - 1)
        )
        self.scenario_source = create_scenario(scenario_seed)

        return super().reset(seed=seed, options=options)


def train_model(
    model_name: str,
    reward_mode: str,
    seed: int,
    output_dir,
) -> None:
    print("\n" + "=" * 72)
    print(f"Training model: {model_name}")
    print(f"Reward mode:    {reward_mode}")
    print(f"Timesteps:      {TRAINING_TIMESTEPS}")
    print("=" * 72)

    env = RandomizedRLEnvironment(
        reward_mode=reward_mode,
        seed=seed,
    )

    check_env(env, skip_render_check=True)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        seed=seed,
        device="cpu",
        verbose=1,
    )

    model.learn(
        total_timesteps=TRAINING_TIMESTEPS,
        progress_bar=False,
    )

    model_path = output_dir / model_name
    model.save(str(model_path))
    env.close()

    print(f"Saved model: {model_path}.zip")


def main() -> None:
    output_dir = PROJECT_ROOT / "experiments" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    configurations = [
        ("ppo_wait_only", "wait_only"),
        ("ppo_core_penalty", "core_penalty"),
        ("ppo_extended_penalty", "extended_penalty"),
    ]

    for index, (model_name, reward_mode) in enumerate(
        configurations
    ):
        train_model(
            model_name=model_name,
            reward_mode=reward_mode,
            seed=BASE_SEED + index,
            output_dir=output_dir,
        )

    print("\nALL THREE TRAINING RUNS COMPLETED.")
    print(f"Models saved under: {output_dir}")


if __name__ == "__main__":
    main()