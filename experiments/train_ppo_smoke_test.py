from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from simulator.config import PROJECT_ROOT
from simulator.rl_environment import RLSchedulingEnvironment


def main():
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

    env = RLSchedulingEnvironment(
        scenario=scenario,
        machine_metadata=machines,
        candidate_count=2,
    )

    check_env(env, skip_render_check=True)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=64,
        batch_size=32,
        n_epochs=5,
        gamma=0.99,
        seed=42,
        device="cpu",
        verbose=1,
    )

    model.learn(total_timesteps=2_000)

    output_dir = PROJECT_ROOT / "experiments" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "ppo_smoke_test"
    model.save(str(model_path))

    print(f"\nPPO smoke-test model saved to: {model_path}.zip")

    # Evaluate one complete deterministic episode.
    observation, info = env.reset(seed=123)
    terminated = False
    truncated = False
    total_reward = 0.0
    actions = []

    while not terminated and not truncated:
        action, _ = model.predict(observation, deterministic=True)
        actions.append(int(action))

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    sim = env.sim
    assert sim is not None

    print("\nPPO SMOKE-TEST EVALUATION")
    print("-" * 45)
    print(f"Actions: {actions}")
    print(f"Total reward: {total_reward:.4f}")
    print(f"Tasks completed: {sim.metrics['tasks_completed']}")
    print(f"Tasks unschedulable: {sim.metrics['tasks_unschedulable']}")
    print(f"Final simulation time: {sim.current_time}")
    print(f"Active allocations: {sim.machine_state.active_allocation_count}")

    sim.machine_state.validate_invariants()


if __name__ == "__main__":
    main()