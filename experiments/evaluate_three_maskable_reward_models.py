from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO

from simulator.config import PROJECT_ROOT
from simulator.environment import SchedulingEnvironment
from simulator.machine_state import MachineStateManager
from simulator.rl_environment import RLSchedulingEnvironment
from experiments.train_three_maskable_reward_models import (
    create_machines,
    create_scenario,
)


EVALUATION_SEEDS = list(range(1000, 1050))
CANDIDATE_COUNT = 2

MODEL_CONFIGS = [
    ("ppo_maskable_wait_only", "wait_only"),
    ("ppo_maskable_core_penalty", "core_penalty"),
    ("ppo_maskable_extended_penalty", "extended_penalty"),
]


def summarize_task_results(
    results: pd.DataFrame,
    scenario: pd.DataFrame,
    machine_metadata: pd.DataFrame,
    final_time: float,
    invalid_actions: int = 0,
) -> dict[str, Any]:
    task_count = len(scenario)

    if results.empty:
        completed = results.copy()
        unschedulable_count = 0
    else:
        completed = results[results["status"] == "completed"].copy()
        unschedulable_count = int(
            (results["status"] == "unschedulable").sum()
        )

    completed_count = len(completed)
    mean_wait = float(completed["wait_time"].mean()) if completed_count else 0.0
    total_wait = float(completed["wait_time"].sum()) if completed_count else 0.0
    mean_flow = float(completed["flow_time"].mean()) if completed_count else 0.0

    cpu_capacity_total = float(machine_metadata["cpu_num"].sum())
    memory_capacity_total = float(machine_metadata["mem_size"].sum())

    cpu_resource_time = 0.0
    memory_resource_time = 0.0
    if completed_count:
        cpu_resource_time = float(
            (
                completed["allocated_cpu"].astype(float)
                * completed["baseline_duration"].astype(float)
            ).sum()
        )
        memory_resource_time = float(
            (
                completed["allocated_memory"].astype(float)
                * completed["baseline_duration"].astype(float)
            ).sum()
        )

    denominator_time = max(float(final_time), 0.0)
    cpu_utilization = (
        cpu_resource_time / (cpu_capacity_total * denominator_time)
        if cpu_capacity_total > 0 and denominator_time > 0 else 0.0
    )
    memory_utilization = (
        memory_resource_time / (memory_capacity_total * denominator_time)
        if memory_capacity_total > 0 and denominator_time > 0 else 0.0
    )

    return {
        "task_count": int(task_count),
        "completed_tasks": int(completed_count),
        "unschedulable_tasks": int(unschedulable_count),
        "completion_rate": completed_count / task_count if task_count else 0.0,
        "mean_wait_time": mean_wait,
        "total_wait_time": total_wait,
        "mean_flow_time": mean_flow,
        "makespan": float(final_time),
        "cpu_time_utilization": float(cpu_utilization),
        "memory_time_utilization": float(memory_utilization),
        "invalid_actions": int(invalid_actions),
    }


def run_rl_episode(
    model: MaskablePPO,
    reward_mode: str,
    scenario: pd.DataFrame,
    machine_metadata: pd.DataFrame,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    env = RLSchedulingEnvironment(
        scenario=scenario,
        machine_metadata=machine_metadata,
        candidate_count=CANDIDATE_COUNT,
        reward_mode=reward_mode,
    )

    observation, _ = env.reset(seed=seed)
    terminated = False
    truncated = False
    invalid_actions = 0
    total_reward = 0.0
    decisions = 0

    while not (terminated or truncated):
        action_mask = np.asarray(env.action_masks(), dtype=bool)
        if not action_mask.any():
            raise RuntimeError(
                f"No valid action at a live decision: seed={seed}, "
                f"reward_mode={reward_mode}, candidates={env.candidates}"
            )

        action, _ = model.predict(
            observation,
            deterministic=True,
            action_masks=action_mask,
        )
        action_index = int(np.asarray(action).item())

        if (
            action_index < 0
            or action_index >= len(env.candidates)
            or not action_mask[action_index]
        ):
            invalid_actions += 1

        observation, reward, terminated, truncated, _ = env.step(action_index)
        total_reward += float(reward)
        decisions += 1

    assert env.sim is not None
    env.sim.machine_state.validate_invariants()
    if env.sim.machine_state.active_allocation_count != 0:
        raise RuntimeError(
            f"Model {reward_mode}, seed {seed}: active allocations remain."
        )

    task_results = pd.DataFrame(env.sim.results.values())
    if len(task_results) != len(scenario):
        raise RuntimeError(
            f"Model {reward_mode}, seed {seed}: expected {len(scenario)} "
            f"task result rows, got {len(task_results)}."
        )

    metrics = summarize_task_results(
        task_results, scenario, machine_metadata,
        float(env.sim.current_time), invalid_actions,
    )
    metrics.update({
        "method": f"ppo_maskable_{reward_mode}",
        "seed": int(seed),
        "episode_reward": float(total_reward),
        "scheduling_decisions": int(decisions),
        "episode_terminated": bool(terminated),
        "resource_invariants_pass": True,
    })

    task_results = task_results.copy()
    task_results["method"] = f"ppo_maskable_{reward_mode}"
    task_results["seed"] = int(seed)
    env.close()
    return metrics, task_results


def run_baseline_episode(
    policy: str,
    scenario: pd.DataFrame,
    machine_metadata: pd.DataFrame,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    manager = MachineStateManager(machine_metadata)
    env = SchedulingEnvironment(
        scenario=scenario,
        machine_state=manager,
        placement_policy=policy,
    )
    outcome = env.run()
    results = outcome["results"]

    manager.validate_invariants()
    if manager.active_allocation_count != 0:
        raise RuntimeError(
            f"Baseline {policy}, seed {seed}: active allocations remain."
        )

    metrics = summarize_task_results(
        results, scenario, machine_metadata, float(outcome["final_time"])
    )
    metrics.update({
        "method": policy,
        "seed": int(seed),
        "episode_reward": np.nan,
        "scheduling_decisions": int(outcome["metrics"]["tasks_started"]),
        "episode_terminated": bool(outcome["finished"]),
        "resource_invariants_pass": True,
    })

    results = results.copy()
    results["method"] = policy
    results["seed"] = int(seed)
    return metrics, results


def main() -> None:
    model_dir = PROJECT_ROOT / "experiments" / "models"
    output_dir = PROJECT_ROOT / "experiments" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    machine_metadata = create_machines()
    episode_rows: list[dict[str, Any]] = []
    detail_frames: list[pd.DataFrame] = []
    models: dict[str, MaskablePPO] = {}

    for model_name, reward_mode in MODEL_CONFIGS:
        model_path = model_dir / f"{model_name}.zip"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found: {model_path}\n"
                "Run experiments.train_three_maskable_reward_models first."
            )

        model = MaskablePPO.load(str(model_path), device="cpu")
        expected_shape = (4 + CANDIDATE_COUNT * 4,)
        if tuple(model.observation_space.shape) != expected_shape:
            raise ValueError(
                f"{model_name} expects observation shape "
                f"{model.observation_space.shape}; expected {expected_shape}."
            )
        if int(model.action_space.n) != CANDIDATE_COUNT:
            raise ValueError(
                f"{model_name} has {model.action_space.n} actions; "
                f"expected {CANDIDATE_COUNT}."
            )

        models[reward_mode] = model
        print(f"Loaded model: {model_path.name}")

    for seed in EVALUATION_SEEDS:
        scenario = create_scenario(seed)
        print(f"\nWorkload seed {seed}: {len(scenario)} tasks")

        for _, reward_mode in MODEL_CONFIGS:
            metrics, task_results = run_rl_episode(
                models[reward_mode], reward_mode, scenario,
                machine_metadata, seed,
            )
            episode_rows.append(metrics)
            detail_frames.append(task_results)
            print(
                f"  ppo_maskable_{reward_mode}: "
                f"completion={metrics['completion_rate']:.3f}, "
                f"mean_wait={metrics['mean_wait_time']:.3f}, "
                f"mean_flow={metrics['mean_flow_time']:.3f}, "
                f"makespan={metrics['makespan']:.3f}, "
                f"invalid_actions={metrics['invalid_actions']}"
            )

        for policy in ("first_fit", "least_loaded"):
            metrics, task_results = run_baseline_episode(
                policy, scenario, machine_metadata, seed
            )
            episode_rows.append(metrics)
            detail_frames.append(task_results)
            print(
                f"  {policy}: completion={metrics['completion_rate']:.3f}, "
                f"mean_wait={metrics['mean_wait_time']:.3f}, "
                f"mean_flow={metrics['mean_flow_time']:.3f}, "
                f"makespan={metrics['makespan']:.3f}"
            )

    episode_df = pd.DataFrame(episode_rows)
    details_df = pd.concat(detail_frames, ignore_index=True)

    summary_df = (
        episode_df.groupby("method", as_index=False)
        .agg(
            episodes=("seed", "count"),
            mean_completion_rate=("completion_rate", "mean"),
            mean_unschedulable_tasks=("unschedulable_tasks", "mean"),
            mean_wait_time=("mean_wait_time", "mean"),
            mean_total_wait_time=("total_wait_time", "mean"),
            mean_flow_time=("mean_flow_time", "mean"),
            mean_makespan=("makespan", "mean"),
            mean_cpu_time_utilization=("cpu_time_utilization", "mean"),
            mean_memory_time_utilization=("memory_time_utilization", "mean"),
            mean_invalid_actions=("invalid_actions", "mean"),
            all_invariants_pass=("resource_invariants_pass", "all"),
        )
    )

    episode_path = output_dir / "maskable_reward_model_evaluation_episodes.csv"
    summary_path = output_dir / "maskable_reward_model_evaluation_summary.csv"
    details_path = output_dir / "maskable_reward_model_evaluation_task_details.csv"

    episode_df.to_csv(episode_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    details_df.to_csv(details_path, index=False)

    print("\n" + "=" * 88)
    print("MASKABLE PPO EVALUATION SUMMARY")
    print("Lower wait, flow time, and makespan are generally better.")
    print("Compare episode reward only within the same reward formulation.")
    print("=" * 88)
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSaved files:")
    print(f"  Per-episode metrics: {episode_path}")
    print(f"  Summary:             {summary_path}")
    print(f"  Per-task details:    {details_path}")
    print(
        "\nNote: baselines and PPO have different placement decision procedures; "
        "interpret their comparison accordingly."
    )


if __name__ == "__main__":
    main()
