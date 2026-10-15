from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO

from simulator.config import PROJECT_ROOT
from simulator.rl_environment import RLSchedulingEnvironment
from experiments.evaluate_three_maskable_reward_models_post_fix import (
    CANDIDATE_COUNT,
    EVALUATION_SEEDS,
    MODEL_CONFIGS,
    create_scenario,
    create_machines,
)


def main():
    root = Path(PROJECT_ROOT)
    model_dir = root / "experiments" / "models"
    output_dir = root / "experiments" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_rows = []
    component_rows = []

    models = {}

    for model_name, reward_mode in MODEL_CONFIGS:
        model_path = model_dir / f"{model_name}.zip"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        models[reward_mode] = MaskablePPO.load(
            str(model_path),
            device="cpu",
        )
        print(f"Loaded {model_path.name}")

    machine_metadata = create_machines()

    for seed in EVALUATION_SEEDS:
        scenario = create_scenario(seed)
        print(f"Auditing seed {seed}, tasks={len(scenario)}")

        for _, reward_mode in MODEL_CONFIGS:
            env = RLSchedulingEnvironment(
                scenario=scenario,
                machine_metadata=machine_metadata,
                candidate_count=CANDIDATE_COUNT,
                reward_mode=reward_mode,
            )

            observation, _ = env.reset(seed=seed)
            terminated = False
            truncated = False
            decision = 0
            episode_reward = 0.0
            episode_components = defaultdict(float)
            episode_nonzero = defaultdict(int)
            episode_max = defaultdict(float)

            while not (terminated or truncated):
                mask = np.asarray(env.action_masks(), dtype=bool)

                action, _ = models[reward_mode].predict(
                    observation,
                    deterministic=True,
                    action_masks=mask,
                )

                observation, reward, terminated, truncated, info = (
                    env.step(int(np.asarray(action).item()))
                )

                decision += 1
                episode_reward += float(reward)

                penalties = info["penalty_components"]

                row = {
                    "seed": seed,
                    "reward_mode": reward_mode,
                    "decision": decision,
                    "task_id": info.get("last_task_id"),
                    "machine_id": info.get("last_machine_id"),
                    "reward": float(reward),
                }

                for name, value in penalties.items():
                    value = float(value)
                    row[name] = value
                    episode_components[name] += value
                    episode_nonzero[name] += int(abs(value) > 1e-12)
                    episode_max[name] = max(
                        episode_max[name], value
                    )

                component_rows.append(row)

            episode_row = {
                "seed": seed,
                "reward_mode": reward_mode,
                "decisions": decision,
                "episode_reward": episode_reward,
            }

            for name, total in episode_components.items():
                episode_row[f"{name}_total"] = total
                episode_row[f"{name}_mean"] = total / max(1, decision)
                episode_row[f"{name}_nonzero_count"] = (
                    episode_nonzero[name]
                )
                episode_row[f"{name}_nonzero_rate"] = (
                    episode_nonzero[name] / max(1, decision)
                )
                episode_row[f"{name}_max"] = episode_max[name]

            episode_rows.append(episode_row)
            env.close()

    steps_df = pd.DataFrame(component_rows)
    episodes_df = pd.DataFrame(episode_rows)

    steps_path = (
        output_dir / "reward_component_audit_post_fix_20k_steps.csv"
    )
    episodes_path = (
        output_dir / "reward_component_audit_post_fix_20k_episodes.csv"
    )
    summary_path = (
        output_dir / "reward_component_audit_post_fix_20k_summary.csv"
    )

    steps_df.to_csv(steps_path, index=False)
    episodes_df.to_csv(episodes_path, index=False)

    component_names = [
        "wait",
        "makespan",
        "invalid",
        "unschedulable",
        "imbalance",
        "fragmentation",
        "saturation",
        "starvation",
    ]

    summary_rows = []

    for mode, group in steps_df.groupby("reward_mode"):
        for component in component_names:
            values = group[component].astype(float)

            summary_rows.append({
                "reward_mode": mode,
                "component": component,
                "decisions": len(values),
                "mean_weighted_penalty": values.mean(),
                "median_weighted_penalty": values.median(),
                "maximum_weighted_penalty": values.max(),
                "nonzero_decisions": int(
                    (values.abs() > 1e-12).sum()
                ),
                "nonzero_rate": float(
                    (values.abs() > 1e-12).mean()
                ),
                "total_weighted_penalty": values.sum(),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)

    print("\nReward component summary")
    print(
        summary_df.round(6).to_string(index=False)
    )

    print("\nEpisode reward totals")
    print(
        episodes_df.groupby("reward_mode")[
            ["episode_reward", "decisions"]
        ].mean().round(6).to_string()
    )

    print("\nSaved files:")
    print(steps_path)
    print(episodes_path)
    print(summary_path)


if __name__ == "__main__":
    main()