from pathlib import Path

import numpy as np
import pandas as pd

from simulator.config import PROJECT_ROOT
from simulator.rl_environment import RLSchedulingEnvironment
from experiments.train_three_maskable_reward_models_post_fix import (
    create_machines,
    create_scenario,
)


SEEDS = list(range(2000, 2050))

REWARD_MODES = [
    "wait_only",
    "core_penalty",
    "extended_penalty",
]

COMPONENTS = [
    "wait",
    "makespan",
    "invalid",
    "unschedulable",
    "imbalance",
    "fragmentation",
    "saturation",
    "starvation",
]


def main():
    rows = []

    for seed in SEEDS:
        scenario = create_scenario(seed)
        machines = create_machines()

        for mode in REWARD_MODES:
            env = RLSchedulingEnvironment(
                scenario=scenario,
                machine_metadata=machines,
                candidate_count=2,
                reward_mode=mode,
            )

            observation, _ = env.reset(seed=seed)
            terminated = False
            truncated = False
            episode_reward = 0.0
            decisions = 0

            while not (terminated or truncated):
                mask = np.asarray(env.action_masks(), dtype=bool)

                if not mask.any():
                    raise RuntimeError(
                        f"No valid action: seed={seed}, mode={mode}"
                    )

                # Use a fixed valid action rule to measure reward signals
                # without introducing different random action sequences.
                valid_actions = np.flatnonzero(mask)
                action = int(valid_actions[0])

                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(action)

                episode_reward += float(reward)
                decisions += 1

                record = {
                    "seed": seed,
                    "reward_mode": mode,
                    "decision": decisions,
                    "reward": float(reward),
                }

                weighted = info.get("penalty_components", {})

                for component in COMPONENTS:
                    record[component] = float(
                        weighted.get(component, 0.0)
                    )

                rows.append(record)

            assert env.sim is not None
            env.sim.machine_state.validate_invariants()

            if env.sim.machine_state.active_allocation_count != 0:
                raise RuntimeError(
                    f"Active allocations remain: seed={seed}, mode={mode}"
                )

            print(
                f"seed={seed}, mode={mode}, "
                f"decisions={decisions}, "
                f"episode_reward={episode_reward:.4f}"
            )

            env.close()

    results = pd.DataFrame(rows)

    output_dir = PROJECT_ROOT / "experiments" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / "reward_component_diagnostic_raw.csv"
    summary_path = output_dir / "reward_component_diagnostic_summary.csv"

    results.to_csv(raw_path, index=False)

    summary_rows = []

    for mode in REWARD_MODES:
        subset = results[results["reward_mode"] == mode]

        for component in COMPONENTS:
            values = subset[component]
            nonzero = values.abs() > 1e-12

            summary_rows.append({
                "reward_mode": mode,
                "component": component,
                "decision_count": int(len(values)),
                "nonzero_decisions": int(nonzero.sum()),
                "nonzero_fraction": float(nonzero.mean()),
                "mean_weighted_penalty": float(values.mean()),
                "median_weighted_penalty": float(values.median()),
                "total_weighted_penalty": float(values.sum()),
                "max_weighted_penalty": float(values.max()),
            })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_path, index=False)

    print("\nREWARD COMPONENT DIAGNOSTIC")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nSaved files:")
    print(raw_path)
    print(summary_path)


if __name__ == "__main__":
    main()