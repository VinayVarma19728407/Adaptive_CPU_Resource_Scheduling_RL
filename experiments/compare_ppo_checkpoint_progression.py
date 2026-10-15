import json
import time
from pathlib import Path

import pandas as pd
from sb3_contrib import MaskablePPO

import experiments.validate_final_candidate_multiseed as validation


OUTPUT_NAME = "ppo_checkpoint_progression_20261014"

RUN_DIR = validation.MODEL_PATH.parent
CHECKPOINT_DIR = RUN_DIR / "checkpoints"
OUTPUT_DIR = (
    validation.PROJECT_ROOT
    / "experiments"
    / "results"
    / OUTPUT_NAME
)

CHECKPOINTS = [
    (10240, CHECKPOINT_DIR / "final_dataset_ppo_10240_steps.zip"),
    (20480, CHECKPOINT_DIR / "final_dataset_ppo_20480_steps.zip"),
    (30720, CHECKPOINT_DIR / "final_dataset_ppo_30720_steps.zip"),
    (40960, CHECKPOINT_DIR / "final_dataset_ppo_40960_steps.zip"),
    (50176, validation.MODEL_PATH),
]


def main():
    start = time.perf_counter()

    # Protect previous experiments and avoid scanning data unnecessarily.
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output directory already exists: {OUTPUT_DIR}"
        )

    for steps, path in CHECKPOINTS:
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    print("PPO CHECKPOINT PROGRESSION EVALUATION")
    print("=" * 72)
    print(f"Output: {OUTPUT_DIR}")
    print(f"Checkpoint stages: {[steps for steps, _ in CHECKPOINTS]}")

    machines = validation.create_machine_pool()

    task_pool = validation.load_bounded_eligible_task_pool(
        max_cpu_capacity=float(machines["cpu_num"].max()),
        max_memory_capacity=float(machines["mem_size"].max()),
        sample_size=validation.TASK_POOL_SIZE,
        seed=validation.TASK_POOL_SEED,
    )

    if len(task_pool) < validation.TASK_COUNT:
        raise ValueError("Insufficient eligible task records.")

    # Generate each scenario only once, then reuse it for every checkpoint.
    scenarios = {}
    for profile in validation.PROFILE_NAMES:
        for seed in validation.SCENARIO_SEEDS:
            scenarios[(profile, seed)] = validation.create_scenario(
                task_pool, profile, seed
            )

    all_metrics = []
    task_outputs = []

    for steps, model_path in CHECKPOINTS:
        print(f"\nLoading checkpoint: {steps} steps")
        model = MaskablePPO.load(str(model_path), device="cpu")

        for (profile, seed), scenario in scenarios.items():
            results, final_time = validation.run_ppo(
                scenario, machines, model, seed
            )

            policy_name = f"ppo_{steps}_steps"
            record = validation.summarize(
                profile,
                seed,
                policy_name,
                scenario,
                results,
                final_time,
            )
            record["checkpoint_steps"] = steps
            all_metrics.append(record)

            results.insert(0, "checkpoint_steps", steps)
            results.insert(0, "evaluation_seed", seed)
            results.insert(0, "profile", profile)
            task_outputs.append(results)

            print(
                f"profile={profile} seed={seed} "
                f"completed={record['completed']}/{record['tasks']} "
                f"mean_wait={record['mean_wait_time']:.5f} "
                f"makespan={record['makespan']:.5f} "
                f"invalid_actions={record['invalid_actions']}"
            )

        del model

    metrics_df = pd.DataFrame(all_metrics)
    tasks_df = pd.concat(task_outputs, ignore_index=True)

    metrics_df.to_csv(
        OUTPUT_DIR / "checkpoint_metrics.csv", index=False
    )
    tasks_df.to_csv(
        OUTPUT_DIR / "checkpoint_task_results.csv", index=False
    )

    config = {
        "checkpoints": [
            {"steps": steps, "path": str(path)}
            for steps, path in CHECKPOINTS
        ],
        "machine_count": len(machines),
        "task_pool_size": len(task_pool),
        "tasks_per_scenario": validation.TASK_COUNT,
        "scenario_seeds": validation.SCENARIO_SEEDS,
        "profiles": validation.PROFILE_NAMES,
        "reward_mode": "extended_penalty",
        "same_scenarios_reused_for_every_checkpoint": True,
        "elapsed_seconds": round(time.perf_counter() - start, 3),
    }
    (OUTPUT_DIR / "evaluation_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print("\nCHECKPOINT MEAN METRICS")
    print(
        metrics_df.groupby(
            ["profile", "checkpoint_steps"]
        )[
            [
                "completion_rate",
                "mean_wait_time",
                "total_wait_time",
                "mean_flow_time",
                "makespan",
                "invalid_actions",
            ]
        ].mean().round(5).to_string()
    )

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"Elapsed seconds: {time.perf_counter() - start:.2f}")


if __name__ == "__main__":
    main()