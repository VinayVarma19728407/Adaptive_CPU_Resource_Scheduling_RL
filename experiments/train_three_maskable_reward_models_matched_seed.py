from pathlib import Path

from simulator.config import PROJECT_ROOT
from experiments.train_three_maskable_reward_models_post_fix import (
    BASE_SEED,
    train_model,
)


TRAINING_SEEDS = [42]

CONFIGURATIONS = [
    (
        "ppo_maskable_wait_only_matched_seed42_20k",
        "wait_only",
    ),
    (
        "ppo_maskable_core_penalty_matched_seed42_20k",
        "core_penalty",
    ),
    (
        "ppo_maskable_extended_penalty_matched_seed42_20k",
        "extended_penalty",
    ),
]


def main() -> None:
    output_dir = PROJECT_ROOT / "experiments" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    for seed in TRAINING_SEEDS:
        print(f"\nTraining all reward modes with matched seed {seed}")

        for model_name, reward_mode in CONFIGURATIONS:
            print(f"\nModel: {model_name}")
            print(f"Reward mode: {reward_mode}")
            print(f"Training seed: {seed}")

            train_model(
                model_name=model_name,
                reward_mode=reward_mode,
                seed=seed,
                output_dir=output_dir,
            )

    print("\nMatched-seed training runs completed.")
    print(f"Models saved under: {output_dir}")


if __name__ == "__main__":
    main()