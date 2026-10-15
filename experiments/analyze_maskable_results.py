from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "experiments" / "evaluation"

episodes = pd.read_csv(
    EVAL / "maskable_reward_model_evaluation_episodes.csv"
)

metrics = [
    "mean_wait_time",
    "total_wait_time",
    "mean_flow_time",
    "makespan",
]

methods = [
    "ppo_maskable_wait_only",
    "ppo_maskable_core_penalty",
    "ppo_maskable_extended_penalty",
]

baselines = ["first_fit", "least_loaded"]

print("\nPAIRED EVALUATION AUDIT")
print("=" * 75)

# Check that each seed has one episode per method.
counts = episodes.groupby("seed")["method"].nunique()
print(f"Seeds: {episodes['seed'].nunique()}")
print(f"Methods present for every seed: {(counts == episodes['method'].nunique()).all()}")

# Task counts should match across methods for a given seed.
task_counts = episodes.pivot(
    index="seed", columns="method", values="task_count"
)
print("\nTask-count consistency across methods:")
print("All methods have equal task counts per seed:",
      task_counts.nunique(axis=1).eq(1).all())

for metric in metrics:
    print(f"\n\nMETRIC: {metric} (lower is better)")
    print("-" * 75)

    pivot = episodes.pivot(
        index="seed", columns="method", values=metric
    )

    for model in methods:
        if model not in pivot.columns:
            print(f"\nMissing model: {model}")
            continue

        for baseline in baselines:
            if baseline not in pivot.columns:
                continue

            delta = pivot[model] - pivot[baseline]
            wins = int((delta < -1e-9).sum())
            ties = int((delta.abs() <= 1e-9).sum())
            losses = int((delta > 1e-9).sum())

            print(
                f"{model} vs {baseline}: "
                f"mean difference={delta.mean():.4f}, "
                f"wins={wins}, ties={ties}, losses={losses}"
            )

# Verify identical task IDs per seed across methods.
details_path = EVAL / "maskable_reward_model_evaluation_task_details.csv"
details = pd.read_csv(details_path)

print("\n\nTASK-SET CONSISTENCY AUDIT")
print("=" * 75)

task_sets = (
    details.groupby(["seed", "method"])["scenario_task_id"]
    .apply(lambda x: frozenset(x))
)

inconsistent = []
for seed in sorted(details["seed"].unique()):
    seed_sets = task_sets.loc[seed]
    if seed_sets.nunique() != 1:
        inconsistent.append(seed)

print("Seeds with different task-ID sets across methods:",
      inconsistent if inconsistent else "None")

print("\nAudit complete.")