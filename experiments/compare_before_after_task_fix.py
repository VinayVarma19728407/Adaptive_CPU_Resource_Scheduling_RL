from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]
folder = root / "experiments" / "evaluation"

before = pd.read_csv(
    folder / "maskable_reward_model_evaluation_episodes_before_task_fix.csv"
)
after = pd.read_csv(
    folder / "maskable_reward_model_evaluation_episodes.csv"
)

metrics = [
    "mean_wait_time",
    "total_wait_time",
    "mean_flow_time",
    "makespan",
]

methods = [
    "first_fit",
    "least_loaded",
    "ppo_maskable_wait_only",
    "ppo_maskable_core_penalty",
    "ppo_maskable_extended_penalty",
]

print("\nBEFORE VS AFTER TASK-SELECTION CHANGE")
print("=" * 75)

for method in methods:
    old = before[before["method"] == method].set_index("seed")
    new = after[after["method"] == method].set_index("seed")

    common = old.index.intersection(new.index)
    old = old.loc[common].sort_index()
    new = new.loc[common].sort_index()

    print(f"\n{method} | matched seeds: {len(common)}")

    for metric in metrics:
        delta = new[metric] - old[metric]
        print(
            f"  {metric}: "
            f"before={old[metric].mean():.4f}, "
            f"after={new[metric].mean():.4f}, "
            f"change={delta.mean():+.4f}"
        )

    if method in ("first_fit", "least_loaded"):
        unchanged = all(
            (old[m] - new[m]).abs().max() < 1e-9
            for m in metrics
        )
        print(f"  Baseline metrics unchanged: {unchanged}")

print("\nComparison complete.")