from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "experiments" / "evaluation"

EPISODES_PATH = (
    EVAL / "maskable_reward_model_evaluation_post_fix_20k_episodes.csv"
)
DETAILS_PATH = (
    EVAL / "maskable_reward_model_evaluation_post_fix_20k_task_details.csv"
)

episodes = pd.read_csv(EPISODES_PATH)
details = pd.read_csv(DETAILS_PATH)

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
all_methods = methods + baselines

print("\nPOST-FIX PPO PAIRED EVALUATION AUDIT")
print("=" * 80)
print(f"Episode file: {EPISODES_PATH.name}")
print(f"Task-detail file: {DETAILS_PATH.name}")

# ---------------------------------------------------------
# 1. Validate required columns
# ---------------------------------------------------------
required_episode_columns = {
    "seed", "method", "task_count", *metrics,
    "completion_rate", "unschedulable_tasks",
    "invalid_actions", "resource_invariants_pass",
}
required_detail_columns = {"seed", "method", "scenario_task_id"}

missing_episode = required_episode_columns - set(episodes.columns)
missing_details = required_detail_columns - set(details.columns)

if missing_episode:
    raise ValueError(
        f"Missing episode columns: {sorted(missing_episode)}"
    )

if missing_details:
    raise ValueError(
        f"Missing task-detail columns: {sorted(missing_details)}"
    )

# ---------------------------------------------------------
# 2. Check duplicate seed/method records
# ---------------------------------------------------------
duplicates = episodes.duplicated(
    subset=["seed", "method"], keep=False
)

print("\nDUPLICATE RECORD AUDIT")
print("-" * 80)

if duplicates.any():
    print(episodes.loc[duplicates, ["seed", "method"]].to_string(index=False))
    raise ValueError(
        "Duplicate seed/method episode rows found. Resolve these first."
    )

print("Duplicate episode rows: None")

# ---------------------------------------------------------
# 3. Check seed coverage and method coverage
# ---------------------------------------------------------
expected_seeds = set(range(1000, 1050))
actual_seeds = set(episodes["seed"].astype(int).unique())

print("\nSEED AND METHOD COVERAGE")
print("-" * 80)
print(f"Unique seeds: {len(actual_seeds)}")
print(f"Expected seeds 1000-1049 present: {actual_seeds == expected_seeds}")

present_methods = set(episodes["method"].unique())
print(f"Methods present: {sorted(present_methods)}")

missing_methods = set(all_methods) - present_methods
unexpected_methods = present_methods - set(all_methods)

print(f"Missing methods: {sorted(missing_methods)}")
print(f"Unexpected methods: {sorted(unexpected_methods)}")

method_counts = episodes.groupby("seed")["method"].nunique()
print(
    "Every seed has all five methods:",
    bool(
        method_counts.eq(len(all_methods)).all()
        and not missing_methods
        and not unexpected_methods
    ),
)

if actual_seeds != expected_seeds or missing_methods or unexpected_methods:
    raise ValueError(
        "Seed or method coverage is incomplete. Inspect the evaluation data."
    )

# ---------------------------------------------------------
# 4. Check task counts and missing metric values
# ---------------------------------------------------------
print("\nDATA COMPLETENESS AUDIT")
print("-" * 80)

task_counts = episodes.pivot(
    index="seed", columns="method", values="task_count"
)

print(
    "Equal task counts across methods for every seed:",
    bool(task_counts.nunique(axis=1).eq(1).all()),
)

missing_metrics = episodes[metrics].isna().sum()
print("Missing metric values:")
print(missing_metrics.to_string())

if missing_metrics.any():
    raise ValueError("Missing metric values found; paired analysis aborted.")

print(
    "All methods have 100% completion:",
    bool((episodes["completion_rate"] == 1.0).all()),
)
print(
    "All methods have zero unschedulable tasks:",
    bool((episodes["unschedulable_tasks"] == 0).all()),
)
print(
    "All methods have zero invalid actions:",
    bool((episodes["invalid_actions"] == 0).all()),
)
print(
    "All resource invariants pass:",
    bool(episodes["resource_invariants_pass"].astype(bool).all()),
)

# ---------------------------------------------------------
# 5. Paired comparisons on identical workload seeds
# ---------------------------------------------------------
print("\nPAIRED PERFORMANCE COMPARISONS")
print("=" * 80)
print("Delta = PPO metric - baseline metric.")
print("Negative delta is better for all four metrics below.")

summary_rows = []

for metric in metrics:
    print(f"\nMETRIC: {metric}")
    print("-" * 80)

    pivot = episodes.pivot(
        index="seed", columns="method", values=metric
    )

    for model in methods:
        for baseline in baselines:
            delta = (
                pivot[model].astype(float)
                - pivot[baseline].astype(float)
            )

            wins = int((delta < -1e-9).sum())
            ties = int((delta.abs() <= 1e-9).sum())
            losses = int((delta > 1e-9).sum())

            # Paired bootstrap confidence interval for the mean difference.
            # Resample workload-level differences, not individual tasks.
            rng = np.random.default_rng(20261011)
            values = delta.to_numpy()
            bootstrap_means = np.empty(10000)

            for i in range(len(bootstrap_means)):
                sample = rng.choice(
                    values, size=len(values), replace=True
                )
                bootstrap_means[i] = sample.mean()

            ci_low, ci_high = np.percentile(
                bootstrap_means, [2.5, 97.5]
            )

            row = {
                "metric": metric,
                "model": model,
                "baseline": baseline,
                "mean_delta": float(delta.mean()),
                "median_delta": float(delta.median()),
                "std_delta": float(delta.std(ddof=1)),
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
                "wins": wins,
                "ties": ties,
                "losses": losses,
            }
            summary_rows.append(row)

            print(
                f"\n{model} vs {baseline}\n"
                f"  Mean paired difference: {delta.mean():.6f}\n"
                f"  Median paired difference: {delta.median():.6f}\n"
                f"  Paired difference SD: {delta.std(ddof=1):.6f}\n"
                f"  Bootstrap 95% CI for mean difference: "
                f"[{ci_low:.6f}, {ci_high:.6f}]\n"
                f"  Wins / ties / losses: {wins} / {ties} / {losses}"
            )

# ---------------------------------------------------------
# 6. Compare the two penalty models directly
# ---------------------------------------------------------
print("\nCORE-PENALTY VS EXTENDED-PENALTY")
print("=" * 80)

penalty_pivot = episodes.pivot(
    index="seed", columns="method", values=metrics
)

for metric in metrics:
    core = penalty_pivot[(metric, "ppo_maskable_core_penalty")]
    extended = penalty_pivot[(metric, "ppo_maskable_extended_penalty")]
    equal = np.allclose(
        core.to_numpy(),
        extended.to_numpy(),
        rtol=0,
        atol=1e-12,
    )
    print(f"{metric}: identical across all seeds = {equal}")

# Check whether both policies chose the same machine for every task.
model_details = details[
    details["method"].isin(
        [
            "ppo_maskable_core_penalty",
            "ppo_maskable_extended_penalty",
        ]
    )
]

# Inspect available result columns before attempting action-level comparison.
print("\nPenalty-model detail columns:")
print(", ".join(model_details.columns))

# ---------------------------------------------------------
# 7. Verify identical task IDs per seed across methods
# ---------------------------------------------------------
print("\nTASK-SET CONSISTENCY AUDIT")
print("=" * 80)

task_sets = (
    details.groupby(["seed", "method"])["scenario_task_id"]
    .apply(lambda x: frozenset(x))
)

inconsistent = []

for seed in sorted(details["seed"].unique()):
    if seed not in task_sets.index.get_level_values("seed"):
        inconsistent.append(seed)
        continue

    seed_sets = task_sets.loc[seed]

    if len(seed_sets) != len(all_methods) or seed_sets.nunique() != 1:
        inconsistent.append(seed)

print(
    "Seeds with missing methods or different task-ID sets:",
    inconsistent if inconsistent else "None",
)

# ---------------------------------------------------------
# 8. Save paired-comparison results
# ---------------------------------------------------------
comparison_path = (
    EVAL / "maskable_reward_model_paired_comparison_post_fix_20k.csv"
)

pd.DataFrame(summary_rows).to_csv(comparison_path, index=False)

print("\nAnalysis complete.")
print(f"Saved paired comparisons to:\n{comparison_path}")