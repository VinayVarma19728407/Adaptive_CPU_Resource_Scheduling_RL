# =============================================================================
# FILE: show_heldout_ppo_comparison.py
# ROLE IN PROJECT:
#   A results-display script (read-only) for examining PPO performance on
#   held-out (unseen) evaluation scenarios after training.
#
#   Reads existing evaluation CSV files produced by an evaluate_*.py script
#   and prints:
#     1. Completion rate and invalid action counts per policy and seed.
#     2. Mean metrics (wait_time, flow_time, makespan, etc.) by profile/policy.
#     3. PAIRED waiting-time differences: PPO minus First-Fit, PPO minus
#        Least-Loaded, on a per-scenario basis.
#     4. Percent change in mean wait time (PPO vs each baseline).
#
#   The script includes an interpretation note: previously-computed 95%
#   confidence intervals included zero for every comparison, so no statistical
#   superiority can be claimed from these descriptive results alone.
#
#   Input:  experiments/results/ppo_20480_checkpoint_heldout_validation_*/
#           episode_metrics.csv
#   Output: printed comparison tables to stdout (no files written)
# =============================================================================
"""
Display held-out PPO metrics and compare PPO against First Fit and Least Loaded.

Run from the project root:
    python show_heldout_ppo_comparison.py

This script reads existing evaluation CSV files only. It does not retrain the model,
rerun the simulator, or modify the existing results.
"""

from pathlib import Path
import sys
import pandas as pd

RESULTS_DIR = Path("experiments/results/ppo_20480_checkpoint_heldout_validation_20261014")
METRICS_FILE = RESULTS_DIR / "episode_metrics.csv"

POLICIES = ["first_fit", "least_loaded", "maskable_ppo"]
METRICS = [
    "completion_rate",
    "mean_wait_time",
    "total_wait_time",
    "mean_flow_time",
    "makespan",
    "invalid_actions",
]


def main():
    if not METRICS_FILE.exists():
        print(f"ERROR: Results file not found:\n  {METRICS_FILE}")
        print("\nRun this script from the project root folder:")
        print(r" C:\Users\gvkva\Desktop\My ongoing projects\Machine Learning Project\Adaptive-Multi-Core-CPU-Scheduler-RL")
        sys.exit(1)

    df = pd.read_csv(METRICS_FILE)

    required = {"profile", "seed", "policy", "completed", *METRICS}
    missing = sorted(required - set(df.columns))
    if missing:
        print("ERROR: The results CSV is missing expected columns:", ", ".join(missing))
        print("Available columns:", ", ".join(df.columns))
        sys.exit(1)

    print("=" * 90)
    print("HELD-OUT PPO CHECKPOINT: METRICS AND BASELINE COMPARISON")
    print("=" * 90)
    print(f"Source: {METRICS_FILE}")
    print(f"Evaluation rows: {len(df)}")
    print(f"Profiles: {', '.join(sorted(df['profile'].unique()))}")
    print(f"Seeds: {', '.join(map(str, sorted(df['seed'].unique())))}")
    print(f"Policies: {', '.join(POLICIES)}")
    print("\nCOMPLETION AND ACTION VALIDITY")
    print("-" * 90)
    cols = ["profile", "seed", "policy", "completed", "completion_rate", "invalid_actions"]
    print(df[cols].sort_values(["profile", "seed", "policy"]).to_string(index=False))

    existing_metrics = [m for m in METRICS if m in df.columns]
    print("\nMEAN METRICS BY PROFILE AND POLICY")
    print("-" * 90)
    summary = (
        df.groupby(["profile", "policy"])[existing_metrics]
        .mean()
        .reset_index()
        .sort_values(["profile", "policy"])
    )
    print(summary.round(6).to_string(index=False))

    # Pair each policy on the same profile and seed, then compare PPO to each baseline.
    pivot = df.pivot(index=["profile", "seed"], columns="policy", values="mean_wait_time")
    expected_policies = set(POLICIES)
    if expected_policies.issubset(set(pivot.columns)):
        pivot["PPO_minus_FirstFit"] = pivot["maskable_ppo"] - pivot["first_fit"]
        pivot["PPO_minus_LeastLoaded"] = pivot["maskable_ppo"] - pivot["least_loaded"]

        print("\nPAIRED WAITING-TIME DIFFERENCES BY SCENARIO")
        print("(Negative differences favor PPO; positive differences favor the baseline.)")
        print("-" * 90)
        print(pivot[
            ["first_fit", "least_loaded", "maskable_ppo",
             "PPO_minus_FirstFit", "PPO_minus_LeastLoaded"]
        ].round(6).to_string())

        print("\nAVERAGE PAIRED WAITING-TIME DIFFERENCE BY PROFILE")
        print("-" * 90)
        paired_summary = pivot.groupby(level="profile")[
            ["PPO_minus_FirstFit", "PPO_minus_LeastLoaded"]
        ].agg(["mean", "std", "min", "max"])
        print(paired_summary.round(6).to_string())

        print("\nPERCENT CHANGE IN MEAN WAITING TIME (PPO VS BASELINE)")
        print("(Negative percentage means PPO reduced waiting time.)")
        print("-" * 90)
        avg_wait = df.groupby(["profile", "policy"])["mean_wait_time"].mean().unstack("policy")
        for profile, row in avg_wait.iterrows():
            for baseline in ["first_fit", "least_loaded"]:
                pct = (row["maskable_ppo"] - row[baseline]) / row[baseline] * 100
                print(f"{profile}: PPO vs {baseline}: {pct:+.3f}%")

    print("\nINTERPRETATION NOTE")
    print("-" * 90)
    print("These are descriptive results from three held-out seeds per profile.")
    print("Do not claim statistical superiority from these results alone; the previously")
    print("calculated paired 95% confidence intervals included zero for every comparison.")
    print("\nNo models or result files were changed.")


if __name__ == "__main__":
    main()
