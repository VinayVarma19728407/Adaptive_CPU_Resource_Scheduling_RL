
from pathlib import Path
import math

import numpy as np
import pandas as pd


INPUT = Path(
    "experiments/results/"
    "final_candidate_multiseed_validation_20261014/"
    "episode_metrics.csv"
)


# Student-t critical values for a two-sided 95% CI.
# Five paired seeds means 4 degrees of freedom.
T_CRITICAL_DF4 = 2.776445105


def paired_summary(differences):
    differences = np.asarray(differences, dtype=float)
    n = len(differences)
    mean_diff = float(np.mean(differences))
    sd_diff = float(np.std(differences, ddof=1))
    se_diff = sd_diff / math.sqrt(n)

    low = mean_diff - T_CRITICAL_DF4 * se_diff
    high = mean_diff + T_CRITICAL_DF4 * se_diff

    return mean_diff, sd_diff, low, high


df = pd.read_csv(INPUT)

print("PAIRED MULTI-SEED STATISTICAL ANALYSIS")
print("=" * 72)
print("Negative PPO-minus-baseline differences favor PPO.")
print("Positive differences favor the baseline.")
print("Each comparison uses five paired scenario seeds.")
print("95% confidence intervals use the t critical value for df=4.")
print("No additional packages are required.")

for profile in df["profile"].unique():
    subset = df[df["profile"] == profile]

    print(f"\nPROFILE: {profile}")
    print("-" * 72)

    for metric in ["mean_wait_time", "total_wait_time", "makespan"]:
        wide = subset.pivot(
            index="seed",
            columns="policy",
            values=metric,
        )

        for baseline in ["first_fit", "least_loaded"]:
            differences = (
                wide["maskable_ppo"] - wide[baseline]
            ).to_numpy(dtype=float)

            mean_diff, sd_diff, low, high = paired_summary(differences)

            print(f"\nMetric: {metric} | Baseline: {baseline}")
            print("  Seed-level PPO-minus-baseline differences:",
                  np.round(differences, 6).tolist())
            print(f"  Mean paired difference: {mean_diff:.6f}")
            print(f"  SD of paired differences: {sd_diff:.6f}")
            print(f"  95% t confidence interval: [{low:.6f}, {high:.6f}]")

print("\nAnalysis complete. Original evaluation files were not modified.")
