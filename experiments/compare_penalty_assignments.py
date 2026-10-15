from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parents[1]
path = (
    root
    / "experiments"
    / "evaluation"
    / "maskable_reward_model_evaluation_post_fix_20k_task_details.csv"
)

details = pd.read_csv(path)

core_name = "ppo_maskable_core_penalty"
extended_name = "ppo_maskable_extended_penalty"

key_columns = ["seed", "scenario_task_id"]
required_columns = key_columns + ["method", "machine_id"]

missing = set(required_columns) - set(details.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

selected = details[
    details["method"].isin([core_name, extended_name])
].copy()

if selected.duplicated(key_columns + ["method"]).any():
    raise ValueError("Duplicate task records found for a seed and method.")

core = (
    selected[selected["method"] == core_name]
    [key_columns + ["machine_id"]]
    .rename(columns={"machine_id": "core_machine"})
)

extended = (
    selected[selected["method"] == extended_name]
    [key_columns + ["machine_id"]]
    .rename(columns={"machine_id": "extended_machine"})
)

comparison = core.merge(
    extended,
    on=key_columns,
    how="outer",
    indicator=True,
    validate="one_to_one",
)

missing_rows = comparison["_merge"] != "both"
different_machine = (
    comparison["core_machine"] != comparison["extended_machine"]
).fillna(True)

print("PENALTY MODEL PLACEMENT COMPARISON")
print(f"Core model task records: {len(core)}")
print(f"Extended model task records: {len(extended)}")
print(f"Unmatched task records: {int(missing_rows.sum())}")
print(
    "Tasks assigned to different machines:",
    int((~missing_rows & different_machine).sum()),
)

if missing_rows.any() or different_machine.any():
    print("\nFirst 20 discrepancies:")
    print(
        comparison.loc[
            missing_rows | different_machine,
            key_columns + [
                "core_machine",
                "extended_machine",
                "_merge",
            ],
        ].head(20).to_string(index=False)
    )
else:
    print("Every task has the same machine assignment in both models.")

comparison_path = (
    root
    / "experiments"
    / "evaluation"
    / "penalty_model_assignment_comparison_post_fix_20k.csv"
)

comparison.to_csv(comparison_path, index=False)
print(f"\nSaved comparison: {comparison_path}")