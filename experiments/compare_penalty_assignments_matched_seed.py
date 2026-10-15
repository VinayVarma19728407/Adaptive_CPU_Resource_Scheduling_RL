from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "experiments" / "evaluation"

DETAILS_PATH = (
    EVAL
    / "maskable_reward_model_evaluation_matched_seed42_20k_task_details.csv"
)

OUTPUT_PATH = (
    EVAL
    / "penalty_model_assignment_comparison_matched_seed42_20k.csv"
)


def main():
    df = pd.read_csv(DETAILS_PATH)

    core = df[
        df["method"] == "ppo_maskable_core_penalty"
    ].copy()

    extended = df[
        df["method"] == "ppo_maskable_extended_penalty"
    ].copy()

    keys = ["seed", "scenario_task_id"]

    if core.duplicated(keys).any():
        raise ValueError("Duplicate core-penalty task records found.")

    if extended.duplicated(keys).any():
        raise ValueError("Duplicate extended-penalty task records found.")

    comparison = core[
        keys + ["machine_id", "status", "start_time", "completion_time"]
    ].merge(
        extended[
            keys + [
                "machine_id",
                "status",
                "start_time",
                "completion_time",
            ]
        ],
        on=keys,
        how="outer",
        suffixes=("_core", "_extended"),
        indicator=True,
        validate="one_to_one",
    )

    unmatched = comparison["_merge"] != "both"

    matched = comparison[~unmatched].copy()

    different_machine = (
        matched["machine_id_core"] != matched["machine_id_extended"]
    )

    different_status = (
        matched["status_core"] != matched["status_extended"]
    )

    different_start = (
        matched["start_time_core"] != matched["start_time_extended"]
    )

    different_completion = (
        matched["completion_time_core"]
        != matched["completion_time_extended"]
    )

    comparison["machine_assignment_differs"] = False
    comparison.loc[matched.index, "machine_assignment_differs"] = (
        different_machine
    )

    comparison["status_differs"] = False
    comparison.loc[matched.index, "status_differs"] = different_status

    comparison.to_csv(OUTPUT_PATH, index=False)

    print("MATCHED-SEED PENALTY MODEL ASSIGNMENT COMPARISON")
    print(f"Core task records: {len(core)}")
    print(f"Extended task records: {len(extended)}")
    print(f"Unmatched task keys: {int(unmatched.sum())}")
    print(f"Matched task records: {len(matched)}")
    print(f"Different machine assignments: {int(different_machine.sum())}")
    print(f"Different task statuses: {int(different_status.sum())}")
    print(f"Different start times: {int(different_start.sum())}")
    print(f"Different completion times: {int(different_completion.sum())}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()