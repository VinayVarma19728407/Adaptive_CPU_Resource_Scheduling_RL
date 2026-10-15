from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "experiments" / "evaluation"

INPUT_PATH = (
    EVAL
    / "maskable_reward_model_evaluation_matched_seed42_20k_task_details.csv"
)

OUTPUT_PATH = (
    EVAL
    / "changed_penalty_assignments_detailed_matched_seed42_20k.csv"
)


def main():
    df = pd.read_csv(INPUT_PATH)

    methods = {
        "core": "ppo_maskable_core_penalty",
        "extended": "ppo_maskable_extended_penalty",
    }

    keys = ["seed", "scenario_task_id"]

    metrics = [
        "machine_id",
        "arrival_time",
        "start_time",
        "completion_time",
        "wait_time",
        "flow_time",
        "status",
    ]

    core = df[df["method"] == methods["core"]][keys + metrics].copy()
    extended = df[
        df["method"] == methods["extended"]
    ][keys + metrics].copy()

    if core.duplicated(keys).any() or extended.duplicated(keys).any():
        raise ValueError("Duplicate task keys detected.")

    comparison = core.merge(
        extended,
        on=keys,
        how="outer",
        suffixes=("_core", "_extended"),
        indicator=True,
        validate="one_to_one",
    )

    if not (comparison["_merge"] == "both").all():
        raise ValueError("Core and extended task sets do not match.")

    changed = comparison[
        comparison["machine_id_core"]
        != comparison["machine_id_extended"]
    ].copy()

    changed["wait_delta_extended_minus_core"] = (
        changed["wait_time_extended"] - changed["wait_time_core"]
    )

    changed["flow_delta_extended_minus_core"] = (
        changed["flow_time_extended"] - changed["flow_time_core"]
    )

    changed["completion_delta_extended_minus_core"] = (
        changed["completion_time_extended"]
        - changed["completion_time_core"]
    )

    changed.to_csv(OUTPUT_PATH, index=False)

    print("CHANGED MACHINE ASSIGNMENTS")
    print(f"Changed assignment records: {len(changed)}")
    print(f"Saved detailed records: {OUTPUT_PATH}")

    if changed.empty:
        print("No changed assignments found.")
        return

    print("\nTiming changes for the changed assignments")
    for column in [
        "wait_delta_extended_minus_core",
        "flow_delta_extended_minus_core",
        "completion_delta_extended_minus_core",
    ]:
        values = changed[column]
        print(
            f"{column}: "
            f"improved={(values < 0).sum()}, "
            f"unchanged={(values == 0).sum()}, "
            f"worsened={(values > 0).sum()}, "
            f"mean_delta={values.mean():.4f}"
        )

    print("\nChanged assignments by seed")
    print(changed.groupby("seed").size().to_string())

    print("\nSample changed assignments")
    print(
        changed[
            keys
            + [
                "machine_id_core",
                "machine_id_extended",
                "wait_delta_extended_minus_core",
                "completion_delta_extended_minus_core",
            ]
        ].head(15).to_string(index=False)
    )


if __name__ == "__main__":
    main()