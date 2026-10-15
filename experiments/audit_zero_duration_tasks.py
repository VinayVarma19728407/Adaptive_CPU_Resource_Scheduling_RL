
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCENARIO_PATH = (
    ROOT / "experiments" / "scenarios" / "heavy_scenario_seed42.csv"
)
RESULTS_DIR = ROOT / "experiments" / "results"

POLICIES = ["first_fit", "least_loaded"]


def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    print(f"\nFile: {path.name}")
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())

    return df


def main():
    print("=" * 70)
    print("ZERO-DURATION TASK INVESTIGATION")
    print("=" * 70)

    scenario = load_csv(SCENARIO_PATH)

    if "scenario_task_id" not in scenario.columns:
        raise ValueError(
            "Scenario has no scenario_task_id column. "
            "Inspect the printed columns before proceeding."
        )

    print("\nSCENARIO SAMPLE (first 5 rows)")
    print(scenario.head().to_string(index=False))

    # The scenario may store duration under a different name.
    duration_candidates = [
        "baseline_duration",
        "duration",
        "task_duration",
        "runtime",
        "duration_seconds",
    ]

    scenario_duration_col = next(
        (col for col in duration_candidates if col in scenario.columns),
        None,
    )

    if scenario_duration_col is None:
        print(
            "\nNo recognized duration column exists in the scenario."
        )
        print(
            "The result files contain baseline_duration, so the "
            "scenario-to-result comparison will be limited to task IDs."
        )
    else:
        scenario[scenario_duration_col] = pd.to_numeric(
            scenario[scenario_duration_col], errors="raise"
        )

        print("\nSCENARIO DURATION COLUMN:", scenario_duration_col)
        print(scenario[scenario_duration_col].describe().to_string())
        print(
            "Zero scenario durations:",
            int((scenario[scenario_duration_col] == 0).sum()),
        )
        print(
            "Negative scenario durations:",
            int((scenario[scenario_duration_col] < 0).sum()),
        )

    for policy in POLICIES:
        result_path = (
            RESULTS_DIR / f"heavy_{policy}_seed42_results.csv"
        )
        results = load_csv(result_path)

        required = {
            "scenario_task_id",
            "start_time",
            "completion_time",
            "baseline_duration",
            "status",
        }

        missing = required - set(results.columns)
        if missing:
            raise ValueError(
                f"{result_path.name} is missing columns: "
                f"{sorted(missing)}"
            )

        completed = results[
            results["status"].astype(str).str.strip().str.lower()
            == "completed"
        ].copy()

        numeric_columns = [
            "start_time",
            "completion_time",
            "baseline_duration",
        ]

        for column in numeric_columns:
            completed[column] = pd.to_numeric(
                completed[column], errors="raise"
            )

        completed["observed_duration"] = (
            completed["completion_time"] - completed["start_time"]
        )

        zero_observed = completed[
            completed["observed_duration"] == 0
        ].copy()

        negative_observed = completed[
            completed["observed_duration"] < 0
        ].copy()

        print("\n" + "=" * 70)
        print(f"RESULT AUDIT: {policy}")
        print("=" * 70)

        print("Completed tasks:", len(completed))
        print("Zero observed durations:", len(zero_observed))
        print("Negative observed durations:", len(negative_observed))
        print(
            "Zero baseline_duration values in results:",
            int((completed["baseline_duration"] == 0).sum()),
        )
        print(
            "Negative baseline_duration values in results:",
            int((completed["baseline_duration"] < 0).sum()),
        )

        if not zero_observed.empty:
            display_columns = [
                "scenario_task_id",
                "task_name",
                "job_name",
                "start_time",
                "completion_time",
                "baseline_duration",
                "wait_time",
                "flow_time",
                "machine_id",
            ]
            display_columns = [
                col for col in display_columns if col in completed.columns
            ]

            print("\nZero observed-duration task records:")
            print(
                zero_observed[display_columns]
                .sort_values("scenario_task_id")
                .to_string(index=False)
            )

        if not negative_observed.empty:
            print("\nERROR: Negative observed durations found:")
            print(negative_observed.to_string(index=False))

        # Confirm that result task IDs exist in the scenario.
        scenario_ids = set(scenario["scenario_task_id"])
        result_ids = set(completed["scenario_task_id"])

        print(
            "\nCompleted result IDs missing from scenario:",
            len(result_ids - scenario_ids),
        )
        print(
            "Scenario IDs without completed results:",
            len(scenario_ids - result_ids),
        )

        if scenario_duration_col is not None:
            scenario_durations = scenario[
                ["scenario_task_id", scenario_duration_col]
            ].rename(
                columns={
                    scenario_duration_col: "scenario_duration"
                }
            )

            comparison = completed.merge(
                scenario_durations,
                on="scenario_task_id",
                how="left",
                validate="one_to_one",
            )

            print(
                "Result/scenario duration mismatches:",
                int(
                    (
                        comparison["baseline_duration"]
                        != comparison["scenario_duration"]
                    ).sum(),
                ),
            )

            if not zero_observed.empty:
                print("\nZero-duration tasks joined to scenario data:")
                zero_comparison = zero_observed.merge(
                    scenario_durations,
                    on="scenario_task_id",
                    how="left",
                    validate="one_to_one",
                )
                print(zero_comparison.to_string(index=False))


if __name__ == "__main__":
    main()
