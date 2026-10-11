
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_DIR = PROJECT_ROOT / "experiments" / "scenarios"

EXPECTED = {
    "light": 16,
    "moderate": 8,
    "heavy": 4,
}

SOURCE_ID_COLUMNS = ["task_name", "job_name", "task_type", "duration",
                     "plan_cpu", "plan_mem"]


def validate_scenario(name: str, interval: int) -> pd.DataFrame:
    path = SCENARIO_DIR / f"{name}_scenario_seed42.csv"

    assert path.exists(), f"Missing scenario file: {path}"

    df = pd.read_csv(path)

    required = [
        "scenario_task_id",
        "task_name",
        "job_name",
        "plan_cpu",
        "plan_mem",
        "duration",
        "arrival_time",
        "scenario_name",
        "scenario_seed",
        "inter_arrival_interval",
    ]

    missing_columns = sorted(set(required) - set(df.columns))
    assert not missing_columns, f"{name}: missing columns {missing_columns}"
    assert len(df) == 500, f"{name}: expected 500 tasks, found {len(df)}"
    assert df["scenario_task_id"].is_unique, f"{name}: duplicate scenario IDs"

    for column in ["plan_cpu", "plan_mem", "duration", "arrival_time"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        assert df[column].notna().all(), f"{name}: invalid {column}"

    assert (df["plan_cpu"] > 0).all(), f"{name}: nonpositive CPU demand"
    assert (df["plan_mem"] > 0).all(), f"{name}: nonpositive memory demand"
    assert (df["duration"] >= 0).all(), f"{name}: negative duration"

    arrivals = df["arrival_time"].tolist()
    assert arrivals[0] == 0, f"{name}: first arrival must be zero"
    assert all(
        arrivals[i] < arrivals[i + 1]
        for i in range(len(arrivals) - 1)
    ), f"{name}: arrival times are not strictly increasing"

    gaps = df["arrival_time"].diff().dropna()
    assert (gaps == interval).all(), f"{name}: unexpected arrival intervals"
    assert (
        df["inter_arrival_interval"] == interval
    ).all(), f"{name}: inconsistent interval metadata"
    assert (df["scenario_name"] == name).all(), f"{name}: wrong scenario label"
    assert (df["scenario_seed"] == 42).all(), f"{name}: wrong seed metadata"

    print(
        f"PASS {name:8s} | tasks={len(df)} | interval={interval} "
        f"| final_arrival={int(df['arrival_time'].iloc[-1])}"
    )
    return df


def main():
    scenarios = {
        name: validate_scenario(name, interval)
        for name, interval in EXPECTED.items()
    }

    # Compare source task characteristics, not synthetic scenario IDs/times.
    reference = scenarios["moderate"][SOURCE_ID_COLUMNS].reset_index(drop=True)

    for name in ["light", "heavy"]:
        comparison = scenarios[name][SOURCE_ID_COLUMNS].reset_index(drop=True)
        assert reference.equals(comparison), (
            f"{name}: selected task records differ from moderate scenario"
        )

    print("\nPASS: all three scenarios satisfy the initial validation contract.")
    print("PASS: task records match across scenarios.")
    print("Note: these checks validate files, not simulator behavior.")


if __name__ == "__main__":
    main()
