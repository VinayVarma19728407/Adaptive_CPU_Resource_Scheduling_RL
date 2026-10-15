from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parents[1]
path = (
    root
    / "experiments"
    / "evaluation"
    / "maskable_reward_model_evaluation_post_fix_20k_task_details.csv"
)

df = pd.read_csv(path)
df = df[df["status"] == "completed"].copy()

df["calculated_flow_time"] = (
    df["completion_time"] - df["arrival_time"]
)
df["calculated_wait_time"] = (
    df["start_time"] - df["arrival_time"]
)
df["execution_time"] = (
    df["completion_time"] - df["start_time"]
)

print("FLOW-TIME CONSISTENCY AUDIT")
print("Completed task records:", len(df))
print(
    "Maximum flow-time calculation error:",
    (df["flow_time"] - df["calculated_flow_time"]).abs().max(),
)
print(
    "Maximum wait-time calculation error:",
    (df["wait_time"] - df["calculated_wait_time"]).abs().max(),
)
print(
    "Maximum flow = wait + execution error:",
    (
        df["flow_time"]
        - df["wait_time"]
        - df["execution_time"]
    ).abs().max(),
)

print("\nMean execution time by method:")
print(
    df.groupby("method")["execution_time"]
    .mean()
    .sort_values()
    .to_string()
)

print("\nMean execution time by method and seed:")
print(
    df.groupby(["seed", "method"])["execution_time"]
    .mean()
    .unstack("method")
    .round(6)
    .to_string()
)