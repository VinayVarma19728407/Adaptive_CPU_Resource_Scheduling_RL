import pandas as pd

DATA_FILE = (
    "C:/Users/gvkva/Desktop/My ongoing projects/"
    "Machine Learning Project/Adaptive-Multi-Core-CPU-Scheduler-RL/"
    "data/processed/ml_cpu_dataset.csv"
)

df = pd.read_csv(DATA_FILE)

df = df.sort_values("Arrival_Time").reset_index(drop=True)

print("=" * 60)
print("TARGET BEHAVIOR OVER TIME")
print("=" * 60)

print(f"Rows: {len(df)}")

# ---------------------------------------------------------
# Divide data into 10 chronological periods
# ---------------------------------------------------------

df["Time_Bin"] = pd.qcut(
    df["Arrival_Time"],
    q=10,
    duplicates="drop"
)

summary = (
    df.groupby("Time_Bin", observed=True)["CPU_Requirement"]
      .agg([
          "count",
          "mean",
          "median",
          "std",
          "min",
          "max"
      ])
)

print("\n" + "=" * 60)
print("CPU REQUIREMENT BY TIME PERIOD")
print("=" * 60)

print(summary.to_string())


# ---------------------------------------------------------
# CPU distribution
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("CPU REQUIREMENT PERCENTILES")
print("=" * 60)

print(
    df["CPU_Requirement"].quantile(
        [0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999]
    )
)


# ---------------------------------------------------------
# Extreme values
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("EXTREME CPU VALUES")
print("=" * 60)

print(
    df.nlargest(
        20,
        "CPU_Requirement"
    )[
        [
            "Arrival_Time",
            "Task_ID",
            "Job_ID",
            "PC_ID",
            "Memory_Requirement",
            "CPU_Requirement"
        ]
    ].to_string(index=False)
)


# ---------------------------------------------------------
# CPU by task
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("TASK CPU STABILITY")
print("=" * 60)

task_stats = (
    df.groupby("Task_ID")["CPU_Requirement"]
      .agg(["count", "mean", "std", "min", "max"])
      .sort_values("count", ascending=False)
)

print(task_stats.head(20).to_string())