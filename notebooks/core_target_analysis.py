import pandas as pd


DATA_FILE = (
    "C:/Users/gvkva/Desktop/My ongoing projects/"
    "Machine Learning Project/Adaptive-Multi-Core-CPU-Scheduler-RL/"
    "data/processed/ml_cpu_dataset.csv"
)


df = pd.read_csv(DATA_FILE)


# =========================================================
# CREATE CORE TARGET
# =========================================================

def cpu_to_cores(cpu):
    if cpu <= 25:
        return 1
    elif cpu <= 50:
        return 2
    elif cpu <= 75:
        return 4
    else:
        return 8


df["Core_Target"] = df["CPU_Requirement"].apply(cpu_to_cores)


print("=" * 60)
print("CORE ALLOCATION TARGET ANALYSIS")
print("=" * 60)


# =========================================================
# CLASS DISTRIBUTION
# =========================================================

print("\n" + "=" * 60)
print("CORE CLASS DISTRIBUTION")
print("=" * 60)

print(
    df["Core_Target"]
    .value_counts()
    .sort_index()
)


print("\nPercentage:")

print(
    (
        df["Core_Target"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    ).round(2)
)


# =========================================================
# CPU RANGE PER CORE CLASS
# =========================================================

print("\n" + "=" * 60)
print("CPU RANGE PER CORE CLASS")
print("=" * 60)

print(
    df.groupby("Core_Target")["CPU_Requirement"]
      .agg([
          "count",
          "min",
          "max",
          "mean",
          "std"
      ])
)


# =========================================================
# CORE TARGET BY TIME
# =========================================================

df = df.sort_values("Arrival_Time")

df["Time_Bin"] = pd.qcut(
    df["Arrival_Time"],
    q=10,
    duplicates="drop"
)


print("\n" + "=" * 60)
print("CORE DISTRIBUTION OVER TIME")
print("=" * 60)

time_core = pd.crosstab(
    df["Time_Bin"],
    df["Core_Target"],
    normalize="index"
) * 100

print(
    time_core.round(2).to_string()
)