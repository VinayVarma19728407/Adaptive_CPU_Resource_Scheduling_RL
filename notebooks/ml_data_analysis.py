import pandas as pd
FILE_PATH = "C:/Users/gvkva/Desktop/My ongoing projects/Machine Learning Project/Adaptive-Multi-Core-CPU-Scheduler-RL/data/processed/processedscheduler_dataset.csv"
df = pd.read_csv(FILE_PATH)
print("=" * 60)
print("COLUMN INFORMATION")
print("=" * 60)
df.info()
print("\n" + "=" * 60)
print("COLUMN NAMES")
print("=" * 60)
print(df.columns.tolist())
print("\n" + "=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)
print(df.head())
print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)
print(df.isnull().sum())
print("\n" + "=" * 60)
print("UNIQUE VALUES PER COLUMN")
print("=" * 60)
print(df.nunique())
print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)
print(df["CPU_Requirement"].value_counts().sort_index())
print("\n" + "=" * 60)
print("REQUIRED CORES vs CPU REQUIREMENT")
print("=" * 60)
print(
    df.groupby("Required_Cores")["CPU_Requirement"]
      .agg(["count", "min", "max", "mean", "std"])
)
print("\n" + "=" * 60)
print("NUMERICAL CORRELATION WITH CPU REQUIREMENT")
print("=" * 60)
numeric_columns = [
    "Arrival_Time",
    "Completion_Time",
    "Burst_Time",
    "CPU_Requirement",
    "Memory_Requirement",
    "Required_Cores"
]
print(df[numeric_columns].corr()["CPU_Requirement"].sort_values(ascending=False))
print("\n" + "=" * 60)
print("CPU REQUIREMENT BY TASK")
print("=" * 60)
print(
    df.groupby("Task_ID")["CPU_Requirement"]
      .agg(["count", "min", "max", "mean"])
      .sort_values("count", ascending=False)
      .head(20)
)