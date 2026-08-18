import pandas as pd

DATA_FILE = (
    "C:/Users/gvkva/Desktop/My ongoing projects/"
    "Machine Learning Project/Adaptive-Multi-Core-CPU-Scheduler-RL/"
    "data/processed/ml_cpu_historical.csv"
)

df = pd.read_csv(DATA_FILE)

historical_columns = [
    "Task_Previous_Count",
    "Task_Historical_Mean",
    "Task_Historical_Std",
    "Task_Historical_Min",
    "Task_Historical_Max",
    "PC_Previous_Count",
    "PC_Historical_Mean",
    "PC_Historical_Std",
    "PC_Historical_Min",
    "PC_Historical_Max",
]

print("=" * 60)
print("HISTORICAL FEATURE CHECK")
print("=" * 60)

print(f"Rows: {len(df)}")

print("\n" + "=" * 60)
print("HISTORICAL FEATURE SUMMARY")
print("=" * 60)

print(df[historical_columns].describe().T)

print("\n" + "=" * 60)
print("ROWS WITH NO PREVIOUS TASK OBSERVATION")
print("=" * 60)

print(
    (df["Task_Previous_Count"] == 0).sum()
)

print("\n" + "=" * 60)
print("ROWS WITH NO PREVIOUS PC OBSERVATION")
print("=" * 60)

print(
    (df["PC_Previous_Count"] == 0).sum()
)

print("\n" + "=" * 60)
print("ROWS WITH TASK HISTORY")
print("=" * 60)

print(
    (df["Task_Previous_Count"] > 0).sum()
)

print("\n" + "=" * 60)
print("ROWS WITH PC HISTORY")
print("=" * 60)

print(
    (df["PC_Previous_Count"] > 0).sum()
)

print("\n" + "=" * 60)
print("CORRELATION WITH CPU REQUIREMENT")
print("=" * 60)

numeric_columns = historical_columns + [
    "Memory_Requirement",
    "Arrival_Time",
    "CPU_Requirement",
]

print(
    df[numeric_columns]
    .corr()["CPU_Requirement"]
    .sort_values(ascending=False)
)