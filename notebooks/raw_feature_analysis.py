import pandas as pd

FILE_PATH = (
    "C:/Users/gvkva/Desktop/My ongoing projects/"
    "Machine Learning Project/Adaptive-Multi-Core-CPU-Scheduler-RL/"
    "data/raw/batch_instance.csv"
)

# Read only the first 1,000,000 rows.
# We already know the complete raw dataset is 1.35B rows,
# so this avoids another full 103GB scan.

columns = [
    "Process_ID",
    "Task_ID",
    "Job_ID",
    "Task_Type",
    "Status",
    "Start_Time",
    "End_Time",
    "PC_ID",
    "Sequence_No",
    "Total_Sequence",
    "CPU_Avg",
    "CPU_Max",
    "Memory_Avg",
    "Memory_Max"
]

df = pd.read_csv(
    FILE_PATH,
    header=None,
    names=columns,
    nrows=1_000_000,
    low_memory=False
)

print("=" * 60)
print("RAW FEATURE ANALYSIS")
print("=" * 60)

print(f"Rows analyzed: {len(df)}")

print("\n" + "=" * 60)
print("TASK TYPE")
print("=" * 60)

print(df["Task_Type"].value_counts().head(20))

print("\nUnique Task Types:", df["Task_Type"].nunique())


print("\n" + "=" * 60)
print("SEQUENCE NUMBER")
print("=" * 60)

print(df["Sequence_No"].describe())

print("\nUnique Sequence Numbers:", df["Sequence_No"].nunique())


print("\n" + "=" * 60)
print("TOTAL SEQUENCE")
print("=" * 60)

print(df["Total_Sequence"].describe())

print("\nUnique Total Sequence Values:", df["Total_Sequence"].nunique())


print("\n" + "=" * 60)
print("CPU AVERAGE vs CPU MAX")
print("=" * 60)

print(
    df[["CPU_Avg", "CPU_Max"]].corr()
)


print("\n" + "=" * 60)
print("MEMORY AVERAGE vs MEMORY MAX")
print("=" * 60)

print(
    df[["Memory_Avg", "Memory_Max"]].corr()
)


print("\n" + "=" * 60)
print("CPU MAX STATISTICS")
print("=" * 60)

print(df["CPU_Max"].describe())


print("\n" + "=" * 60)
print("MEMORY MAX STATISTICS")
print("=" * 60)

print(df["Memory_Max"].describe())