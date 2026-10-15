# =============================================================================
# FILE: audit_task_multiplicity.py
# ROLE IN PROJECT:
#   A focused data-quality audit on the batch_task.csv dataset. It examines
#   the instance_num field (how many instances each task declares) and the
#   resource demand fields (plan_cpu, plan_mem, duration) to understand:
#     - Distribution and range of instance_num values.
#     - How many task definitions declare more than 1 instance.
#     - Whether any (task_name, job_name) composite keys are duplicated in
#       the task definition file, which would indicate data anomalies.
#
#   This is a one-off diagnostic run during the data understanding phase.
#   Results are printed to stdout and no files are written.
#
#   Input:  dataset/preprocessed/batch_task.csv
#   Output: printed summary statistics to stdout
# =============================================================================
import pandas as pd
from pathlib import Path

path = Path("dataset/preprocessed/batch_task.csv")

if not path.exists():
    raise FileNotFoundError(path)

print("READING TASK MULTIPLICITY AND RESOURCE FIELDS")
df = pd.read_csv(
    path,
    usecols=[
        "task_name",
        "job_name",
        "instance_num",
        "plan_cpu",
        "plan_mem",
        "duration",
    ],
    dtype={
        "task_name": "string",
        "job_name": "string",
    },
)

print(f"\nRows: {len(df):,}")
print(f"Unique (task_name, job_name) keys: "
      f"{df[['task_name', 'job_name']].drop_duplicates().shape[0]:,}")

for col in ["instance_num", "plan_cpu", "plan_mem", "duration"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"\n{col} summary:")
    print(df[col].describe(percentiles=[0.01, 0.25, 0.50, 0.75, 0.99]))

print("\nINSTANCE_NUM VALUE COUNTS (top 20):")
print(df["instance_num"].value_counts(dropna=False).head(20))

print("\nINSTANCE_NUM GROUPS:")
print(f"Missing: {df['instance_num'].isna().sum():,}")
print(f"Zero or negative: {(df['instance_num'] <= 0).sum():,}")
print(f"Exactly 1: {(df['instance_num'] == 1).sum():,}")
print(f"Greater than 1: {(df['instance_num'] > 1).sum():,}")

duplicate_keys = df.duplicated(["task_name", "job_name"], keep=False)
print(f"\nRows participating in duplicate composite keys: "
      f"{duplicate_keys.sum():,}")

if duplicate_keys.any():
    print("Example duplicate keys:")
    print(
        df.loc[duplicate_keys, ["task_name", "job_name"]]
          .drop_duplicates()
          .head(10)
          .to_string(index=False)
    )

print("\nAUDIT COMPLETE")
