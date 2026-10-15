# =============================================================================
# FILE: compare_declared_instance_counts.py
# ROLE IN PROJECT:
#   A lightweight sanity-check script that compares the total number of
#   instance rows expected (sum of instance_num from batch_task.csv) against
#   the total rows actually present across the three preprocessed split files:
#     - batch_instance_training.csv
#     - batch_instance_validation.csv
#     - batch_instance_testing.csv
#
#   The declared-vs-observed difference helps confirm whether the preprocessing
#   pipeline preserved all instance records or dropped some (e.g., due to
#   invalid durations, missing fields, or split-boundary rounding).
#
#   Input:  dataset/preprocessed/batch_task.csv and the three split files
#   Output: printed count comparison to stdout (no files written)
# =============================================================================
import pandas as pd
from pathlib import Path

task_file = Path("dataset/preprocessed/batch_task.csv")

instance_files = [
    Path("dataset/preprocessed/batch_instance_training.csv"),
    Path("dataset/preprocessed/batch_instance_validation.csv"),
    Path("dataset/preprocessed/batch_instance_testing.csv"),
]

print("CALCULATING EXPECTED INSTANCE COUNT")
tasks = pd.read_csv(
    task_file,
    usecols=["instance_num"],
    dtype={"instance_num": "int64"},
)

expected = int(tasks["instance_num"].sum())

print(f"Task definitions: {len(tasks):,}")
print(f"Sum of instance_num: {expected:,}")
print(f"Definitions with instance_num > 1: "
      f"{(tasks['instance_num'] > 1).sum():,}")

print("\nCOMPARING WITH INSTANCE SPLIT ROW COUNTS")
total_rows = 0

for path in instance_files:
    if not path.exists():
        print(f"MISSING: {path}")
        continue

    # Count CSV data records without loading the file into pandas.
    rows = 0
    with path.open("rb") as f:
        for line in f:
            rows += 1

    rows = max(0, rows - 1)  # Exclude header
    total_rows += rows
    print(f"{path.name}: {rows:,} rows")

print(f"\nTotal instance rows across splits: {total_rows:,}")
print(f"Declared-to-observed row difference: {expected - total_rows:,}")
print("NOTE: A difference is diagnostic, not proof of a data error.")
