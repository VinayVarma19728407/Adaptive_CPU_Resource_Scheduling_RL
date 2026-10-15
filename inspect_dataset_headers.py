# =============================================================================
# FILE: inspect_dataset_headers.py
# ROLE IN PROJECT:
#   A tiny utility script that prints the column header of every major dataset
#   CSV file in the project. Used during the data understanding phase to
#   quickly verify:
#     - Which columns are present in each file.
#     - Whether expected files exist on disk.
#
#   Files inspected (preprocessed and cleaned datasets):
#     - dataset/preprocessed/batch_task.csv
#     - dataset/preprocessed/batch_instance_training.csv
#     - dataset/preprocessed/batch_instance_validation.csv
#     - dataset/preprocessed/batch_instance_testing.csv
#     - dataset/preprocessed/machine_meta.csv
#     - dataset/cleaned/machine_usage.csv
#
#   Output: column names printed to stdout; "MISSING" if the file is absent.
# =============================================================================
import csv
from pathlib import Path

files = [
    Path("dataset/preprocessed/batch_task.csv"),
    Path("dataset/preprocessed/batch_instance_training.csv"),
    Path("dataset/preprocessed/batch_instance_validation.csv"),
    Path("dataset/preprocessed/batch_instance_testing.csv"),
    Path("dataset/preprocessed/machine_meta.csv"),
    Path("dataset/cleaned/machine_usage.csv"),
]

for path in files:
    print(f"\n{'=' * 80}")
    print(f"FILE: {path}")
    if not path.exists():
        print("MISSING")
        continue

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, None)

    print("COLUMNS:")
    print(header)
