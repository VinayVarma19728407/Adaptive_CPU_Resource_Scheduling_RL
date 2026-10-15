from pathlib import Path

import pandas as pd

from simulator.config import (
    TRAINING_FILE,
    VALIDATION_FILE,
    TESTING_FILE,
    BATCH_TASK_FILE,
)
from simulator.task_data import load_machine_metadata

ROOT = Path(__file__).resolve().parents[1]
POOL_FILE = ROOT / "experiments" / "data" / "eligible_task_pool_seed42.csv"


def count_csv_rows(path, chunksize=250_000):
    if not path.exists():
        print(f"Missing: {path}")
        return None

    total = 0
    for chunk in pd.read_csv(path, chunksize=chunksize):
        total += len(chunk)

    return total


def main():
    print("DATASET SCALE AUDIT")

    machines = load_machine_metadata()
    print(f"Eligible machines: {len(machines):,}")
    print("Machine CPU capacity:")
    print(machines["cpu_num"].describe().to_string())
    print("Machine memory capacity:")
    print(machines["mem_size"].describe().to_string())

    for name, path in [
        ("Batch task definitions", BATCH_TASK_FILE),
        ("Training instances", TRAINING_FILE),
        ("Validation instances", VALIDATION_FILE),
        ("Testing instances", TESTING_FILE),
        ("Eligible task pool", POOL_FILE),
    ]:
        count = count_csv_rows(path)
        if count is not None:
            print(f"{name}: {count:,} rows")

    if POOL_FILE.exists():
        pool = pd.read_csv(POOL_FILE, nrows=5)
        print("\nTask-pool columns:")
        print(pool.columns.tolist())
        print("\nSample task-pool rows:")
        print(pool.to_string(index=False))


if __name__ == "__main__":
    main()