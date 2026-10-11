
from pathlib import Path
import sys
import pandas as pd

# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from simulator.config import (
    MACHINE_META_FILE,
    ELIGIBLE_MACHINE_STATUS,
)

df = pd.read_csv(
    MACHINE_META_FILE,
    usecols=[
        "machine_id",
        "time_stamp",
        "cpu_num",
        "mem_size",
        "status",
    ],
)

print("MACHINE METADATA AUDIT")
print("-" * 60)
print(f"Total metadata rows: {len(df):,}")
print(f"Distinct machine IDs in source: {df['machine_id'].nunique():,}")

for col in ["machine_id", "time_stamp", "cpu_num", "mem_size", "status"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

valid_keys = df.dropna(
    subset=["machine_id", "time_stamp"]
).copy()

latest = (
    valid_keys
    .sort_values(["machine_id", "time_stamp"])
    .drop_duplicates("machine_id", keep="last")
)

latest_using = latest[
    latest["status"] == ELIGIBLE_MACHINE_STATUS
]

valid_capacity = (
    latest["cpu_num"].gt(0)
    & latest["mem_size"].gt(0)
    & latest["cpu_num"].notna()
    & latest["mem_size"].notna()
)

eligible = latest[
    (latest["status"] == ELIGIBLE_MACHINE_STATUS)
    & valid_capacity
]

print(f"Machines with valid ID and timestamp: {len(latest):,}")
print(f"Machines whose latest status is USING: {len(latest_using):,}")
print(
    "Machines whose latest status is not USING:",
    int((latest["status"] != ELIGIBLE_MACHINE_STATUS).sum()),
)
print(
    f"Latest records with invalid capacity: "
    f"{int((~valid_capacity).sum()):,}"
)
print(f"Final eligible machine count: {eligible['machine_id'].nunique():,}")

print("\nLatest status distribution:")
print(latest["status"].value_counts(dropna=False).to_string())

print("\nSource dataset was not modified.")
