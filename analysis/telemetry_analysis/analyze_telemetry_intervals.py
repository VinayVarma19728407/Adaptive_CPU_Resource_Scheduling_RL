
from pathlib import Path
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

FILE = Path("dataset/cleaned/machine_usage.csv")
CHUNK_SIZE = 1_000_000

REQUIRED_COLUMNS = {"machine_id", "time_stamp"}

print("=" * 70)
print("TELEMETRY SAMPLING INTERVAL ANALYSIS")
print("=" * 70)

if not FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {FILE.resolve()}")

# ---------------------------------------------------------
# PASS 1: CHECK ORDER AND BASIC DATA QUALITY
# ---------------------------------------------------------

print("\nPASS 1: CHECKING TIMESTAMPS AND FILE ORDER")

last_machine = None
last_timestamp = None
previous_key = None

rows = 0
invalid_machine_rows = 0
invalid_timestamp_rows = 0
duplicate_or_nonincreasing = 0
global_order_violations = 0

# Track the last timestamp for every machine.
last_seen = {}
gap_seconds = []

# Timestamp unit is not assumed here.
# This pass measures gaps in the original timestamp units.
# We determine the unit only after inspecting the output.

reader = pd.read_csv(
    FILE,
    usecols=["machine_id", "time_stamp"],
    chunksize=CHUNK_SIZE,
    low_memory=False
)

for chunk_number, chunk in enumerate(reader, start=1):
    rows += len(chunk)

    chunk["machine_id"] = chunk["machine_id"].astype("string")
    chunk["time_stamp"] = pd.to_numeric(
        chunk["time_stamp"], errors="coerce"
    )

    invalid_machine_rows += int(chunk["machine_id"].isna().sum())
    invalid_timestamp_rows += int(chunk["time_stamp"].isna().sum())

    chunk = chunk.dropna(subset=["machine_id", "time_stamp"])

    # Check whether rows are globally ordered by machine_id,
    # then timestamp. This is diagnostic, not an assumption.
    for machine, timestamp in zip(
        chunk["machine_id"], chunk["time_stamp"]
    ):
        key = (str(machine), float(timestamp))

        if previous_key is not None and key < previous_key:
            global_order_violations += 1

        previous_key = key

        machine = str(machine)
        timestamp = float(timestamp)

        if machine in last_seen:
            gap = timestamp - last_seen[machine]

            if gap <= 0:
                duplicate_or_nonincreasing += 1
            else:
                gap_seconds.append(gap)

        last_seen[machine] = timestamp

    if chunk_number % 5 == 0:
        print(f"Processed {rows:,} rows...")

print("\nPASS 1 SUMMARY")
print("-" * 70)
print(f"Rows scanned: {rows:,}")
print(f"Machines encountered: {len(last_seen):,}")
print(f"Rows with missing machine ID: {invalid_machine_rows:,}")
print(f"Rows with invalid timestamp: {invalid_timestamp_rows:,}")
print(f"Duplicate/non-increasing per-machine gaps: "
      f"{duplicate_or_nonincreasing:,}")
print(f"Global machine/timestamp ordering violations: "
      f"{global_order_violations:,}")

if not gap_seconds:
    raise RuntimeError(
        "No positive timestamp gaps found. Inspect timestamp values "
        "and dataset ordering before continuing."
    )

# ---------------------------------------------------------
# GAP DISTRIBUTION
# ---------------------------------------------------------

gaps = pd.Series(gap_seconds, dtype="float64")

print("\nPOSITIVE GAP DISTRIBUTION")
print("-" * 70)
print(gaps.describe(percentiles=[
    0.50, 0.75, 0.90, 0.95, 0.99, 0.999
]).to_string())

print("\nInterpretation warning:")
print(
    "These gaps are trustworthy only if observations are ordered "
    "by machine and timestamp. If ordering violations are present, "
    "do not use this distribution to select the freshness threshold."
)

print("\nTimestamp examples from the source must be inspected to determine")
print("whether time_stamp represents seconds, milliseconds, or another unit.")

print("\nAnalysis complete. No source dataset was modified.")
