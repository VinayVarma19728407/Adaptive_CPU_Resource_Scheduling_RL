
from pathlib import Path
import pandas as pd

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

FILE = Path("dataset/cleaned/machine_usage.csv")
CHUNK_SIZE = 1_000_000

# ---------------------------------------------------------
# SUMMARY VARIABLES ONLY
# ---------------------------------------------------------

rows = 0
invalid_machine_ids = 0
invalid_timestamps = 0

minimum_timestamp = None
maximum_timestamp = None

first_values = []
last_values = []

print("=" * 70)
print("MEMORY-EFFICIENT TELEMETRY TIMESTAMP INSPECTION")
print("=" * 70)

if not FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {FILE.resolve()}"
    )

# ---------------------------------------------------------
# READ THE FILE IN CHUNKS
# ---------------------------------------------------------

reader = pd.read_csv(
    FILE,
    usecols=["machine_id", "time_stamp"],
    chunksize=CHUNK_SIZE,
    low_memory=False
)

for chunk_number, chunk in enumerate(reader, start=1):

    rows += len(chunk)

    # Check machine IDs
    invalid_machine_ids += int(
        chunk["machine_id"].isna().sum()
    )

    # Convert timestamps only within the current chunk
    timestamps = pd.to_numeric(
        chunk["time_stamp"],
        errors="coerce"
    )

    invalid_timestamps += int(
        timestamps.isna().sum()
    )

    valid = timestamps.dropna()

    if not valid.empty:

        chunk_min = valid.min()
        chunk_max = valid.max()

        if minimum_timestamp is None:
            minimum_timestamp = chunk_min
            maximum_timestamp = chunk_max
        else:
            minimum_timestamp = min(
                minimum_timestamp, chunk_min
            )
            maximum_timestamp = max(
                maximum_timestamp, chunk_max
            )

        # Keep only the first 20 valid timestamp values
        if len(first_values) < 20:
            needed = 20 - len(first_values)
            first_values.extend(
                valid.iloc[:needed].tolist()
            )

        # Keep only the last 20 valid timestamp values
        last_values.extend(
            valid.tail(20).tolist()
        )
        last_values = last_values[-20:]

    # Release references to temporary chunk data
    del timestamps, valid, chunk

    if chunk_number % 10 == 0:
        print(f"Processed {rows:,} rows")

# ---------------------------------------------------------
# REPORT
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TIMESTAMP INSPECTION SUMMARY")
print("=" * 70)

print(f"Rows scanned: {rows:,}")
print(f"Missing machine IDs: {invalid_machine_ids:,}")
print(f"Invalid or missing timestamps: {invalid_timestamps:,}")

print(f"Minimum timestamp: {minimum_timestamp}")
print(f"Maximum timestamp: {maximum_timestamp}")

if minimum_timestamp is not None:
    print(
        "Timestamp range in source units: "
        f"{maximum_timestamp - minimum_timestamp}"
    )

print(f"\nFirst timestamp values: {first_values}")
print(f"Last timestamp values: {last_values}")

print("\nMemory behavior:")
print("- Only the current CSV chunk is processed.")
print("- No complete history of machine/timestamp pairs is stored.")
print("- Only summary values and small timestamp samples are retained.")

print("\nSource dataset was not modified.")
print("Timestamp units have NOT yet been confirmed.")
