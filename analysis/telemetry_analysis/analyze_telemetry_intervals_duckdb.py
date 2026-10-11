
from pathlib import Path
import duckdb

ROOT = Path.cwd()
FILE = ROOT / "dataset" / "cleaned" / "machine_usage.csv"
TEMP_DIR = ROOT / "duckdb_temp"

MEMORY_LIMIT = "4GB"
THREADS = 4

if not FILE.exists():
    raise FileNotFoundError(f"CSV not found: {FILE}")

TEMP_DIR.mkdir(parents=True, exist_ok=True)

def sql_path(path):
    return str(path).replace("\\", "/").replace("'", "''")

file_path = sql_path(FILE)
temp_path = sql_path(TEMP_DIR)

db = duckdb.connect(database=":memory:")
db.execute(f"SET memory_limit = '{MEMORY_LIMIT}'")
db.execute(f"SET threads = {THREADS}")
db.execute(f"SET temp_directory = '{temp_path}'")

# Explicitly specify the CSV delimiter and the complete seven-column
# schema. Only machine_id and time_stamp will be queried.
source = f"""
read_csv(
    '{file_path}',
    delim = ',',
    header = true,
    columns = {{
        'machine_id': 'BIGINT',
        'time_stamp': 'BIGINT',
        'cpu_util_percent': 'DOUBLE',
        'mem_util_percent': 'DOUBLE',
        'net_in': 'DOUBLE',
        'net_out': 'DOUBLE',
        'disk_io_percent': 'DOUBLE'
    }}
)
"""

print("=" * 72)
print("DUCKDB TELEMETRY INTERVAL ANALYSIS")
print("=" * 72)
print(f"File: {FILE}")
print(f"Memory limit: {MEMORY_LIMIT}")
print(f"Threads: {THREADS}")
print("Source data will not be modified.")

try:
    # First verify that DuckDB can read the file.
    print("\nChecking CSV reader configuration...")

    print(db.execute(f"""
        SELECT machine_id, time_stamp
        FROM {source}
        LIMIT 5
    """).fetchdf().to_string(index=False))

    # Sort observations by machine and timestamp, then calculate
    # consecutive gaps within each machine.
    query = f"""
    WITH ordered AS (
        SELECT
            machine_id,
            time_stamp,
            LAG(time_stamp) OVER (
                PARTITION BY machine_id
                ORDER BY time_stamp
            ) AS previous_timestamp
        FROM {source}
        WHERE machine_id IS NOT NULL
          AND time_stamp IS NOT NULL
    ),
    gaps AS (
        SELECT
            machine_id,
            time_stamp - previous_timestamp AS gap
        FROM ordered
        WHERE previous_timestamp IS NOT NULL
    )
    SELECT
        COUNT(*) AS consecutive_pairs,
        COUNT(DISTINCT machine_id) AS machines_with_gaps,
        SUM(CASE WHEN gap = 0 THEN 1 ELSE 0 END)
            AS duplicate_timestamp_gaps,
        SUM(CASE WHEN gap < 0 THEN 1 ELSE 0 END)
            AS negative_gaps,
        MIN(gap) AS minimum_gap,
        MAX(gap) AS maximum_gap,
        QUANTILE_CONT(gap, 0.50) AS p50_gap,
        QUANTILE_CONT(gap, 0.75) AS p75_gap,
        QUANTILE_CONT(gap, 0.90) AS p90_gap,
        QUANTILE_CONT(gap, 0.95) AS p95_gap,
        QUANTILE_CONT(gap, 0.99) AS p99_gap,
        QUANTILE_CONT(gap, 0.999) AS p999_gap,
        SUM(CASE WHEN gap > 60 THEN 1 ELSE 0 END)
            AS gaps_over_60_units,
        SUM(CASE WHEN gap > 300 THEN 1 ELSE 0 END)
            AS gaps_over_300_units,
        SUM(CASE WHEN gap > 600 THEN 1 ELSE 0 END)
            AS gaps_over_600_units,
        SUM(CASE WHEN gap > 3600 THEN 1 ELSE 0 END)
            AS gaps_over_3600_units
    FROM gaps
    """

    print("\nCalculating sorted per-machine timestamp gaps.")
    print("This may take considerable time and temporary disk space.")

    result = db.execute(query).fetchdf()

    print("\nPER-MACHINE GAP SUMMARY")
    print("-" * 72)
    print(result.to_string(index=False))

    summary = db.execute(f"""
        SELECT
            COUNT(*) AS valid_rows,
            COUNT(DISTINCT machine_id) AS machine_count,
            MIN(time_stamp) AS minimum_timestamp,
            MAX(time_stamp) AS maximum_timestamp
        FROM {source}
        WHERE machine_id IS NOT NULL
          AND time_stamp IS NOT NULL
    """).fetchdf()

    print("\nSOURCE SUMMARY")
    print("-" * 72)
    print(summary.to_string(index=False))

    print("\nGap values are still in source timestamp units.")
    print("Timestamp units have not been inferred.")
    print("No source data was modified.")

finally:
    db.close()
