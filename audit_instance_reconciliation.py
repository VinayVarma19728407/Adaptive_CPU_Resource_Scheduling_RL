# =============================================================================
# FILE: audit_instance_reconciliation.py
# ROLE IN PROJECT:
#   A data-quality audit script that checks whether the number of batch
#   instance rows across the train/validation/test splits actually matches
#   the declared instance_num counts in batch_task.csv.
#
#   Uses DuckDB for memory-efficient SQL joins across large CSV files.
#   Steps:
#     1. Load batch_task.csv as a reference table (declared_instances).
#     2. Count observed instance rows by (task_name, job_name) across all
#        three split files (training + validation + testing).
#     3. Full-outer-join the two to classify each key as:
#          match / observed_less_than_declared /
#          observed_greater_than_declared /
#          definition_without_observed_instances /
#          instance_key_without_definition
#     4. Save a summary text file and a CSV of all mismatches.
#
#   Outputs:
#     experiments/audits/instance_reconciliation_summary.txt
#     experiments/audits/instance_count_mismatches.csv
# =============================================================================
import duckdb
from pathlib import Path

ROOT = Path(".").resolve()
TASK_FILE = (ROOT / "dataset/preprocessed/batch_task.csv").as_posix()
TRAIN_FILE = (ROOT / "dataset/preprocessed/batch_instance_training.csv").as_posix()
VAL_FILE = (ROOT / "dataset/preprocessed/batch_instance_validation.csv").as_posix()
TEST_FILE = (ROOT / "dataset/preprocessed/batch_instance_testing.csv").as_posix()

OUT_DIR = ROOT / "experiments" / "audits"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DB_FILE = (OUT_DIR / "instance_reconciliation.duckdb").as_posix()
SUMMARY_FILE = OUT_DIR / "instance_reconciliation_summary.txt"
MISMATCH_FILE = OUT_DIR / "instance_count_mismatches.csv"

con = duckdb.connect(DB_FILE)

# Keep memory bounded. DuckDB can spill larger operations to disk.
con.execute("SET memory_limit = '4GB'")
con.execute(f"SET temp_directory = '{(OUT_DIR / 'duckdb_tmp').as_posix()}'")
con.execute("SET threads = 4")

print("STEP 1/4: Loading task-definition reference")
con.execute(f"""
    CREATE OR REPLACE TABLE task_reference AS
    SELECT
        CAST(task_name AS VARCHAR) AS task_name,
        CAST(job_name AS VARCHAR) AS job_name,
        TRY_CAST(instance_num AS BIGINT) AS declared_instances
    FROM read_csv(
        '{TASK_FILE}',
        header = true,
        all_varchar = true
    )
""")

task_count = con.execute(
    "SELECT COUNT(*) FROM task_reference"
).fetchone()[0]

unique_keys = con.execute("""
    SELECT COUNT(*)
    FROM (
        SELECT task_name, job_name
        FROM task_reference
        GROUP BY task_name, job_name
    )
""").fetchone()[0]

print(f"Task rows: {task_count:,}")
print(f"Unique composite keys: {unique_keys:,}")

print("STEP 2/4: Counting observed instances by composite key")
con.execute(f"""
    CREATE OR REPLACE TABLE observed_counts AS
    SELECT task_name, job_name, COUNT(*)::BIGINT AS observed_instances
    FROM (
        SELECT CAST(task_name AS VARCHAR) AS task_name,
               CAST(job_name AS VARCHAR) AS job_name
        FROM read_csv('{TRAIN_FILE}', header=true, all_varchar=true)

        UNION ALL

        SELECT CAST(task_name AS VARCHAR), CAST(job_name AS VARCHAR)
        FROM read_csv('{VAL_FILE}', header=true, all_varchar=true)

        UNION ALL

        SELECT CAST(task_name AS VARCHAR), CAST(job_name AS VARCHAR)
        FROM read_csv('{TEST_FILE}', header=true, all_varchar=true)
    ) instances
    GROUP BY task_name, job_name
""")

observed_total = con.execute(
    "SELECT COALESCE(SUM(observed_instances), 0) FROM observed_counts"
).fetchone()[0]

print(f"Observed instance rows counted: {observed_total:,}")

print("STEP 3/4: Reconciling declared and observed counts")
con.execute("""
    CREATE OR REPLACE TABLE reconciliation AS
    SELECT
        COALESCE(t.task_name, o.task_name) AS task_name,
        COALESCE(t.job_name, o.job_name) AS job_name,
        t.declared_instances,
        o.observed_instances,
        CASE
            WHEN t.task_name IS NULL THEN 'instance_key_without_definition'
            WHEN o.task_name IS NULL THEN 'definition_without_observed_instances'
            WHEN t.declared_instances = o.observed_instances THEN 'match'
            WHEN o.observed_instances < t.declared_instances THEN 'observed_less_than_declared'
            ELSE 'observed_greater_than_declared'
        END AS reconciliation_status,
        CASE
            WHEN t.task_name IS NOT NULL AND o.task_name IS NOT NULL
            THEN t.declared_instances - o.observed_instances
            ELSE NULL
        END AS declared_minus_observed
    FROM task_reference t
    FULL OUTER JOIN observed_counts o
      ON t.task_name = o.task_name
     AND t.job_name = o.job_name
""")

summary = con.execute("""
    SELECT
        reconciliation_status,
        COUNT(*) AS key_count,
        SUM(COALESCE(declared_instances, 0)) AS declared_instance_sum,
        SUM(COALESCE(observed_instances, 0)) AS observed_instance_sum,
        SUM(COALESCE(declared_minus_observed, 0)) AS difference_sum
    FROM reconciliation
    GROUP BY reconciliation_status
    ORDER BY reconciliation_status
""").fetchall()

mismatch_count = con.execute("""
    SELECT COUNT(*)
    FROM reconciliation
    WHERE reconciliation_status <> 'match'
""").fetchone()[0]

print("STEP 4/4: Saving summary and mismatch examples")
con.execute(f"""
    COPY (
        SELECT *
        FROM reconciliation
        WHERE reconciliation_status <> 'match'
        ORDER BY reconciliation_status, ABS(COALESCE(declared_minus_observed, 0)) DESC
    )
    TO '{MISMATCH_FILE.as_posix()}'
    (HEADER, DELIMITER ',')
""")

with SUMMARY_FILE.open("w", encoding="utf-8") as f:
    f.write(f"Task-definition rows: {task_count:,}\n")
    f.write(f"Unique task composite keys: {unique_keys:,}\n")
    f.write(f"Observed instance rows: {observed_total:,}\n")
    f.write(f"Reconciliation keys: {len(summary):,} status groups\n")
    f.write(f"Keys with non-matching status: {mismatch_count:,}\n\n")
    f.write("STATUS SUMMARY\n")
    for row in summary:
        f.write(
            f"{row[0]} | keys={row[1]:,} | "
            f"declared={row[2]:,} | observed={row[3]:,} | "
            f"difference={row[4]:,}\n"
        )

print("\nSTATUS SUMMARY")
for row in summary:
    print(
        f"{row[0]}: keys={row[1]:,}, "
        f"declared={row[2]:,}, observed={row[3]:,}, "
        f"difference={row[4]:,}"
    )

print(f"\nKeys with non-matching status: {mismatch_count:,}")
print(f"Summary saved to: {SUMMARY_FILE}")
print(f"Mismatch records saved to: {MISMATCH_FILE}")
print("AUDIT COMPLETE")

con.close()
