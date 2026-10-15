# =============================================================================
# FILE: inspect_reconciliation_mismatches.py
# ROLE IN PROJECT:
#   A read-only follow-up inspection script meant to be run AFTER
#   audit_instance_reconciliation.py has built the DuckDB database.
#
#   Queries the reconciliation table in the DuckDB file and prints:
#     1. Overall counts by reconciliation_status category.
#     2. Top 20 keys where observed < declared (largest deficits).
#     3. Top 20 keys where observed > declared (largest excesses).
#     4. Top 20 task definitions that have zero observed instances.
#     5. Top 20 instance keys that have no matching task definition.
#     6. Distribution of mismatch sizes (min/median/max absolute difference).
#
#   Input:  experiments/audits/instance_reconciliation.duckdb
#   Output: printed SQL query results to stdout (no files written or modified)
# =============================================================================
import duckdb
from pathlib import Path

db_path = Path("experiments/audits/instance_reconciliation.duckdb")
out_dir = Path("experiments/audits")
con = duckdb.connect(str(db_path), read_only=True)

print("1. RECONCILIATION COUNTS")
print(con.execute("""
    SELECT reconciliation_status,
           COUNT(*) AS keys,
           SUM(COALESCE(declared_instances, 0)) AS declared,
           SUM(COALESCE(observed_instances, 0)) AS observed,
           SUM(COALESCE(declared_minus_observed, 0)) AS net_difference
    FROM reconciliation
    GROUP BY reconciliation_status
    ORDER BY reconciliation_status
""").df().to_string(index=False))

print("\n2. OBSERVED-LESS-THAN-DECLARED: LARGEST DEFICITS")
print(con.execute("""
    SELECT task_name, job_name, declared_instances,
           observed_instances, declared_minus_observed
    FROM reconciliation
    WHERE reconciliation_status = 'observed_less_than_declared'
    ORDER BY declared_minus_observed DESC
    LIMIT 20
""").df().to_string(index=False))

print("\n3. OBSERVED-GREATER-THAN-DECLARED: LARGEST EXCESSES")
print(con.execute("""
    SELECT task_name, job_name, declared_instances,
           observed_instances, declared_minus_observed
    FROM reconciliation
    WHERE reconciliation_status = 'observed_greater_than_declared'
    ORDER BY declared_minus_observed ASC
    LIMIT 20
""").df().to_string(index=False))

print("\n4. DEFINITIONS WITHOUT OBSERVED INSTANCES: EXAMPLES")
print(con.execute("""
    SELECT task_name, job_name, declared_instances
    FROM reconciliation
    WHERE reconciliation_status = 'definition_without_observed_instances'
    ORDER BY declared_instances DESC
    LIMIT 20
""").df().to_string(index=False))

print("\n5. INSTANCE KEYS WITHOUT DEFINITIONS: EXAMPLES")
print(con.execute("""
    SELECT task_name, job_name, observed_instances
    FROM reconciliation
    WHERE reconciliation_status = 'instance_key_without_definition'
    ORDER BY observed_instances DESC
    LIMIT 20
""").df().to_string(index=False))

print("\n6. MISMATCH SIZE DISTRIBUTION")
print(con.execute("""
    SELECT reconciliation_status,
           MIN(ABS(declared_minus_observed)) AS min_abs_difference,
           MEDIAN(ABS(declared_minus_observed)) AS median_abs_difference,
           MAX(ABS(declared_minus_observed)) AS max_abs_difference
    FROM reconciliation
    WHERE reconciliation_status IN (
        'observed_less_than_declared',
        'observed_greater_than_declared'
    )
    GROUP BY reconciliation_status
    ORDER BY reconciliation_status
""").df().to_string(index=False))

con.close()
print("\nINSPECTION COMPLETE")
