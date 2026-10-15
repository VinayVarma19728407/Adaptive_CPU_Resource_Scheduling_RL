# =============================================================================
# FILE: inspect_unmatched_key_relationships.py
# ROLE IN PROJECT:
#   A targeted investigation script (read-only DuckDB queries) that digs
#   deeper into the two most interesting mismatch categories found by
#   audit_instance_reconciliation.py:
#
#     1. "instance_key_without_definition": instance rows whose
#        (task_name, job_name) key does not appear in batch_task.csv at all.
#        This script checks whether at least the task_name exists under a
#        different job — a common pattern in cluster traces where jobs are
#        renamed.
#
#     2. "definition_without_observed_instances": task keys that are declared
#        in batch_task.csv but have zero rows in the instance splits. Checks
#        if those task names appear at least somewhere in the observed data.
#
#   Input:  experiments/audits/instance_reconciliation.duckdb
#   Output: printed diagnostic tables to stdout (no files written)
# =============================================================================
import duckdb
from pathlib import Path

db_path = Path("experiments/audits/instance_reconciliation.duckdb")
con = duckdb.connect(str(db_path), read_only=True)

print("Preparing distinct task-name references...", flush=True)

# These tables reduce repeated comparisons across millions of task keys.
con.execute("""
    CREATE TEMP TABLE reference_task_names AS
    SELECT DISTINCT task_name
    FROM task_reference
""")

con.execute("""
    CREATE TEMP TABLE observed_task_names AS
    SELECT DISTINCT task_name
    FROM observed_counts
""")

print("\n1. UNMATCHED INSTANCE KEYS", flush=True)
print(con.execute("""
    SELECT
        COUNT(*) AS unmatched_keys,
        SUM(r.observed_instances) AS observed_instances,
        COUNT(DISTINCT r.task_name) AS distinct_task_names,
        COUNT(DISTINCT CASE
            WHEN t.task_name IS NOT NULL THEN r.task_name
        END) AS task_names_found_in_reference
    FROM reconciliation r
    LEFT JOIN reference_task_names t
      ON r.task_name = t.task_name
    WHERE r.reconciliation_status = 'instance_key_without_definition'
""").df().to_string(index=False), flush=True)

print("\n2. DEFINITIONS WITHOUT OBSERVED INSTANCES", flush=True)
print(con.execute("""
    SELECT
        COUNT(*) AS unmatched_keys,
        SUM(r.declared_instances) AS declared_instances,
        COUNT(DISTINCT r.task_name) AS distinct_task_names,
        COUNT(DISTINCT CASE
            WHEN o.task_name IS NOT NULL THEN r.task_name
        END) AS task_names_found_in_observed_data
    FROM reconciliation r
    LEFT JOIN observed_task_names o
      ON r.task_name = o.task_name
    WHERE r.reconciliation_status = 'definition_without_observed_instances'
""").df().to_string(index=False), flush=True)

print("\n3. EXAMPLES OF UNMATCHED INSTANCE KEYS", flush=True)
print(con.execute("""
    SELECT
        r.task_name,
        r.job_name,
        r.observed_instances,
        CASE WHEN t.task_name IS NOT NULL
             THEN 'task_name exists under another job'
             ELSE 'task_name absent from reference'
        END AS finding
    FROM reconciliation r
    LEFT JOIN reference_task_names t
      ON r.task_name = t.task_name
    WHERE r.reconciliation_status = 'instance_key_without_definition'
    ORDER BY r.observed_instances DESC
    LIMIT 20
""").df().to_string(index=False), flush=True)

print("\nAUDIT COMPLETE", flush=True)
con.close()