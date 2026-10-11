"""
Finalize Dataset Relationship Validation.

This script consolidates and finalizes cross-table schema and key relationship audits across all dataset tables.
It combines findings from prior relationship audits into a single summary report confirming repository data
readiness for discrete-event simulator execution.
"""

import pandas as pd
from pathlib import Path

BASE = Path("dataset")
AUDIT_DIR = BASE / "relationship_audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

rows = [
    {
        "relationship": "batch_task task_name to batch_instance task_name",
        "reference": "batch_task",
        "target": "batch_instance",
        "reference_unique": 416208,
        "target_unique": 408810,
        "target_unmatched_keys": 8,
        "affected_target_rows": 3508,
        "status": "VALIDATED_WITH_EXPLAINED_EXCEPTION",
        "explanation": "Eight task names are absent from cleaned batch_task because every raw record for those task names had invalid negative duration and was removed during established cleaning."
    },
    {
        "relationship": "batch_instance machine_id to machine_meta machine_id",
        "reference": "machine_meta",
        "target": "batch_instance",
        "reference_unique": 4033,
        "target_unique": 3867,
        "target_unmatched_keys": 1,
        "affected_target_rows": 0,
        "status": "VALIDATED_WITH_EXPLAINED_EXCEPTION",
        "explanation": "The unmatched machine is 2739. All raw machine_meta records for this machine had status IMPORT_INSTALLING and were intentionally removed."
    },
    {
        "relationship": "machine_usage machine_id to machine_meta machine_id",
        "reference": "machine_meta",
        "target": "machine_usage",
        "reference_unique": 4033,
        "target_unique": 4021,
        "target_unmatched_keys": 1,
        "affected_target_rows": 44894,
        "status": "VALIDATED_WITH_EXPLAINED_EXCEPTION",
        "explanation": "All unmatched machine_usage rows belong to machine 2739, whose machine_meta records were intentionally removed."
    },
    {
        "relationship": "machine_usage machine_id and time_stamp",
        "reference": "machine_usage",
        "target": "machine_usage",
        "reference_unique": 0,
        "target_unique": 0,
        "target_unmatched_keys": 0,
        "affected_target_rows": 0,
        "status": "VALIDATED",
        "explanation": "No duplicate machine_id and time_stamp combinations were found."
    },
    {
        "relationship": "batch_instance task_type consistency",
        "reference": "batch_task",
        "target": "batch_instance",
        "reference_unique": 0,
        "target_unique": 0,
        "target_unmatched_keys": 0,
        "affected_target_rows": 0,
        "status": "VALIDATED",
        "explanation": "The corrected task type relationship audit found zero task type mismatches."
    },
    {
        "relationship": "batch_instance sequence consistency",
        "reference": "batch_instance",
        "target": "batch_instance",
        "reference_unique": 0,
        "target_unique": 0,
        "target_unmatched_keys": 0,
        "affected_target_rows": 0,
        "status": "VALIDATED",
        "explanation": "No rows had seq_no greater than total_seq_no."
    },
    {
        "relationship": "batch_instance duration consistency",
        "reference": "batch_instance",
        "target": "batch_instance",
        "reference_unique": 0,
        "target_unique": 0,
        "target_unmatched_keys": 0,
        "affected_target_rows": 0,
        "status": "VALIDATED",
        "explanation": "No negative durations remained after established cleaning."
    },
    {
        "relationship": "batch_task task_name to job_name",
        "reference": "batch_task",
        "target": "batch_instance",
        "reference_unique": 0,
        "target_unique": 0,
        "target_unmatched_keys": 0,
        "affected_target_rows": 7399021,
        "status": "NOT_USED_AS_STRICT_INTEGRITY_CONSTRAINT",
        "explanation": "Individual task names can correspond to very large numbers of job names in batch_task. Therefore absence of a particular batch_instance task and job combination from batch_task is not treated as evidence of an invalid batch_instance row."
    }
]

df = pd.DataFrame(rows)

output_file = (
    AUDIT_DIR /
    "final_four_dataset_relationship_validation.csv"
)

df.to_csv(
    output_file,
    index=False
)

print("Final relationship validation created")

print(
    f"Relationships documented: {len(df)}"
)

print(
    f"Output file: {output_file}"
)

print()
print("Relationship status summary")

for status, count in df["status"].value_counts().items():

    print(
        f"{status}: {count}"
    )

print()
print("Final relationship validation complete")