"""
============================================================
File: task_data.py
Module: simulator
============================================================
Raw data access layer for the RL-based CPU scheduling simulator.
This module is responsible for loading and lightly validating
the two primary source datasets — machine metadata and batch
task records — and for producing a reproducible reservoir sample
of task records that later stages convert into workload scenarios.

Key Components:
  - TASK_COLUMNS: List of column names selected from batch_task.csv
    when reading task records (task_name, job_name, task_type,
    status, start_time, end_time, plan_cpu, plan_mem, duration).
  - MACHINE_COLUMNS: List of column names selected from
    machine_meta.csv (machine_id, time_stamp, failure domains,
    cpu_num, mem_size, status).
  - load_machine_metadata() -> pd.DataFrame:
      Reads machine_meta.csv, converts numeric columns, drops
      rows with missing critical fields, de-duplicates by keeping
      only the latest-timestamped record per machine, and filters
      to machines with status == ELIGIBLE_MACHINE_STATUS (1),
      positive cpu_num, and positive mem_size. Raises ValueError
      if no eligible machines remain.
  - sample_task_records(sample_size, seed) -> pd.DataFrame:
      Streams batch_task.csv in configurable chunks and performs
      reservoir sampling to obtain a reproducible, fixed-size
      sample without loading the entire file into memory. Does
      not fill missing demand/duration values — that is deferred
      to the scenario loader.
  - summarize_data(tasks, machines) -> None:
      Prints a human-readable summary of the sampled task records
      and eligible machine pool (counts, missing values, capacity
      statistics) to stdout.

Inputs:
  - BATCH_TASK_FILE (batch_task.csv): Preprocessed batch task
    records, potentially very large; read in chunks.
  - MACHINE_META_FILE (machine_meta.csv): Small machine metadata
    file read in one pass.
  - sample_size (int): Number of task records to retain in the
    reservoir sample (default 10,000).
  - seed (int): Random seed for reproducible sampling (default 42).

Outputs:
  - pd.DataFrame of eligible machines consumed by
    MachineStateManager.
  - pd.DataFrame of sampled task records consumed by
    scenario_loader.py to generate workload scenarios.
============================================================
"""

import random
from typing import Dict, List

import pandas as pd

from simulator.config import (
    BATCH_TASK_FILE,
    MACHINE_META_FILE,
    CSV_CHUNK_SIZE,
    RANDOM_SEED,
    ELIGIBLE_MACHINE_STATUS,
    validate_required_files,
)


TASK_COLUMNS = [
    "task_name",
    "job_name",
    "task_type",
    "status",
    "start_time",
    "end_time",
    "plan_cpu",
    "plan_mem",
    "duration",
]

MACHINE_COLUMNS = [
    "machine_id",
    "time_stamp",
    "failure_domain_1",
    "failure_domain_2",
    "cpu_num",
    "mem_size",
    "status",
]



def load_machine_metadata() -> pd.DataFrame:
    """
    Select the latest valid metadata record per machine,
    then retain machines whose latest status is USING (1).
    """
    validate_required_files()

    machines = pd.read_csv(
        MACHINE_META_FILE,
        usecols=MACHINE_COLUMNS,
        low_memory=False,
    )

    machines["machine_id"] = pd.to_numeric(
        machines["machine_id"], errors="coerce"
    )
    machines["time_stamp"] = pd.to_numeric(
        machines["time_stamp"], errors="coerce"
    )
    machines["cpu_num"] = pd.to_numeric(
        machines["cpu_num"], errors="coerce"
    )
    machines["mem_size"] = pd.to_numeric(
        machines["mem_size"], errors="coerce"
    )
    machines["status"] = pd.to_numeric(
        machines["status"], errors="coerce"
    )

    # A record needs a valid machine ID and timestamp to determine
    # which metadata record is latest.
    machines = machines.dropna(
        subset=[
            "machine_id",
            "time_stamp",
            "cpu_num",
            "mem_size",
            "status",
        ]
    )

    # Establish the latest metadata record for each machine first.
    machines = machines.sort_values(
        ["machine_id", "time_stamp"]
    )

    machines = machines.drop_duplicates(
        subset=["machine_id"],
        keep="last",
    )

    # Apply eligibility to the latest record only.
    machines = machines[
        (machines["status"] == ELIGIBLE_MACHINE_STATUS)
        & (machines["cpu_num"] > 0)
        & (machines["mem_size"] > 0)
    ].copy()

    machines = machines.reset_index(drop=True)

    if machines.empty:
        raise ValueError(
            "No eligible machines remain after latest-record validation."
        )

    if machines["machine_id"].duplicated().any():
        raise ValueError("Duplicate machine IDs remain in the eligible pool.")

    return machines

    """
    Load the small machine metadata file and return eligible machines.

    The original capacity values are preserved. No utilization
    percentages are converted into machine capacities here.
    """
    validate_required_files()

    machines = pd.read_csv(
        MACHINE_META_FILE,
        usecols=MACHINE_COLUMNS,
        low_memory=False,
    )

    machines["machine_id"] = pd.to_numeric(
        machines["machine_id"], errors="coerce"
    )
    machines["cpu_num"] = pd.to_numeric(
        machines["cpu_num"], errors="coerce"
    )
    machines["mem_size"] = pd.to_numeric(
        machines["mem_size"], errors="coerce"
    )
    machines["status"] = pd.to_numeric(
        machines["status"], errors="coerce"
    )

    machines = machines.dropna(
        subset=["machine_id", "cpu_num", "mem_size", "status"]
    )

    machines = machines[
        (machines["cpu_num"] > 0)
        & (machines["mem_size"] > 0)
        & (machines["status"] == ELIGIBLE_MACHINE_STATUS)
    ].copy()

    # One current metadata row per machine is required for the
    # initial simulator pool. If history contains multiple rows,
    # retain the row with the latest metadata timestamp.
    machines = machines.sort_values("time_stamp")
    machines = machines.drop_duplicates(
        subset=["machine_id"], keep="last"
    )

    machines = machines.reset_index(drop=True)

    if machines.empty:
        raise ValueError(
            "No eligible machines remain after metadata validation."
        )

    return machines


def sample_task_records(
    sample_size: int = 10_000,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Obtain a reproducible reservoir sample from batch_task.csv.

    Memory usage depends primarily on sample_size and chunk size,
    rather than the total number of task rows.

    This is a sample of task records, not yet a generated scenario.
    Arrival times and scenario semantics are added in a later stage.
    """
    validate_required_files()

    if sample_size <= 0:
        raise ValueError("sample_size must be greater than zero.")

    rng = random.Random(seed)
    reservoir: List[Dict] = []
    rows_seen = 0

    reader = pd.read_csv(
        BATCH_TASK_FILE,
        usecols=TASK_COLUMNS,
        chunksize=CSV_CHUNK_SIZE,
        low_memory=False,
    )

    for chunk in reader:
        # Normalize numeric columns without silently inventing values.
        for col in [
            "task_type",
            "start_time",
            "end_time",
            "plan_cpu",
            "plan_mem",
            "duration",
        ]:
            chunk[col] = pd.to_numeric(
                chunk[col], errors="coerce"
            )

        # Reservoir sampling gives each row an equal selection
        # probability without retaining the complete dataset.
        for record in chunk.to_dict(orient="records"):
            rows_seen += 1

            if len(reservoir) < sample_size:
                reservoir.append(record)
            else:
                j = rng.randrange(rows_seen)
                if j < sample_size:
                    reservoir[j] = record

        del chunk

    if rows_seen == 0:
        raise ValueError("batch_task.csv contains no task records.")

    result = pd.DataFrame(reservoir)

    # Do not silently fill missing demand or duration here.
    # Scenario eligibility will be checked by the scenario loader.
    result.attrs["source_rows_seen"] = rows_seen
    result.attrs["sample_seed"] = seed

    return result


def summarize_data(
    tasks: pd.DataFrame,
    machines: pd.DataFrame,
) -> None:
    print("\nTASK SAMPLE SUMMARY")
    print("-" * 60)
    print(f"Sampled task records: {len(tasks):,}")
    print(f"Source rows scanned: {tasks.attrs.get('source_rows_seen', 'unknown'):,}")
    print(f"Sample seed: {tasks.attrs.get('sample_seed', 'unknown')}")

    print("\nMissing values in sampled task records:")
    print(tasks[TASK_COLUMNS].isna().sum().to_string())

    print("\nMACHINE POOL SUMMARY")
    print("-" * 60)
    print(f"Eligible machines: {len(machines):,}")
    print(f"Distinct machine IDs: {machines['machine_id'].nunique():,}")

    print("\nConfigured machine capacity summary:")
    print(
        machines[["cpu_num", "mem_size"]]
        .describe()
        .to_string()
    )


if __name__ == "__main__":
    print("LOADING SIMULATOR DATA")

    machines = load_machine_metadata()
    print(f"Eligible machines loaded: {len(machines):,}")

    print("\nSampling task records from batch_task.csv...")
    tasks = sample_task_records(sample_size=10_000)

    summarize_data(tasks, machines)

    print("\nStage 1 data access test completed.")
    print("Source datasets were not modified.")
