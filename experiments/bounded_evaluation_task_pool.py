from pathlib import Path

import numpy as np
import pandas as pd

from simulator.config import BATCH_TASK_FILE


def load_bounded_eligible_task_pool(
    max_cpu_capacity: float,
    max_memory_capacity: float,
    sample_size: int = 50_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Reservoir-sample eligible source records without retaining the full pool."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    if not Path(BATCH_TASK_FILE).is_file():
        raise FileNotFoundError(f"Canonical dataset not found: {BATCH_TASK_FILE}")

    required = {"task_name", "job_name", "plan_cpu", "plan_mem", "duration"}
    rng = np.random.default_rng(seed)
    reservoir = []
    eligible_seen = 0
    rows_read = 0

    for chunk in pd.read_csv(BATCH_TASK_FILE, chunksize=250_000, low_memory=False):
        missing = required - set(chunk.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        rows_read += len(chunk)
        for col in ("plan_cpu", "plan_mem", "duration"):
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        mask = (
            chunk["task_name"].notna()
            & chunk["job_name"].notna()
            & chunk["plan_cpu"].notna()
            & chunk["plan_mem"].notna()
            & chunk["duration"].notna()
            & np.isfinite(chunk["plan_cpu"])
            & np.isfinite(chunk["plan_mem"])
            & np.isfinite(chunk["duration"])
            & (chunk["plan_cpu"] > 0)
            & (chunk["plan_mem"] > 0)
            & (chunk["duration"] > 0)
            & (chunk["plan_cpu"] / 100.0 <= max_cpu_capacity)
            & (chunk["plan_mem"] <= max_memory_capacity)
        )

        eligible = chunk.loc[
            mask, ["task_name", "job_name", "plan_cpu", "plan_mem", "duration"]
        ]

        for row in eligible.itertuples(index=False, name=None):
            eligible_seen += 1
            if len(reservoir) < sample_size:
                reservoir.append(row)
            else:
                j = int(rng.integers(0, eligible_seen))
                if j < sample_size:
                    reservoir[j] = row

        print(
            f"Rows scanned: {rows_read:,}; eligible records: {eligible_seen:,}; "
            f"sample retained: {len(reservoir):,}"
        )

    if not reservoir:
        raise ValueError("No eligible records found.")

    result = pd.DataFrame(
        reservoir,
        columns=["task_name", "job_name", "plan_cpu", "plan_mem", "duration"],
    )
    print(f"Final evaluation sample: {len(result):,} of {eligible_seen:,} eligible records")
    return result
