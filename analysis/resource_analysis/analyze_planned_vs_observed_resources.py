
"""
Memory-conscious planned-versus-observed resource analysis.

The full training split is processed in chunks. Only a bounded,
stratified sample of rows is retained for ratio distributions and
correlations.

Important:
- Planned resource references are aggregated by task_name, as in
  the original script.
- This is an exploratory association, not a validated row-level
  task/job identity match.
- Percentiles and correlations are estimates from the retained sample.
"""

from pathlib import Path
import gc

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent

BATCH_TASK = ROOT / "dataset" / "preprocessed" / "batch_task.csv"
BATCH_INSTANCE = (
    ROOT / "dataset" / "preprocessed" / "batch_instance_training.csv"
)

CHUNK_SIZE = 100_000

# Keep at most this many sampled rows per processed chunk.
SAMPLE_PER_CHUNK = 100
RANDOM_SEED = 42

OBSERVED_COLUMNS = [
    "duration",
    "cpu_avg",
    "cpu_max",
    "mem_avg",
    "mem_max",
]

PLAN_COLUMNS = [
    "plan_cpu_median",
    "plan_cpu_mean",
    "plan_mem_median",
    "plan_mem_mean",
    "task_rows",
]


def main():
    print("BUILDING PLANNED RESOURCE REFERENCE")

    if not BATCH_TASK.exists():
        raise FileNotFoundError(BATCH_TASK)

    if not BATCH_INSTANCE.exists():
        raise FileNotFoundError(BATCH_INSTANCE)

    batch_task = pd.read_csv(
        BATCH_TASK,
        usecols=["task_name", "plan_cpu", "plan_mem"],
        low_memory=True,
    )

    print(f"batch_task rows: {len(batch_task):,}")

    for column in ["plan_cpu", "plan_mem"]:
        batch_task[column] = pd.to_numeric(
            batch_task[column], errors="coerce"
        )

    # Preserve the original analysis approach: aggregate by task_name.
    reference = (
        batch_task.groupby("task_name", sort=False)
        .agg(
            plan_cpu_median=("plan_cpu", "median"),
            plan_cpu_mean=("plan_cpu", "mean"),
            plan_mem_median=("plan_mem", "median"),
            plan_mem_mean=("plan_mem", "mean"),
            task_rows=("task_name", "size"),
        )
    )

    # Use indexed lookups instead of materializing merge outputs.
    del batch_task
    gc.collect()

    print(f"Unique task_name references: {len(reference):,}")
    print("\nPROCESSING TRAINING DATA IN CHUNKS")

    total_rows = 0
    cpu_matches = 0
    memory_matches = 0
    chunks_processed = 0

    cpu_ratio_invalid = 0
    memory_ratio_invalid = 0

    sampled_chunks = []

    usecols = ["task_name"] + OBSERVED_COLUMNS

    for chunk in pd.read_csv(
        BATCH_INSTANCE,
        usecols=usecols,
        chunksize=CHUNK_SIZE,
        low_memory=True,
    ):
        chunks_processed += 1
        total_rows += len(chunk)

        # Numeric conversions are performed only on the current chunk.
        for column in OBSERVED_COLUMNS:
            chunk[column] = pd.to_numeric(
                chunk[column], errors="coerce"
            )

        # Indexed mapping avoids a full DataFrame merge.
        chunk["plan_cpu_median"] = chunk["task_name"].map(
            reference["plan_cpu_median"]
        )
        chunk["plan_mem_median"] = chunk["task_name"].map(
            reference["plan_mem_median"]
        )

        cpu_matches += int(chunk["plan_cpu_median"].notna().sum())
        memory_matches += int(chunk["plan_mem_median"].notna().sum())

        valid_cpu_plan = (
            chunk["plan_cpu_median"].notna()
            & (chunk["plan_cpu_median"] > 0)
        )
        valid_memory_plan = (
            chunk["plan_mem_median"].notna()
            & (chunk["plan_mem_median"] > 0)
        )

        cpu_ratio_invalid += int((~valid_cpu_plan).sum())
        memory_ratio_invalid += int((~valid_memory_plan).sum())

        # Avoid division by missing, zero, or negative planned values.
        chunk["cpu_avg_to_plan"] = np.nan
        chunk["cpu_max_to_plan"] = np.nan
        chunk["mem_avg_to_plan"] = np.nan
        chunk["mem_max_to_plan"] = np.nan

        chunk.loc[valid_cpu_plan, "cpu_avg_to_plan"] = (
            chunk.loc[valid_cpu_plan, "cpu_avg"]
            / chunk.loc[valid_cpu_plan, "plan_cpu_median"]
        )
        chunk.loc[valid_cpu_plan, "cpu_max_to_plan"] = (
            chunk.loc[valid_cpu_plan, "cpu_max"]
            / chunk.loc[valid_cpu_plan, "plan_cpu_median"]
        )

        chunk.loc[valid_memory_plan, "mem_avg_to_plan"] = (
            chunk.loc[valid_memory_plan, "mem_avg"]
            / chunk.loc[valid_memory_plan, "plan_mem_median"]
        )
        chunk.loc[valid_memory_plan, "mem_max_to_plan"] = (
            chunk.loc[valid_memory_plan, "mem_max"]
            / chunk.loc[valid_memory_plan, "plan_mem_median"]
        )

        # Keep a small random sample from each chunk.
        # This bounds memory use while representing the entire file.
        sample = chunk.sample(
            n=min(SAMPLE_PER_CHUNK, len(chunk)),
            random_state=RANDOM_SEED + chunks_processed,
        )

        sampled_chunks.append(
            sample[
                OBSERVED_COLUMNS
                + [
                    "plan_cpu_median",
                    "plan_mem_median",
                    "cpu_avg_to_plan",
                    "cpu_max_to_plan",
                    "mem_avg_to_plan",
                    "mem_max_to_plan",
                ]
            ].copy()
        )

        if chunks_processed % 100 == 0:
            print(
                f"Processed {total_rows:,} rows; "
                f"chunks={chunks_processed:,}; "
                f"sampled rows={sum(len(x) for x in sampled_chunks):,}"
            )

        del chunk
        gc.collect()

    print("\nFULL-DATA MATCH COVERAGE")
    print(f"Rows processed: {total_rows:,}")
    print(f"Chunks processed: {chunks_processed:,}")

    if total_rows:
        print(
            f"CPU planned-reference match: {cpu_matches:,} "
            f"({cpu_matches / total_rows * 100:.4f}%)"
        )
        print(
            f"Memory planned-reference match: {memory_matches:,} "
            f"({memory_matches / total_rows * 100:.4f}%)"
        )

    print(f"Rows with invalid CPU planned denominator: {cpu_ratio_invalid:,}")
    print(
        "Rows with invalid memory planned denominator: "
        f"{memory_ratio_invalid:,}"
    )

    if not sampled_chunks:
        print("No training rows were sampled.")
        return

    sample_data = pd.concat(sampled_chunks, ignore_index=True)
    del sampled_chunks
    gc.collect()

    print(f"\nBOUNDED ANALYSIS SAMPLE: {len(sample_data):,} rows")

    ratio_columns = [
        "cpu_avg_to_plan",
        "cpu_max_to_plan",
        "mem_avg_to_plan",
        "mem_max_to_plan",
    ]

    print("\nOBSERVED-TO-PLANNED RESOURCE RATIOS")
    print(
        sample_data[ratio_columns].describe(
            percentiles=[
                0.01, 0.05, 0.25, 0.50,
                0.75, 0.90, 0.95, 0.99,
            ]
        ).to_string()
    )

    correlation_columns = [
        "plan_cpu_median",
        "plan_mem_median",
        "cpu_avg",
        "cpu_max",
        "mem_avg",
        "mem_max",
        "duration",
    ]

    print("\nCORRELATION MATRIX (SAMPLED DATA)")
    print(
        sample_data[correlation_columns]
        .corr(min_periods=2)
        .to_string()
    )

    print("\nSELECTED SAMPLED CORRELATIONS")
    pairs = [
        ("plan_cpu_median", "cpu_avg"),
        ("plan_cpu_median", "cpu_max"),
        ("plan_mem_median", "mem_avg"),
        ("plan_mem_median", "mem_max"),
        ("plan_cpu_median", "duration"),
        ("plan_mem_median", "duration"),
    ]

    for left, right in pairs:
        valid = sample_data[[left, right]].dropna()

        if len(valid) < 2:
            print(f"{left} <-> {right}: insufficient data")
            continue

        print(
            f"{left} <-> {right}: "
            f"{valid[left].corr(valid[right]):.6f} "
            f"(sample pairs={len(valid):,})"
        )

    print("\nANALYSIS COMPLETE")
    print(
        "Match coverage uses all training rows. "
        "Ratio percentiles and correlations use the bounded sample."
    )
    print(
        "The task_name aggregation is exploratory; it does not establish "
        "a unique task/job identity or causal runtime effects."
    )


if __name__ == "__main__":
    main()
