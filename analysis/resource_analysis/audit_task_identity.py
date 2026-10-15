
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TASK_FILE = ROOT / "dataset" / "preprocessed" / "batch_task.csv"
POOL_FILE = ROOT / "experiments" / "data" / "task_pool_seed42.csv"
OUTPUT_DIR = ROOT / "analysis" / "resource_analysis"

print("=" * 72)
print("TASK IDENTITY AND RESOURCE REFERENCE AUDIT")
print("=" * 72)

# Load the smaller task reference dataset.
print("\n[1/5] Loading batch_task reference...")
tasks = pd.read_csv(
    TASK_FILE,
    usecols=["task_name", "job_name", "plan_cpu", "plan_mem"],
    dtype={"task_name": "string", "job_name": "string"},
)

for col in ["plan_cpu", "plan_mem"]:
    tasks[col] = pd.to_numeric(tasks[col], errors="coerce")

print(f"Rows: {len(tasks):,}")
print(f"Unique task_name values: {tasks['task_name'].nunique():,}")
print(f"Unique task/job pairs: {tasks[['task_name', 'job_name']].drop_duplicates().shape[0]:,}")

# Count task names associated with multiple jobs.
print("\n[2/5] Checking task_name-to-job_name relationships...")
jobs_per_task = tasks.groupby("task_name", dropna=False)["job_name"].nunique()
multi_job_names = jobs_per_task[jobs_per_task > 1]

print(f"task_name values associated with multiple jobs: {len(multi_job_names):,}")
print(f"Maximum distinct jobs for one task_name: {jobs_per_task.max()}")
print("\n[IDENTIFIER QUALITY CHECK]")
print("Missing task_name rows:", int(tasks["task_name"].isna().sum()))
print("Missing job_name rows:", int(tasks["job_name"].isna().sum()))

print("\nTop 10 task_name values by distinct job count:")
print(jobs_per_task.sort_values(ascending=False).head(10).to_string())

print("\nTop 10 non-missing task names by distinct job count:")
print(
    jobs_per_task.loc[jobs_per_task.index.notna()]
    .sort_values(ascending=False)
    .head(10)
    .to_string()
)
# Build a diagnostic task_name-only reference.
# This aggregation is for comparison only; it does not assert that
# the median is the correct demand for any individual task instance.
name_reference = (
    tasks.groupby("task_name", dropna=False)
    .agg(
        name_plan_cpu=("plan_cpu", "median"),
        name_plan_mem=("plan_mem", "median"),
        name_job_count=("job_name", "nunique"),
        name_cpu_values=("plan_cpu", "nunique"),
        name_mem_values=("plan_mem", "nunique"),
    )
    .reset_index()
)

# Build a composite-key reference.
pair_reference = (
    tasks.groupby(["task_name", "job_name"], dropna=False)
    .agg(
        pair_plan_cpu=("plan_cpu", "median"),
        pair_plan_mem=("plan_mem", "median"),
        pair_source_rows=("plan_cpu", "size"),
        pair_cpu_values=("plan_cpu", "nunique"),
        pair_mem_values=("plan_mem", "nunique"),
    )
    .reset_index()
)

# Compare both lookup approaches on the existing bounded task pool.
print("\n[3/5] Comparing lookups on the 10,000-row task pool...")
pool = pd.read_csv(
    POOL_FILE,
    usecols=["task_name", "job_name", "plan_cpu", "plan_mem"],
    dtype={"task_name": "string", "job_name": "string"},
)

pool["plan_cpu"] = pd.to_numeric(pool["plan_cpu"], errors="coerce")
pool["plan_mem"] = pd.to_numeric(pool["plan_mem"], errors="coerce")

pool = pool.merge(name_reference, on="task_name", how="left", validate="many_to_one")
pool = pool.merge(
    pair_reference,
    on=["task_name", "job_name"],
    how="left",
    validate="many_to_one",
)

name_match = pool["name_plan_cpu"].notna() & pool["name_plan_mem"].notna()
pair_match = pool["pair_plan_cpu"].notna() & pool["pair_plan_mem"].notna()

cpu_diff = (
    pool["name_plan_cpu"].notna()
    & pool["pair_plan_cpu"].notna()
    & (pool["name_plan_cpu"] != pool["pair_plan_cpu"])
)
mem_diff = (
    pool["name_plan_mem"].notna()
    & pool["pair_plan_mem"].notna()
    & (pool["name_plan_mem"] != pool["pair_plan_mem"])
)

print(f"Pool rows: {len(pool):,}")
print(f"Rows with both resource values under task_name lookup: {name_match.sum():,}")
print(f"Rows with both resource values under composite lookup: {pair_match.sum():,}")
print(f"CPU reference values differing between approaches: {cpu_diff.sum():,}")
print(f"Memory reference values differing between approaches: {mem_diff.sum():,}")
print(
    "Rows whose task_name is associated with multiple jobs in reference: "
    f"{pool['name_job_count'].fillna(0).gt(1).sum():,}"
)

# Report ambiguity within composite keys too.
print("\n[4/5] Checking resource variation within each key...")
print(
    "Composite keys with multiple distinct plan_cpu values: "
    f"{(pair_reference['pair_cpu_values'] > 1).sum():,}"
)
print(
    "Composite keys with multiple distinct plan_mem values: "
    f"{(pair_reference['pair_mem_values'] > 1).sum():,}"
)

# Save small audit artifacts.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

summary = pd.DataFrame(
    [
        ("batch_task_rows", len(tasks)),
        ("unique_task_names", tasks["task_name"].nunique()),
        ("unique_task_job_pairs", tasks[["task_name", "job_name"]].drop_duplicates().shape[0]),
        ("task_names_with_multiple_jobs", len(multi_job_names)),
        ("pool_rows", len(pool)),
        ("pool_name_lookup_complete", int(name_match.sum())),
        ("pool_composite_lookup_complete", int(pair_match.sum())),
        ("pool_cpu_reference_differences", int(cpu_diff.sum())),
        ("pool_memory_reference_differences", int(mem_diff.sum())),
        ("composite_keys_with_multiple_cpu_values", int((pair_reference["pair_cpu_values"] > 1).sum())),
        ("composite_keys_with_multiple_memory_values", int((pair_reference["pair_mem_values"] > 1).sum())),
    ],
    columns=["metric", "value"],
)

summary_path = OUTPUT_DIR / "task_identity_audit_summary.csv"
summary.to_csv(summary_path, index=False)

examples_path = OUTPUT_DIR / "task_identity_audit_examples.csv"
examples = pool.loc[
    pool["name_job_count"].fillna(0).gt(1)
    | cpu_diff
    | mem_diff,
    [
        "task_name", "job_name",
        "plan_cpu", "plan_mem",
        "name_plan_cpu", "pair_plan_cpu",
        "name_plan_mem", "pair_plan_mem",
        "name_job_count",
    ],
].head(200)
examples.to_csv(examples_path, index=False)

print("\n[5/5] Audit files saved:")
print(summary_path)
print(examples_path)
print("\nAUDIT COMPLETE")
print("No source datasets or simulator code were modified.")
