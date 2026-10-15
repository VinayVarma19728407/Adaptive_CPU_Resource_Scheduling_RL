"""
PPO v3 episode-capacity benchmark.

Run from the project root:
    python .\experiments\benchmark_v3_episode_capacity.py

This benchmark does NOT train PPO, modify existing models, or write to the
dataset. It benchmarks progressively larger task samples in isolated child
processes so memory from one trial is released before the next.

It uses the actual project SchedulingEnvironment and a representative subset
of eligible rows from dataset/preprocessed/batch_task.csv. The results are
machine-specific; do not treat them as a universal capacity guarantee.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "dataset" / "preprocessed" / "batch_task.csv"
RESULT_PATH = ROOT / "experiments" / "ppo_v3_episode_capacity_benchmark.json"
CHILD_FLAG = "--benchmark-child"

# Candidate sizes grow gradually. The benchmark stops on the first failure or
# when the safety-reserve policy says not to try a larger size.
DEFAULT_CANDIDATES = [256, 512, 1_024, 2_048, 4_096, 8_192, 16_384, 32_768]


def _child(task_count: int) -> int:
    import pandas as pd
    import psutil

    # Import after launching the child so each trial starts with a clean process.
    from simulator.rl_environment import RLSchedulingEnvironment
    from simulator.task_data import load_machine_metadata

    if not DATA_PATH.exists():
        print(f"ERROR: dataset not found: {DATA_PATH}", flush=True)
        return 2

    # Use the same 8-machine capacity bounds as the PPO v3 default.
    all_machines = load_machine_metadata()
    machines = all_machines.sample(n=min(8, len(all_machines)), random_state=42).copy()
    max_cpu_capacity = float(machines["cpu_num"].max())
    max_memory_capacity = float(machines["mem_size"].max())

    # Read in chunks and stop once enough eligible rows are found. Keep only
    # the sample needed for this trial, rather than loading the whole dataset.
    chunks = []
    remaining = task_count
    for chunk in pd.read_csv(DATA_PATH, chunksize=250_000, low_memory=False):
        required = {"task_name", "plan_cpu", "plan_mem", "duration"}
        missing = required.difference(chunk.columns)
        if missing:
            print(f"ERROR: required columns missing: {sorted(missing)}", flush=True)
            return 2

        for col in ("plan_cpu", "plan_mem", "duration"):
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        eligible = chunk.dropna(subset=["task_name", "plan_cpu", "plan_mem", "duration"])
        eligible = eligible[
            (eligible["plan_cpu"] > 0)
            & (eligible["plan_mem"] > 0)
            & (eligible["duration"] > 0)
            & (eligible["plan_cpu"] / 100 <= 1.0)
            & (eligible["plan_mem"] <= 1.0)
        ]
        if not eligible.empty:
            take = eligible.head(remaining)
            chunks.append(take.copy())
            remaining -= len(take)
        if remaining <= 0:
            break

    if remaining > 0:
        print(
            f"ERROR: only {task_count - remaining} eligible rows found for requested "
            f"sample of {task_count}. Confirm resource-capacity units/filters.",
            flush=True,
        )
        return 2

    sample = pd.concat(chunks, ignore_index=True)
    # Build the same scenario schema used by the PPO v3 trainer.
    sample = sample.reset_index(drop=True)
    sample["scenario_task_id"] = [f"capacity_bench_task_{i}" for i in range(len(sample))]
    sample["task_name"] = sample["task_name"].astype(str)
    sample["job_name"] = sample["job_name"].astype(str)
    durations = sample["duration"].to_numpy(dtype=float)
    median_duration = max(float(__import__("numpy").median(durations)), 1e-6)
    rng = __import__("numpy").random.default_rng(42)
    gaps = rng.exponential(scale=max(median_duration * 0.02, 0.01), size=len(sample))
    sample["arrival_time"] = __import__("numpy").cumsum(gaps)
    sample["arrival_time"] -= sample["arrival_time"].iloc[0]
    scenario = sample[
        ["scenario_task_id", "task_name", "job_name", "plan_cpu",
         "plan_mem", "duration", "arrival_time"]
    ].copy()

    process = psutil.Process(os.getpid())
    rss_before_mb = process.memory_info().rss / (1024**2)
    started = time.perf_counter()

    env = RLSchedulingEnvironment(
        scenario=scenario,
        machine_metadata=machines,
        candidate_count=min(8, len(machines)),
        reward_mode="extended_penalty",
    )
    obs, info = env.reset(seed=42)

    # Measure environment creation/reset, not PPO learning. The task sample
    # and simulator event queue are the principal objects being capacity-tested.
    elapsed = time.perf_counter() - started
    rss_after_mb = process.memory_info().rss / (1024**2)
    print(json.dumps({
        "tasks": task_count,
        "status": "PASS",
        "elapsed_seconds": round(elapsed, 3),
        "rss_before_mb": round(rss_before_mb, 1),
        "rss_after_mb": round(rss_after_mb, 1),
        "rss_increase_mb": round(max(0, rss_after_mb - rss_before_mb), 1),
        "observation_shape": getattr(obs, "shape", None),
        "machine_count": len(machines),
    }), flush=True)
    try:
        env.close()
    except Exception:
        pass
    return 0


def main() -> int:
    if CHILD_FLAG in sys.argv:
        index = sys.argv.index(CHILD_FLAG)
        try:
            size = int(sys.argv[index + 1])
        except (IndexError, ValueError):
            print("ERROR: child mode requires a task count.", flush=True)
            return 2
        return _child(size)

    if not DATA_PATH.exists():
        print(f"ERROR: dataset not found: {DATA_PATH}")
        print(f"Expected: {DATA_PATH}")
        return 2

    try:
        import psutil
    except ImportError:
        print("ERROR: psutil is required. Install it in the active venv with:")
        print("python -m pip install psutil")
        return 2

    available_mb = psutil.virtual_memory().available / (1024**2)
    total_mb = psutil.virtual_memory().total / (1024**2)
    # Conservative benchmark policy: retain at least 2 GiB or 35% of currently
    # available memory, whichever is larger. This is a benchmark stop condition,
    # not a claim that every workload can safely use the remaining memory.
    reserve_mb = max(2048, available_mb * 0.35)
    usable_budget_mb = max(0, available_mb - reserve_mb)

    print("=" * 72)
    print("PPO V3 EPISODE CAPACITY BENCHMARK — NO TRAINING")
    print("=" * 72)
    print(f"Project root: {ROOT}")
    print(f"Dataset: {DATA_PATH}")
    print(f"Physical RAM: {total_mb:.0f} MB")
    print(f"Currently available RAM: {available_mb:.0f} MB")
    print(f"Reserved RAM policy: {reserve_mb:.0f} MB")
    print(f"Approximate benchmark growth budget: {usable_budget_mb:.0f} MB")
    print("Each candidate runs in a fresh child process.")
    print("This tests simulator construction/reset, not PPO training throughput.")
    print()

    results = []
    last_pass = None
    for size in DEFAULT_CANDIDATES:
        # Check RAM again before each candidate; skip growth trials if current
        # free RAM has fallen below the reserve.
        current_available_mb = psutil.virtual_memory().available / (1024**2)
        if current_available_mb <= reserve_mb:
            results.append({
                "tasks": size,
                "status": "SKIPPED_LOW_AVAILABLE_RAM",
                "available_ram_mb": round(current_available_mb, 1),
            })
            print(f"STOP: available RAM {current_available_mb:.0f} MB reached reserve.")
            break

        print(f"Testing episode candidate: {size:,} tasks ...", flush=True)
        try:
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), CHILD_FLAG, str(size)],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=180,
            )
            stdout = completed.stdout.strip()
            stderr = completed.stderr.strip()
            if stdout:
                print(stdout)
            if completed.returncode == 0:
                try:
                    child_result = json.loads(stdout.splitlines()[-1])
                except Exception:
                    child_result = {"tasks": size, "status": "PASS", "raw_output": stdout}
                results.append(child_result)
                last_pass = size
            else:
                results.append({
                    "tasks": size,
                    "status": "FAIL",
                    "return_code": completed.returncode,
                    "stderr_tail": stderr[-3000:],
                    "stdout_tail": stdout[-1500:],
                })
                print("STOP: candidate failed; larger sizes will not be attempted.")
                if stderr:
                    print("Child error (last 3000 characters):")
                    print(stderr[-3000:])
                break
        except subprocess.TimeoutExpired:
            results.append({"tasks": size, "status": "TIMEOUT", "timeout_seconds": 180})
            print("STOP: candidate exceeded 180 seconds; larger sizes will not be attempted.")
            break

        # Persist progress after every candidate, so partial results survive
        # interruption.
        RESULT_PATH.write_text(json.dumps({
            "benchmark": "PPO v3 simulator episode capacity",
            "dataset": str(DATA_PATH),
            "physical_ram_mb": round(total_mb, 1),
            "available_ram_at_start_mb": round(available_mb, 1),
            "reserve_policy_mb": round(reserve_mb, 1),
            "last_passing_candidate": last_pass,
            "results": results,
            "notes": [
                "No PPO training was run.",
                "This benchmark tests simulator construction/reset on an eligible sample.",
                "Episode size is not a claim of full-dataset coverage.",
                "The largest passing size is only a candidate for a short smoke test.",
            ],
        }, indent=2), encoding="utf-8")

    final = {
        "benchmark": "PPO v3 simulator episode capacity",
        "dataset": str(DATA_PATH),
        "physical_ram_mb": round(total_mb, 1),
        "available_ram_at_start_mb": round(available_mb, 1),
        "reserve_policy_mb": round(reserve_mb, 1),
        "last_passing_candidate": last_pass,
        "results": results,
        "notes": [
            "No PPO training was run.",
            "This benchmark tests simulator construction/reset on an eligible sample.",
            "Episode size is not a claim of full-dataset coverage.",
            "The largest passing size is only a candidate for a short smoke test.",
        ],
    }
    RESULT_PATH.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print()
    print("=" * 72)
    print(f"Last passing candidate: {last_pass if last_pass is not None else 'NONE'} tasks")
    print(f"Detailed report: {RESULT_PATH}")
    print("Do not start full PPO training from this benchmark result alone.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
