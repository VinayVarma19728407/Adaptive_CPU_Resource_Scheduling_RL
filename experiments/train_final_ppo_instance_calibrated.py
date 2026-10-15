from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback

from simulator.config import PROJECT_ROOT, RANDOM_SEED, BATCH_TASK_FILE, TRAINING_FILE, CSV_CHUNK_SIZE
from simulator.rl_environment import RLSchedulingEnvironment
from simulator.task_data import load_machine_metadata


FEATURES_PER_RECORD = 3  # plan_cpu, plan_mem, duration
DEFAULT_TIMESTEPS = 14_000_000
DEFAULT_TASKS_PER_EPISODE = 512
DEFAULT_MACHINE_COUNT = 8
DEFAULT_CANDIDATE_COUNT = 8
DEFAULT_ARRIVAL_FRACTION = 0.02
DEFAULT_REWARD_MODE = "extended_penalty"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train final MaskablePPO by traversing every eligible batch_task record."
    )
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--tasks-per-episode", type=int, default=DEFAULT_TASKS_PER_EPISODE)
    parser.add_argument("--machine-count", type=int, default=DEFAULT_MACHINE_COUNT)
    parser.add_argument("--candidate-count", type=int, default=DEFAULT_CANDIDATE_COUNT)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--arrival-fraction", type=float, default=DEFAULT_ARRIVAL_FRACTION)
    parser.add_argument("--reward-mode", choices=["core_penalty", "extended_penalty"], default=DEFAULT_REWARD_MODE)
    parser.add_argument("--run-name", default="final_ppo_full_dataset_20261012")
    parser.add_argument("--chunk-size", type=int, default=CSV_CHUNK_SIZE)
    return parser.parse_args()


def select_machine_pool(machine_count: int, seed: int) -> pd.DataFrame:
    machines = load_machine_metadata()
    if machine_count <= 0 or machine_count > len(machines):
        raise ValueError(f"machine-count must be between 1 and {len(machines)}")
    chosen = machines.sample(n=machine_count, random_state=seed).copy()
    chosen = chosen.sort_values("machine_id").reset_index(drop=True)
    if chosen["machine_id"].duplicated().any():
        raise ValueError("Duplicate machine IDs in selected machine pool")
    return chosen


def build_instance_calibration_profile(source_path: Path, cache_path: Path, chunk_size: int) -> dict[tuple[str, str], tuple[float, float, float, int]]:
    """Stream the instance training split and aggregate completed-instance observations.

    Calibration key is (task_name, task_type), because project audits found task_name
    can occur under multiple job names; task_type matching was validated. Duration is
    the observed instance duration in the same timestamp units as the source data.
    CPU/memory utilization means are retained in the profile/report for calibration
    traceability, but are not converted into task demand because their normalized
    units are not directly interchangeable with plan_cpu/plan_mem.
    """
    if not source_path.is_file():
        raise FileNotFoundError(f"Required instance training split not found: {source_path}")
    # Reuse the completed calibration profile on restart; the source contains hundreds
    # of millions of rows, so rescanning it after a downstream preprocessing error is wasteful.
    if cache_path.is_file() and cache_path.stat().st_size > 0:
        cached = pd.read_csv(cache_path, low_memory=False)
        needed = {"task_name", "task_type", "observed_mean_duration", "observed_mean_cpu_avg", "observed_mean_mem_avg", "completed_instance_rows"}
        if needed.issubset(cached.columns) and len(cached):
            profile = {}
            for row in cached.itertuples(index=False):
                profile[(str(row.task_name).strip(), str(row.task_type).strip())] = (
                    float(row.observed_mean_duration),
                    float(row.observed_mean_cpu_avg) if pd.notna(row.observed_mean_cpu_avg) else float("nan"),
                    float(row.observed_mean_mem_avg) if pd.notna(row.observed_mean_mem_avg) else float("nan"),
                    int(row.completed_instance_rows),
                )
            print(f"Loaded cached instance calibration profile: {len(profile):,} keys from {cache_path}")
            return profile
    sums: dict[tuple[str, str], list[float]] = {}
    processed = 0
    used = 0
    print(f"Streaming batch_instance calibration data from: {source_path}")
    cols = ["task_name", "task_type", "status", "duration", "cpu_avg", "mem_avg"]
    for chunk_no, chunk in enumerate(pd.read_csv(source_path, usecols=cols, chunksize=chunk_size, low_memory=False), start=1):
        for col in ("status", "duration", "cpu_avg", "mem_avg"):
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        # Status 4 is Terminated in the project's established preprocessing encoding.
        valid = (chunk["status"] == 4) & np.isfinite(chunk["duration"]) & (chunk["duration"] > 0)
        valid &= chunk["task_name"].notna() & chunk["task_type"].notna()
        chosen = chunk.loc[valid, ["task_name", "task_type", "duration", "cpu_avg", "mem_avg"]]
        for row in chosen.itertuples(index=False, name=None):
            task_name, task_type, duration, cpu_avg, mem_avg = row
            key = (str(task_name).strip(), str(task_type).strip())
            # sum duration, sum finite cpu_avg, sum finite mem_avg, and separate counts
            entry = sums.get(key)
            if entry is None:
                entry = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                sums[key] = entry
            entry[0] += float(duration); entry[3] += 1.0
            if pd.notna(cpu_avg) and np.isfinite(cpu_avg):
                entry[1] += float(cpu_avg); entry[4] += 1.0
            if pd.notna(mem_avg) and np.isfinite(mem_avg):
                entry[2] += float(mem_avg); entry[5] += 1.0
            used += 1
        processed += len(chunk)
        if chunk_no % 20 == 0:
            print(f"  instance chunks: {chunk_no:,}; rows scanned: {processed:,}; valid completed rows: {used:,}; profile keys: {len(sums):,}")
    if not sums:
        raise ValueError("No valid completed batch_instance rows were found for calibration.")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    profile = {}
    for (task_name, task_type), v in sums.items():
        mean_duration = v[0] / v[3]
        mean_cpu = v[1] / v[4] if v[4] else float("nan")
        mean_mem = v[2] / v[5] if v[5] else float("nan")
        profile[(task_name, task_type)] = (mean_duration, mean_cpu, mean_mem, int(v[3]))
        rows.append((task_name, task_type, mean_duration, mean_cpu, mean_mem, int(v[3])))
    pd.DataFrame(rows, columns=["task_name", "task_type", "observed_mean_duration", "observed_mean_cpu_avg", "observed_mean_mem_avg", "completed_instance_rows"]).to_csv(cache_path, index=False)
    print(f"Instance calibration profile keys: {len(profile):,}; cache: {cache_path}")
    return profile


def build_numeric_eligible_pool(
    source_path: Path,
    binary_path: Path,
    max_cpu_capacity: float,
    max_memory_capacity: float,
    chunk_size: int,
    instance_profile: dict[tuple[str, str], tuple[float, float, float, int]],
) -> tuple[int, dict[str, int]]:
    """Stream source CSV and persist only eligible numeric features to disk.

    Task/job names are checked for eligibility but not retained in RAM because
    the simulator's policy uses CPU, memory, duration and generated task IDs.
    """
    required = {"task_name", "job_name", "task_type", "plan_cpu", "plan_mem", "duration"}
    if not source_path.is_file():
        raise FileNotFoundError(f"Required dataset not found: {source_path}")
    binary_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = binary_path.with_suffix(binary_path.suffix + ".tmp")
    if temp_path.exists():
        temp_path.unlink()
    total = 0
    matched = 0
    print(f"Streaming eligible records from: {source_path}")
    with temp_path.open("wb") as out:
        for chunk_no, chunk in enumerate(
            pd.read_csv(source_path, usecols=lambda c: c in required, chunksize=chunk_size, low_memory=False),
            start=1,
        ):
            missing = required - set(chunk.columns)
            if missing:
                raise ValueError(f"batch_task.csv is missing columns: {sorted(missing)}")
            for col in ("plan_cpu", "plan_mem", "duration"):
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
            valid = (
                chunk["task_name"].notna()
                & chunk["job_name"].notna()
                & np.isfinite(chunk["plan_cpu"])
                & np.isfinite(chunk["plan_mem"])
                & np.isfinite(chunk["duration"])
                & (chunk["plan_cpu"] > 0)
                & (chunk["plan_mem"] > 0)
                & (chunk["duration"] > 0)
                & ((chunk["plan_cpu"] / 100.0) <= max_cpu_capacity)
                & (chunk["plan_mem"] <= max_memory_capacity)
            )
            eligible = chunk.loc[valid, ["task_name", "task_type", "plan_cpu", "plan_mem", "duration"]].copy()
            # Use observed completed-instance duration when the validated task_name+task_type
            # key exists. Planned CPU/memory remain the simulator's resource demands.
            keys = zip(eligible["task_name"].astype(str).str.strip(), eligible["task_type"].astype(str).str.strip())
            calibrated_durations = np.fromiter(
                (instance_profile.get(key, (float("nan"), float("nan"), float("nan"), 0))[0] for key in keys),
                dtype=np.float64,
                count=len(eligible),
            )
            has_calibration = np.isfinite(calibrated_durations) & (calibrated_durations > 0)
            # Observed durations can be fractional. Promote before assignment so pandas
            # does not attempt to write float values into an int64 column.
            eligible["duration"] = pd.to_numeric(eligible["duration"], errors="coerce").astype(np.float64)
            eligible.loc[has_calibration, "duration"] = calibrated_durations[has_calibration]
            matched += int(has_calibration.sum())
            values = eligible[["plan_cpu", "plan_mem", "duration"]].to_numpy(dtype=np.float64, copy=True)
            if values.size:
                values.tofile(out)
                total += len(values)
            if chunk_no % 10 == 0:
                print(f"  chunks processed: {chunk_no:,}; eligible records: {total:,}")
    if total == 0:
        temp_path.unlink(missing_ok=True)
        raise ValueError("No eligible task records remain after resource and duration filters")
    temp_path.replace(binary_path)
    manifest = {
        "source": str(source_path),
        "source_size_bytes": source_path.stat().st_size,
        "source_mtime_ns": source_path.stat().st_mtime_ns,
        "eligible_records": total,
        "feature_columns": ["plan_cpu", "plan_mem", "duration"],
        "dtype": "float64",
        "eligibility": [
            "task_name and job_name non-null",
            "finite plan_cpu, plan_mem, duration",
            "plan_cpu > 0, plan_mem > 0, duration > 0",
            "plan_cpu / 100 <= maximum selected machine CPU capacity",
            "plan_mem <= maximum selected machine memory capacity",
        ],
    }
    manifest["instance_calibration_source"] = str(TRAINING_FILE)
    manifest["records_duration_calibrated_from_instances"] = matched
    manifest["duration_calibration_match_percent"] = round(100.0 * matched / total, 6)
    binary_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return total, {"records_duration_calibrated_from_instances": matched, "duration_calibration_match_percent": round(100.0 * matched / total, 6)}


class FullDatasetTraversalEnv(RLSchedulingEnvironment):
    """Traverses every eligible row once per pass, then reshuffles for another pass."""

    def __init__(self, feature_memmap: np.memmap, pool_size: int, machine_metadata: pd.DataFrame,
                 tasks_per_episode: int, candidate_count: int, arrival_fraction: float,
                 seed: int, reward_mode: str):
        if tasks_per_episode <= 0:
            raise ValueError("tasks-per-episode must be positive")
        if not 0 < arrival_fraction <= 1:
            raise ValueError("arrival-fraction must be in (0, 1]")
        self.features = feature_memmap
        self.pool_size = int(pool_size)
        self.tasks_per_episode = int(tasks_per_episode)
        self.arrival_fraction = float(arrival_fraction)
        self.rng = np.random.default_rng(seed)
        self.order = np.arange(self.pool_size, dtype=np.int32)
        self.rng.shuffle(self.order)
        self.cursor = 0
        self.dataset_pass = 0
        self.episode_number = 0
        self.exposed = np.zeros(self.pool_size, dtype=np.bool_)
        self.placement_decisions = 0
        self.records_consumed_current_pass = 0
        self._pending_initial_scenario = True
        initial = self._make_scenario()
        super().__init__(scenario=initial, machine_metadata=machine_metadata,
                         candidate_count=candidate_count, reward_mode=reward_mode)

    def _next_indices(self) -> np.ndarray:
        if self.cursor >= self.pool_size:
            self.rng.shuffle(self.order)
            self.cursor = 0
            self.dataset_pass += 1
            self.records_consumed_current_pass = 0
        end = min(self.cursor + self.tasks_per_episode, self.pool_size)
        indices = self.order[self.cursor:end].copy()
        self.cursor = end
        self.records_consumed_current_pass += len(indices)
        return indices

    def _make_scenario(self) -> pd.DataFrame:
        indices = self._next_indices()
        selected = np.asarray(self.features[indices], dtype=np.float64)
        durations = selected[:, 2]
        median_duration = max(float(np.median(durations)), 1e-6)
        mean_gap = max(median_duration * self.arrival_fraction, 0.01)
        gaps = self.rng.exponential(scale=mean_gap, size=len(indices))
        arrivals = np.cumsum(gaps)
        arrivals -= arrivals[0]
        # ID contains source-pool row index so successful placements can be tracked.
        scenario = pd.DataFrame({
            "scenario_task_id": [f"poolrow_{int(i)}" for i in indices],
            "task_name": [f"task_{int(i)}" for i in indices],
            "job_name": [f"job_{int(i)}" for i in indices],
            "plan_cpu": selected[:, 0],
            "plan_mem": selected[:, 1],
            "duration": durations,
            "arrival_time": arrivals.astype(float),
        })
        self.episode_number += 1
        return scenario

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self._pending_initial_scenario:
            self._pending_initial_scenario = False
        else:
            self.scenario_source = self._make_scenario()
        return super().reset(seed=seed, options=options)

    def step(self, action: int):
        task_id = self.current_task_id
        result = super().step(action)
        if task_id and task_id.startswith("poolrow_"):
            row_idx = int(task_id.removeprefix("poolrow_"))
            if not self.exposed[row_idx]:
                self.exposed[row_idx] = True
            self.placement_decisions += 1
        return result

    def exposure_summary(self) -> dict:
        unique = int(self.exposed.sum())
        return {
            "eligible_pool_records": self.pool_size,
            "placement_decisions": self.placement_decisions,
            "unique_records_exposed": unique,
            "coverage_percent": round(100.0 * unique / self.pool_size, 6),
            "dataset_pass_started_zero_based": self.dataset_pass,
            "records_consumed_current_pass": self.records_consumed_current_pass,
            "full_unique_record_coverage": unique == self.pool_size,
        }


def main() -> None:
    args = parse_args()
    if args.timesteps <= 0:
        raise ValueError("timesteps must be positive")
    if args.candidate_count <= 0 or args.candidate_count > args.machine_count:
        raise ValueError("candidate-count must be between 1 and machine-count")

    run_root = PROJECT_ROOT / "experiments" / "models" / "final_dataset_ppo_runs"
    run_dir = run_root / args.run_name
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}; choose a new --run-name")

    machines = select_machine_pool(args.machine_count, args.seed)
    data_dir = PROJECT_ROOT / "experiments" / "data"
    numeric_cache = data_dir / "eligible_task_features_instance_calibrated.f64"
    profile_cache = data_dir / "batch_instance_training_calibration_profile.csv"
    instance_profile = build_instance_calibration_profile(TRAINING_FILE, profile_cache, args.chunk_size)
    pool_size, calibration_summary = build_numeric_eligible_pool(
        BATCH_TASK_FILE, numeric_cache,
        max_cpu_capacity=float(machines["cpu_num"].max()),
        max_memory_capacity=float(machines["mem_size"].max()),
        chunk_size=args.chunk_size,
        instance_profile=instance_profile,
    )
    features = np.memmap(numeric_cache, dtype=np.float64, mode="r", shape=(pool_size, FEATURES_PER_RECORD))

    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "algorithm": "MaskablePPO",
        "dataset": str(BATCH_TASK_FILE),
        "instance_calibration_dataset": str(TRAINING_FILE),
        "instance_calibration_profile": str(profile_cache),
        "instance_calibration": calibration_summary,
        "calibration_method": "For matched task_name+task_type records, replace planned duration with mean positive duration from completed (status=4) batch_instance training rows. Planned CPU/memory demands remain from batch_task because observed normalized utilization units are not directly interchangeable with plan_cpu/plan_mem.",
        "eligible_records": pool_size,
        "full_dataset_traversal": True,
        "timesteps_requested": args.timesteps,
        "tasks_per_episode": args.tasks_per_episode,
        "machine_count": args.machine_count,
        "candidate_count": args.candidate_count,
        "reward_mode": args.reward_mode,
        "seed": args.seed,
        "network_architecture": {"pi": [256, 128], "vf": [256, 128]},
        "n_steps": 512,
        "batch_size": 128,
        "n_epochs": 10,
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "arrival_fraction_of_median_duration": args.arrival_fraction,
        "note": "Arrival times are generated; task resource requests and durations are sourced from eligible batch_task records.",
    }
    (run_dir / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("\nFINAL FULL-DATASET MASKABLE PPO TRAINING")
    print(f"Eligible task records: {pool_size:,}")
    print(f"Instance duration calibration: {calibration_summary['records_duration_calibrated_from_instances']:,} records ({calibration_summary['duration_calibration_match_percent']:.3f}%)")
    print(f"Requested timesteps: {args.timesteps:,}")
    print(f"Tasks per episode: {args.tasks_per_episode:,}")
    print(f"Reward mode: {args.reward_mode}")
    print(f"Run directory: {run_dir}")

    env = FullDatasetTraversalEnv(
        feature_memmap=features, pool_size=pool_size, machine_metadata=machines,
        tasks_per_episode=args.tasks_per_episode, candidate_count=args.candidate_count,
        arrival_fraction=args.arrival_fraction, seed=args.seed, reward_mode=args.reward_mode,
    )
    callback = CheckpointCallback(
        save_freq=100_000, save_path=str(checkpoint_dir), name_prefix="final_ppo_full_dataset",
        save_replay_buffer=False, save_vecnormalize=False,
    )
    model = MaskablePPO(
        "MlpPolicy", env, learning_rate=3e-4, n_steps=512, batch_size=128,
        n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
        policy_kwargs={"net_arch": {"pi": [256, 128], "vf": [256, 128]}},
        seed=args.seed, device="cpu", verbose=1,
    )
    started = time.perf_counter()
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=False, reset_num_timesteps=True)
    elapsed = time.perf_counter() - started
    model_path = run_dir / "final_ppo_full_dataset"
    model.save(str(model_path))
    exposure = env.exposure_summary()
    (run_dir / "record_exposure_summary.json").write_text(json.dumps(exposure, indent=2), encoding="utf-8")
    summary = {
        "status": "training_completed",
        "timesteps_requested": args.timesteps,
        "timesteps_completed": int(model.num_timesteps),
        "elapsed_seconds": round(elapsed, 3),
        "elapsed_minutes": round(elapsed / 60.0, 3),
        "model_file": str(model_path.with_suffix(".zip")),
        "run_directory": str(run_dir),
        "dataset_exposure": exposure,
    }
    (run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    env.close()
    print("\nFinal PPO training completed.")
    print(f"Timesteps: {model.num_timesteps:,}")
    print(f"Model: {model_path}.zip")
    print(f"Unique records exposed: {exposure['unique_records_exposed']:,}/{pool_size:,}")
    print(f"Dataset coverage: {exposure['coverage_percent']:.6f}%")
    print(f"Exposure report: {run_dir / 'record_exposure_summary.json'}")
    print(f"Training summary: {run_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()
