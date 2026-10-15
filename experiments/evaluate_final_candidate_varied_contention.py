
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO

from simulator.config import PROJECT_ROOT
from simulator.environment import SchedulingEnvironment
from simulator.machine_state import MachineStateManager
from simulator.rl_environment import RLSchedulingEnvironment
from simulator.task_data import load_machine_metadata
from experiments.bounded_evaluation_task_pool import (
    load_bounded_eligible_task_pool,
)
from experiments.calibrate_workload_profiles import build_profiles


RUN_NAME = (
    "final_candidate_4000_machines_10_candidates_50000_20261014"
)
MODEL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "models"
    / "final_dataset_ppo_runs"
    / RUN_NAME
    / "final_dataset_ppo.zip"
)
CONFIG_PATH = MODEL_PATH.parent / "training_config.json"

OUTPUT_NAME = "final_candidate_varied_contention_10_machines_20261014"

MACHINE_COUNT = 10
CANDIDATE_COUNT = 10
TASK_POOL_SIZE = 50_000
TASK_COUNT = 5_000

SAMPLE_SEED = 42
MACHINE_SEED = 50001
ARRIVAL_SEED = 50001
SCENARIO_SEEDS = [50001]

PROFILE_NAMES = [
    "high_exponential",
    "bursty_100_per_unit",
]


def create_machine_pool(machine_count: int) -> pd.DataFrame:
    all_machines = load_machine_metadata()

    if machine_count > len(all_machines):
        raise ValueError(
            f"Requested {machine_count} machines, but only "
            f"{len(all_machines)} eligible machines are available."
        )

    machines = all_machines.sample(
        n=machine_count,
        random_state=MACHINE_SEED,
    ).copy()

    machines = machines.sort_values(
        "machine_id"
    ).reset_index(drop=True)

    if machines["machine_id"].duplicated().any():
        raise ValueError("Duplicate machine IDs in machine pool.")

    return machines


def create_scenario(
    task_pool: pd.DataFrame,
    arrivals: np.ndarray,
    seed: int,
    profile: str,
) -> pd.DataFrame:
    if TASK_COUNT > len(task_pool):
        raise ValueError("Task count exceeds the available task pool.")

    if len(arrivals) < TASK_COUNT:
        raise ValueError(
            f"Profile {profile} has fewer than {TASK_COUNT} arrivals."
        )

    rng = np.random.default_rng(seed)
    task_sample_seed = int(rng.integers(0, 2**31 - 1))

    selected = task_pool.sample(
        n=TASK_COUNT,
        replace=False,
        random_state=task_sample_seed,
    ).reset_index(drop=True)

    selected_arrivals = np.asarray(
        arrivals[:TASK_COUNT], dtype=float
    )
    selected_arrivals -= selected_arrivals.min()

    if (
        not np.isfinite(selected_arrivals).all()
        or (selected_arrivals < 0).any()
    ):
        raise ValueError("Generated scenario has invalid arrival times.")

    return pd.DataFrame({
        "scenario_task_id": [
            f"{profile}_{seed:06d}_{i:06d}"
            for i in range(TASK_COUNT)
        ],
        "task_name": selected["task_name"].astype(str).to_numpy(),
        "job_name": selected["job_name"].astype(str).to_numpy(),
        "plan_cpu": pd.to_numeric(
            selected["plan_cpu"], errors="raise"
        ).to_numpy(),
        "plan_mem": pd.to_numeric(
            selected["plan_mem"], errors="raise"
        ).to_numpy(),
        "duration": pd.to_numeric(
            selected["duration"], errors="raise"
        ).to_numpy(dtype=float),
        "arrival_time": selected_arrivals,
    })


def summarize(
    policy: str,
    profile: str,
    seed: int,
    scenario: pd.DataFrame,
    results: pd.DataFrame,
    final_time: float,
    manager: MachineStateManager,
    invalid_actions: int = 0,
) -> dict:
    completed = results[results["status"] == "completed"]
    completed_count = len(completed)
    task_count = len(scenario)

    waits = pd.to_numeric(
        completed["wait_time"], errors="coerce"
    ).dropna()

    flows = pd.to_numeric(
        completed["flow_time"], errors="coerce"
    ).dropna()

    manager.validate_invariants()
    active_allocations = int(manager.active_allocation_count)

    if active_allocations:
        raise RuntimeError(
            f"{policy}/{profile}: active allocations remain."
        )

    return {
        "profile": profile,
        "seed": seed,
        "policy": policy,
        "tasks": task_count,
        "completed": completed_count,
        "unschedulable": int(
            (results["status"] == "unschedulable").sum()
        ),
        "completion_rate": (
            completed_count / task_count if task_count else 0.0
        ),
        "mean_wait_time": float(waits.mean()) if len(waits) else 0.0,
        "total_wait_time": float(waits.sum()) if len(waits) else 0.0,
        "mean_flow_time": float(flows.mean()) if len(flows) else 0.0,
        "total_flow_time": float(flows.sum()) if len(flows) else 0.0,
        "makespan": float(final_time),
        "invalid_actions": int(invalid_actions),
        "resource_invariants_passed": True,
        "active_allocations_at_end": active_allocations,
    }


def run_baseline(
    scenario: pd.DataFrame,
    machines: pd.DataFrame,
    policy: str,
    profile: str,
    seed: int,
) -> tuple[dict, pd.DataFrame]:
    manager = MachineStateManager(machines.copy())

    env = SchedulingEnvironment(
        scenario=scenario.copy(),
        machine_state=manager,
        placement_policy=policy,
    )

    outcome = env.run()

    record = summarize(
        policy=policy,
        profile=profile,
        seed=seed,
        scenario=scenario,
        results=outcome["results"],
        final_time=float(outcome["final_time"]),
        manager=manager,
    )

    return record, outcome["results"].copy()


def run_ppo(
    scenario: pd.DataFrame,
    machines: pd.DataFrame,
    model: MaskablePPO,
    profile: str,
    seed: int,
) -> tuple[dict, pd.DataFrame]:
    env = RLSchedulingEnvironment(
        scenario=scenario.copy(),
        machine_metadata=machines.copy(),
        candidate_count=CANDIDATE_COUNT,
        reward_mode="extended_penalty",
    )

    observation, _ = env.reset(seed=seed)
    terminated = False
    truncated = False
    invalid_actions = 0
    steps = 0

    while not (terminated or truncated):
        masks = np.asarray(env.action_masks(), dtype=bool)

        if masks.shape != (CANDIDATE_COUNT,) or not masks.any():
            raise RuntimeError(
                f"Invalid action mask for {profile}, seed {seed}."
            )

        action, _ = model.predict(
            observation,
            deterministic=True,
            action_masks=masks,
        )

        action_index = int(np.asarray(action).item())

        if (
            action_index < 0
            or action_index >= len(masks)
            or not masks[action_index]
        ):
            invalid_actions += 1

        observation, _, terminated, truncated, _ = env.step(action_index)
        steps += 1

        if steps > len(scenario) * 3:
            raise RuntimeError(
                f"PPO exceeded expected decision count for {profile}."
            )

    if env.sim is None:
        raise RuntimeError("PPO simulator was not initialized.")

    results = pd.DataFrame(env.sim.results.values())
    manager = env.sim.machine_state

    record = summarize(
        policy="maskable_ppo",
        profile=profile,
        seed=seed,
        scenario=scenario,
        results=results,
        final_time=float(env.sim.current_time),
        manager=manager,
        invalid_actions=invalid_actions,
    )

    env.close()
    return record, results


def main() -> None:
    start = time.perf_counter()

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Training config not found: {CONFIG_PATH}")

    training_config = json.loads(
        CONFIG_PATH.read_text(encoding="utf-8")
    )

    if int(training_config["candidate_count"]) != CANDIDATE_COUNT:
        raise ValueError("Candidate count does not match the trained model.")

    output_dir = PROJECT_ROOT / "experiments" / "results" / OUTPUT_NAME
    output_dir.mkdir(parents=True, exist_ok=False)

    print("LOADING FINAL PPO CANDIDATE AND EVALUATION DATA")

    machines = create_machine_pool(MACHINE_COUNT)

    task_pool = load_bounded_eligible_task_pool(
        max_cpu_capacity=float(machines["cpu_num"].max()),
        max_memory_capacity=float(machines["mem_size"].max()),
        sample_size=TASK_POOL_SIZE,
        seed=SAMPLE_SEED,
    )

    durations = task_pool["duration"].to_numpy(dtype=float)
    cpu = task_pool["plan_cpu"].to_numpy(dtype=float) / 100.0
    memory = task_pool["plan_mem"].to_numpy(dtype=float)

    if not (
        np.isfinite(durations).all()
        and np.isfinite(cpu).all()
        and np.isfinite(memory).all()
        and (durations > 0).all()
        and (cpu > 0).all()
        and (memory > 0).all()
    ):
        raise ValueError("Task pool contains invalid resource values.")

    profiles = build_profiles(durations)
    model = MaskablePPO.load(str(MODEL_PATH), device="cpu")

    records = []
    task_outputs = []
    profile_summaries = []

    print(f"Model: {MODEL_PATH}")
    print(f"Machines: {len(machines):,}")
    print(f"Eligible task pool: {len(task_pool):,}")
    print(f"Tasks per episode: {TASK_COUNT:,}")
    print(f"Profiles: {PROFILE_NAMES}")
    print(f"Scenario seeds: {SCENARIO_SEEDS}")

    for profile in PROFILE_NAMES:
        profile_arrivals = profiles[profile]

        for seed in SCENARIO_SEEDS:
            scenario = create_scenario(
                task_pool=task_pool,
                arrivals=profile_arrivals,
                seed=seed,
                profile=profile,
            )

            profile_summaries.append({
                "profile": profile,
                "seed": seed,
                "machines": MACHINE_COUNT,
                "tasks": len(scenario),
                "arrival_window": float(
                    scenario["arrival_time"].max()
                    - scenario["arrival_time"].min()
                ),
                "mean_interarrival_gap": float(
                    scenario["arrival_time"].diff().dropna().mean()
                ),
            })

            print(f"\nProfile: {profile} | seed: {seed}")

            for policy in ("first_fit", "least_loaded"):
                record, task_results = run_baseline(
                    scenario, machines, policy, profile, seed
                )

                records.append(record)
                task_results.insert(0, "profile", profile)
                task_results.insert(1, "policy", policy)
                task_results.insert(2, "evaluation_seed", seed)
                task_outputs.append(task_results)

                print(
                    f"{policy}: completed={record['completed']}, "
                    f"wait={record['mean_wait_time']:.4f}, "
                    f"makespan={record['makespan']:.4f}"
                )

            record, task_results = run_ppo(
                scenario, machines, model, profile, seed
            )

            records.append(record)
            task_results.insert(0, "profile", profile)
            task_results.insert(1, "policy", "maskable_ppo")
            task_results.insert(2, "evaluation_seed", seed)
            task_outputs.append(task_results)

            print(
                f"maskable_ppo: completed={record['completed']}, "
                f"wait={record['mean_wait_time']:.4f}, "
                f"makespan={record['makespan']:.4f}, "
                f"invalid_actions={record['invalid_actions']}"
            )

    metrics_df = pd.DataFrame(records)
    tasks_df = pd.concat(task_outputs, ignore_index=True)
    profiles_df = pd.DataFrame(profile_summaries)

    metrics_df.to_csv(output_dir / "episode_metrics.csv", index=False)
    tasks_df.to_csv(output_dir / "task_results.csv", index=False)
    profiles_df.to_csv(output_dir / "profile_summary.csv", index=False)

    config = {
        "model_path": str(MODEL_PATH),
        "training_config_path": str(CONFIG_PATH),
        "machine_count": MACHINE_COUNT,
        "machine_seed": MACHINE_SEED,
        "candidate_count": CANDIDATE_COUNT,
        "task_pool_size": TASK_POOL_SIZE,
        "task_pool_sample_seed": SAMPLE_SEED,
        "tasks_per_episode": TASK_COUNT,
        "arrival_seed": ARRIVAL_SEED,
        "scenario_seeds": SCENARIO_SEEDS,
        "profiles": PROFILE_NAMES,
        "policies": ["first_fit", "least_loaded", "maskable_ppo"],
        "notes": [
            "The model is the final trained candidate, not the pilot.",
            "Each policy receives identical task rows and arrival times within each profile and seed.",
            "Task demands and durations are dataset-derived.",
            "Arrival times are synthetic, not historical trace replay.",
            "This is a 10-machine stress test, not the intended 4,000-machine fleet evaluation.",
            "Only one scenario seed is used; results are exploratory.",
        ],
        "elapsed_seconds": round(time.perf_counter() - start, 3),
    }

    (output_dir / "evaluation_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print("\n=== CONTENTION BENCHMARK RESULTS ===")
    print(
        metrics_df[
            [
                "profile",
                "policy",
                "completed",
                "completion_rate",
                "mean_wait_time",
                "mean_flow_time",
                "makespan",
                "invalid_actions",
            ]
        ].to_string(index=False)
    )

    print(f"\nResults saved to: {output_dir}")
    print(f"Elapsed seconds: {time.perf_counter() - start:.2f}")


if __name__ == "__main__":
    main()
