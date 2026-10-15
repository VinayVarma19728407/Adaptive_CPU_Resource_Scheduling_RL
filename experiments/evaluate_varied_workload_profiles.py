
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

OUTPUT_NAME = "varied_workload_profiles_final_candidate_fixed_20261014"

MACHINE_COUNT = 4000
CANDIDATE_COUNT = 10
TASK_COUNT = 5000

SAMPLE_SEED = 42
ARRIVAL_SEED = 50001
MACHINE_SEED = 43
SCENARIO_SEEDS = [50001]

PROFILE_NAMES = [
    "low_exponential",
    "moderate_exponential",
    "high_exponential",
    "bursty_100_per_unit",
]


def create_machine_pool(machine_count: int) -> pd.DataFrame:
    machines = load_machine_metadata()

    if machine_count > len(machines):
        raise ValueError(
            f"Requested {machine_count} machines, but only "
            f"{len(machines)} eligible machines are available."
        )

    machines = machines.sample(
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
    if len(arrivals) < TASK_COUNT:
        raise ValueError("Arrival profile is smaller than TASK_COUNT.")

    rng = np.random.default_rng(seed)
    selected_seed = int(rng.integers(0, 2**31 - 1))

    selected = task_pool.sample(
        n=TASK_COUNT,
        replace=False,
        random_state=selected_seed,
    ).reset_index(drop=True)

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
        "arrival_time": np.asarray(arrivals[:TASK_COUNT], dtype=float),
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
    total_tasks = len(scenario)

    waits = pd.to_numeric(
        completed["wait_time"], errors="coerce"
    ).dropna()

    flows = pd.to_numeric(
        completed["flow_time"], errors="coerce"
    ).dropna()

    manager.validate_invariants()

    active_allocations = int(manager.active_allocation_count)
    if active_allocations != 0:
        raise RuntimeError(
            f"{policy}/{profile}: active allocations remain."
        )

    return {
        "profile": profile,
        "seed": seed,
        "policy": policy,
        "tasks": total_tasks,
        "completed": completed_count,
        "unschedulable": int(
            (results["status"] == "unschedulable").sum()
        ),
        "completion_rate": (
            completed_count / total_tasks if total_tasks else 0.0
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
                f"PPO exceeded expected decisions for {profile}."
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

    trained_candidates = int(training_config["candidate_count"])
    if trained_candidates != CANDIDATE_COUNT:
        raise ValueError(
            f"Model uses {trained_candidates} candidates, expected "
            f"{CANDIDATE_COUNT}."
        )

    output_dir = PROJECT_ROOT / "experiments" / "results" / OUTPUT_NAME

    # Refuse to overwrite a prior experiment.
    output_dir.mkdir(parents=True, exist_ok=False)

    print("LOADING ELIGIBLE MACHINES AND TASK SAMPLE")

    machines = create_machine_pool(MACHINE_COUNT)

    task_pool = load_bounded_eligible_task_pool(
        max_cpu_capacity=float(machines["cpu_num"].max()),
        max_memory_capacity=float(machines["mem_size"].max()),
        sample_size=50_000,
        seed=SAMPLE_SEED,
    )

    if len(task_pool) < TASK_COUNT:
        raise ValueError("Insufficient eligible tasks for evaluation.")

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

    # Generate all four profiles using the same calibrated generator.
    all_profiles = build_profiles(durations)

    model = MaskablePPO.load(str(MODEL_PATH), device="cpu")

    records = []
    task_outputs = []
    profile_outputs = []

    print(f"Model: {MODEL_PATH}")
    print(f"Machines: {len(machines):,}")
    print(f"Task pool: {len(task_pool):,}")
    print(f"Tasks per profile/episode: {TASK_COUNT:,}")
    print(f"Scenario seeds: {SCENARIO_SEEDS}")
    print("Policies: first_fit, least_loaded, maskable_ppo")

    for profile in PROFILE_NAMES:
        full_arrivals = all_profiles[profile]

        if len(full_arrivals) < TASK_COUNT:
            raise ValueError(f"Profile {profile} has too few arrivals.")

        # Select one reproducible subset of arrival timestamps for the
        # chosen tasks-per-episode setting. Reuse it for all policies.
        profile_rng = np.random.default_rng(ARRIVAL_SEED)
        selected_arrivals = np.asarray(
            full_arrivals[:TASK_COUNT], dtype=float
        )
        selected_arrivals = selected_arrivals - selected_arrivals.min()

        profile_outputs.append({
            "profile": profile,
            "calibrated_task_count": len(full_arrivals),
            "evaluated_task_count": TASK_COUNT,
            "arrival_window": float(
                selected_arrivals.max() - selected_arrivals.min()
            ),
            "mean_interarrival_gap": float(
                np.diff(selected_arrivals).mean()
            ) if len(selected_arrivals) > 1 else 0.0,
        })

        for seed in SCENARIO_SEEDS:
            scenario = create_scenario(
                task_pool=task_pool,
                arrivals=selected_arrivals,
                seed=seed,
                profile=profile,
            )

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
    profiles_df = pd.DataFrame(profile_outputs)

    metrics_df.to_csv(output_dir / "episode_metrics.csv", index=False)
    tasks_df.to_csv(output_dir / "task_results.csv", index=False)
    profiles_df.to_csv(output_dir / "profile_summary.csv", index=False)

    config = {
        "model_path": str(MODEL_PATH),
        "training_config_path": str(CONFIG_PATH),
        "machine_count": MACHINE_COUNT,
        "machine_seed": MACHINE_SEED,
        "candidate_count": CANDIDATE_COUNT,
        "task_pool_sample_size": 50_000,
        "task_pool_sample_seed": SAMPLE_SEED,
        "tasks_per_episode": TASK_COUNT,
        "arrival_seed": ARRIVAL_SEED,
        "scenario_seeds": SCENARIO_SEEDS,
        "profiles": PROFILE_NAMES,
        "policies": ["first_fit", "least_loaded", "maskable_ppo"],
        "notes": [
            "Task resource requests and durations are dataset-derived.",
            "Arrival times are synthetic; they are not historical trace replay.",
            "Within each profile and seed, every policy receives identical tasks and arrival times.",
            "PPO uses deterministic prediction and action masks.",
            "This benchmark uses 4,000 machines; smaller-fleet contention tests are separate.",
            "A single seed per profile is an exploratory benchmark, not statistical proof of superiority.",
        ],
        "elapsed_seconds": round(time.perf_counter() - start, 3),
    }

    (output_dir / "evaluation_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print("\n=== VARIED-WORKLOAD RESULTS ===")
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
    print(f"\nOutputs saved to: {output_dir}")
    print(f"Elapsed seconds: {time.perf_counter() - start:.2f}")


if __name__ == "__main__":
    main()
