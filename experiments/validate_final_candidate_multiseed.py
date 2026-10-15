
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


# ------------------------------------------------------------
# REPRODUCIBLE MULTI-SEED VALIDATION
# ------------------------------------------------------------

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

OUTPUT_NAME = "final_candidate_multiseed_validation_20261014"

MACHINE_COUNT = 10
CANDIDATE_COUNT = 10
TASK_POOL_SIZE = 50_000
TASK_COUNT = 5_000

TASK_POOL_SEED = 42
MACHINE_SEED = 50001

SCENARIO_SEEDS = [11001, 22002, 33003, 44004, 55005]

PROFILE_NAMES = [
    "high_exponential",
    "bursty_100_per_unit",
]

POLICIES = ["first_fit", "least_loaded", "maskable_ppo"]


def create_machine_pool() -> pd.DataFrame:
    machines = load_machine_metadata()

    if MACHINE_COUNT > len(machines):
        raise ValueError(
            f"Requested {MACHINE_COUNT} machines, "
            f"but only {len(machines)} are available."
        )

    machines = machines.sample(
        n=MACHINE_COUNT,
        random_state=MACHINE_SEED,
    ).copy()

    machines = machines.sort_values(
        "machine_id"
    ).reset_index(drop=True)

    if machines["machine_id"].duplicated().any():
        raise ValueError("Duplicate machine IDs detected.")

    return machines


def build_arrivals(
    profile: str,
    seed: int,
    count: int,
) -> np.ndarray:
    """Generate reproducible, seed-specific synthetic arrival times."""
    rng = np.random.default_rng(seed)

    if profile == "high_exponential":
        gaps = rng.exponential(scale=0.01, size=count)
        arrivals = np.cumsum(gaps)
        arrivals -= arrivals[0]

    elif profile == "bursty_100_per_unit":
        burst_size = 100
        burst_number = np.arange(count) // burst_size
        within_burst = rng.uniform(0.0, 0.05, size=count)
        arrivals = burst_number.astype(float) + within_burst
        arrivals = np.sort(arrivals)
        arrivals -= arrivals.min()

    else:
        raise ValueError(f"Unknown profile: {profile}")

    if (
        not np.isfinite(arrivals).all()
        or (arrivals < 0).any()
        or np.any(np.diff(arrivals) < 0)
    ):
        raise ValueError(f"Invalid arrivals generated for {profile}/{seed}.")

    return arrivals


def create_scenario(
    task_pool: pd.DataFrame,
    profile: str,
    seed: int,
) -> pd.DataFrame:
    if TASK_COUNT > len(task_pool):
        raise ValueError("Task count exceeds task pool size.")

    rng = np.random.default_rng(seed)
    task_sample_seed = int(rng.integers(0, 2**31 - 1))

    selected = task_pool.sample(
        n=TASK_COUNT,
        replace=False,
        random_state=task_sample_seed,
    ).reset_index(drop=True)

    arrivals = build_arrivals(profile, seed, TASK_COUNT)

    scenario = pd.DataFrame({
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
        "arrival_time": arrivals,
    })

    if scenario["scenario_task_id"].duplicated().any():
        raise ValueError("Duplicate scenario task IDs.")

    return scenario


def summarize(
    profile: str,
    seed: int,
    policy: str,
    scenario: pd.DataFrame,
    results: pd.DataFrame,
    final_time: float,
) -> dict:
    completed = results[results["status"] == "completed"]
    completed_count = len(completed)
    task_count = len(scenario)

    if completed_count:
        mean_wait = float(completed["wait_time"].mean())
        total_wait = float(completed["wait_time"].sum())
        mean_flow = float(completed["flow_time"].mean())
    else:
        mean_wait = 0.0
        total_wait = 0.0
        mean_flow = 0.0

    invalid_actions = (
        int(results["invalid_action_count"].sum())
        if "invalid_action_count" in results.columns
        else 0
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
        "completion_rate": completed_count / task_count,
        "mean_wait_time": mean_wait,
        "total_wait_time": total_wait,
        "mean_flow_time": mean_flow,
        "makespan": float(final_time),
        "invalid_actions": invalid_actions,
    }


def run_baseline(
    scenario: pd.DataFrame,
    machines: pd.DataFrame,
    policy: str,
) -> tuple[pd.DataFrame, float]:
    manager = MachineStateManager(machines.copy())

    env = SchedulingEnvironment(
        scenario=scenario.copy(),
        machine_state=manager,
        placement_policy=policy,
    )

    outcome = env.run()

    if manager.active_allocation_count != 0:
        raise RuntimeError(
            f"{policy}: active allocations remain after episode."
        )

    return outcome["results"].copy(), float(outcome["final_time"])


def run_ppo(
    scenario: pd.DataFrame,
    machines: pd.DataFrame,
    model: MaskablePPO,
    seed: int,
) -> tuple[pd.DataFrame, float]:
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

    try:
        while not (terminated or truncated):
            action_masks = env.action_masks()

            if not np.asarray(action_masks, dtype=bool).any():
                raise RuntimeError(
                    f"No valid PPO actions for seed {seed}."
                )

            action, _ = model.predict(
                observation,
                deterministic=True,
                action_masks=action_masks,
            )

            observation, _, terminated, truncated, info = env.step(
                int(action)
            )

            invalid_actions += int(info.get("penalty_components", {}).get("invalid", 0.0) > 0)

        if env.sim is None:
            raise RuntimeError("PPO simulator was not initialized.")

        results = pd.DataFrame(env.sim.results.values())
        final_time = float(env.sim.current_time)

        results["invalid_action_count"] = 0
        if len(results):
            results.loc[results.index[0], "invalid_action_count"] = invalid_actions

        if env.sim.machine_state.active_allocation_count != 0:
            raise RuntimeError("PPO episode ended with active allocations.")

        return results, final_time

    finally:
        env.close()


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
        raise ValueError("Candidate count differs from trained model config.")

    output_dir = PROJECT_ROOT / "experiments" / "results" / OUTPUT_NAME

    # Never overwrite an existing experiment.
    output_dir.mkdir(parents=True, exist_ok=False)

    print("LOADING PPO MODEL AND VALIDATION DATA")
    print(f"Model: {MODEL_PATH}")

    machines = create_machine_pool()

    task_pool = load_bounded_eligible_task_pool(
        max_cpu_capacity=float(machines["cpu_num"].max()),
        max_memory_capacity=float(machines["mem_size"].max()),
        sample_size=TASK_POOL_SIZE,
        seed=TASK_POOL_SEED,
    )

    if len(task_pool) < TASK_COUNT:
        raise ValueError("Insufficient eligible task records.")

    model = MaskablePPO.load(str(MODEL_PATH), device="cpu")

    metrics = []
    task_outputs = []
    scenario_summaries = []

    print(f"Machines: {len(machines)}")
    print(f"Task pool: {len(task_pool):,}")
    print(f"Tasks per scenario: {TASK_COUNT:,}")
    print(f"Scenario seeds: {SCENARIO_SEEDS}")
    print(f"Profiles: {PROFILE_NAMES}")
    print(f"Policies: {POLICIES}")

    for profile in PROFILE_NAMES:
        for seed in SCENARIO_SEEDS:
            scenario = create_scenario(task_pool, profile, seed)

            scenario_summaries.append({
                "profile": profile,
                "seed": seed,
                "machine_count": len(machines),
                "task_count": len(scenario),
                "arrival_window": float(
                    scenario["arrival_time"].max()
                    - scenario["arrival_time"].min()
                ),
                "mean_interarrival_gap": float(
                    scenario["arrival_time"].diff().dropna().mean()
                ),
                "task_sample_fingerprint": str(
                    pd.util.hash_pandas_object(
                        scenario[
                            ["task_name", "job_name", "plan_cpu",
                             "plan_mem", "duration"]
                        ],
                        index=False,
                    ).sum()
                ),
            })

            print(f"\nProfile={profile} | seed={seed}")

            for policy in ("first_fit", "least_loaded"):
                results, final_time = run_baseline(
                    scenario, machines, policy
                )

                record = summarize(
                    profile, seed, policy,
                    scenario, results, final_time,
                )
                metrics.append(record)

                results.insert(0, "policy", policy)
                results.insert(0, "evaluation_seed", seed)
                results.insert(0, "profile", profile)
                task_outputs.append(results)

                print(
                    f"{policy}: completed={record['completed']}, "
                    f"mean_wait={record['mean_wait_time']:.4f}, "
                    f"makespan={record['makespan']:.4f}"
                )

            results, final_time = run_ppo(
                scenario, machines, model, seed
            )

            record = summarize(
                profile, seed, "maskable_ppo",
                scenario, results, final_time,
            )
            metrics.append(record)

            results.insert(0, "policy", "maskable_ppo")
            results.insert(0, "evaluation_seed", seed)
            results.insert(0, "profile", profile)
            task_outputs.append(results)

            print(
                f"maskable_ppo: completed={record['completed']}, "
                f"mean_wait={record['mean_wait_time']:.4f}, "
                f"makespan={record['makespan']:.4f}, "
                f"invalid_actions={record['invalid_actions']}"
            )

    metrics_df = pd.DataFrame(metrics)
    tasks_df = pd.concat(task_outputs, ignore_index=True)
    scenarios_df = pd.DataFrame(scenario_summaries)

    metrics_df.to_csv(output_dir / "episode_metrics.csv", index=False)
    tasks_df.to_csv(output_dir / "task_results.csv", index=False)
    scenarios_df.to_csv(output_dir / "scenario_summary.csv", index=False)

    config = {
        "model_path": str(MODEL_PATH),
        "machine_count": MACHINE_COUNT,
        "machine_seed": MACHINE_SEED,
        "task_pool_size": len(task_pool),
        "task_pool_seed": TASK_POOL_SEED,
        "tasks_per_scenario": TASK_COUNT,
        "scenario_seeds": SCENARIO_SEEDS,
        "profiles": PROFILE_NAMES,
        "policies": POLICIES,
        "candidate_count": CANDIDATE_COUNT,
        "reward_mode": "extended_penalty",
        "arrival_generation": {
            "high_exponential": "seed-specific exponential gaps, scale=0.01",
            "bursty_100_per_unit": (
                "100 arrivals per time unit with seed-specific "
                "uniform offsets in [0, 0.05)"
            ),
        },
        "notes": [
            "Synthetic arrivals are not historical trace replay.",
            "All policies receive the same scenario for each profile and seed.",
            "The trained PPO model is evaluated without further training.",
            "The machine pool is fixed across scenarios.",
        ],
        "elapsed_seconds": round(time.perf_counter() - start, 3),
    }

    (output_dir / "evaluation_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print("\n=== MULTI-SEED VALIDATION SUMMARY ===")
    print(
        metrics_df[
            [
                "profile", "seed", "policy", "completed",
                "completion_rate", "mean_wait_time",
                "total_wait_time", "makespan", "invalid_actions",
            ]
        ].to_string(index=False)
    )

    print(f"\nResults saved to: {output_dir}")
    print(f"Elapsed seconds: {time.perf_counter() - start:.2f}")


if __name__ == "__main__":
    main()