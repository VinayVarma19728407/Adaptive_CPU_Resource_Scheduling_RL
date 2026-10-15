
from __future__ import annotations

import sys
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from simulator.environment import SchedulingEnvironment
from simulator.machine_state import MachineStateManager
from simulator.rl_environment import RLSchedulingEnvironment
from simulator.task_data import load_machine_metadata


MODEL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "models"
    / "final_dataset_ppo_runs"
    / "final_candidate_4000_machines_10_candidates_50000_20261014"
    / "checkpoints"
    / "final_dataset_ppo_20480_steps.zip"
)

CANDIDATE_COUNT = 10


def ask_float(prompt, minimum=None, allow_zero=False):
    while True:
        try:
            value = float(input(prompt))
            if not np.isfinite(value):
                raise ValueError

            if allow_zero and value == 0:
                return value

            if minimum is not None and value < minimum:
                print(f"Enter a value >= {minimum}.")
                continue

            if not allow_zero and value <= 0:
                print("Enter a positive value.")
                continue

            return value
        except ValueError:
            print("Enter a valid finite number.")


def ask_int(prompt, minimum=1):
    while True:
        try:
            value = int(input(prompt))
            if value < minimum:
                print(f"Enter an integer >= {minimum}.")
                continue
            return value
        except ValueError:
            print("Enter a valid integer.")


def create_tasks():
    print("\nTASK INPUT")
    print("CPU is entered in cores; memory uses the dataset's memory units.")
    print("Arrival time and duration use the simulator's time units.")

    count = ask_int("Number of tasks: ", 1)
    tasks = []

    for i in range(count):
        print(f"\n--- Task {i + 1} of {count} ---")

        name = input(f"Task name [task_{i + 1}]: ").strip()
        if not name:
            name = f"task_{i + 1}"

        job = input(f"Job name [job_{i + 1}]: ").strip()
        if not job:
            job = f"job_{i + 1}"

        cpu_cores = ask_float("CPU demand in cores: ", minimum=0)
        memory = ask_float("Memory demand: ", minimum=0)
        duration = ask_float(
            "Duration: ", minimum=0, allow_zero=True
        )
        arrival = ask_float(
            "Arrival time: ", minimum=0, allow_zero=True
        )

        tasks.append({
            "scenario_task_id": f"manual_{i + 1:04d}",
            "task_name": name,
            "job_name": job,
            "plan_cpu": cpu_cores * 100.0,
            "plan_mem": memory,
            "duration": duration,
            "arrival_time": arrival,
        })

    return pd.DataFrame(tasks)


def create_machines():
    print("\nMACHINE CONFIGURATION")
    print("1. Use real eligible dataset machines")
    print("2. Use identical custom machines")

    while True:
        choice = input("Choose 1 or 2: ").strip()
        if choice in {"1", "2"}:
            break
        print("Enter 1 or 2.")

    if choice == "1":
        available = load_machine_metadata()
        count = ask_int(
            f"Number of machines (1-{len(available)}): ", 1
        )

        if count > len(available):
            raise ValueError("Requested more machines than available.")

        seed = ask_int("Machine sampling seed [50001]: ", 0) \
            if input("Change sampling seed? (y/N): ").strip().lower() == "y" \
            else 50001

        machines = available.sample(
            n=count, random_state=seed
        ).copy()

        machines = machines.sort_values(
            "machine_id"
        ).reset_index(drop=True)

        return machines

    count = ask_int("Number of machines: ")
    cpu = ask_float("CPU capacity per machine (cores): ")
    memory = ask_float("Memory capacity per machine: ")

    return pd.DataFrame({
        "machine_id": [f"manual_machine_{i + 1:03d}" for i in range(count)],
        "cpu_num": [cpu] * count,
        "mem_size": [memory] * count,
        "status": [1] * count,
    })


def run_baseline(scenario, machines, policy):
    manager = MachineStateManager(machines.copy())

    env = SchedulingEnvironment(
        scenario=scenario.copy(),
        machine_state=manager,
        placement_policy=policy,
    )

    outcome = env.run()

    if manager.active_allocation_count != 0:
        raise RuntimeError(f"{policy}: resource allocations remain active.")

    return outcome["results"].copy(), float(outcome["final_time"])


def run_ppo(scenario, machines, model):
    env = RLSchedulingEnvironment(
        scenario=scenario.copy(),
        machine_metadata=machines.copy(),
        candidate_count=CANDIDATE_COUNT,
        reward_mode="extended_penalty",
    )

    observation, _ = env.reset(seed=42)
    terminated = False
    truncated = False
    invalid_actions = 0

    try:
        while not (terminated or truncated):
            masks = env.action_masks()

            if not np.asarray(masks, dtype=bool).any():
                raise RuntimeError("PPO has no valid action.")

            action, _ = model.predict(
                observation,
                deterministic=True,
                action_masks=masks,
            )

            observation, _, terminated, truncated, info = env.step(
                int(action)
            )

            invalid_actions += int(
                info.get("penalty_components", {}).get("invalid", 0.0) > 0
            )

        if env.sim is None:
            raise RuntimeError("PPO simulator was not initialized.")

        results = pd.DataFrame(env.sim.results.values())
        final_time = float(env.sim.current_time)

        results["invalid_action_count"] = 0
        if len(results):
            results.loc[results.index[0], "invalid_action_count"] = (
                invalid_actions
            )

        if env.sim.machine_state.active_allocation_count != 0:
            raise RuntimeError("PPO ended with active allocations.")

        return results, final_time

    finally:
        env.close()


def summarize(policy, scenario, results, final_time):
    completed = results[results["status"] == "completed"]
    count = len(scenario)
    completed_count = len(completed)

    return {
        "policy": policy,
        "tasks": count,
        "completed": completed_count,
        "unschedulable": int(
            (results["status"] == "unschedulable").sum()
        ),
        "completion_rate": completed_count / count if count else 0.0,
        "mean_wait_time": (
            float(completed["wait_time"].mean())
            if completed_count else 0.0
        ),
        "total_wait_time": (
            float(completed["wait_time"].sum())
            if completed_count else 0.0
        ),
        "mean_flow_time": (
            float(completed["flow_time"].mean())
            if completed_count else 0.0
        ),
        "makespan": final_time,
        "invalid_actions": int(
            results["invalid_action_count"].sum()
        ) if "invalid_action_count" in results.columns else 0,
    }


def main():
    print("=" * 72)
    print("INTERACTIVE CPU SCHEDULER COMPARISON")
    print("PPO vs First Fit vs Least Loaded")
    print("=" * 72)

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"PPO checkpoint not found: {MODEL_PATH}")

    print(f"\nLoading checkpoint:\n{MODEL_PATH}")
    model = MaskablePPO.load(str(MODEL_PATH), device="cpu")

    scenario = create_tasks()
    machines = create_machines()

    # Validate the same scenario and machine metadata before evaluation.
    # Both environments use plan_cpu / 100 as CPU demand.
    print("\nVALIDATING WORKLOAD")
    print(f"Tasks: {len(scenario)}")
    print(f"Machines: {len(machines)}")

    print("\nMachine capacities:")
    print(
        machines[["machine_id", "cpu_num", "mem_size", "status"]]
        .to_string(index=False)
    )

    print("\nTasks:")
    task_view = scenario.copy()
    task_view["cpu_cores"] = task_view["plan_cpu"] / 100.0
    print(
        task_view[
            ["scenario_task_id", "task_name", "cpu_cores",
             "plan_mem", "duration", "arrival_time"]
        ].to_string(index=False)
    )

    print("\nRunning First Fit...")
    ff_results, ff_time = run_baseline(
        scenario, machines, "first_fit"
    )

    print("Running Least Loaded...")
    ll_results, ll_time = run_baseline(
        scenario, machines, "least_loaded"
    )

    print("Running PPO...")
    ppo_results, ppo_time = run_ppo(
        scenario, machines, model
    )

    outcomes = {
        "First Fit": (ff_results, ff_time),
        "Least Loaded": (ll_results, ll_time),
        "PPO 20480": (ppo_results, ppo_time),
    }

    summaries = []
    task_outputs = []

    for policy, (results, final_time) in outcomes.items():
        summaries.append(
            summarize(policy, scenario, results, final_time)
        )

        results = results.copy()
        results.insert(0, "policy", policy)
        task_outputs.append(results)

    summary_df = pd.DataFrame(summaries)

    print("\n" + "=" * 90)
    print("SUMMARY COMPARISON")
    print("=" * 90)
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    combined = pd.concat(task_outputs, ignore_index=True)

    print("\n" + "=" * 90)
    print("PER-TASK RESULTS")
    print("=" * 90)

    columns = [
        "policy", "scenario_task_id", "task_name", "status",
        "machine_id", "arrival_time", "start_time", "completion_time",
        "wait_time", "flow_time", "reason",
    ]
    available_columns = [c for c in columns if c in combined.columns]
    print(combined[available_columns].to_string(index=False))

    # Create a unique output directory; never overwrite a previous run.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        PROJECT_ROOT / "experiments" / "results"
        / f"interactive_workload_comparison_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    scenario.to_csv(output_dir / "manual_workload.csv", index=False)
    machines.to_csv(output_dir / "machine_configuration.csv", index=False)
    summary_df.to_csv(output_dir / "policy_summary.csv", index=False)
    combined.to_csv(output_dir / "per_task_results.csv", index=False)

    config = {
        "checkpoint": str(MODEL_PATH),
        "candidate_count": CANDIDATE_COUNT,
        "task_count": len(scenario),
        "machine_count": len(machines),
        "cpu_input_unit": "cores; stored as plan_cpu = cores * 100",
        "memory_input_unit": "same units as machine mem_size",
        "duration_and_arrival_units": "simulator time units",
        "policies": list(outcomes.keys()),
        "timestamp": timestamp,
    }

    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print(f"\nResults saved to:\n{output_dir}")
    print("\nInterpretation:")
    print("- Lower mean waiting time is better.")
    print("- Lower mean flow time is better.")
    print("- Lower makespan is better only when the workload is comparable.")
    print("- A higher completion rate is better.")
    print("- One manual workload is a demonstration, not statistical proof.")


if __name__ == "__main__":
    main()