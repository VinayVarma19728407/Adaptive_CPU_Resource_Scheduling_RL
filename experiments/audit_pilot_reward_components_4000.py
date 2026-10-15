from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO

from simulator.config import PROJECT_ROOT
from simulator.rl_environment import RLSchedulingEnvironment
from experiments.evaluate_4000_machine_stress_test import (
    MODEL_PATH,
    CANDIDATE_COUNT,
    create_machine_pool,
    create_task_pool,
    make_fixed_scenario,
)

RUN_NAME = "reward_component_audit_pilot_4000_1000_20261013"
OUTPUT_DIR = (
    PROJECT_ROOT / "experiments" / "evaluation" / RUN_NAME
)
MACHINE_COUNT = 4000
TASK_COUNT = 1000
MACHINE_SEED = 43
EVALUATION_SEED = 50001
ARRIVAL_FRACTION = 0.001

COMPONENTS = [
    "wait", "makespan", "invalid", "unschedulable",
    "imbalance", "fragmentation", "saturation", "starvation",
]


def main() -> None:
    start = time.perf_counter()

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing audit directory: {OUTPUT_DIR}"
        )
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Pilot model not found: {MODEL_PATH}")

    config_path = MODEL_PATH.parent / "training_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Pilot config not found: {config_path}")

    training_config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(training_config["machine_count"]) != MACHINE_COUNT:
        raise ValueError("Pilot configuration machine count is not 4000.")
    if int(training_config["candidate_count"]) != CANDIDATE_COUNT:
        raise ValueError("Pilot configuration candidate count mismatch.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    machines = create_machine_pool(MACHINE_SEED, MACHINE_COUNT)
    task_pool = create_task_pool(machines)
    scenario = make_fixed_scenario(
        task_pool=task_pool,
        tasks_per_episode=TASK_COUNT,
        arrival_fraction=ARRIVAL_FRACTION,
        seed=EVALUATION_SEED,
    )

    model = MaskablePPO.load(str(MODEL_PATH), device="cpu")
    env = RLSchedulingEnvironment(
        scenario=scenario.copy(),
        machine_metadata=machines.copy(),
        candidate_count=CANDIDATE_COUNT,
        reward_mode="extended_penalty",
    )

    observation, _ = env.reset(seed=EVALUATION_SEED)
    terminated = truncated = False
    step_rows = []
    episode_reward = 0.0
    invalid_actions = 0
    decisions = 0

    while not (terminated or truncated):
        masks = np.asarray(env.action_masks(), dtype=bool)
        if masks.shape != (CANDIDATE_COUNT,) or not masks.any():
            raise RuntimeError(
                f"Invalid action mask at decision {decisions + 1}: {masks}"
            )

        action, _ = model.predict(
            observation, deterministic=True, action_masks=masks
        )
        action_index = int(np.asarray(action).item())
        if (
            action_index < 0
            or action_index >= len(masks)
            or not masks[action_index]
        ):
            invalid_actions += 1

        observation, reward, terminated, truncated, info = env.step(
            action_index
        )
        decisions += 1
        episode_reward += float(reward)

        row = {
            "decision": decisions,
            "task_id": info.get("last_task_id"),
            "machine_id": info.get("last_machine_id"),
            "simulation_time": info.get("simulation_time"),
            "last_wait_time": info.get("last_wait_time"),
            "reward": float(reward),
            "candidate_count": len(info.get("candidate_machine_ids", [])),
            "waiting_tasks": info.get("waiting_tasks"),
            "running_tasks": info.get("running_tasks"),
        }
        penalties = info.get("penalty_components", {})
        for component in COMPONENTS:
            row[component] = float(penalties.get(component, 0.0))

        sim = env.sim
        if sim is None:
            raise RuntimeError("Simulator unexpectedly missing.")

        machine_states = list(sim.machine_state.machines.values())
        cpu_utils = [
            1.0 - m.cpu_available / m.cpu_capacity for m in machine_states
        ]
        mem_utils = [
            1.0 - m.memory_available / m.memory_capacity
            for m in machine_states
        ]
        row["fleet_mean_cpu_utilization"] = float(np.mean(cpu_utils))
        row["fleet_mean_memory_utilization"] = float(np.mean(mem_utils))
        row["peak_machine_cpu_utilization"] = float(max(cpu_utils, default=0))
        row["peak_machine_memory_utilization"] = float(max(mem_utils, default=0))
        row["active_allocations"] = sim.machine_state.active_allocation_count
        step_rows.append(row)

        if decisions > TASK_COUNT * 3:
            raise RuntimeError("Decision count exceeded safety limit.")

    if env.sim is None:
        raise RuntimeError("Simulator missing after episode.")

    env.sim.machine_state.validate_invariants()
    if env.sim.machine_state.active_allocation_count != 0:
        raise RuntimeError("Active allocations remain after episode.")

    steps = pd.DataFrame(step_rows)
    results = pd.DataFrame(env.sim.results.values())

    steps.to_csv(OUTPUT_DIR / "step_reward_components.csv", index=False)
    results.to_csv(OUTPUT_DIR / "task_results.csv", index=False)

    summary_rows = []
    for component in COMPONENTS:
        values = steps[component].astype(float)
        summary_rows.append({
            "component": component,
            "decision_count": len(values),
            "nonzero_decisions": int((values.abs() > 1e-12).sum()),
            "nonzero_rate": float((values.abs() > 1e-12).mean()),
            "mean_weighted_penalty": float(values.mean()),
            "median_weighted_penalty": float(values.median()),
            "total_weighted_penalty": float(values.sum()),
            "maximum_weighted_penalty": float(values.max()),
        })
    pd.DataFrame(summary_rows).to_csv(
        OUTPUT_DIR / "component_summary.csv", index=False
    )

    completed = int((results["status"] == "completed").sum())
    unschedulable = int((results["status"] == "unschedulable").sum())
    config = {
        "model_path": str(MODEL_PATH),
        "training_config": training_config,
        "machine_count": len(machines),
        "eligible_task_pool_size": len(task_pool),
        "tasks_requested": TASK_COUNT,
        "scenario_tasks": len(scenario),
        "machine_seed": MACHINE_SEED,
        "evaluation_seed": EVALUATION_SEED,
        "arrival_fraction": ARRIVAL_FRACTION,
        "reward_mode": "extended_penalty",
        "reward_weights": env.reward_weights,
        "duration_scale": env._duration_scale,
        "time_scale": env._time_scale,
        "decisions": decisions,
        "episode_reward": episode_reward,
        "completed_tasks": completed,
        "unschedulable_tasks": unschedulable,
        "completion_rate": completed / len(scenario),
        "invalid_actions": invalid_actions,
        "makespan": float(env.sim.current_time),
        "resource_invariants_passed": True,
        "active_allocations_at_end": env.sim.machine_state.active_allocation_count,
        "elapsed_seconds": time.perf_counter() - start,
        "note": (
            "Single deterministic PPO trajectory. Synthetic exponential "
            "arrivals; not a claim of historical arrival-time replay."
        ),
    }
    (OUTPUT_DIR / "audit_config_and_metrics.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    env.close()
    print("REWARD AUDIT COMPLETED")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Decisions: {decisions}")
    print(f"Completed: {completed}/{len(scenario)}")
    print(f"Unschedulable: {unschedulable}")
    print(f"Episode reward: {episode_reward:.6f}")
    print(f"Makespan: {config['makespan']:.6f}")
    print("Resource invariants: passed")
    print("Files: step_reward_components.csv, component_summary.csv,")
    print("       task_results.csv, audit_config_and_metrics.json")


if __name__ == "__main__":
    main()
