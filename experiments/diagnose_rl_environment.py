from __future__ import annotations

import numpy as np

from simulator.config import MAX_CANDIDATE_MACHINES
from simulator.environment import load_scenario
from simulator.rl_environment import RLSchedulingEnvironment
from simulator.task_data import load_machine_metadata


def main():
    scenario_path = load_scenario("heavy", seed=42)
    machine_metadata = load_machine_metadata()

    env = RLSchedulingEnvironment(
        scenario=scenario_path,
        machine_metadata=machine_metadata,
        candidate_count=MAX_CANDIDATE_MACHINES,
    )

    observation, info = env.reset(seed=42)

    rewards = []
    candidate_counts = []
    terminated = False
    truncated = False
    decisions = 0

    while not terminated and not truncated:
        # Capture the valid candidate count before taking the action.
        candidate_counts.append(len(env.candidates))

        # Deterministic diagnostic policy: always choose candidate zero.
        observation, reward, terminated, truncated, info = env.step(0)
        rewards.append(reward)
        decisions += 1

    sim = env.sim
    assert sim is not None

    results = list(sim.results.values())
    completed = [
        row for row in results if row["status"] == "completed"
    ]
    unschedulable = [
        row for row in results if row["status"] == "unschedulable"
    ]

    wait_times = [row["wait_time"] for row in completed]
    flow_times = [row["flow_time"] for row in completed]

    print("\nRL ENVIRONMENT DIAGNOSTIC")
    print("=" * 55)
    print(f"Scenario: {scenario_path.name}")
    print(f"Tasks in scenario: {len(sim.tasks):,}")
    print(f"Eligible machines: {sim.machine_state.machine_count:,}")
    print(f"Scheduling decisions: {decisions:,}")
    print(f"Episode terminated: {terminated}")
    print(f"Episode truncated: {truncated}")
    print(f"Tasks completed: {len(completed):,}")
    print(f"Tasks unschedulable: {len(unschedulable):,}")
    print(f"Completion rate: {len(completed) / len(sim.tasks):.4f}")
    print(f"Final simulation time: {sim.current_time:.4f}")
    print(f"Total reward: {sum(rewards):.4f}")
    print(f"Mean reward per decision: {np.mean(rewards):.6f}")

    if candidate_counts:
        print(f"Minimum candidates: {min(candidate_counts)}")
        print(f"Maximum candidates: {max(candidate_counts)}")
        print(
            "Decisions with fewer than 20 candidates: "
            f"{sum(n < MAX_CANDIDATE_MACHINES for n in candidate_counts):,}"
        )

    print(
        f"Mean wait time: {np.mean(wait_times):.4f}"
        if wait_times else "Mean wait time: N/A"
    )
    print(
        f"Mean flow time: {np.mean(flow_times):.4f}"
        if flow_times else "Mean flow time: N/A"
    )
    print(
        "Active allocations at end:",
        sim.machine_state.active_allocation_count,
    )

    sim.machine_state.validate_invariants()
    print("Resource invariants: PASS")


if __name__ == "__main__":
    main()