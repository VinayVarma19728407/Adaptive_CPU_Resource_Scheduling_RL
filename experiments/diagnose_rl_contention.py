import pandas as pd

from simulator.rl_environment import RLSchedulingEnvironment


def main():
    # One machine with limited capacity, plus several tasks arriving together.
    machines = pd.DataFrame([
        {
            "machine_id": "machine_0",
            "cpu_num": 10,
            "mem_size": 10,
            "status": 1,
        }
    ])

    # plan_cpu=1000 means 10 CPU cores under the current simulator contract.
    # Each task therefore occupies the entire machine until it completes.
    scenario = pd.DataFrame([
        {
            "scenario_task_id": f"task_{i}",
            "task_name": f"task_{i}",
            "job_name": "contention_test",
            "plan_cpu": 1000,
            "plan_mem": 10,
            "duration": 5,
            "arrival_time": 0,
        }
        for i in range(4)
    ])

    env = RLSchedulingEnvironment(
        scenario=scenario,
        machine_metadata=machines,
        candidate_count=1,
    )

    observation, info = env.reset(seed=42)

    rewards = []
    terminated = False
    truncated = False

    while not terminated and not truncated:
        observation, reward, terminated, truncated, info = env.step(0)
        rewards.append(reward)

    sim = env.sim
    assert sim is not None

    results = sim.results
    completed = [
        row for row in results.values()
        if row["status"] == "completed"
    ]

    print("\nRESOURCE-CONTENTION DIAGNOSTIC")
    print("=" * 45)
    print(f"Tasks: {len(scenario)}")
    print(f"Completed: {len(completed)}")
    print(f"Terminated: {terminated}")
    print(f"Rewards: {rewards}")
    print(f"Total reward: {sum(rewards):.4f}")
    print(f"Mean wait time: {sim.metrics['mean_wait_time'] if 'mean_wait_time' in sim.metrics else sum(r['wait_time'] for r in completed) / max(len(completed), 1):.4f}")
    print(f"Final simulation time: {sim.current_time}")

    for row in sorted(completed, key=lambda r: r["start_time"]):
        print(
            f"{row['scenario_task_id']}: "
            f"start={row['start_time']}, "
            f"complete={row['completion_time']}, "
            f"wait={row['wait_time']}"
        )

    print("Active allocations:", sim.machine_state.active_allocation_count)
    sim.machine_state.validate_invariants()
    print("Resource invariants: PASS")

    assert len(completed) == 4
    assert any(row["wait_time"] > 0 for row in completed)
    assert any(reward < 0 for reward in rewards)
    assert sim.machine_state.active_allocation_count == 0


if __name__ == "__main__":
    main()