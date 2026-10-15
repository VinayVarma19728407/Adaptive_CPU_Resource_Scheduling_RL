from experiments.train_final_dataset_maskable_ppo import (
    select_machine_pool,
    load_eligible_task_pool,
    DatasetRandomizedRLEnvironment,
)
from simulator.config import RANDOM_SEED


def main():
    machines = select_machine_pool(8, RANDOM_SEED)
    pool = load_eligible_task_pool(
        float(machines["cpu_num"].max()),
        float(machines["mem_size"].max()),
    )

    env = DatasetRandomizedRLEnvironment(
        task_pool=pool,
        machine_metadata=machines,
        tasks_per_episode=512,
        candidate_count=8,
        arrival_fraction=0.02,
        seed=RANDOM_SEED,
    )

    env.reset(seed=RANDOM_SEED)
    total_reward = 0.0
    steps = 0
    terminated = truncated = False

    while not terminated and not truncated:
        mask = env.action_masks()
        valid_actions = [i for i, valid in enumerate(mask) if valid]
        if not valid_actions:
            raise RuntimeError("No valid actions available.")

        observation, reward, terminated, truncated, info = env.step(
            valid_actions[0]
        )
        total_reward += float(reward)
        steps += 1

    sim = env.sim
    if sim is None:
        raise RuntimeError("Simulation was not initialized.")

    results = list(sim.results.values())
    completed = [r for r in results if r.get("status") == "completed"]
    waits = [float(r.get("wait_time", 0)) for r in completed]

    print("TRAINING-WORKLOAD CONTENTION CHECK")
    print("Steps:", steps)
    print("Completed:", len(completed))
    print("Total reward:", round(total_reward, 4))
    print("Mean wait:", round(sum(waits) / len(waits), 4) if waits else "N/A")
    print("Tasks with waiting:", sum(w > 0 for w in waits))
    print("Active allocations:", sim.machine_state.active_allocation_count)

    sim.machine_state.validate_invariants()


if __name__ == "__main__":
    main()