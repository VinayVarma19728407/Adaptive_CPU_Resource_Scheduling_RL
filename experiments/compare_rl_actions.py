import pandas as pd

from simulator.rl_environment import RLSchedulingEnvironment


def run_policy(first_action):
    machines = pd.DataFrame([
        {"machine_id": "A", "cpu_num": 10, "mem_size": 10, "status": 1},
        {"machine_id": "B", "cpu_num": 6, "mem_size": 10, "status": 1},
    ])

    scenario = pd.DataFrame([
        {
            "scenario_task_id": "task_1",
            "task_name": "task_1",
            "job_name": "job_1",
            "plan_cpu": 600,
            "plan_mem": 4,
            "duration": 10,
            "arrival_time": 0,
        },
        {
            "scenario_task_id": "task_2",
            "task_name": "task_2",
            "job_name": "job_2",
            "plan_cpu": 500,
            "plan_mem": 4,
            "duration": 1,
            "arrival_time": 0,
        },
        {
            "scenario_task_id": "task_3",
            "task_name": "task_3",
            "job_name": "job_3",
            "plan_cpu": 800,
            "plan_mem": 4,
            "duration": 1,
            "arrival_time": 0,
        },
    ])

    env = RLSchedulingEnvironment(
        scenario=scenario,
        machine_metadata=machines,
        candidate_count=2,
    )

    env.reset(seed=42)
    rewards = []
    decisions = 0
    terminated = False

    while not terminated:
        # Change only the first action; use candidate zero afterward.
        action = first_action if decisions == 0 else 0
        _, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        decisions += 1

    sim = env.sim
    assert sim is not None

    results = pd.DataFrame(sim.results.values()).sort_values(
        "scenario_task_id"
    )

    return {
        "first_action": first_action,
        "total_reward": sum(rewards),
        "mean_wait": results["wait_time"].mean(),
        "final_time": sim.current_time,
        "results": results,
    }


def main():
    for first_action in (0, 1):
        outcome = run_policy(first_action)

        print(f"\nFIRST ACTION = {outcome['first_action']}")
        print("-" * 45)
        print(f"Total reward: {outcome['total_reward']:.4f}")
        print(f"Mean wait: {outcome['mean_wait']:.4f}")
        print(f"Final simulation time: {outcome['final_time']:.4f}")
        print(
            outcome["results"][
                ["scenario_task_id", "machine_id", "start_time",
                 "completion_time", "wait_time", "status"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()