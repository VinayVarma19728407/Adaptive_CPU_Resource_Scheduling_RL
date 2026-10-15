import numpy as np
import pandas as pd

from experiments.train_final_dataset_maskable_ppo_v3 import DatasetRandomizedRLEnvironment

pool = pd.DataFrame({
    "task_name": [f"task{i}" for i in range(7)],
    "job_name": [f"job{i}" for i in range(7)],
    "plan_cpu": [100.0] * 7,
    "plan_mem": [1.0] * 7,
    "duration": [1.0] * 7,
})

env = DatasetRandomizedRLEnvironment.__new__(DatasetRandomizedRLEnvironment)
env.task_pool = pool
env.tasks_per_episode = 3
env.scenario_rng = np.random.default_rng(42)
env.episode_number = 0
env.arrival_fraction = 0.02

episodes = [env._make_scenario() for _ in range(3)]
names = [name for episode in episodes for name in episode["task_name"]]

assert [len(episode) for episode in episodes] == [3, 3, 1], [
    len(episode) for episode in episodes
]
assert len(names) == 7, len(names)
assert len(set(names)) == 7, names
assert set(names) == set(pool["task_name"]), names

print("PASS: episode sizes are 3, 3, 1")
print("PASS: all 7 records are covered exactly once")
print("PASS: no record repeats before the dataset pass ends")
env.arrival_fraction = 0.02
