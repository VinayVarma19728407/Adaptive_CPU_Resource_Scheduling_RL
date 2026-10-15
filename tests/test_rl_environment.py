import unittest

import numpy as np
import pandas as pd
from gymnasium.utils.env_checker import check_env

from simulator.rl_environment import RLSchedulingEnvironment


class TestRLSchedulingEnvironment(unittest.TestCase):
    def setUp(self):
        self.machines = pd.DataFrame([
            {
                "machine_id": f"m{i}",
                "cpu_num": 10,
                "mem_size": 10,
                "status": 1,
            }
            for i in range(3)
        ])

        self.scenario = pd.DataFrame([
            {
                "scenario_task_id": "t1",
                "task_name": "task1",
                "job_name": "job1",
                "plan_cpu": 400,
                "plan_mem": 3,
                "duration": 2,
                "arrival_time": 0,
            },
            {
                "scenario_task_id": "t2",
                "task_name": "task2",
                "job_name": "job2",
                "plan_cpu": 300,
                "plan_mem": 2,
                "duration": 1,
                "arrival_time": 1,
            },
        ])

    def test_reset_observation_matches_space(self):
        env = RLSchedulingEnvironment(
            self.scenario, self.machines, candidate_count=3
        )
        observation, info = env.reset(seed=42)

        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(len(info["candidate_machine_ids"]), 3)

    def test_episode_completes_and_releases_resources(self):
        env = RLSchedulingEnvironment(
            self.scenario, self.machines, candidate_count=3
        )
        observation, _ = env.reset(seed=42)
        terminated = False

        while not terminated:
            observation, reward, terminated, truncated, info = env.step(0)
            self.assertFalse(truncated)
            self.assertTrue(env.observation_space.contains(observation))

        self.assertEqual(info["tasks_completed"], 2)
        self.assertEqual(env.sim.machine_state.active_allocation_count, 0)
        env.sim.machine_state.validate_invariants()

    def test_action_mask_marks_unavailable_candidate_slots_invalid(self):
        # Only one machine is feasible, while the action space has two slots.
        one_machine = self.machines.iloc[:1].copy()

        env = RLSchedulingEnvironment(
            self.scenario,
            one_machine,
            candidate_count=2,
        )
        observation, _ = env.reset(seed=42)

        mask = env.action_masks()

        self.assertEqual(mask, [True, False])
        self.assertTrue(env.observation_space.contains(observation))

    def test_action_mask_updates_after_each_decision(self):
        env = RLSchedulingEnvironment(
            self.scenario,
            self.machines,
            candidate_count=3,
        )
        observation, _ = env.reset(seed=42)

        mask = env.action_masks()
        self.assertEqual(mask, [True, True, True])

        observation, reward, terminated, truncated, info = env.step(0)

        self.assertTrue(env.observation_space.contains(observation))
        if not terminated:
            mask = env.action_masks()
            self.assertEqual(len(mask), env.action_space.n)
            self.assertTrue(any(mask))
    def test_feasible_task_bypasses_temporarily_blocked_queue_head(self):
        machines = pd.DataFrame([
            {
                "machine_id": "m0",
                "cpu_num": 10,
                "mem_size": 10,
                "status": 1,
            }
        ])

        scenario = pd.DataFrame([
            {
                "scenario_task_id": "t0",
                "task_name": "task0",
                "job_name": "job0",
                "plan_cpu": 600,
                "plan_mem": 6,
                "duration": 5,
                "arrival_time": 0,
            },
            {
                "scenario_task_id": "t1",
                "task_name": "task1",
                "job_name": "job1",
                "plan_cpu": 600,
                "plan_mem": 6,
                "duration": 2,
                "arrival_time": 0,
            },
            {
                "scenario_task_id": "t2",
                "task_name": "task2",
                "job_name": "job2",
                "plan_cpu": 400,
                "plan_mem": 4,
                "duration": 1,
                "arrival_time": 0,
            },
        ])

        env = RLSchedulingEnvironment(
            scenario,
            machines,
            candidate_count=1,
        )

        env.reset(seed=42)

        # Start t0. It occupies 6 of the 10 CPU units and memory units.
        _, _, terminated, truncated, _ = env.step(0)

        self.assertFalse(terminated)
        self.assertFalse(truncated)

        # t1 needs 6 units and cannot fit in the remaining 4.
        # t2 needs 4 units and should start at time 0.
        self.assertEqual(env.sim.current_time, 0)
        self.assertEqual(env.current_task_id, "t2")

        _, _, terminated, truncated, _ = env.step(0)

        self.assertFalse(truncated)

        # The environment may process t2's completion before step() returns.
        # Check the authoritative task result rather than current running state.
        result = env.sim.results["t2"]
        self.assertEqual(result["start_time"], 0)
        self.assertEqual(result["wait_time"], 0)

        env.close()
    
    def test_gymnasium_checker(self):
        env = RLSchedulingEnvironment(
            self.scenario, self.machines, candidate_count=3
        )
        check_env(env, skip_render_check=True)


if __name__ == "__main__":
    unittest.main()