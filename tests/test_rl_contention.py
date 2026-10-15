import unittest
import pandas as pd

from simulator.rl_environment import RLSchedulingEnvironment


class TestRLContention(unittest.TestCase):
    def test_core_reward_under_resource_contention(self):
        machines = pd.DataFrame([
            {
                "machine_id": "m0",
                "cpu_num": 10,
                "mem_size": 10,
                "status": 1,
            }
        ])

        # Each task requests 6 CPU cores and 6 memory units.
        # Only one task can run at a time.
        tasks = pd.DataFrame([
            {
                "scenario_task_id": f"t{i}",
                "task_name": f"task{i}",
                "job_name": "job",
                "plan_cpu": 600,
                "plan_mem": 6,
                "duration": 10,
                "arrival_time": 0,
            }
            for i in range(3)
        ])

        env = RLSchedulingEnvironment(
            scenario=tasks,
            machine_metadata=machines,
            candidate_count=1,
            reward_mode="core_penalty",
        )

        env.reset(seed=42)

        rewards = []
        penalty_history = []
        waiting_history = []

        try:
            while True:
                masks = env.action_masks()
                self.assertTrue(any(masks))

                _, reward, terminated, truncated, info = env.step(0)

                rewards.append(reward)
                penalty_history.append(info["penalty_components"])
                waiting_history.append(info["last_wait_time"])

                if terminated or truncated:
                    break

            # Verify task completion and simulator timing.
            self.assertEqual(len(rewards), 3)
            self.assertEqual(env.sim.metrics["tasks_completed"], 3)
            self.assertEqual(env.sim.metrics["tasks_unschedulable"], 0)

            self.assertAlmostEqual(
                env.sim.metrics["total_wait_time"], 30.0
            )
            self.assertAlmostEqual(env.sim.current_time, 30.0)

            # Tasks wait 0, 10, and 20 time units.
            self.assertEqual(waiting_history, [0.0, 10.0, 20.0])

            # Waiting penalties are normalized by median duration (10).
            self.assertAlmostEqual(penalty_history[0]["wait"], 0.0)
            self.assertAlmostEqual(penalty_history[1]["wait"], 1.0)
            self.assertAlmostEqual(penalty_history[2]["wait"], 2.0)

            # Makespan penalty applies only on the terminal decision.
            self.assertAlmostEqual(
                penalty_history[0]["makespan"], 0.0
            )
            self.assertAlmostEqual(
                penalty_history[1]["makespan"], 0.0
            )
            self.assertAlmostEqual(
                penalty_history[2]["makespan"], 1.5
            )

            # Expected rewards: 0, -1, and -(2 + 0.5 * 1.5).
            self.assertAlmostEqual(rewards[0], 0.0)
            self.assertAlmostEqual(rewards[1], -1.0)
            self.assertAlmostEqual(rewards[2], -3.5)
            self.assertAlmostEqual(sum(rewards), -4.5)

            # Confirm all allocations have been released.
            self.assertEqual(
                env.sim.machine_state.active_allocation_count, 0
            )
            env.sim.machine_state.validate_invariants()

        finally:
            env.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)