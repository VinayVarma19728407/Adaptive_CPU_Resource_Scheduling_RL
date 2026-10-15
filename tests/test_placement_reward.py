import unittest
import pandas as pd

from simulator.rl_environment import RLSchedulingEnvironment


class TestPlacementReward(unittest.TestCase):
    def test_placement_choices_under_heterogeneous_contention(self):
        machines = pd.DataFrame([
            {
                "machine_id": "m0",
                "cpu_num": 10,
                "mem_size": 10,
                "status": 1,
            },
            {
                "machine_id": "m1",
                "cpu_num": 20,
                "mem_size": 20,
                "status": 1,
            },
        ])

        tasks = pd.DataFrame([
            {
                "scenario_task_id": "t0",
                "task_name": "large_task",
                "job_name": "job",
                "plan_cpu": 800,
                "plan_mem": 8,
                "duration": 20,
                "arrival_time": 0,
            },
            {
                "scenario_task_id": "t1",
                "task_name": "small_task_1",
                "job_name": "job",
                "plan_cpu": 300,
                "plan_mem": 3,
                "duration": 10,
                "arrival_time": 0,
            },
            {
                "scenario_task_id": "t2",
                "task_name": "small_task_2",
                "job_name": "job",
                "plan_cpu": 300,
                "plan_mem": 3,
                "duration": 10,
                "arrival_time": 0,
            },
        ])

        env = RLSchedulingEnvironment(
            scenario=tasks,
            machine_metadata=machines,
            candidate_count=2,
            reward_mode="core_penalty",
        )

        try:
            observation, info = env.reset(seed=42)

            # Record the initial decision state.
            self.assertEqual(len(env.candidates), 2)
            self.assertEqual(len(observation), 12)

            # Place the first task on candidate index 0.
            observation, reward, terminated, truncated, info = env.step(0)

            self.assertIn(
                info["last_machine_id"], ["m0", "m1"]
            )
            self.assertEqual(info["last_task_id"], "t0")

            # Continue using the first valid candidate.
            while not (terminated or truncated):
                masks = env.action_masks()
                self.assertTrue(any(masks))

                observation, reward, terminated, truncated, info = (
                    env.step(0)
                )

            results = pd.DataFrame(env.sim.results.values())

            self.assertEqual(len(results), 3)
            self.assertEqual(
                int((results["status"] == "completed").sum()), 3
            )
            self.assertEqual(
                int((results["status"] == "unschedulable").sum()), 0
            )

            # Verify final resource integrity.
            self.assertEqual(
                env.sim.machine_state.active_allocation_count, 0
            )
            env.sim.machine_state.validate_invariants()

            # Print observations for analysis; do not assume the
            # two placements must have different makespans.
            print("\nPLACEMENT REWARD DIAGNOSTIC")
            print("Task results:")
            print(
                results[
                    [
                        "scenario_task_id",
                        "start_time",
                        "completion_time",
                        "wait_time",
                        "machine_id",
                        "status",
                    ]
                ].to_string(index=False)
            )
            print("\nFinal simulation time:", env.sim.current_time)

        finally:
            env.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)