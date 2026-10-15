import unittest
import pandas as pd

from simulator.rl_environment import RLSchedulingEnvironment


class TestPlacementComparison(unittest.TestCase):
    def run_scenario(self, first_action):
        machines = pd.DataFrame([
            {"machine_id": "m0", "cpu_num": 10,
             "mem_size": 10, "status": 1},
            {"machine_id": "m1", "cpu_num": 20,
             "mem_size": 20, "status": 1},
        ])


        tasks = pd.DataFrame([
            {
                "scenario_task_id": "large",
                "job_name": "job_large",
                "task_name": "task_large",
                "arrival_time": 0.0,
                "plan_cpu": 8,
                "plan_mem": 8,
                "duration": 30.0,
            },
            {
                "scenario_task_id": "small",
                "job_name": "job_small",
                "task_name": "task_small",
                "arrival_time": 1.0,
                "plan_cpu": 13,
                "plan_mem": 13,
                "duration": 10.0,
            },
        ])


        env = RLSchedulingEnvironment(
            scenario=tasks,
            machine_metadata=machines,
            candidate_count=2,
            reward_mode="core_penalty",
        )

        try:
            env.reset(seed=42)
            total_reward = 0.0
            decisions = []
            first_decision = True
            terminated = False
            truncated = False

            while not (terminated or truncated):
                masks = env.action_masks()
                valid_actions = [
                    i for i, valid in enumerate(masks) if valid
                ]
                self.assertTrue(valid_actions)

                if first_decision:
                    self.assertIn(first_action, valid_actions)
                    action = first_action
                    first_decision = False
                else:
                    action = valid_actions[0]

                _, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                decisions.append({
                    "task_id": info["last_task_id"],
                    "machine_id": info["last_machine_id"],
                    "reward": reward,
                    "wait": info["last_wait_time"],
                })

            results = pd.DataFrame(env.sim.results.values())
            env.sim.machine_state.validate_invariants()

            self.assertEqual(len(results), 2)
            self.assertEqual(
                int((results["status"] == "completed").sum()), 2
            )
            self.assertEqual(
                env.sim.machine_state.active_allocation_count, 0
            )

            return {
                "results": results,
                "decisions": decisions,
                "total_reward": total_reward,
                "makespan": float(env.sim.current_time),
                "total_wait": float(results["wait_time"].sum()),
            }

        finally:
            env.close()

    def test_compare_first_placement_choices(self):
        run_a = self.run_scenario(0)
        run_b = self.run_scenario(1)

        print("\nSTAGGERED-ARRIVAL PLACEMENT COMPARISON")

        for label, run in [("RUN A", run_a), ("RUN B", run_b)]:
            print(f"\n{label}")
            print("First machine:", run["decisions"][0]["machine_id"])
            print("Total reward:", run["total_reward"])
            print("Makespan:", run["makespan"])
            print("Total wait:", run["total_wait"])
            print(
                run["results"][
                    ["scenario_task_id", "machine_id",
                     "arrival_time", "start_time",
                     "completion_time", "wait_time", "status"]
                ].to_string(index=False)
            )

        self.assertNotEqual(
            run_a["decisions"][0]["machine_id"],
            run_b["decisions"][0]["machine_id"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)