import unittest
import pandas as pd

from simulator.environment import SchedulingEnvironment
from simulator.machine_state import MachineStateManager


def machines(cpu=10, memory=10, count=1):
    return pd.DataFrame([
        {"machine_id": f"m{i}", "cpu_num": cpu, "mem_size": memory, "status": 1}
        for i in range(count)
    ])


def scenario(rows):
    return pd.DataFrame(rows, columns=[
        "scenario_task_id", "task_name", "job_name", "plan_cpu", "plan_mem",
        "duration", "arrival_time"
    ])


class TestSchedulingEnvironment(unittest.TestCase):
    def make_env(self, rows, cpu=10, memory=10, count=1):
        return SchedulingEnvironment(
            scenario(rows), machine_state=MachineStateManager(machines(cpu, memory, count))
        )

    def test_single_task_completes_and_releases_resources(self):
        env = self.make_env([["t1", "task", "job", 4, 3, 5, 0]])
        out = env.run()
        self.assertEqual(out["metrics"]["tasks_completed"], 1)
        self.assertEqual(out["results"].iloc[0]["completion_time"], 5)
        self.assertEqual(env.machine_state.active_allocation_count, 0)
        self.assertEqual(env.machine_state.get_machine("m0").cpu_available, 10)

    def test_zero_duration_completes_without_loop(self):
        env = self.make_env([["t1", "task", "job", 4, 3, 0, 0]])
        out = env.run()
        self.assertEqual(out["metrics"]["tasks_completed"], 1)
        self.assertEqual(out["results"].iloc[0]["completion_time"], 0)
        self.assertEqual(env.machine_state.active_allocation_count, 0)

    def test_completion_before_same_time_arrival_scheduling(self):
        env = self.make_env([
            ["t1", "task1", "job1", 10, 10, 5, 0],
            ["t2", "task2", "job2", 10, 10, 2, 5],
        ])
        out = env.run()
        results = out["results"].set_index("scenario_task_id")
        self.assertEqual(results.loc["t1", "completion_time"], 5)
        self.assertEqual(results.loc["t2", "start_time"], 5)
        self.assertEqual(results.loc["t2", "completion_time"], 7)

    def test_waiting_task_starts_after_resources_release(self):
        env = self.make_env([
            ["t1", "task1", "job1", 7, 7, 4, 0],
            ["t2", "task2", "job2", 7, 7, 2, 1],
        ])
        out = env.run()
        results = out["results"].set_index("scenario_task_id")
        self.assertEqual(results.loc["t2", "start_time"], 4)
        self.assertEqual(results.loc["t2", "wait_time"], 3)

    def test_permanently_infeasible_task_is_marked(self):
        env = self.make_env([["t1", "task", "job", 11, 1, 3, 0]])
        out = env.run()
        self.assertEqual(out["metrics"]["tasks_unschedulable"], 1)
        self.assertEqual(out["results"].iloc[0]["status"], "unschedulable")

    def test_invalid_negative_duration_rejected(self):
        with self.assertRaises(ValueError):
            self.make_env([["t1", "task", "job", 1, 1, -1, 0]])

    def test_duplicate_task_ids_rejected(self):
        with self.assertRaises(ValueError):
            self.make_env([
                ["t1", "task1", "job1", 1, 1, 1, 0],
                ["t1", "task2", "job2", 1, 1, 1, 1],
            ])

    def test_all_tasks_resolved_and_no_leaked_allocations(self):
        env = self.make_env([
            ["t1", "task1", "job1", 4, 4, 2, 0],
            ["t2", "task2", "job2", 4, 4, 2, 0],
            ["t3", "task3", "job3", 4, 4, 1, 1],
        ])
        out = env.run()
        self.assertEqual(out["metrics"]["resolved_tasks"], 3)
        self.assertEqual(env.machine_state.active_allocation_count, 0)
        env.machine_state.validate_invariants()


if __name__ == "__main__":
    unittest.main(verbosity=2)