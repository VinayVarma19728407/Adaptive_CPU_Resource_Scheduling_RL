"""
============================================================
File: test_contention.py
Module: tests (Simulator Integration Tests)
============================================================
Unit/integration test for resource contention behaviour in the
core (non-RL) SchedulingEnvironment simulator. The tests verify that
when multiple tasks arrive simultaneously and compete for a single
machine's CPU and memory, the scheduler correctly queues excess tasks
and runs them sequentially once resources become available.

The test scenario uses a single machine with 10 CPU units and 10 memory
units, and submits three tasks that each require 6 CPU cores and
6 memory units — meaning only one task can run at any given time.
Expected behaviour: tasks start at times 0, 10, and 20, with wait times
of 0, 10, and 20 respectively, and the simulation completes at time 30.

Key Components:
  - TestSchedulingContention (unittest.TestCase): test class containing
    one parameterized contention scenario.
    - test_simultaneous_tasks_wait_for_cpu_and_memory: verifies correct
      sequential scheduling under contention, including start times,
      wait times, total wait, mean wait, final simulation time, and
      that all resource allocations are released cleanly.

Inputs:
  - Inline pandas DataFrames defining a single machine and three tasks
    with identical resource demands and arrival times.
  - simulator.environment.SchedulingEnvironment
  - simulator.machine_state.MachineStateManager

Outputs:
  - unittest pass/fail assertions; no files written.
============================================================
"""
import unittest
import pandas as pd

from simulator.environment import SchedulingEnvironment
from simulator.machine_state import MachineStateManager


class TestSchedulingContention(unittest.TestCase):
    def test_simultaneous_tasks_wait_for_cpu_and_memory(self):
        machines = pd.DataFrame([
            {
                "machine_id": "m0",
                "cpu_num": 10,
                "mem_size": 10,
                "status": 1,
            }
        ])

        # plan_cpu is divided by 100 by the simulator.
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

        manager = MachineStateManager(machines)
        env = SchedulingEnvironment(tasks, machine_state=manager)
        outcome = env.run()

        results = outcome["results"].set_index("scenario_task_id")

        self.assertEqual(outcome["metrics"]["tasks_completed"], 3)
        self.assertEqual(outcome["metrics"]["tasks_unschedulable"], 0)

        # First task starts immediately; subsequent tasks wait.
        self.assertEqual(results["start_time"].min(), 0)
        self.assertEqual(sorted(results["start_time"].tolist()), [0, 10, 20])
        self.assertEqual(sorted(results["wait_time"].tolist()), [0, 10, 20])

        self.assertAlmostEqual(outcome["metrics"]["total_wait_time"], 30)
        self.assertAlmostEqual(outcome["metrics"]["mean_wait_time"], 10)
        self.assertAlmostEqual(outcome["final_time"], 30)

        self.assertEqual(manager.active_allocation_count, 0)
        manager.validate_invariants()


if __name__ == "__main__":
    unittest.main(verbosity=2)