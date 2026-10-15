# =============================================================================
# FILE: test_simulator_contention.py
# ROLE IN PROJECT:
#   A targeted unit test verifying the core resource contention handling of
#   SchedulingEnvironment. It creates a single-machine scenario:
#     - 1 machine with 1 normalised CPU and 100 GB memory.
#     - 2 tasks (A and B), both requesting 100% CPU and 50 GB memory,
#       arriving simultaneously at time 0.
#
#   Because only one task can run at a time (CPU is the bottleneck), task B
#   must wait until task A completes before it can start. The test asserts:
#     - Task A starts at time 0 and completes at time 10.
#     - Task B starts at time 10 (after A completes) and waits 10 time units.
#     - Final simulation time is 15 (10 + 5 for task B's duration).
#     - No resource allocations are leaked at the end.
#
#   This validates the wait-then-schedule logic in SchedulingEnvironment.run().
#
#   Run with: pytest test_simulator_contention.py -v
#             OR python test_simulator_contention.py
# =============================================================================
import pandas as pd

from simulator.environment import SchedulingEnvironment
from simulator.machine_state import MachineStateManager
from simulator.config import ELIGIBLE_MACHINE_STATUS


def test_task_waits_until_machine_resources_are_released():
    # One machine with capacity for exactly one task at a time.
    metadata = pd.DataFrame([{
        "machine_id": "test_machine",
        "cpu_num": 1.0,
        "mem_size": 100.0,
        "status": ELIGIBLE_MACHINE_STATUS,
    }])

    machines = MachineStateManager(machine_metadata=metadata)

    # Both tasks arrive together and each needs all available CPU.
    scenario = pd.DataFrame([
        {
            "scenario_task_id": "task_A",
            "task_name": "A",
            "job_name": "job_A",
            "plan_cpu": 100.0,
            "plan_mem": 50.0,
            "duration": 10.0,
            "arrival_time": 0.0,
        },
        {
            "scenario_task_id": "task_B",
            "task_name": "B",
            "job_name": "job_B",
            "plan_cpu": 100.0,
            "plan_mem": 50.0,
            "duration": 5.0,
            "arrival_time": 0.0,
        },
    ])

    env = SchedulingEnvironment(
        scenario=scenario,
        machine_state=machines,
        placement_policy="first_fit",
    )
    outcome = env.run()
    results = outcome["results"].set_index("scenario_task_id")

    assert outcome["metrics"]["tasks_completed"] == 2
    assert results.loc["task_A", "start_time"] == 0.0
    assert results.loc["task_B", "start_time"] == 10.0
    assert results.loc["task_B", "wait_time"] == 10.0
    assert outcome["final_time"] == 15.0
    assert machines.active_allocation_count == 0

    print("\nContention test passed:")
    print(results[["start_time", "completion_time", "wait_time"]])
    print("Final simulation time:", outcome["final_time"])