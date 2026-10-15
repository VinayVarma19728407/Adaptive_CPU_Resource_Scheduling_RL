import pandas as pd

from simulator.config import ELIGIBLE_MACHINE_STATUS
from simulator.environment import SchedulingEnvironment
from simulator.machine_state import MachineStateManager


def test_twenty_tasks_contend_for_one_machine():
    metadata = pd.DataFrame([{
        "machine_id": "contention_machine",
        "cpu_num": 96.0,
        "mem_size": 100.0,
        "status": ELIGIBLE_MACHINE_STATUS,
    }])

    manager = MachineStateManager(machine_metadata=metadata)

    scenario = pd.DataFrame([
        {
            "scenario_task_id": f"task_{i:02d}",
            "task_name": f"task_{i:02d}",
            "job_name": f"job_{i:02d}",
            "plan_cpu": 800.0,
            "plan_mem": 1.0,
            "duration": 100.0,
            "arrival_time": 0.0,
        }
        for i in range(20)
    ])

    env = SchedulingEnvironment(
        scenario=scenario,
        machine_state=manager,
        placement_policy="first_fit",
    )
    outcome = env.run()
    results = outcome["results"]

    assert outcome["metrics"]["tasks_completed"] == 20
    assert (results["status"] == "completed").all()
    assert (results["wait_time"] > 0).any()
    assert results["wait_time"].max() > 0
    assert manager.active_allocation_count == 0
    manager.validate_invariants()

    print("\n20-task contention test passed")
    print(f"Tasks completed: {len(results)}")
    print(f"Tasks that waited: {(results['wait_time'] > 0).sum()}")
    print(f"Maximum wait: {results['wait_time'].max()}")
    print(f"Final simulation time: {outcome['final_time']}")