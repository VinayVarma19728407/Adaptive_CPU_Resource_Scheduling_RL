import pandas as pd
from simulator.machine_state import MachineStateManager
from simulator.environment import SchedulingEnvironment

machines = pd.DataFrame([
    {
        "machine_id": "test_machine",
        "cpu_num": 96,
        "mem_size": 100,
        "status": 1,
    }
])

scenario = pd.DataFrame([
    {
        "scenario_task_id": f"task_{i:02d}",
        "task_name": f"task_{i:02d}",
        "job_name": f"job_{i:02d}",
        "plan_cpu": 800,
        "plan_mem": 1,
        "duration": 100,
        "arrival_time": 0,
    }
    for i in range(20)
])

for policy in ("first_fit", "least_loaded"):
    manager = MachineStateManager(machines.copy())
    env = SchedulingEnvironment(
        scenario=scenario.copy(),
        machine_state=manager,
        placement_policy=policy,
    )
    outcome = env.run()
    results = outcome["results"]

    waits = pd.to_numeric(results["wait_time"])
    completed = results["status"].eq("completed").sum()

    print(f"\nPolicy: {policy}")
    print(f"Completed: {completed}/{len(scenario)}")
    print(f"Tasks that waited: {(waits > 0).sum()}")
    print(f"Maximum wait: {waits.max()}")
    print(f"Final simulation time: {outcome['final_time']}")

