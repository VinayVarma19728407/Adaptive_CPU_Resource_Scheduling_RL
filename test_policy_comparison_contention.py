import pandas as pd

from simulator.config import ELIGIBLE_MACHINE_STATUS
from simulator.environment import SchedulingEnvironment
from simulator.machine_state import MachineStateManager


def test_baseline_policies_under_identical_contention():
    metadata = pd.DataFrame([{
        "machine_id": "contention_machine",
        "cpu_num": 96.0,
        "mem_size": 100.0,
        "status": ELIGIBLE_MACHINE_STATUS,
    }])

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

    summaries = []

    for policy in ("first_fit", "least_loaded"):
        manager = MachineStateManager(machine_metadata=metadata.copy())
        env = SchedulingEnvironment(
            scenario=scenario.copy(),
            machine_state=manager,
            placement_policy=policy,
        )
        outcome = env.run()
        results = outcome["results"]

        manager.validate_invariants()
        assert manager.active_allocation_count == 0
        assert outcome["metrics"]["tasks_completed"] == 20
        assert (results["status"] == "completed").all()
        assert (results["wait_time"] >= 0).all()

        summary = {
            "policy": policy,
            "completed": len(results),
            "machines_used": results["machine_id"].nunique(),
            "tasks_that_waited": int((results["wait_time"] > 0).sum()),
            "mean_wait": float(results["wait_time"].mean()),
            "max_wait": float(results["wait_time"].max()),
            "final_time": float(outcome["final_time"]),
        }
        summaries.append(summary)

    report = pd.DataFrame(summaries)
    print("\n=== BASELINE CONTENTION COMPARISON ===")
    print(report.to_string(index=False))

    assert report["completed"].eq(20).all()
    assert report["tasks_that_waited"].eq(8).all()
    assert report["max_wait"].eq(100.0).all()
    assert report["final_time"].eq(200.0).all()
