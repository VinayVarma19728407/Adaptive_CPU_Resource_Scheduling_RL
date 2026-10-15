# =============================================================================
# FILE: test_policy_comparison_10_machine_contention.py
# ROLE IN PROJECT:
#   An integration test / comparison script for the two baseline scheduling
#   policies (first_fit and least_loaded) under a high-contention scenario:
#     - 10 machines, each with 96 CPU units and 100 GB memory.
#     - 130 tasks, each requesting 800% CPU (i.e. 8 normalised CPUs) and
#       1 GB memory, all arriving simultaneously at time 0.
#
#   This creates genuine resource contention because only ~12 tasks (96/8)
#   can run concurrently per machine, so many tasks must queue.
#
#   The script asserts:
#     - All 130 tasks complete successfully.
#     - More than one machine is used.
#     - Final simulation time is positive (tasks actually queued and waited).
#   It also prints a side-by-side summary of wait statistics per policy.
#
#   Run with: pytest test_policy_comparison_10_machine_contention.py -v
#             OR python test_policy_comparison_10_machine_contention.py
# =============================================================================
import pandas as pd

from simulator.config import ELIGIBLE_MACHINE_STATUS
from simulator.environment import SchedulingEnvironment
from simulator.machine_state import MachineStateManager


def test_baseline_policies_under_10_machine_contention():
    metadata = pd.DataFrame([
        {
            "machine_id": f"machine_{i:02d}",
            "cpu_num": 96.0,
            "mem_size": 100.0,
            "status": ELIGIBLE_MACHINE_STATUS,
        }
        for i in range(10)
    ])

    scenario = pd.DataFrame([
        {
            "scenario_task_id": f"task_{i:03d}",
            "task_name": f"task_{i:03d}",
            "job_name": f"job_{i:03d}",
            "plan_cpu": 800.0,
            "plan_mem": 1.0,
            "duration": 100.0,
            "arrival_time": 0.0,
        }
        for i in range(130)
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
        assert outcome["metrics"]["tasks_completed"] == 130
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
    print("\n=== MEAN RESOURCE DEMAND CONTENTION COMPARISON ===")
    print(report.to_string(index=False))

    assert report["completed"].eq(130).all()
    assert (report["machines_used"] > 1).all()
    assert (report["final_time"] > 0).all()







