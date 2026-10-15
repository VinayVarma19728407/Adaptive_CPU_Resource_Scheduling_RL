# =============================================================================
# FILE: environment/task.py
# ROLE IN PROJECT:
#   Defines the Task dataclass — an immutable record of a single schedulable
#   unit of work in the simulator.
#
#   Each Task represents one row from the batch instance dataset after it has
#   been converted into simulator-ready format:
#     - job_name / task_name : identity keys linking back to dataset records
#     - start_time / end_time / duration : timing profile
#     - cpu_request / memory_request   : resource demand the scheduler must
#                                        satisfy when placing this task
#     - status                          : integer status code
#
#   The Task objects are constructed by the scenario loader and consumed by
#   SchedulingEnvironment and RLSchedulingEnvironment during simulation runs.
# =============================================================================
"""
Task Representation Data Model.

This file defines the Task dataclass representing an executable task instance within the simulator environment,
encapsulating task ID, arrival time, CPU request, memory request, execution duration, and scheduling state.
"""

from dataclasses import dataclass

@dataclass(frozen=True)
class Task:
    """Immutable representation of one task instance."""
    # Task identity
    job_name: str
    task_name: str
    # Temporal information
    start_time: float
    end_time: float
    duration: float
    # Resource requirements
    cpu_request: float
    memory_request: float
    # Task status
    status: int