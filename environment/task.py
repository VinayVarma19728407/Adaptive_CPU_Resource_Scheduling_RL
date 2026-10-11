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