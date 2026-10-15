# =============================================================================
# FILE: environment/__init__.py
# ROLE IN PROJECT:
#   This is the package initializer for the 'environment' module.
#   It marks the 'environment/' folder as a proper Python package so its
#   sub-modules (config.py, scenario.py, task.py) can be imported with
#   dot-notation (e.g., `from environment.task import Task`).
#
#   Exported components:
#     - config.py  : Global simulation constants (mode, resources, preemption)
#     - scenario.py: The Scenario dataclass — an immutable workload window used
#                    by the RL agent and simulator as an episode descriptor.
#     - task.py    : The Task dataclass — one executable task with CPU/memory
#                    demand and timing info.
#
#   This file itself contains only a docstring and exports nothing explicitly;
#   direct imports (e.g. `from environment.task import Task`) are used instead.
# =============================================================================
"""
Environment Module Initialization.

This package initialization file exports core simulation environment components including configuration settings,
scenario representations, and task data structures for the Adaptive Multi-Core CPU Scheduler.
"""
