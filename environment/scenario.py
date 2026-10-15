# =============================================================================
# FILE: environment/scenario.py
# ROLE IN PROJECT:
#   Defines the Scenario dataclass — an immutable, frozen description of one
#   workload episode used by both the RL agent and the simulator.
#
#   A Scenario acts as the "map" for an RL training/evaluation episode:
#     - It records the temporal window (time_start, time_end) of real cluster
#       data that tasks are drawn from.
#     - It carries workload-regime labels (intensity, resource_dominance,
#       variability) used by the scenario manifest and curriculum-learning logic.
#     - Its contains_time() method is the authoritative boundary check used by
#       the simulator to decide whether a task belongs to this episode.
#
#   Created by:  scenario_management/create_workload_scenario_manifest.py
#   Consumed by: simulator/rl_environment.py and experiment scripts.
# =============================================================================
"""
Scenario Representation Data Model.

This file defines the Scenario dataclass representing a validated temporal workload window for RL scheduler evaluation,
encapsulating scenario metadata, start/end timestamps, task lists, and temporal boundary checks.
"""

from dataclasses import dataclass

@dataclass(frozen=True)
class Scenario:
    """Immutable representation of one simulator workload scenario."""

    # Scenario identification and selection metadata
    scenario_id: str
    split: str
    scenario_type: str
    scenario_axis: str
    regime: str
    time_bin: int

    # Temporal workload window
    time_start: float
    time_end: float

    # Continuous workload characteristics
    workload_intensity: float
    resource_dominance: float
    variability_score: float

    # Workload regime labels
    intensity_regime: str
    resource_regime: str
    variability_regime: str

    # Deterministic scenario-selection seed
    selection_seed: int

    def contains_time(self, timestamp: float) -> bool:
        """Return True when a timestamp belongs to this scenario.
        The simulator contract uses the half-open interval:
            time_start <= timestamp < time_end"""
        return self.time_start <= timestamp < self.time_end