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