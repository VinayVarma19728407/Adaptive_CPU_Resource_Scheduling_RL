# =============================================================================
# FILE: environment/config.py
# ROLE IN PROJECT:
#   Defines global constants for the simulator's *environment* layer
#   (distinct from simulator/config.py which handles dataset paths).
#   Any module in the project that needs to know the simulation mode,
#   which resource dimensions to track, how many candidate machines to
#   show the RL agent, or whether preemption is allowed simply imports
#   these constants from here.
#
#   Key constants:
#     SIMULATION_MODE        = "event_driven"  - controls how the simulator
#                                               processes time (discrete events)
#     PRIMARY_RESOURCES      = ("cpu","memory") - the two resource axes tracked
#     MAX_CANDIDATE_MACHINES = 20              - max machines presented to the
#                                               RL agent as valid action choices
#     PREEMPTION_ENABLED     = False           - tasks run to completion; no
#                                               preemption is modelled in v1
# =============================================================================

"""
Simulator Environment Configuration.

This file defines global configuration parameters and constants for the discrete-event CPU scheduler simulator,
including simulation mode, primary resources (CPU, memory), core allocations, and time step settings.
"""

# Simulation architecture
SIMULATION_MODE = "event_driven"

# Resource model
PRIMARY_RESOURCES = ("cpu","memory",)

# Machine candidate selection
MAX_CANDIDATE_MACHINES = 20

# Task execution model
PREEMPTION_ENABLED = False