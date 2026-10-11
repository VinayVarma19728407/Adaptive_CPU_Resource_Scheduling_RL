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