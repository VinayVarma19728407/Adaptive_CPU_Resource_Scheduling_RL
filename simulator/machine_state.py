# =============================================================================
# FILE: simulator/machine_state.py
# ROLE IN PROJECT:
#   Implements the authoritative resource ledger for all machines in the
#   simulator. Every CPU and memory allocation/release goes through this module.
#
#   Key classes:
#     MachineState         — a single machine's CPU and memory (capacity and
#                            available). Invariants are enforced on construction.
#     MachineStateManager  — manages ALL machines; exposes:
#         can_allocate()   — check if a machine can accept a given task's demand
#         allocate()       — atomically reserve CPU + memory and return an
#                            allocation_id (used later for release)
#         release()        — free resources when a task completes
#         validate_invariants() — assertion that the ledger is consistent
#                            (called at episode start/end by the simulator)
#
#   Consumed by:
#     - simulator/environment.py  (baseline scheduling)
#     - simulator/rl_environment.py (RL step loop)
#     - tests/test_machine_state.py (unit tests)
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import math

import pandas as pd

from simulator.config import ELIGIBLE_MACHINE_STATUS
from simulator.task_data import load_machine_metadata


@dataclass
class MachineState:
    """Current resource state of one eligible machine."""

    machine_id: str
    cpu_capacity: float
    memory_capacity: float
    cpu_available: float
    memory_available: float

    def __post_init__(self) -> None:
        values = (
            self.cpu_capacity,
            self.memory_capacity,
            self.cpu_available,
            self.memory_available,
        )

        if not all(math.isfinite(value) for value in values):
            raise ValueError("Machine resource values must be finite.")

        if self.cpu_capacity <= 0 or self.memory_capacity <= 0:
            raise ValueError("Machine capacities must be positive.")

        if not 0 <= self.cpu_available <= self.cpu_capacity:
            raise ValueError("Available CPU is outside machine capacity.")

        if not 0 <= self.memory_available <= self.memory_capacity:
            raise ValueError("Available memory is outside machine capacity.")


class MachineStateManager:
    """
    Maintain authoritative CPU and memory availability for machines.

    Allocations are atomic: CPU and memory are both reserved, or neither
    is reserved. Each successful allocation receives a unique allocation
    ID and must be released exactly once.
    """

    def __init__(
        self,
        machine_metadata: Optional[pd.DataFrame] = None,
        tolerance: float = 1e-9,
    ) -> None:
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Tolerance must be finite and nonnegative.")

        self.tolerance = tolerance
        self.machines: Dict[str, MachineState] = {}
        self._allocations: Dict[str, dict] = {}
        self._next_allocation_number = 1

        if machine_metadata is None:
            machine_metadata = load_machine_metadata()

        self._initialize_machines(machine_metadata)

    @staticmethod
    def _normalize_machine_id(value) -> str:
        if pd.isna(value):
            raise ValueError("Machine ID cannot be missing.")

        # Normalize numeric IDs consistently, e.g. 123.0 -> "123".
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric_value = float(value)

            if not math.isfinite(numeric_value):
                raise ValueError("Machine ID must be finite.")

            if numeric_value.is_integer():
                return str(int(numeric_value))

        normalized = str(value).strip()

        if not normalized:
            raise ValueError("Machine ID cannot be empty.")

        return normalized

    def _initialize_machines(self, metadata: pd.DataFrame) -> None:
        required = {
            "machine_id",
            "cpu_num",
            "mem_size",
            "status",
        }

        missing = required - set(metadata.columns)

        if missing:
            raise ValueError(
                f"Machine metadata is missing columns: {sorted(missing)}"
            )

        if metadata.empty:
            raise ValueError("Machine metadata contains no machines.")

        data = metadata.copy()

        for column in ("cpu_num", "mem_size", "status"):
            data[column] = pd.to_numeric(data[column], errors="coerce")

        if data[list(required)].isna().any().any():
            raise ValueError(
                "Machine metadata contains missing or nonnumeric required values."
            )

        if not (data["status"] == ELIGIBLE_MACHINE_STATUS).all():
            raise ValueError(
                "Machine metadata contains machines that are not eligible."
            )

        for row in data.itertuples(index=False):
            machine_id = self._normalize_machine_id(
                getattr(row, "machine_id")
            )
            cpu_capacity = float(getattr(row, "cpu_num"))
            memory_capacity = float(getattr(row, "mem_size"))

            if (
                not math.isfinite(cpu_capacity)
                or not math.isfinite(memory_capacity)
                or cpu_capacity <= 0
                or memory_capacity <= 0
            ):
                raise ValueError(
                    f"Machine {machine_id} has invalid resource capacity."
                )

            if machine_id in self.machines:
                raise ValueError(
                    f"Duplicate machine ID in metadata: {machine_id}"
                )

            self.machines[machine_id] = MachineState(
                machine_id=machine_id,
                cpu_capacity=cpu_capacity,
                memory_capacity=memory_capacity,
                cpu_available=cpu_capacity,
                memory_available=memory_capacity,
            )

        if not self.machines:
            raise ValueError("No eligible machines were initialized.")

    @property
    def machine_count(self) -> int:
        return len(self.machines)

    @property
    def active_allocation_count(self) -> int:
        return len(self._allocations)

    def get_machine(self, machine_id) -> MachineState:
        normalized_id = self._normalize_machine_id(machine_id)

        if normalized_id not in self.machines:
            raise KeyError(f"Unknown or ineligible machine: {normalized_id}")

        return self.machines[normalized_id]

    def can_allocate(
        self,
        machine_id,
        cpu_amount: float,
        memory_amount: float,
    ) -> bool:
        """Return whether both resource requests fit right now."""

        try:
            cpu_amount = float(cpu_amount)
            memory_amount = float(memory_amount)
            machine = self.get_machine(machine_id)
        except (TypeError, ValueError, KeyError):
            return False

        if (
            not math.isfinite(cpu_amount)
            or not math.isfinite(memory_amount)
            or cpu_amount <= 0
            or memory_amount <= 0
        ):
            return False

        return (
            cpu_amount <= machine.cpu_available + self.tolerance
            and memory_amount
            <= machine.memory_available + self.tolerance
        )

    def allocate(
        self,
        machine_id,
        cpu_amount: float,
        memory_amount: float,
        task_id: str,
    ) -> str:
        """
        Reserve CPU and memory together and return an allocation ID.

        Raises ValueError if the request is invalid or does not fit.
        Raises KeyError if the machine is unknown or ineligible.
        """

        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("task_id must be a nonempty string.")

        cpu_amount = float(cpu_amount)
        memory_amount = float(memory_amount)
        machine = self.get_machine(machine_id)

        if (
            not math.isfinite(cpu_amount)
            or not math.isfinite(memory_amount)
            or cpu_amount <= 0
            or memory_amount <= 0
        ):
            raise ValueError("Allocation amounts must be finite and positive.")

        # Do not let numerical tolerance authorize over-allocation.
        if (
            cpu_amount > machine.cpu_available + self.tolerance
            or memory_amount > machine.memory_available + self.tolerance
        ):
            raise ValueError(
                f"Insufficient resources on machine {machine.machine_id}: "
                f"requested CPU={cpu_amount}, memory={memory_amount}; "
                f"available CPU={machine.cpu_available}, "
                f"memory={machine.memory_available}."
            )

        allocation_id = f"alloc_{self._next_allocation_number:09d}"
        self._next_allocation_number += 1

        # Both checks have passed. Commit both resource changes together.
        machine.cpu_available -= cpu_amount
        machine.memory_available -= memory_amount

        # Prevent tiny negative floating-point residue.
        if abs(machine.cpu_available) <= self.tolerance:
            machine.cpu_available = 0.0
        if abs(machine.memory_available) <= self.tolerance:
            machine.memory_available = 0.0

        self._allocations[allocation_id] = {
            "machine_id": machine.machine_id,
            "task_id": task_id,
            "cpu_amount": cpu_amount,
            "memory_amount": memory_amount,
        }

        return allocation_id

    def release(self, allocation_id: str) -> dict:
        """
        Release the exact resources associated with one allocation.

        Each allocation can be released only once.
        """

        if allocation_id not in self._allocations:
            raise KeyError(
                f"Unknown or already released allocation: {allocation_id}"
            )

        allocation = self._allocations[allocation_id]
        machine = self.machines[allocation["machine_id"]]

        new_cpu = machine.cpu_available + allocation["cpu_amount"]
        new_memory = (
            machine.memory_available + allocation["memory_amount"]
        )

        # Check before changing state.
        if (
            new_cpu > machine.cpu_capacity + self.tolerance
            or new_memory > machine.memory_capacity + self.tolerance
        ):
            raise RuntimeError(
                "Release would exceed machine capacity; state was not changed."
            )

        machine.cpu_available = min(new_cpu, machine.cpu_capacity)
        machine.memory_available = min(
            new_memory, machine.memory_capacity
        )

        del self._allocations[allocation_id]

        return allocation.copy()

    def get_snapshot(self) -> pd.DataFrame:
        """Return a tabular snapshot of current machine resources."""

        rows = []

        for machine in self.machines.values():
            rows.append(
                {
                    "machine_id": machine.machine_id,
                    "cpu_capacity": machine.cpu_capacity,
                    "cpu_available": machine.cpu_available,
                    "cpu_allocated": (
                        machine.cpu_capacity - machine.cpu_available
                    ),
                    "memory_capacity": machine.memory_capacity,
                    "memory_available": machine.memory_available,
                    "memory_allocated": (
                        machine.memory_capacity - machine.memory_available
                    ),
                }
            )

        return pd.DataFrame(rows)

    def validate_invariants(self) -> None:
        """Raise an error if tracked resource state is inconsistent."""

        allocated_cpu = {machine_id: 0.0 for machine_id in self.machines}
        allocated_memory = {
            machine_id: 0.0 for machine_id in self.machines
        }

        for allocation in self._allocations.values():
            machine_id = allocation["machine_id"]

            allocated_cpu[machine_id] += allocation["cpu_amount"]
            allocated_memory[machine_id] += allocation["memory_amount"]

        for machine_id, machine in self.machines.items():
            if not (
                -self.tolerance
                <= machine.cpu_available
                <= machine.cpu_capacity + self.tolerance
            ):
                raise AssertionError(
                    f"CPU availability invariant failed: {machine_id}"
                )

            if not (
                -self.tolerance
                <= machine.memory_available
                <= machine.memory_capacity + self.tolerance
            ):
                raise AssertionError(
                    f"Memory availability invariant failed: {machine_id}"
                )

            expected_cpu = machine.cpu_capacity - allocated_cpu[machine_id]
            expected_memory = (
                machine.memory_capacity - allocated_memory[machine_id]
            )

            if not math.isclose(
                machine.cpu_available,
                expected_cpu,
                rel_tol=0.0,
                abs_tol=self.tolerance,
            ):
                raise AssertionError(
                    f"CPU allocation ledger mismatch: {machine_id}"
                )

            if not math.isclose(
                machine.memory_available,
                expected_memory,
                rel_tol=0.0,
                abs_tol=self.tolerance,
            ):
                raise AssertionError(
                    f"Memory allocation ledger mismatch: {machine_id}"
                )


if __name__ == "__main__":
    manager = MachineStateManager()
    manager.validate_invariants()

    print("MACHINE STATE INITIALIZATION")
    print("-" * 50)
    print(f"Eligible machines loaded: {manager.machine_count:,}")
    print(f"Active allocations: {manager.active_allocation_count}")
    print("Initial resource invariants: PASS")
    print("\nFirst five machines:")
    print(manager.get_snapshot().head().to_string(index=False))
