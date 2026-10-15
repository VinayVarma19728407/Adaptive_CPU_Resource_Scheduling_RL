# =============================================================================
# FILE: simulator/rl_environment.py
# ROLE IN PROJECT:
#   This is the Gymnasium-compatible RL environment used to train the PPO
#   (and MaskablePPO) agent. It wraps SchedulingEnvironment with:
#
#     - Observation space: a flat float32 vector of [task features, machine
#       features] — 4 task features + 4 features per candidate machine.
#     - Action space: Discrete(candidate_count) — which machine to place the
#       current waiting task on.
#     - Action masking: action_masks() marks invalid machine slots as False so
#       MaskablePPO never selects infeasible machines.
#     - Reward modes (configurable):
#         "wait_only"       — penalise only task waiting time
#         "core_penalty"    — also penalise invalid actions, unschedulable tasks
#         "extended_penalty"— all above plus imbalance, fragmentation, saturation
#
#   The training scripts (experiments/train_*.py) instantiate this class.
#   Evaluation scripts (experiments/evaluate_*.py) also use it to score
#   trained PPO checkpoints against baseline policies.
# =============================================================================
from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from simulator.config import MAX_CANDIDATE_MACHINES
from simulator.environment import SchedulingEnvironment
from simulator.machine_state import MachineStateManager


class RLSchedulingEnvironment(gym.Env):
    """Gymnasium wrapper for PPO-based machine placement.

    One action selects a machine for the first waiting task.
    The existing simulator remains responsible for resource allocation,
    arrivals, completions, and event ordering.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: pd.DataFrame | str,
        machine_metadata: pd.DataFrame,
        candidate_count: int = MAX_CANDIDATE_MACHINES,
        invalid_action_penalty: float = 1.0,
        reward_mode: str = "wait_only",
        reward_weights: dict[str, float] | None = None,
        starvation_threshold: float = 10.0,
    ):
        super().__init__()

        if candidate_count <= 0:
            raise ValueError("candidate_count must be positive.")
        if invalid_action_penalty <= 0:
            raise ValueError("invalid_action_penalty must be positive.")

        self.scenario_source = scenario
        self.machine_metadata = machine_metadata.copy()
        self.candidate_count = int(candidate_count)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.reward_mode = reward_mode
        self.starvation_threshold = float(starvation_threshold)

        if reward_mode not in {
            "wait_only",
            "core_penalty",
            "extended_penalty",
        }:
            raise ValueError(f"Unknown reward mode: {reward_mode}")

        self.reward_weights = {
            "wait": 1.0,
            "makespan": 0.5,
            "invalid": 1.0,
            "unschedulable": 2.0,
            "imbalance": 0.10,
            "fragmentation": 0.20,
            "saturation": 0.10,
            "starvation": 0.10,
        }

        if reward_weights is not None:
            unknown = set(reward_weights) - set(self.reward_weights)
            if unknown:
                raise ValueError(f"Unknown reward weights: {sorted(unknown)}")
            self.reward_weights.update(reward_weights)

        if any(not np.isfinite(value) or value < 0
            for value in self.reward_weights.values()):
            raise ValueError("Reward weights must be finite and nonnegative.")

        self.reward_components: dict[str, float] = {}

        # Task features: CPU demand, memory demand, duration, waiting time.
        # Per-machine features: available CPU, available memory,
        # projected maximum utilization, projected total utilization.
        self.features_per_machine = 4
        observation_size = 4 + self.candidate_count * self.features_per_machine

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(observation_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.candidate_count)

        self.sim: SchedulingEnvironment | None = None
        self.candidates: list[str] = []
        self.current_task_id: str | None = None
        self._duration_scale = 1.0
        self._time_scale = 1.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        # Reconstruct the manager for a clean resource state each episode.
        manager = MachineStateManager(self.machine_metadata)

        self.sim = SchedulingEnvironment(
            self.scenario_source,
            machine_state=manager,
        )

        durations = self.sim.tasks["duration"].to_numpy(dtype=float)
        arrivals = self.sim.tasks["arrival_time"].to_numpy(dtype=float)

        self._duration_scale = max(float(np.median(durations)), 1.0)
        self._time_scale = max(
            float(np.max(arrivals)) if len(arrivals) else 0.0,
            self._duration_scale,
            1.0,
        )

        self.candidates = []
        self.current_task_id = None
        self.reward_components = {}
        observation = self._prepare_decision()
        return observation, self._info()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.sim is None:
            raise RuntimeError("Call reset() before step().")
        if self.sim.finished:
            raise RuntimeError("Episode is finished; call reset().")
        if self.current_task_id is None or not self.candidates:
            raise RuntimeError("No scheduling decision is available.")

        sim = self.sim
        task_id = self.current_task_id
        action_index = int(action)

        invalid = action_index < 0 or action_index >= len(self.candidates)
        if invalid:
            action_index = 0

        machine_id = self.candidates[action_index]
        row = sim.task_rows[task_id]

        wait_time = max(
            0.0,
            sim.current_time - float(row["arrival_time"]),
        )
        wait_penalty = wait_time / self._duration_scale

        # Snapshot waiting tasks that are not the task being placed.
        other_waiting = [
            waiting_id
            for waiting_id in sim.waiting
            if waiting_id != task_id
        ]
        before_feasible = self._feasible_waiting_tasks(other_waiting)

        # Measure the state after allocation but before advancing events.
        unschedulable_before = int(sim.metrics["tasks_unschedulable"])
        sim.schedule_task_on_machine(task_id, machine_id)

        imbalance_penalty = self._resource_imbalance()
        saturation_penalty = self._resource_saturation()
        after_feasible = self._feasible_waiting_tasks(other_waiting)

        if other_waiting:
            newly_blocked = len(before_feasible - after_feasible)
            fragmentation_penalty = newly_blocked / len(other_waiting)
        else:
            fragmentation_penalty = 0.0

        starvation_penalty = max(
            0.0, wait_time - self.starvation_threshold
        ) / self._duration_scale

        # Advance the event-driven simulator to the next decision or episode end.
        observation = self._prepare_decision()
        terminated = bool(sim.finished)
        truncated = False

        newly_unschedulable = max(
            0,
            int(sim.metrics["tasks_unschedulable"]) - unschedulable_before,
        )
        unschedulable_penalty = (
            newly_unschedulable / max(1, len(sim.tasks))
        )

        makespan_penalty = (
            max(0.0, float(sim.current_time)) / self._duration_scale
            if terminated else 0.0
        )

        components = {
            "wait": wait_penalty,
            "makespan": makespan_penalty,
            "invalid": float(invalid),
            "unschedulable": unschedulable_penalty,
            "imbalance": imbalance_penalty,
            "fragmentation": fragmentation_penalty,
            "saturation": saturation_penalty,
            "starvation": starvation_penalty,
        }

        if self.reward_mode == "wait_only":
            active_terms = {"wait"}
        elif self.reward_mode == "core_penalty":
            active_terms = {
                "wait", "makespan", "invalid", "unschedulable"
            }
        else:
            active_terms = set(components)

        reward = -sum(
            self.reward_weights[name] * components[name]
            for name in active_terms
        )

        self.reward_components = {
            name: (
                self.reward_weights[name] * components[name]
                if name in active_terms else 0.0
            )
            for name in components
        }

        info = self._info()
        info.update({
            "last_task_id": task_id,
            "last_machine_id": machine_id,
            "last_wait_time": wait_time,
            "reward_mode": self.reward_mode,
            "penalty_components": dict(self.reward_components),
        })

        return observation, float(reward), terminated, truncated, info

    def _feasible_waiting_tasks(
        self, task_ids: list[str]
    ) -> set[str]:
        """Return waiting tasks that fit at least one machine now."""
        assert self.sim is not None
        feasible = set()

        for task_id in task_ids:
            row = self.sim.task_rows[task_id]
            cpu = float(row["plan_cpu"]) / 100.0
            memory = float(row["plan_mem"])

            for machine_id in self.sim.machine_state.machines:
                if self.sim.machine_state.can_allocate(
                    machine_id, cpu, memory
                ):
                    feasible.add(task_id)
                    break

        return feasible

    def _resource_imbalance(self) -> float:
        """Measure CPU/memory utilization imbalance after placement."""
        assert self.sim is not None

        cpu_utilizations = []
        memory_utilizations = []

        for machine in self.sim.machine_state.machines.values():
            cpu_utilizations.append(
                1.0 - machine.cpu_available / machine.cpu_capacity
            )
            memory_utilizations.append(
                1.0 - machine.memory_available / machine.memory_capacity
            )

        cpu_imbalance = (
            max(cpu_utilizations) - min(cpu_utilizations)
            if cpu_utilizations else 0.0
        )
        memory_imbalance = (
            max(memory_utilizations) - min(memory_utilizations)
            if memory_utilizations else 0.0
        )

        return float(max(cpu_imbalance, memory_imbalance))

    def _resource_saturation(self) -> float:
        """Penalize CPU or memory utilization above 90 percent."""
        assert self.sim is not None
        threshold = 0.90
        penalties = []

        for machine in self.sim.machine_state.machines.values():
            cpu_util = 1.0 - machine.cpu_available / machine.cpu_capacity
            memory_util = (
                1.0 - machine.memory_available / machine.memory_capacity
            )

            for utilization in (cpu_util, memory_util):
                excess = max(0.0, utilization - threshold) / (1.0 - threshold)
                penalties.append(excess ** 2)

        return float(np.mean(penalties)) if penalties else 0.0

    def _prepare_decision(self) -> np.ndarray:
        """Advance events until a task can be placed or the episode ends."""
        assert self.sim is not None

        while True:
            self.sim._handle_completions_at_current_time()
            self.sim._handle_arrivals_at_current_time()
            self.sim._handle_completions_at_current_time()

            resolved = (
                self.sim.metrics["tasks_completed"]
                + self.sim.metrics["tasks_unschedulable"]
            )

            if resolved == len(self.sim.tasks):
                self.sim.finished = True
                self.sim.machine_state.validate_invariants()
                if self.sim.machine_state.active_allocation_count != 0:
                    raise RuntimeError(
                        "Episode ended with active allocations."
                    )
                self.current_task_id = None
                self.candidates = []
                return self._empty_observation()

            if self.sim.waiting:
                # Select the first waiting task that can be scheduled now.
                # Do not let an infeasible queue-head task block other tasks.
                for task_id in self.sim.waiting:
                    candidates = self._get_candidates(task_id)

                    if candidates:
                        self.current_task_id = task_id
                        self.candidates = candidates
                        return self._make_observation(
                            task_id, candidates
                        )

            if self.sim.events.is_empty():
                raise RuntimeError(
                    "Unresolved tasks remain but no events can advance "
                    "the simulation."
                )

            next_time = float(self.sim.events.peek().time)
            if next_time < self.sim.current_time - 1e-9:
                raise RuntimeError("Event queue contains an event in the past.")

            self.sim.current_time = max(self.sim.current_time, next_time)
    def action_masks(self) -> list[bool]:
        """Return valid candidate indices for MaskablePPO.

        True means the corresponding action selects a currently feasible
        candidate machine. False means that action index has no candidate.
        """
        if self.sim is None or self.sim.finished:
            return [False] * self.candidate_count

        valid_count = len(self.candidates)

        if valid_count == 0:
            return [False] * self.candidate_count

        return [
            index < valid_count
            for index in range(self.candidate_count)
        ]
    def _get_candidates(self, task_id: str) -> list[str]:
        """Rank currently feasible machines and return a stable shortlist."""
        assert self.sim is not None

        row = self.sim.task_rows[task_id]
        cpu = float(row["plan_cpu"]) / 100.0
        memory = float(row["plan_mem"])

        ranked: list[tuple[float, float, str]] = []

        for machine_id in sorted(self.sim.machine_state.machines):
            machine = self.sim.machine_state.get_machine(machine_id)

            if not self.sim.machine_state.can_allocate(
                machine_id, cpu, memory
            ):
                continue

            cpu_util_after = (
                machine.cpu_capacity - machine.cpu_available + cpu
            ) / machine.cpu_capacity
            memory_util_after = (
                machine.memory_capacity - machine.memory_available + memory
            ) / machine.memory_capacity

            ranked.append(
                (
                    max(cpu_util_after, memory_util_after),
                    cpu_util_after + memory_util_after,
                    machine_id,
                )
            )

        ranked.sort()
        return [
            item[2] for item in ranked[: self.candidate_count]
        ]

    def _make_observation(
        self, task_id: str, candidates: list[str]
    ) -> np.ndarray:
        assert self.sim is not None

        row = self.sim.task_rows[task_id]
        cpu = float(row["plan_cpu"]) / 100.0
        memory = float(row["plan_mem"])
        duration = float(row["duration"])
        waiting = max(
            0.0,
            self.sim.current_time - float(row["arrival_time"]),
        )

        max_cpu = max(
            machine.cpu_capacity
            for machine in self.sim.machine_state.machines.values()
        )
        max_memory = max(
            machine.memory_capacity
            for machine in self.sim.machine_state.machines.values()
        )

        features = [
            np.clip(cpu / max_cpu, 0.0, 1.0),
            np.clip(memory / max_memory, 0.0, 1.0),
            np.clip(duration / self._duration_scale, 0.0, 1.0),
            np.clip(waiting / self._time_scale, 0.0, 1.0),
        ]

        for index in range(self.candidate_count):
            if index >= len(candidates):
                features.extend([0.0, 0.0, 0.0, 0.0])
                continue

            machine = self.sim.machine_state.get_machine(
                candidates[index]
            )

            cpu_after = (
                machine.cpu_capacity - machine.cpu_available + cpu
            ) / machine.cpu_capacity
            memory_after = (
                machine.memory_capacity - machine.memory_available + memory
            ) / machine.memory_capacity

            features.extend([
                np.clip(
                    machine.cpu_available / machine.cpu_capacity,
                    0.0, 1.0,
                ),
                np.clip(
                    machine.memory_available / machine.memory_capacity,
                    0.0, 1.0,
                ),
                np.clip(max(cpu_after, memory_after), 0.0, 1.0),
                np.clip(cpu_after + memory_after, 0.0, 2.0) / 2.0,
            ])

        return np.asarray(features, dtype=np.float32)

    def _empty_observation(self) -> np.ndarray:
        return np.zeros(
            self.observation_space.shape, dtype=np.float32
        )

    def _info(self) -> dict[str, Any]:
        assert self.sim is not None
        return {
            "simulation_time": self.sim.current_time,
            "waiting_tasks": len(self.sim.waiting),
            "running_tasks": len(self.sim.running),
            "tasks_completed": self.sim.metrics["tasks_completed"],
            "tasks_unschedulable": self.sim.metrics["tasks_unschedulable"],
            "candidate_machine_ids": list(self.candidates),
        }