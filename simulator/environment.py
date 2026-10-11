from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import math

import pandas as pd

from simulator.config import PROJECT_ROOT, ELIGIBLE_MACHINE_STATUS
from simulator.event_queue import EventQueue
from simulator.machine_state import MachineStateManager


@dataclass
class RunningTask:
    task_id: str
    allocation_id: str
    machine_id: str
    start_time: float
    completion_time: float
    cpu_amount: float
    memory_amount: float
    baseline_duration: float


class SchedulingEnvironment:
    """Small event-driven scheduling environment with a deterministic baseline.

    This is a simulator prototype, not yet a Gymnasium/PPO environment.
    Tasks use their planned CPU/memory demand and baseline duration. Task
    dependencies, preemption, failure recovery, and allocation-dependent
    runtime are intentionally not modeled in this initial version.
    """

    REQUIRED_COLUMNS = {
        "scenario_task_id", "task_name", "job_name", "plan_cpu",
        "plan_mem", "duration", "arrival_time",
    }

    def __init__(
        self,
        scenario: pd.DataFrame | str | Path,
        machine_state: Optional[MachineStateManager] = None,
        max_events: int = 2_000_000,
    ) -> None:
        self.scenario_source = scenario
        self._provided_machine_state = machine_state
        self.max_events = int(max_events)
        if self.max_events <= 0:
            raise ValueError("max_events must be positive.")
        self.reset()

    def _load_scenario(self) -> pd.DataFrame:
        if isinstance(self.scenario_source, pd.DataFrame):
            frame = self.scenario_source.copy()
        else:
            path = Path(self.scenario_source)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            if not path.exists():
                raise FileNotFoundError(f"Scenario file not found: {path}")
            frame = pd.read_csv(path)

        missing = self.REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"Scenario missing required columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError("Scenario must contain at least one task.")

        frame = frame.copy().reset_index(drop=True)
        if frame["scenario_task_id"].isna().any():
            raise ValueError("scenario_task_id contains missing values.")
        frame["scenario_task_id"] = frame["scenario_task_id"].astype(str).str.strip()
        if (frame["scenario_task_id"] == "").any():
            raise ValueError("scenario_task_id cannot be empty.")
        if not frame["scenario_task_id"].is_unique:
            raise ValueError("scenario_task_id must be unique within a scenario.")

        for column in ("plan_cpu", "plan_mem", "duration", "arrival_time"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            if frame[column].isna().any():
                raise ValueError(f"Scenario has missing/nonnumeric values in {column}.")
            if not all(math.isfinite(float(x)) for x in frame[column]):
                raise ValueError(f"Scenario has non-finite values in {column}.")

        if (frame["plan_cpu"] <= 0).any() or (frame["plan_mem"] <= 0).any():
            raise ValueError("Task CPU and memory demands must be positive.")
        if (frame["duration"] < 0).any():
            raise ValueError("Task durations cannot be negative.")
        if (frame["arrival_time"] < 0).any():
            raise ValueError("Arrival times cannot be negative.")

        # Preserve row order for simultaneous arrivals; sort only by arrival time.
        frame["_input_order"] = range(len(frame))
        frame = frame.sort_values(["arrival_time", "_input_order"], kind="stable")
        return frame.drop(columns="_input_order").reset_index(drop=True)

    def reset(self) -> dict[str, Any]:
        """Reset events, waiting/running tasks, machine resources, and metrics."""
        self.tasks = self._load_scenario()
        self.machine_state = (
            self._provided_machine_state
            if self._provided_machine_state is not None
            else MachineStateManager()
        )
        # A supplied manager must be fresh; reset resource state by requiring
        # no active allocations, avoiding silent leaks across episodes.
        if self.machine_state.active_allocation_count:
            raise ValueError("MachineStateManager must have no active allocations at reset.")
        self.machine_state.validate_invariants()

        self.events = EventQueue()
        self.waiting: deque[str] = deque()
        self.running: dict[str, RunningTask] = {}
        self.task_rows = {
            row["scenario_task_id"]: row
            for row in self.tasks.to_dict(orient="records")
        }
        self.results: dict[str, dict[str, Any]] = {}
        self.current_time = 0.0
        self.finished = False
        self._event_count = 0

        self.metrics = {
            "tasks_submitted": 0,
            "tasks_started": 0,
            "tasks_completed": 0,
            "tasks_unschedulable": 0,
            "total_wait_time": 0.0,
            "total_flow_time": 0.0,
        }

        for row in self.tasks.to_dict(orient="records"):
            self.events.push(
                float(row["arrival_time"]), "arrival", row["scenario_task_id"]
            )

        return self._observation()

    def _observation(self) -> dict[str, Any]:
        """Return a simple state summary; PPO observation design comes later."""
        return {
            "time": self.current_time,
            "waiting_tasks": list(self.waiting),
            "running_tasks": list(self.running),
            "pending_events": len(self.events),
            "machines": self.machine_state.get_snapshot(),
            "finished": self.finished,
        }

    def _record_unschedulable(self, task_id: str, reason: str) -> None:
        row = self.task_rows[task_id]
        self.results[task_id] = {
            "scenario_task_id": task_id,
            "task_name": row["task_name"],
            "job_name": row["job_name"],
            "arrival_time": float(row["arrival_time"]),
            "start_time": None,
            "completion_time": None,
            "wait_time": None,
            "flow_time": None,
            "machine_id": None,
            "allocated_cpu": None,
            "allocated_memory": None,
            "baseline_duration": float(row["duration"]),
            "status": "unschedulable",
            "reason": reason,
        }
        self.metrics["tasks_unschedulable"] += 1

    def _is_permanently_infeasible(self, row: dict[str, Any]) -> bool:
        cpu = float(row["plan_cpu"])
        memory = float(row["plan_mem"])
        # Eligibility is already enforced when MachineStateManager is built.
        return not any(
            cpu <= machine.cpu_capacity and memory <= machine.memory_capacity
            for machine in self.machine_state.machines.values()
        )

    def _handle_arrivals_at_current_time(self) -> None:
        """Consume every arrival at current time, preserving input order."""
        while not self.events.is_empty() and math.isclose(
            self.events.peek().time, self.current_time, rel_tol=0.0, abs_tol=1e-9
        ) and self.events.peek().event_type == "arrival":
            event = self.events.pop()
            self._event_count += 1
            self._check_event_limit()
            row = self.task_rows[event.task_id]
            self.metrics["tasks_submitted"] += 1
            if self._is_permanently_infeasible(row):
                self._record_unschedulable(
                    event.task_id,
                    "Planned CPU/memory demand exceeds every eligible machine capacity.",
                )
            else:
                self.waiting.append(event.task_id)

    def _handle_completions_at_current_time(self) -> None:
        """Complete all due tasks before scheduling at the same time."""
        while not self.events.is_empty():
            event = self.events.peek()
            if not math.isclose(
                event.time, self.current_time, rel_tol=0.0, abs_tol=1e-9
            ):
                break
            if event.event_type != "completion":
                break

            event = self.events.pop()
            self._event_count += 1
            self._check_event_limit()
            running = self.running.pop(event.task_id, None)
            if running is None:
                raise RuntimeError(f"Completion event for non-running task {event.task_id}")

            allocation = self.machine_state.release(running.allocation_id)
            row = self.task_rows[event.task_id]
            wait_time = running.start_time - float(row["arrival_time"])
            flow_time = self.current_time - float(row["arrival_time"])
            self.results[event.task_id] = {
                "scenario_task_id": event.task_id,
                "task_name": row["task_name"],
                "job_name": row["job_name"],
                "arrival_time": float(row["arrival_time"]),
                "start_time": running.start_time,
                "completion_time": self.current_time,
                "wait_time": wait_time,
                "flow_time": flow_time,
                "machine_id": running.machine_id,
                "allocated_cpu": allocation["cpu_amount"],
                "allocated_memory": allocation["memory_amount"],
                "baseline_duration": running.baseline_duration,
                "status": "completed",
                "reason": None,
            }
            self.metrics["tasks_completed"] += 1
            self.metrics["total_wait_time"] += wait_time
            self.metrics["total_flow_time"] += flow_time

    def _schedule_waiting_tasks(self) -> None:
        """Deterministic FCFS baseline; scan waiting tasks for feasible fits."""
        if not self.waiting:
            return

        # Tasks that cannot fit now remain waiting if they fit some machine's
        # full capacity. Later completions may free the needed resources.
        remaining = deque()
        while self.waiting:
            task_id = self.waiting.popleft()
            row = self.task_rows[task_id]
            cpu = float(row["plan_cpu"])
            memory = float(row["plan_mem"])

            selected_machine = None
            for machine_id in sorted(self.machine_state.machines):
                if self.machine_state.can_allocate(machine_id, cpu, memory):
                    selected_machine = machine_id
                    break

            if selected_machine is None:
                remaining.append(task_id)
                continue

            allocation_id = self.machine_state.allocate(
                selected_machine, cpu, memory, task_id
            )
            duration = float(row["duration"])
            completion_time = self.current_time + duration
            running = RunningTask(
                task_id=task_id,
                allocation_id=allocation_id,
                machine_id=selected_machine,
                start_time=self.current_time,
                completion_time=completion_time,
                cpu_amount=cpu,
                memory_amount=memory,
                baseline_duration=duration,
            )
            self.running[task_id] = running
            self.metrics["tasks_started"] += 1

            # Zero-duration tasks are completed by an event at current time.
            self.events.push(completion_time, "completion", task_id)

        self.waiting = remaining

    def _check_event_limit(self) -> None:
        if self._event_count > self.max_events:
            raise RuntimeError(
                "Maximum event limit exceeded; possible event-loop defect."
            )

    def run(self) -> dict[str, Any]:
        """Run until all tasks are completed or marked unschedulable."""
        while not self.finished:
            # Process all due completions first. This includes zero-duration
            # completions generated by the previous scheduling pass.
            self._handle_completions_at_current_time()
            self._handle_arrivals_at_current_time()
            self._handle_completions_at_current_time()
            self._schedule_waiting_tasks()

            if (
                self.metrics["tasks_completed"] + self.metrics["tasks_unschedulable"]
                == len(self.tasks)
            ):
                self.finished = True
                break

            if self.events.is_empty():
                # Remaining tasks fit some machine at full capacity but cannot
                # progress: this indicates inconsistent state, not infeasibility.
                if self.waiting:
                    raise RuntimeError(
                        "Waiting tasks remain but no events can free resources."
                    )
                raise RuntimeError("Tasks remain unresolved but the event queue is empty.")

            next_event_time = self.events.peek().time
            if next_event_time < self.current_time - 1e-9:
                raise RuntimeError("Event queue contains an event in the past.")
            self.current_time = max(self.current_time, float(next_event_time))

        self.machine_state.validate_invariants()
        if self.machine_state.active_allocation_count != 0:
            raise RuntimeError("Episode ended with active resource allocations.")

        results_df = pd.DataFrame(self.results.values())
        if not results_df.empty:
            results_df = results_df.sort_values(
                ["arrival_time", "scenario_task_id"], kind="stable"
            ).reset_index(drop=True)

        completed = self.metrics["tasks_completed"]
        self.metrics["makespan"] = self.current_time
        self.metrics["mean_wait_time"] = (
            self.metrics["total_wait_time"] / completed if completed else 0.0
        )
        self.metrics["mean_flow_time"] = (
            self.metrics["total_flow_time"] / completed if completed else 0.0
        )
        self.metrics["completion_rate"] = completed / len(self.tasks)
        self.metrics["resolved_tasks"] = (
            self.metrics["tasks_completed"] + self.metrics["tasks_unschedulable"]
        )

        return {
            "metrics": dict(self.metrics),
            "results": results_df,
            "final_time": self.current_time,
            "finished": self.finished,
        }


def load_scenario(name: str, seed: int = 42) -> Path:
    """Resolve a generated scenario by its standard name and seed."""
    if name not in {"light", "moderate", "heavy"}:
        raise ValueError("name must be light, moderate, or heavy")
    path = PROJECT_ROOT / "experiments" / "scenarios" / f"{name}_scenario_seed{seed}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Generated scenario not found: {path}")
    return path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the deterministic scheduling simulator.")
    parser.add_argument("--scenario", choices=["light", "moderate", "heavy"], default="light")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="experiments/results")
    args = parser.parse_args()

    scenario_path = load_scenario(args.scenario, args.seed)
    environment = SchedulingEnvironment(scenario_path)
    outcome = environment.run()

    output_dir = PROJECT_ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{args.scenario}_baseline_seed{args.seed}_results.csv"
    outcome["results"].to_csv(result_path, index=False)

    print("EVENT-DRIVEN BASELINE SIMULATION")
    print("-" * 60)
    for key, value in outcome["metrics"].items():
        print(f"{key}: {value}")
    print(f"Results saved: {result_path}")