
from dataclasses import dataclass, field
from itertools import count
from typing import Any, Optional
import heapq
import math


@dataclass(order=True)
class Event:
    """An event ordered by simulation time and insertion sequence."""

    time: float
    sequence: int
    event_type: str = field(compare=False)
    task_id: str = field(compare=False)
    payload: Optional[Any] = field(default=None, compare=False)


class EventQueue:
    """
    Priority queue for simulation events.

    Supported event types:
      - arrival
      - completion

    Events at the same time are processed in insertion order.
    """

    VALID_EVENT_TYPES = {"arrival", "completion"}

    def __init__(self):
        self._heap = []
        self._sequence = count()

    def __len__(self) -> int:
        return len(self._heap)

    def is_empty(self) -> bool:
        return not self._heap

    def push(
        self,
        time: float,
        event_type: str,
        task_id: str,
        payload: Optional[Any] = None,
    ) -> None:
        """Insert a validated event into the queue."""

        try:
            time = float(time)
        except (TypeError, ValueError) as exc:
            raise ValueError("Event time must be numeric.") from exc

        if not math.isfinite(time) or time < 0:
            raise ValueError("Event time must be finite and nonnegative.")

        if event_type not in self.VALID_EVENT_TYPES:
            raise ValueError(f"Unsupported event type: {event_type}")

        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("task_id must be a nonempty string.")

        event = Event(
            time=time,
            sequence=next(self._sequence),
            event_type=event_type,
            task_id=task_id,
            payload=payload,
        )
        heapq.heappush(self._heap, event)

    def peek(self) -> Event:
        """Return the next event without removing it."""

        if self.is_empty():
            raise IndexError("Cannot peek at an empty event queue.")

        return self._heap[0]

    def pop(self) -> Event:
        """Remove and return the earliest event."""

        if self.is_empty():
            raise IndexError("Cannot pop from an empty event queue.")

        return heapq.heappop(self._heap)

    def clear(self) -> None:
        """Remove all queued events and reset insertion ordering."""

        self._heap.clear()
        self._sequence = count()


if __name__ == "__main__":
    queue = EventQueue()

    queue.push(16, "arrival", "task_2")
    queue.push(8, "arrival", "task_1")
    queue.push(8, "completion", "task_0")

    print("EVENT QUEUE DEMONSTRATION")
    print("-" * 40)

    while not queue.is_empty():
        event = queue.pop()
        print(
            f"time={event.time:g}, "
            f"type={event.event_type}, "
            f"task={event.task_id}"
        )
