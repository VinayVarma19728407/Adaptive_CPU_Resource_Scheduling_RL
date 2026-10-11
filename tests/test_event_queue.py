
import unittest

from simulator.event_queue import EventQueue


class TestEventQueue(unittest.TestCase):
    def setUp(self):
        self.queue = EventQueue()

    def test_events_are_ordered_by_time(self):
        self.queue.push(16, "arrival", "task_3")
        self.queue.push(4, "arrival", "task_1")
        self.queue.push(8, "completion", "task_2")

        self.assertEqual(
            [self.queue.pop().time for _ in range(3)],
            [4.0, 8.0, 16.0],
        )

    def test_equal_time_events_preserve_insertion_order(self):
        self.queue.push(8, "arrival", "task_1")
        self.queue.push(8, "completion", "task_2")
        self.queue.push(8, "arrival", "task_3")

        self.assertEqual(
            [self.queue.pop().task_id for _ in range(3)],
            ["task_1", "task_2", "task_3"],
        )

    def test_peek_does_not_remove_event(self):
        self.queue.push(5, "arrival", "task_1")

        self.assertEqual(self.queue.peek().task_id, "task_1")
        self.assertEqual(len(self.queue), 1)

    def test_empty_queue_errors(self):
        with self.assertRaises(IndexError):
            self.queue.pop()

        with self.assertRaises(IndexError):
            self.queue.peek()

    def test_invalid_times_are_rejected(self):
        for time in [-1, float("inf"), float("nan"), "invalid"]:
            with self.subTest(time=time):
                with self.assertRaises(ValueError):
                    self.queue.push(time, "arrival", "task_1")

    def test_invalid_event_type_is_rejected(self):
        with self.assertRaises(ValueError):
            self.queue.push(1, "unknown", "task_1")

    def test_invalid_task_id_is_rejected(self):
        with self.assertRaises(ValueError):
            self.queue.push(1, "arrival", "")

    def test_clear_resets_queue(self):
        self.queue.push(2, "arrival", "task_1")
        self.queue.clear()

        self.assertTrue(self.queue.is_empty())
        self.assertEqual(len(self.queue), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
