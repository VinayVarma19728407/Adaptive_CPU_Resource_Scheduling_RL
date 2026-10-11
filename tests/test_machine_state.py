
import unittest

import pandas as pd

from simulator.machine_state import MachineStateManager


def sample_metadata():
    return pd.DataFrame(
        [
            {
                "machine_id": "machine_A",
                "cpu_num": 96,
                "mem_size": 100,
                "status": 1,
            },
            {
                "machine_id": "machine_B",
                "cpu_num": 64,
                "mem_size": 80,
                "status": 1,
            },
        ]
    )


class TestMachineStateManager(unittest.TestCase):
    def setUp(self):
        self.manager = MachineStateManager(sample_metadata())

    def test_initial_capacity(self):
        machine = self.manager.get_machine("machine_A")

        self.assertEqual(machine.cpu_available, 96)
        self.assertEqual(machine.memory_available, 100)
        self.manager.validate_invariants()

    def test_successful_allocation(self):
        allocation_id = self.manager.allocate(
            "machine_A", 24, 30, "task_001"
        )

        machine = self.manager.get_machine("machine_A")
        self.assertEqual(machine.cpu_available, 72)
        self.assertEqual(machine.memory_available, 70)
        self.assertEqual(self.manager.active_allocation_count, 1)
        self.manager.validate_invariants()

    def test_insufficient_cpu_changes_nothing(self):
        before = self.manager.get_snapshot()

        with self.assertRaises(ValueError):
            self.manager.allocate("machine_A", 97, 10, "task_002")

        pd.testing.assert_frame_equal(before, self.manager.get_snapshot())
        self.assertEqual(self.manager.active_allocation_count, 0)
        self.manager.validate_invariants()

    def test_insufficient_memory_changes_nothing(self):
        before = self.manager.get_snapshot()

        with self.assertRaises(ValueError):
            self.manager.allocate("machine_A", 10, 101, "task_003")

        pd.testing.assert_frame_equal(before, self.manager.get_snapshot())
        self.assertEqual(self.manager.active_allocation_count, 0)
        self.manager.validate_invariants()

    def test_can_allocate(self):
        self.assertTrue(self.manager.can_allocate("machine_A", 24, 30))
        self.assertFalse(self.manager.can_allocate("machine_A", 97, 30))
        self.assertFalse(self.manager.can_allocate("machine_A", 24, 101))
        self.assertFalse(self.manager.can_allocate("machine_A", 0, 10))
        self.assertFalse(self.manager.can_allocate("unknown", 1, 1))

    def test_release_restores_exact_resources(self):
        allocation_id = self.manager.allocate(
            "machine_A", 24, 30, "task_004"
        )

        released = self.manager.release(allocation_id)

        machine = self.manager.get_machine("machine_A")
        self.assertEqual(released["task_id"], "task_004")
        self.assertEqual(machine.cpu_available, 96)
        self.assertEqual(machine.memory_available, 100)
        self.assertEqual(self.manager.active_allocation_count, 0)
        self.manager.validate_invariants()

    def test_cannot_release_twice(self):
        allocation_id = self.manager.allocate(
            "machine_A", 10, 10, "task_005"
        )
        self.manager.release(allocation_id)

        with self.assertRaises(KeyError):
            self.manager.release(allocation_id)

        self.manager.validate_invariants()

    def test_unknown_machine_is_rejected(self):
        with self.assertRaises(KeyError):
            self.manager.allocate("unknown", 1, 1, "task_006")

    def test_invalid_amounts_are_rejected(self):
        for cpu, memory in [
            (0, 1),
            (1, 0),
            (-1, 1),
            (1, -1),
            (float("inf"), 1),
            (1, float("nan")),
        ]:
            with self.subTest(cpu=cpu, memory=memory):
                with self.assertRaises(ValueError):
                    self.manager.allocate(
                        "machine_A", cpu, memory, "invalid_task"
                    )

        self.manager.validate_invariants()

    def test_multiple_allocations_and_releases(self):
        first = self.manager.allocate(
            "machine_A", 20, 25, "task_007"
        )
        second = self.manager.allocate(
            "machine_A", 30, 35, "task_008"
        )

        machine = self.manager.get_machine("machine_A")
        self.assertEqual(machine.cpu_available, 46)
        self.assertEqual(machine.memory_available, 40)

        self.manager.release(first)
        self.manager.release(second)

        machine = self.manager.get_machine("machine_A")
        self.assertEqual(machine.cpu_available, 96)
        self.assertEqual(machine.memory_available, 100)
        self.manager.validate_invariants()


if __name__ == "__main__":
    unittest.main(verbosity=2)
