import numpy as np
import pandas as pd

from experiments.bounded_evaluation_task_pool import (
    load_bounded_eligible_task_pool,
)

TASK_COUNT = 50_000
SAMPLE_SEED = 42
ARRIVAL_SEED = 50001
MACHINE_COUNT = 4000
CPU_PER_MACHINE = 96.0
MEMORY_PER_MACHINE = 100.0


def build_profiles(durations):
    rng = np.random.default_rng(ARRIVAL_SEED)
    n = len(durations)

    profiles = {}

    # Existing generator's minimum-gap behavior.
    for name, mean_gap in [
        ("low_exponential", 0.18),
        ("moderate_exponential", 0.045),
        ("high_exponential", 0.01),
    ]:
        gaps = rng.exponential(scale=mean_gap, size=n)
        arrivals = np.cumsum(gaps)
        arrivals -= arrivals[0]
        profiles[name] = arrivals

    # Bursty profile: groups of 100 arrivals, with small within-burst spread.
    # This is a synthetic stress scenario, not a historical arrival trace.
    burst_rng = np.random.default_rng(ARRIVAL_SEED)
    burst_size = 100
    burst_number = np.arange(n) // burst_size
    within_burst = burst_rng.uniform(0.0, 0.05, size=n)
    arrivals = burst_number.astype(float) * 1.0 + within_burst
    arrivals -= arrivals.min()
    profiles["bursty_100_per_unit"] = arrivals

    return profiles


def measure_profile(name, arrivals, durations, cpu, memory):
    # Each task contributes a start and an end event.
    # End events are processed before start events at equal timestamps.
    times = np.concatenate([arrivals, arrivals + durations])
    kinds = np.concatenate([
        np.ones(len(arrivals), dtype=np.int8),
        np.zeros(len(arrivals), dtype=np.int8),
    ])
    task_delta = np.concatenate([
        np.ones(len(arrivals), dtype=np.int64),
        -np.ones(len(arrivals), dtype=np.int64),
    ])
    cpu_delta = np.concatenate([cpu, -cpu])
    memory_delta = np.concatenate([memory, -memory])

    order = np.lexsort((kinds, times))
    times = times[order]
    task_delta = task_delta[order]
    cpu_delta = cpu_delta[order]
    memory_delta = memory_delta[order]

    active_tasks = np.cumsum(task_delta)
    active_cpu = np.cumsum(cpu_delta)
    active_memory = np.cumsum(memory_delta)

    peak_tasks_index = int(np.argmax(active_tasks))
    peak_cpu_index = int(np.argmax(active_cpu))
    peak_memory_index = int(np.argmax(active_memory))

    fleet_cpu = MACHINE_COUNT * CPU_PER_MACHINE
    fleet_memory = MACHINE_COUNT * MEMORY_PER_MACHINE

    return {
        "profile": name,
        "tasks": len(arrivals),
        "arrival_window": float(arrivals.max() - arrivals.min()),
        "peak_concurrent_tasks": int(active_tasks[peak_tasks_index]),
        "peak_concurrency_time": float(times[peak_tasks_index]),
        "peak_cpu_demand": float(active_cpu[peak_cpu_index]),
        "peak_cpu_time": float(times[peak_cpu_index]),
        "peak_cpu_pct_of_fleet": float(
            100.0 * active_cpu[peak_cpu_index] / fleet_cpu
        ),
        "peak_memory_demand": float(active_memory[peak_memory_index]),
        "peak_memory_time": float(times[peak_memory_index]),
        "peak_memory_pct_of_fleet": float(
            100.0 * active_memory[peak_memory_index] / fleet_memory
        ),
        "tasks_overlapping_cpu_capacity": bool(
            active_cpu[peak_cpu_index] > fleet_cpu
        ),
        "tasks_overlapping_memory_capacity": bool(
            active_memory[peak_memory_index] > fleet_memory
        ),
    }


def main():
    print("Loading the reproducible eligible task sample...")
    tasks = load_bounded_eligible_task_pool(
        max_cpu_capacity=CPU_PER_MACHINE,
        max_memory_capacity=MEMORY_PER_MACHINE,
        sample_size=TASK_COUNT,
        seed=SAMPLE_SEED,
    )

    durations = tasks["duration"].to_numpy(dtype=float)
    cpu = tasks["plan_cpu"].to_numpy(dtype=float) / 100.0
    memory = tasks["plan_mem"].to_numpy(dtype=float)

    if not (
        np.isfinite(durations).all()
        and np.isfinite(cpu).all()
        and np.isfinite(memory).all()
        and (durations > 0).all()
        and (cpu > 0).all()
        and (memory > 0).all()
    ):
        raise ValueError("Sample contains invalid durations or resource requests.")

    print(f"Sample size: {len(tasks):,}")
    print(f"Sample seed: {SAMPLE_SEED}")
    print(f"Median duration: {np.median(durations):.4f}")
    print(f"Fleet: {MACHINE_COUNT:,} machines")
    print(f"Fleet CPU capacity: {MACHINE_COUNT * CPU_PER_MACHINE:,.1f}")
    print(f"Fleet memory capacity: {MACHINE_COUNT * MEMORY_PER_MACHINE:,.1f}")
    print("\n=== SYNTHETIC WORKLOAD PROFILE CALIBRATION ===")
    print("Demand figures are aggregate requests, not measured machine utilization.")
    print("Burst arrivals are synthetic and are not verified historical arrivals.\n")

    profiles = build_profiles(durations)
    results = []

    for name, arrivals in profiles.items():
        result = measure_profile(
            name, arrivals, durations, cpu, memory
        )
        results.append(result)

        print(f"--- {name} ---")
        print(f"Arrival window: {result['arrival_window']:.3f}")
        print(f"Peak concurrent tasks: {result['peak_concurrent_tasks']:,}")
        print(f"Peak CPU demand: {result['peak_cpu_demand']:,.3f}")
        print(f"Peak CPU as fleet capacity: {result['peak_cpu_pct_of_fleet']:.4f}%")
        print(f"Peak memory demand: {result['peak_memory_demand']:,.3f}")
        print(
            "Peak memory as fleet capacity: "
            f"{result['peak_memory_pct_of_fleet']:.4f}%"
        )
        print(
            "Aggregate CPU capacity exceeded: "
            f"{result['tasks_overlapping_cpu_capacity']}"
        )
        print(
            "Aggregate memory capacity exceeded: "
            f"{result['tasks_overlapping_memory_capacity']}"
        )
        print()

    output = pd.DataFrame(results)
    output_path = (
        "experiments/workload_profile_calibration_50000_seed42.csv"
    )
    output.to_csv(output_path, index=False)
    print(f"Calibration results saved to: {output_path}")
    print("No scheduler was run and no existing evaluation result was modified.")


if __name__ == "__main__":
    main()
