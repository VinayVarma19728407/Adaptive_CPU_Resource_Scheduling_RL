import numpy as np
from experiments.bounded_evaluation_task_pool import load_bounded_eligible_task_pool

tasks = load_bounded_eligible_task_pool(
    max_cpu_capacity=96,
    max_memory_capacity=100,
    sample_size=50_000,
    seed=42,
)

durations = tasks["duration"].to_numpy(dtype=float)
median_duration = float(np.median(durations))
mean_gap = max(median_duration * 0.001, 0.01)

rng = np.random.default_rng(50001)
gaps = rng.exponential(scale=mean_gap, size=len(tasks))
arrivals = np.cumsum(gaps)
arrivals -= arrivals[0]

cpu = tasks["plan_cpu"].to_numpy(dtype=float) / 100.0
memory = tasks["plan_mem"].to_numpy(dtype=float)

print("\n=== 50,000-TASK WORKLOAD DIAGNOSTIC ===")
print(f"Task count: {len(tasks):,}")
print(f"Machine capacity: 96 CPU, 100 memory per machine")
print(f"Fleet capacity (4,000 machines): {4000 * 96:,} CPU, {4000 * 100:,} memory")
print(f"Median duration: {median_duration:.3f}")
print(f"Mean duration: {durations.mean():.3f}")
print(f"Mean inter-arrival gap: {mean_gap:.5f}")
print(f"Generated arrival window: {arrivals[-1]:.3f}")
print(f"Mean CPU request: {cpu.mean():.5f}")
print(f"Mean memory request: {memory.mean():.5f}")
print(f"Total requested CPU-time: {(cpu * durations).sum():,.3f}")
print(f"Total requested memory-time: {(memory * durations).sum():,.3f}")
print(f"Total CPU-time / fleet CPU-time capacity: {(cpu * durations).sum() / (4000 * 96 * max(arrivals[-1] + durations.max(), 1)):.8%}")
print(f"Total memory-time / fleet memory-time capacity: {(memory * durations).sum() / (4000 * 100 * max(arrivals[-1] + durations.max(), 1)):.8%}")
