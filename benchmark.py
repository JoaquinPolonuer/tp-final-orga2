#!/usr/bin/env python3
"""
Performance benchmark for different wave simulation backends
"""

import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # Use non-interactive backend
from backends.wave_simulation_numpy import NumpyWaveSimulation2D
from backends.wave_simulation_python import PythonWaveSimulation2D
from backends.wave_simulation_c import CWaveSimulation2D
from backends.wave_simulation_c_optimized import OptimizedCWaveSimulation2D
from backends.wave_simulation_asm import ASMWaveSimulation2D
from backends.wave_simulation_asm_avx import ASMAVXWaveSimulation2D

def benchmark_backend(backend_name, size=64, steps=100):
    """Benchmark a specific backend"""
    print(f"\nBenchmarking {backend_name} backend...")
    print(f"Grid size: {size}x{size}, Steps: {steps}")

    try:
        # Create simulation
        start_time = time.time()
        if backend_name == "numpy":
            sim = NumpyWaveSimulation2D(size=size, domain_size=8.0, wave_speed=2.0, dt=0.02)
        elif backend_name == "python":
            sim = PythonWaveSimulation2D(size=size, domain_size=8.0, wave_speed=2.0, dt=0.02)
        elif backend_name == "c":
            sim = CWaveSimulation2D(size=size, domain_size=8.0, wave_speed=2.0, dt=0.02)
        elif backend_name == "optimized_c":
            sim = OptimizedCWaveSimulation2D(size=size, domain_size=8.0, wave_speed=2.0, dt=0.02)
        elif backend_name == "asm":
            sim = ASMWaveSimulation2D(size=size, domain_size=8.0, wave_speed=2.0, dt=0.02)
        elif backend_name == "asm_avx":
            sim = ASMAVXWaveSimulation2D(size=size, domain_size=8.0, wave_speed=2.0, dt=0.02)        
        else:
            raise ImportError(f"Unknown backend: {backend_name}")
        
        init_time = time.time() - start_time

        # Add a wave source
        sim.add_wave_source(0, 0, amplitude=1.0, frequency=3.0)

        # Run simulation steps
        start_time = time.time()
        for i in range(steps):
            sim.step()
            if (i + 1) % 20 == 0:
                print(f"  Step {i + 1}/{steps}")

        simulation_time = time.time() - start_time

        # Calculate metrics
        steps_per_second = steps / simulation_time
        time_per_step = simulation_time / steps * 1000  # ms

        print(f"Results for {backend_name}:")
        print(f"  Initialization time: {init_time:.3f}s")
        print(f"  Simulation time: {simulation_time:.3f}s")
        print(f"  Steps per second: {steps_per_second:.1f}")
        print(f"  Time per step: {time_per_step:.2f}ms")

        return {
            "backend": backend_name,
            "init_time": init_time,
            "simulation_time": simulation_time,
            "steps_per_second": steps_per_second,
            "time_per_step": time_per_step,
        }

    except ImportError as e:
        print(f"  ERROR: {backend_name} backend not available - {e}")
        return None
    except Exception as e:
        print(f"  ERROR: Failed to benchmark {backend_name} - {e}")
        return None


print("Wave Simulation Backend Performance Benchmark")
print("=" * 50)

backends = ["asm_avx", "asm"]
sizes = [16, 32, 64, 128, 256, 512]
steps = 20

results = {}

for size in sizes:
    print(f"\n{'='*20} GRID SIZE {size}x{size} {'='*20}")
    results[size] = {}

    for backend in backends:
        result = benchmark_backend(backend, size=size, steps=steps)
        if result:
            results[size][backend] = result

# Print summary
print(f"\n{'='*20} PERFORMANCE SUMMARY {'='*20}")

for size in sizes:
    print(f"\nGrid Size {size}x{size}:")
    print(f"{'Backend':<10} {'Steps/sec':<12} {'ms/step':<10} {'Speedup':<8}")
    print("-" * 45)

    if "numpy" in results[size]:
        numpy_sps = results[size]["numpy"]["steps_per_second"]
    else:
        numpy_sps = None

    for backend in backends:
        if backend in results[size]:
            result = results[size][backend]
            sps = result["steps_per_second"]
            ms_per_step = result["time_per_step"]

            if numpy_sps and backend != "numpy":
                speedup = f"{sps/numpy_sps:.1f}x"
            elif backend == "numpy":
                speedup = "baseline"
            else:
                speedup = "N/A"

            print(f"{backend:<10} {sps:<12.1f} {ms_per_step:<10.2f} {speedup:<8}")
        else:
            print(f"{backend:<10} {'FAILED':<12} {'FAILED':<10} {'N/A':<8}")

# Create plots
print(f"\n{'='*20} GENERATING PLOTS {'='*20}")

# Prepare data for plotting
plot_sizes = []
plot_data = {backend: {"steps_per_sec": [], "ms_per_step": []} for backend in backends}

for size in sizes:
    if results[size]:  # Only include sizes that have results
        plot_sizes.append(size)
        for backend in backends:
            if backend in results[size]:
                plot_data[backend]["steps_per_sec"].append(results[size][backend]["steps_per_second"])
                plot_data[backend]["ms_per_step"].append(results[size][backend]["time_per_step"])
            else:
                # Use NaN for failed backends to maintain array alignment
                plot_data[backend]["steps_per_sec"].append(np.nan)
                plot_data[backend]["ms_per_step"].append(np.nan)

# Plot 1: Steps per second
plt.figure(figsize=(12, 8))
colors = ["blue", "orange", "green", "red", "purple"]
markers = ["o", "s", "^", "D", "x"]

for i, backend in enumerate(backends):
    steps_data = plot_data[backend]["steps_per_sec"]
    # Only plot if we have valid data
    valid_indices = ~np.isnan(steps_data)
    if np.any(valid_indices):
        valid_sizes = np.array(plot_sizes)[valid_indices]
        valid_steps = np.array(steps_data)[valid_indices]
        plt.plot(
            valid_sizes,
            valid_steps,
            marker=markers[i],
            color=colors[i],
            linewidth=2,
            markersize=8,
            label=backend,
        )

plt.xlabel("Grid Size (N×N)", fontsize=12)
plt.ylabel("Steps per Second", fontsize=12)
plt.title("Performance Comparison: Steps per Second vs Grid Size", fontsize=14, fontweight="bold")
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.yscale("log")
plt.xscale("log")
plt.xticks(plot_sizes, [f"{s}×{s}" for s in plot_sizes])
plt.tight_layout()
plt.savefig(
    "/home/joacopolo/Documents/tp-final-orga2/results/steps_per_second.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# Plot 2: Milliseconds per step
plt.figure(figsize=(12, 8))

for i, backend in enumerate(backends):
    ms_data = plot_data[backend]["ms_per_step"]
    # Only plot if we have valid data
    valid_indices = ~np.isnan(ms_data)
    if np.any(valid_indices):
        valid_sizes = np.array(plot_sizes)[valid_indices]
        valid_ms = np.array(ms_data)[valid_indices]
        plt.plot(
            valid_sizes,
            valid_ms,
            marker=markers[i],
            color=colors[i],
            linewidth=2,
            markersize=8,
            label=backend,
        )

plt.xlabel("Grid Size (N×N)", fontsize=12)
plt.ylabel("Milliseconds per Step", fontsize=12)
plt.title("Performance Comparison: Time per Step vs Grid Size", fontsize=14, fontweight="bold")
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.yscale("log")
plt.xscale("log")
plt.xticks(plot_sizes, [f"{s}×{s}" for s in plot_sizes])
plt.tight_layout()
plt.savefig(
    "/home/joacopolo/Documents/tp-final-orga2/results/milliseconds_per_step.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# Combined plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Steps per second subplot
for i, backend in enumerate(backends):
    steps_data = plot_data[backend]["steps_per_sec"]
    valid_indices = ~np.isnan(steps_data)
    if np.any(valid_indices):
        valid_sizes = np.array(plot_sizes)[valid_indices]
        valid_steps = np.array(steps_data)[valid_indices]
        ax1.plot(
            valid_sizes,
            valid_steps,
            marker=markers[i],
            color=colors[i],
            linewidth=2,
            markersize=8,
            label=backend,
        )

ax1.set_xlabel("Grid Size (N×N)", fontsize=12)
ax1.set_ylabel("Steps per Second", fontsize=12)
ax1.set_title("Steps per Second vs Grid Size", fontsize=12, fontweight="bold")
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_yscale("log")
ax1.set_xscale("log")
ax1.set_xticks(plot_sizes)
ax1.set_xticklabels([f"{s}×{s}" for s in plot_sizes])

# Milliseconds per step subplot
for i, backend in enumerate(backends):
    ms_data = plot_data[backend]["ms_per_step"]
    valid_indices = ~np.isnan(ms_data)
    if np.any(valid_indices):
        valid_sizes = np.array(plot_sizes)[valid_indices]
        valid_ms = np.array(ms_data)[valid_indices]
        ax2.plot(
            valid_sizes,
            valid_ms,
            marker=markers[i],
            color=colors[i],
            linewidth=2,
            markersize=8,
            label=backend,
        )

ax2.set_xlabel("Grid Size (N×N)", fontsize=12)
ax2.set_ylabel("Milliseconds per Step", fontsize=12)
ax2.set_title("Time per Step vs Grid Size", fontsize=12, fontweight="bold")
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_yscale("log")
ax2.set_xscale("log")
ax2.set_xticks(plot_sizes)
ax2.set_xticklabels([f"{s}×{s}" for s in plot_sizes])

plt.tight_layout()
plt.savefig(
    "/home/joacopolo/Documents/tp-final-orga2/results/combined_performance.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

print("Plots saved to results/ directory:")
print("- steps_per_second.png")
print("- milliseconds_per_step.png")
print("- combined_performance.png")
