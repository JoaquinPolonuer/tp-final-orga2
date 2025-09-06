#!/usr/bin/env python3
"""
Performance benchmark for different wave simulation backends
"""

import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

matplotlib.use("Agg")  # Use non-interactive backend
from backends.wave_simulation_numpy import NumpyWaveSimulation2D
from backends.wave_simulation_python import PythonWaveSimulation2D
from backends.wave_simulation_c import CWaveSimulation2D
from backends.wave_simulation_c_avx import OptimizedCWaveSimulation2D
from backends.wave_simulation_asm import ASMWaveSimulation2D
from backends.wave_simulation_asm_simd import ASMSIMDWaveSimulation2D


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
        elif backend_name == "c_avx":
            sim = OptimizedCWaveSimulation2D(size=size, domain_size=8.0, wave_speed=2.0, dt=0.02)
        elif backend_name == "asm":
            sim = ASMWaveSimulation2D(size=size, domain_size=8.0, wave_speed=2.0, dt=0.02)
        elif backend_name == "asm_simd":
            sim = ASMSIMDWaveSimulation2D(size=size, domain_size=8.0, wave_speed=2.0, dt=0.02)
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

        print(f"Results for {backend_name}:")
        print(f"  Initialization time: {init_time:.3f}s")
        print(f"  Simulation time: {simulation_time:.3f}s")
        print(f"  Steps per second: {steps_per_second:.1f}")

        return {
            "backend": backend_name,
            "init_time": init_time,
            "simulation_time": simulation_time,
            "steps_per_second": steps_per_second,
        }

    except ImportError as e:
        print(f"  ERROR: {backend_name} backend not available - {e}")
        return None
    except Exception as e:
        print(f"  ERROR: Failed to benchmark {backend_name} - {e}")
        return None


print("Wave Simulation Backend Performance Benchmark")
print("=" * 50)

backends = ["python", "numpy", "c", "c_avx", "asm_simd", "asm"]
sizes = [16, 32, 64, 128, 256, 512, 1024]
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
    print(f"{'Backend':<10} {'Steps/sec':<12} {'Speedup':<8}")
    print("-" * 35)

    if "numpy" in results[size]:
        numpy_sps = results[size]["numpy"]["steps_per_second"]
    else:
        numpy_sps = None

    for backend in backends:
        if backend in results[size]:
            result = results[size][backend]
            sps = result["steps_per_second"]

            if numpy_sps and backend != "numpy":
                speedup = f"{sps/numpy_sps:.1f}x"
            elif backend == "numpy":
                speedup = "baseline"
            else:
                speedup = "N/A"

            print(f"{backend:<10} {sps:<12.1f} {speedup:<8}")
        else:
            print(f"{backend:<10} {'FAILED':<12} {'N/A':<8}")

# Save results to CSV
print(f"\n{'='*20} SAVING RESULTS TO CSV {'='*20}")

# Ensure results directory exists
os.makedirs("/home/joacopolo/Documents/tp-final-orga2/results", exist_ok=True)

# Create DataFrame with results
data = {}
for backend in backends:
    data[backend] = []
    for size in sizes:
        if backend in results[size]:
            sps = results[size][backend]["steps_per_second"]
            data[backend].append(sps)
        else:
            data[backend].append(None)

# Create DataFrame
df = pd.DataFrame(data, index=[f"{size}x{size}" for size in sizes])

# Transpose so methods are rows and sizes are columns
df = df.T

# Sort by performance on largest size (descending)
largest_size_col = df.columns[-1]
df = df.sort_values(by=largest_size_col, na_position="last")

# Format the data for display (round to 1 decimal place, replace NaN with "FAILED")
df_formatted = df.round(1).fillna("FAILED")

# Save to CSV
df_formatted.to_csv("/home/joacopolo/Documents/tp-final-orga2/results/table_of_results.csv", index_label="Method")

print("Results saved to results/table_of_results.csv")

# Create plots
print(f"\n{'='*20} GENERATING PLOTS {'='*20}")

# Prepare data for plotting
plot_sizes = []
plot_data = {backend: {"steps_per_sec": []} for backend in backends}

for size in sizes:
    if results[size]:  # Only include sizes that have results
        plot_sizes.append(size)
        for backend in backends:
            if backend in results[size]:
                plot_data[backend]["steps_per_sec"].append(results[size][backend]["steps_per_second"])
            else:
                # Use NaN for failed backends to maintain array alignment
                plot_data[backend]["steps_per_sec"].append(np.nan)

# Plot 1: Steps per second
plt.figure(figsize=(12, 8))
colors = ["blue", "orange", "green", "red", "purple", "brown"]
markers = ["o", "s", "^", "D", "x", "*", "P"]

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


print("Plots saved to results/ directory:")
print("- steps_per_second.png")
