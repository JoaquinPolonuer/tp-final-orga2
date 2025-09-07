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


def benchmark_backend(backend_name, size=64, steps=100):
    print(f"\nBenchmarking {backend_name} backend...")
    print(f"Grid size: {size}x{size}, Steps: {steps}")

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


def create_performance_dataframe(backends, sizes, steps=20):
    df = pd.DataFrame(index=backends, columns=sizes, dtype=float)

    for backend in backends:
        for size in sizes:
            result = benchmark_backend(backend, size=size, steps=steps)
            df.loc[backend, size] = result["steps_per_second"]

    return df


def print_performance_summary(df):
    print(f"\n{'='*20} PERFORMANCE SUMMARY {'='*20}")

    for size in df.columns:
        print(f"\nGrid Size {size}x{size}:")
        print(f"{'Backend':<10} {'Steps/sec':<12} {'Speedup':<8}")
        print("-" * 35)

        numpy_sps = df.loc["numpy", size]
        size_data = df[size].sort_values(ascending=False, na_position="last")

        for backend in size_data.index:
            sps = size_data[backend]

            if backend == "numpy":
                speedup = "baseline"
            else:
                speedup = f"{sps/numpy_sps:.1f}x"

            print(f"{backend:<10} {sps:<12.1f} {speedup:<8}")


def save_results_to_csv(df, output_dir="/home/joacopolo/Documents/tp-final-orga2/results"):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "table_of_results.csv")

    # Sort by performance on largest size (descending)
    largest_size = df.columns[-1]
    df.sort_values(by=largest_size, na_position="last").round(1).to_csv(csv_path, index_label="Method")

    print(f"Results saved to {csv_path}")
    return csv_path


def create_performance_plots(df, output_dir="/home/joacopolo/Documents/tp-final-orga2/results"):
    print(f"\n{'='*20} GENERATING PLOTS {'='*20}")

    # Ensure results directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Steps per second (all sizes, all backends)
    plt.figure(figsize=(12, 8))
    colors = ["blue", "orange", "green", "red", "purple", "brown"]
    markers = ["o", "s", "^", "D", "x", "*", "P"]

    for i, backend in enumerate(df.index):
        plt.plot(
            np.log(df.columns),
            df.loc[backend].values,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            linewidth=2,
            markersize=8,
            label=backend,
        )

    plt.xlabel("Tamaño de la grilla (N×N)", fontsize=12)
    plt.ylabel("Iteraciones por segundo", fontsize=12)
    plt.title(
        "Comparación de rendimiento: Tamaño de la grilla vs Iteraciones por segundo",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(np.log(df.columns), [f"{s}×{s}" for s in df.columns])
    plt.yscale("log")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "steps_per_second.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 2: Only 3 largest sizes, excluding 'python'
    largest_sizes = df.columns[-3:]
    backends_no_python = [b for b in df.index if b != "python"]

    plt.figure(figsize=(10, 6))
    for i, backend in enumerate(backends_no_python):
        plt.plot(
            np.log(largest_sizes),
            df.loc[backend, largest_sizes].values,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            linewidth=2,
            markersize=8,
            label=backend,
        )

    plt.xlabel("Tamaño de la grilla (N×N)", fontsize=12)
    plt.ylabel("Iteraciones por segundo", fontsize=12)
    plt.title(
        "Rendimiento para grillas grandes (excluyendo Python)",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(np.log(largest_sizes), [f"{s}×{s}" for s in largest_sizes])
    plt.yscale("log")
    plt.tight_layout()

    plot_path_top3 = os.path.join(output_dir, "steps_per_second_top3.png")
    plt.savefig(plot_path_top3, dpi=300, bbox_inches="tight")
    plt.close()

    return plot_path


if __name__ == "__main__":
    backends = ["numpy", "c", "c_avx", "asm", "python"]
    sizes = [16, 32, 64, 128, 256, 512, 1024]
    steps = 50

    # if not os.path.exists("results/table_of_results.csv"):
    df = create_performance_dataframe(backends, sizes, steps)
    # else:
    #     df = pd.read_csv("results/table_of_results.csv", index_col="Method")
    #     df.columns = [int(col) for col in df.columns]

    print_performance_summary(df)

    save_results_to_csv(df)

    create_performance_plots(df)

    print(f"\n{'='*20} BENCHMARK COMPLETE {'='*20}")
    print("All results saved to results/ directory")
