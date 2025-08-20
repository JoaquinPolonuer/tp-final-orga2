#!/usr/bin/env python3
"""
Performance benchmark for different wave simulation backends
"""

import time
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from main import WaveSimulation2D

def benchmark_backend(backend_name, size=64, steps=100):
    """Benchmark a specific backend"""
    print(f"\nBenchmarking {backend_name} backend...")
    print(f"Grid size: {size}x{size}, Steps: {steps}")
    
    try:
        # Create simulation
        start_time = time.time()
        sim = WaveSimulation2D(
            backend=backend_name,
            size=size,
            domain_size=8.0,
            wave_speed=2.0,
            dt=0.02
        )
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
            'backend': backend_name,
            'init_time': init_time,
            'simulation_time': simulation_time,
            'steps_per_second': steps_per_second,
            'time_per_step': time_per_step
        }
        
    except ImportError as e:
        print(f"  ERROR: {backend_name} backend not available - {e}")
        return None
    except Exception as e:
        print(f"  ERROR: Failed to benchmark {backend_name} - {e}")
        return None

def main():
    print("Wave Simulation Backend Performance Benchmark")
    print("=" * 50)
    
    backends = ['numpy', 'python', 'c']
    sizes = [64, 128]
    steps = 50
    
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
        
        if 'numpy' in results[size]:
            numpy_sps = results[size]['numpy']['steps_per_second']
        else:
            numpy_sps = None
            
        for backend in backends:
            if backend in results[size]:
                result = results[size][backend]
                sps = result['steps_per_second']
                ms_per_step = result['time_per_step']
                
                if numpy_sps and backend != 'numpy':
                    speedup = f"{sps/numpy_sps:.1f}x"
                elif backend == 'numpy':
                    speedup = "baseline"
                else:
                    speedup = "N/A"
                
                print(f"{backend:<10} {sps:<12.1f} {ms_per_step:<10.2f} {speedup:<8}")
            else:
                print(f"{backend:<10} {'FAILED':<12} {'FAILED':<10} {'N/A':<8}")

if __name__ == "__main__":
    main()