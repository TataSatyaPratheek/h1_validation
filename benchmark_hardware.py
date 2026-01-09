"""
Phase 1: Hardware Capability Benchmark
Find the maximum qubit count and circuit depth this M4 Mac can handle.
"""
import torch
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorcircuit as tc
from utils import setup_mps_environment

K = tc.set_backend("pytorch")

def benchmark_qubits(max_qubits=16, samples=10, depth=1):
    """Test how many qubits we can simulate in reasonable time."""
    results = []
    
    for n_qubits in range(2, max_qubits + 1, 2):
        print(f"Testing {n_qubits} qubits, depth {depth}...")
        
        try:
            times = []
            for _ in range(samples):
                inputs = torch.randn(n_qubits)
                
                start = time.time()
                c = tc.Circuit(n_qubits)
                
                # Encoding layer
                for i in range(n_qubits):
                    c.ry(i, theta=inputs[i])
                
                # Entanglement layers
                for d in range(depth):
                    for i in range(n_qubits):
                        c.cnot(i, (i + 1) % n_qubits)
                
                # Measure all qubits
                exps = [c.expectation([tc.gates.z(), [i]]) for i in range(n_qubits)]
                _ = torch.stack([e.real for e in exps])
                
                elapsed = time.time() - start
                times.append(elapsed)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            memory = 2 ** n_qubits * 16 / 1024 / 1024  # MB (complex128)
            
            results.append({
                "qubits": n_qubits,
                "depth": depth,
                "avg_time_ms": avg_time * 1000,
                "std_time_ms": std_time * 1000,
                "memory_mb": memory,
                "status": "ok"
            })
            print(f"  -> {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms, ~{memory:.2f}MB")
            
            # Stop if too slow
            if avg_time > 5.0:
                print(f"  -> Too slow, stopping qubit scaling")
                break
                
        except Exception as e:
            results.append({
                "qubits": n_qubits,
                "depth": depth,
                "status": "failed",
                "error": str(e)
            })
            print(f"  -> FAILED: {e}")
            break
    
    return results

def benchmark_depth(n_qubits=4, max_depth=32, samples=10):
    """Test how deep a circuit we can run."""
    results = []
    
    for depth in [1, 2, 4, 8, 16, 32]:
        if depth > max_depth:
            break
            
        print(f"Testing depth {depth} with {n_qubits} qubits...")
        
        try:
            times = []
            for _ in range(samples):
                inputs = torch.randn(n_qubits)
                
                start = time.time()
                c = tc.Circuit(n_qubits)
                
                for i in range(n_qubits):
                    c.ry(i, theta=inputs[i])
                
                for d in range(depth):
                    for i in range(n_qubits):
                        c.ry(i, theta=inputs[i] * (d + 1))
                    for i in range(n_qubits):
                        c.cnot(i, (i + 1) % n_qubits)
                
                exps = [c.expectation([tc.gates.z(), [i]]) for i in range(n_qubits)]
                _ = torch.stack([e.real for e in exps])
                
                elapsed = time.time() - start
                times.append(elapsed)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            results.append({
                "qubits": n_qubits,
                "depth": depth,
                "avg_time_ms": avg_time * 1000,
                "std_time_ms": std_time * 1000,
                "status": "ok"
            })
            print(f"  -> {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
            
        except Exception as e:
            results.append({
                "qubits": n_qubits,
                "depth": depth,
                "status": "failed",
                "error": str(e)
            })
            print(f"  -> FAILED: {e}")
            break
    
    return results

def benchmark_combined(max_qubits=12, max_depth=16, samples=5):
    """Create a heatmap of qubits × depth → time."""
    qubit_range = list(range(2, max_qubits + 1, 2))
    depth_range = [1, 2, 4, 8, 16]
    
    heatmap = np.zeros((len(depth_range), len(qubit_range)))
    
    for i, depth in enumerate(depth_range):
        for j, n_qubits in enumerate(qubit_range):
            print(f"Benchmarking {n_qubits} qubits × {depth} depth...", end=" ")
            
            try:
                times = []
                for _ in range(samples):
                    inputs = torch.randn(n_qubits)
                    
                    start = time.time()
                    c = tc.Circuit(n_qubits)
                    
                    for q in range(n_qubits):
                        c.ry(q, theta=inputs[q])
                    
                    for d in range(depth):
                        for q in range(n_qubits):
                            c.cnot(q, (q + 1) % n_qubits)
                        for q in range(n_qubits):
                            c.ry(q, theta=inputs[q])
                    
                    exps = [c.expectation([tc.gates.z(), [q]]) for q in range(n_qubits)]
                    _ = torch.stack([e.real for e in exps])
                    
                    elapsed = time.time() - start
                    times.append(elapsed)
                
                avg_time = np.mean(times) * 1000  # ms
                heatmap[i, j] = avg_time
                print(f"{avg_time:.1f}ms")
                
            except Exception as e:
                heatmap[i, j] = np.nan
                print(f"FAILED")
    
    return heatmap, qubit_range, depth_range

def plot_results(heatmap, qubit_range, depth_range, save_path):
    """Create visualization of hardware limits."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Replace inf/nan with max for visualization
    plot_data = np.nan_to_num(heatmap, nan=np.nanmax(heatmap) * 1.5)
    
    im = ax.imshow(plot_data, cmap='RdYlGn_r', aspect='auto')
    
    ax.set_xticks(range(len(qubit_range)))
    ax.set_xticklabels(qubit_range)
    ax.set_yticks(range(len(depth_range)))
    ax.set_yticklabels(depth_range)
    
    ax.set_xlabel("Number of Qubits")
    ax.set_ylabel("Circuit Depth (layers)")
    ax.set_title("M4 Mac Hardware Limits: Time per Forward Pass (ms)")
    
    # Add text annotations
    for i in range(len(depth_range)):
        for j in range(len(qubit_range)):
            val = heatmap[i, j]
            if np.isnan(val):
                text = "X"
            elif val > 1000:
                text = f"{val/1000:.1f}s"
            else:
                text = f"{val:.0f}ms"
            color = "white" if val > np.nanmedian(heatmap) else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)
    
    # Add "safe zone" boundary (< 100ms)
    plt.colorbar(im, label="Time (ms)")
    
    # Draw safe zone
    safe_mask = heatmap < 100
    for i in range(len(depth_range)):
        for j in range(len(qubit_range)):
            if safe_mask[i, j]:
                rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, 
                                      edgecolor='green', linewidth=2)
                ax.add_patch(rect)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()

def main():
    setup_mps_environment()
    print("="*60)
    print("M4 Mac Hardware Capability Benchmark")
    print("="*60)
    
    # Quick qubit scaling test
    print("\n--- Qubit Scaling (depth=1) ---")
    qubit_results = benchmark_qubits(max_qubits=16, samples=5, depth=1)
    
    # Depth scaling test
    print("\n--- Depth Scaling (qubits=4) ---")
    depth_results = benchmark_depth(n_qubits=4, max_depth=32, samples=5)
    
    # Combined heatmap
    print("\n--- Combined Benchmark (Heatmap) ---")
    heatmap, qubit_range, depth_range = benchmark_combined(max_qubits=12, max_depth=16, samples=3)
    
    # Save results
    results = {
        "qubit_scaling": qubit_results,
        "depth_scaling": depth_results,
        "heatmap": heatmap.tolist(),
        "qubit_range": qubit_range,
        "depth_range": depth_range
    }
    
    with open("results/hardware_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: results/hardware_benchmark.json")
    
    # Plot
    plot_results(heatmap, qubit_range, depth_range, "results/hardware_limits.png")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Safe Operating Zone for M4 Mac")
    print("="*60)
    
    safe_configs = []
    for i, depth in enumerate(depth_range):
        for j, qubits in enumerate(qubit_range):
            if heatmap[i, j] < 100:  # < 100ms is "safe"
                safe_configs.append((qubits, depth, heatmap[i, j]))
    
    print("\nConfigurations under 100ms:")
    for q, d, t in sorted(safe_configs, key=lambda x: x[2]):
        print(f"  {q} qubits × {d} depth: {t:.1f}ms")
    
    max_qubits_safe = max(c[0] for c in safe_configs) if safe_configs else 4
    max_depth_safe = max(c[1] for c in safe_configs) if safe_configs else 1
    
    print(f"\nRecommended limits:")
    print(f"  Max Qubits: {max_qubits_safe}")
    print(f"  Max Depth: {max_depth_safe}")
    print(f"  Safe Zone: {len(safe_configs)} configurations")

if __name__ == "__main__":
    main()
