"""
Phase 4: Comparative Circuit Analysis
Compare different quantum circuit architectures visually.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import tensorcircuit as tc
from utils import setup_mps_environment

K = tc.set_backend("pytorch")

# ============= CIRCUIT DEFINITIONS =============

def simple_ring_circuit(inputs, n_qubits=4):
    """Simple Ry encoding + CNOT ring."""
    c = tc.Circuit(n_qubits)
    for i in range(n_qubits):
        c.ry(i, theta=inputs[i])
    for i in range(n_qubits):
        c.cnot(i, (i + 1) % n_qubits)
    return c

def circuit_14(inputs, n_qubits=4, depth=2):
    """High expressivity ansatz (Sim et al. Circuit 14)."""
    c = tc.Circuit(n_qubits)
    for d in range(depth):
        for i in range(n_qubits):
            c.ry(i, theta=inputs[i] * (d + 1))
            c.rz(i, theta=inputs[i] * 0.5 * (d + 1))
        for i in range(n_qubits):
            c.cnot(i, (i + 1) % n_qubits)
        for i in range(0, n_qubits - 1, 2):
            if i + 1 < n_qubits:
                c.cnot(i, i + 1)
    return c

def matchgate_circuit(inputs, n_qubits=4):
    """Matchgate (fermionic) circuit."""
    c = tc.Circuit(n_qubits)
    for i in range(n_qubits):
        c.ry(i, theta=inputs[i])
    # Matchgate pairs (only nearest-neighbor XX+YY)
    for i in range(0, n_qubits - 1, 2):
        theta = inputs[i] * 0.5
        c.rxx(i, i + 1, theta=theta)
        c.ryy(i, i + 1, theta=theta)
    for i in range(1, n_qubits - 1, 2):
        theta = inputs[i] * 0.5
        c.rxx(i, i + 1, theta=theta)
        c.ryy(i, i + 1, theta=theta)
    return c

def linear_entanglement_circuit(inputs, n_qubits=4):
    """Linear chain entanglement (no ring)."""
    c = tc.Circuit(n_qubits)
    for i in range(n_qubits):
        c.ry(i, theta=inputs[i])
    for i in range(n_qubits - 1):
        c.cnot(i, i + 1)
    return c

CIRCUITS = {
    "Simple Ring": simple_ring_circuit,
    "Circuit 14": circuit_14,
    "Matchgate": matchgate_circuit,
    "Linear Chain": linear_entanglement_circuit,
}

# ============= ANALYSIS FUNCTIONS =============

def get_expectations(circuit, n_qubits=4):
    """Get all single and two-body expectations."""
    single = [circuit.expectation([tc.gates.z(), [i]]).cpu().numpy().real for i in range(n_qubits)]
    two_body = []
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            two_body.append(circuit.expectation([tc.gates.z(), [i]], [tc.gates.z(), [j]]).cpu().numpy().real)
    return np.array(single), np.array(two_body)

def compute_variance_over_inputs(circuit_fn, n_samples=100, n_qubits=4):
    """Compute feature variance across random inputs."""
    all_outputs = []
    for _ in range(n_samples):
        inputs = torch.randn(n_qubits)
        c = circuit_fn(inputs, n_qubits)
        single, two_body = get_expectations(c, n_qubits)
        all_outputs.append(np.concatenate([single, two_body]))
    all_outputs = np.array(all_outputs)
    return np.var(all_outputs, axis=0), np.mean(all_outputs, axis=0)

def compute_entanglement_score(circuit_fn, n_samples=50, n_qubits=4):
    """Measure entanglement via correlation strength."""
    correlations = []
    for _ in range(n_samples):
        inputs = torch.randn(n_qubits)
        c = circuit_fn(inputs, n_qubits)
        single, two_body = get_expectations(c, n_qubits)
        
        # Compute <ZiZj> - <Zi><Zj> for connected pairs
        corr = []
        idx = 0
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                zizj = two_body[idx]
                zi_zj = single[i] * single[j]
                corr.append(abs(zizj - zi_zj))
                idx += 1
        correlations.append(np.mean(corr))
    return np.mean(correlations), np.std(correlations)

# ============= VISUALIZATION =============

def compare_circuits_bar(save_path):
    """Bar chart comparing variance and entanglement across circuits."""
    
    circuits = [
        {"name": "4Q Ring (Baseline)", "n_qubits": 4, "depth": 2, "topology": "ring"},
        {"name": "8Q Chain (Best)", "n_qubits": 8, "depth": 1, "topology": "chain"},
        {"name": "8Q All2All", "n_qubits": 8, "depth": 1, "topology": "all2all"},
    ]
    
    results = {}
    
    print(f"{'Circuit':<20} | {'Variance':<8} | {'Entanglement':<8}")
    print("-" * 45)
    
    for config in circuits:
        name = config["name"]
        
        # Factory function
        def circuit_fn(inputs, weights):
            n_qubits = config["n_qubits"]
            c = tc.Circuit(n_qubits)
            
            # Enc (Redundant)
            input_dim = inputs.shape[0]
            for i in range(n_qubits):
                c.ry(i, theta=inputs[i % input_dim])
            
            # Var
            for d in range(config["depth"]):
                if config["topology"] == "ring":
                     for i in range(n_qubits):
                        c.cnot(i, (i + 1) % n_qubits)
                elif config["topology"] == "chain":
                    for i in range(n_qubits - 1):
                        c.cnot(i, i + 1)
                elif config["topology"] == "all2all":
                    for i in range(n_qubits):
                        for j in range(i + 1, n_qubits):
                            c.cnot(i, j)
                
                # Fixed weight layer for comparison
                for i in range(n_qubits):
                    c.ry(i, theta=1.0) 
            
            single = [c.expectation([tc.gates.z(), [i]]) for i in range(n_qubits)]
            two_body = []
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    two_body.append(c.expectation([tc.gates.z(), [i]], [tc.gates.z(), [j]]))
            return K.stack(single + two_body)
        
        vmap_circuit = K.vmap(circuit_fn, vectorized_argnums=0)
        
        # Generate inputs
        # Note: Previous function expected n_qubits input, but now we have redundant encoding
        # For fair comparison, let's feed 4 inputs to all (since dataset is 4D)
        inputs = torch.randn(100, 4) 
        
        # 1. Variance
        outputs = vmap_circuit(inputs, None).real
        variance = torch.var(outputs).mean().item()
        
        # 2. Entanglement (Correlation Matrix Mean)
        corr_matrix = torch.corrcoef(outputs.T)
        n_feats = outputs.shape[1]
        mask = torch.eye(n_feats, dtype=torch.bool)
        mean_corr = torch.abs(corr_matrix[~mask]).mean().item() if n_feats > 1 else 0.0
        
        print(f"{name:<20} | {variance:.4f}   | {mean_corr:.4f}")
        
        results[name] = {
            "variance": variance,
            "entanglement": mean_corr,
            "ent_std": 0.0 # Simplified
        }
    
    # Plotting code adapted for new results dict
    names = list(results.keys())
    variances = [results[n]["variance"] for n in names]
    entanglements = [results[n]["entanglement"] for n in names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(names))
    width = 0.6
    
    # Variance
    bars1 = ax1.bar(x, variances, width, color=['gray', 'green', 'purple'])
    ax1.set_ylabel('Mean Feature Variance')
    ax1.set_title('Output Variance (Expressivity)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15)
    
    # Entanglement
    bars2 = ax2.bar(x, entanglements, width, color=['gray', 'green', 'purple'])
    ax2.set_ylabel('Mean Feature Correlation')
    ax2.set_title('Entanglement (Feature Coupling)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()
    
    return results

def compare_outputs_heatmap(inputs, save_path):
    """Heatmap of outputs across different circuits."""
    n_qubits = 4
    n_outputs = n_qubits + n_qubits * (n_qubits - 1) // 2
    
    output_labels = [f"⟨Z{i}⟩" for i in range(n_qubits)]
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            output_labels.append(f"⟨Z{i}Z{j}⟩")
    
    outputs = np.zeros((len(CIRCUITS), n_outputs))
    
    for idx, (name, fn) in enumerate(CIRCUITS.items()):
        c = fn(inputs, n_qubits)
        single, two_body = get_expectations(c, n_qubits)
        outputs[idx, :] = np.concatenate([single, two_body])
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    im = ax.imshow(outputs, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    
    ax.set_yticks(range(len(CIRCUITS)))
    ax.set_yticklabels(list(CIRCUITS.keys()))
    ax.set_xticks(range(n_outputs))
    ax.set_xticklabels(output_labels, rotation=45, ha='right')
    
    ax.set_xlabel("Output Observable")
    ax.set_ylabel("Circuit Architecture")
    ax.set_title(f"Output Comparison for Input: {inputs.cpu().numpy().round(2)}")
    
    for i in range(len(CIRCUITS)):
        for j in range(n_outputs):
            ax.text(j, i, f"{outputs[i,j]:.2f}", ha='center', va='center', 
                    color='white' if abs(outputs[i,j]) > 0.5 else 'black', fontsize=8)
    
    plt.colorbar(im, label="Expectation Value")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()

def compare_bloch_spheres(inputs, save_path):
    """Side-by-side Bloch sphere comparison."""
    n_qubits = 4
    
    fig = plt.figure(figsize=(16, 12))
    
    for c_idx, (name, fn) in enumerate(CIRCUITS.items()):
        c = fn(inputs, n_qubits)
        state = c.state().cpu().numpy()
        
        for q in range(n_qubits):
            ax = fig.add_subplot(len(CIRCUITS), n_qubits, c_idx * n_qubits + q + 1, projection='3d')
            
            # Draw sphere
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 15)
            xs = np.outer(np.cos(u), np.sin(v))
            ys = np.outer(np.sin(u), np.sin(v))
            zs = np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_wireframe(xs, ys, zs, color='lightgray', alpha=0.2, linewidth=0.3)
            
            # Bloch vector (simplified - just use Z expectation)
            z_exp = c.expectation([tc.gates.z(), [q]]).cpu().numpy().real
            ax.scatter([0], [0], [z_exp], color='red', s=80)
            ax.quiver(0, 0, 0, 0, 0, z_exp, color='red', arrow_length_ratio=0.2)
            
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            ax.set_title(f"{name[:8]} Q{q}" if q == 0 else f"Q{q}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.suptitle(f"Bloch Sphere Comparison | Input: {inputs.cpu().numpy().round(2)}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()

def main():
    setup_mps_environment()
    print("="*60)
    print("Phase 4: Comparative Circuit Analysis")
    print("="*60)
    
    output_dir = "results/phase4_comparison"
    n_qubits = 4
    
    # 1. Variance & Entanglement Comparison
    print("\n--- Variance & Entanglement Analysis ---")
    results = compare_circuits_bar(f"{output_dir}/circuit_comparison_bar.png")
    
    # Print summary
    print("\nSummary:")
    for name, data in results.items():
        print(f"  {name}: Var={data['variance']:.4f}, Ent={data['entanglement']:.3f}")
    
    # 2. Output Heatmaps for different inputs
    print("\n--- Output Heatmaps ---")
    test_inputs = [
        ("uniform", torch.ones(n_qubits) * 0.5),
        ("random", torch.randn(n_qubits)),
        ("parity", torch.tensor([0.0, np.pi, 0.0, np.pi])),
    ]
    for name, inputs in test_inputs:
        compare_outputs_heatmap(inputs, f"{output_dir}/output_heatmap_{name}.png")
    
    # 3. Bloch Sphere Comparison
    print("\n--- Bloch Sphere Comparison ---")
    compare_bloch_spheres(torch.randn(n_qubits), f"{output_dir}/bloch_comparison.png")
    
    print("\n" + "="*60)
    print("Phase 4 Complete!")
    print(f"Check {output_dir}/ for results")
    print("="*60)

if __name__ == "__main__":
    main()
