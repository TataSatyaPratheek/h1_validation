"""
Phase 3: Sensitivity Analysis
Understand how input variations affect quantum outputs.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import tensorcircuit as tc
from utils import setup_mps_environment

K = tc.set_backend("pytorch")

def create_qfm_circuit(inputs, n_qubits=4):
    """Create the standard QFM circuit and return expectations."""
    c = tc.Circuit(n_qubits)
    for i in range(n_qubits):
        c.ry(i, theta=inputs[i])
    for i in range(n_qubits):
        c.cnot(i, (i + 1) % n_qubits)
    
    # Single body
    single = [c.expectation([tc.gates.z(), [i]]) for i in range(n_qubits)]
    # Two body
    two_body = []
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            two_body.append(c.expectation([tc.gates.z(), [i]], [tc.gates.z(), [j]]))
    
    return torch.stack([e.real for e in single + two_body])

def compute_jacobian(inputs, n_qubits=4, epsilon=0.01):
    """Compute Jacobian matrix: d(output)/d(input) via finite differences."""
    n_outputs = n_qubits + n_qubits * (n_qubits - 1) // 2  # 4 + 6 = 10
    jacobian = np.zeros((n_outputs, n_qubits))
    
    base_output = create_qfm_circuit(inputs, n_qubits).cpu().numpy()
    
    for i in range(n_qubits):
        perturbed = inputs.clone()
        perturbed[i] += epsilon
        perturbed_output = create_qfm_circuit(perturbed, n_qubits).cpu().numpy()
        jacobian[:, i] = (perturbed_output - base_output) / epsilon
    
    return jacobian

def visualize_jacobian(jacobian, save_path, title="Input-Output Jacobian"):
    """Heatmap of Jacobian matrix."""
    n_qubits = jacobian.shape[1]
    n_outputs = jacobian.shape[0]
    
    fig, ax = plt.subplots(figsize=(8, 10))
    
    im = ax.imshow(jacobian, cmap='RdBu', aspect='auto', vmin=-2, vmax=2)
    
    # Labels
    input_labels = [f"Input {i}" for i in range(n_qubits)]
    output_labels = [f"⟨Z{i}⟩" for i in range(n_qubits)]
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            output_labels.append(f"⟨Z{i}Z{j}⟩")
    
    ax.set_xticks(range(n_qubits))
    ax.set_xticklabels(input_labels)
    ax.set_yticks(range(n_outputs))
    ax.set_yticklabels(output_labels)
    
    ax.set_xlabel("Input Feature")
    ax.set_ylabel("Output Observable")
    ax.set_title(title)
    
    # Add values
    for i in range(n_outputs):
        for j in range(n_qubits):
            val = jacobian[i, j]
            color = 'white' if abs(val) > 1 else 'black'
            ax.text(j, i, f"{val:.2f}", ha='center', va='center', color=color, fontsize=8)
    
    plt.colorbar(im, label="Sensitivity (∂output/∂input)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()

def interpolate_inputs(input_a, input_b, n_steps=20):
    """Generate interpolation between two inputs."""
    steps = []
    for t in np.linspace(0, 1, n_steps):
        steps.append((1 - t) * input_a + t * input_b)
    return steps

def visualize_interpolation(input_a, input_b, n_qubits=4, save_path="interpolation.png"):
    """Visualize how outputs change along interpolation path."""
    steps = interpolate_inputs(input_a, input_b, n_steps=30)
    
    outputs = []
    for inp in steps:
        out = create_qfm_circuit(inp, n_qubits).cpu().numpy()
        outputs.append(out)
    outputs = np.array(outputs)
    
    n_outputs = outputs.shape[1]
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Top: Single-body observables
    for i in range(n_qubits):
        axes[0].plot(outputs[:, i], label=f"⟨Z{i}⟩", linewidth=2)
    axes[0].set_xlabel("Interpolation Step (t=0 → t=1)")
    axes[0].set_ylabel("Expectation Value")
    axes[0].set_title("Single-Body Observables Along Interpolation Path")
    axes[0].legend(loc='upper right')
    axes[0].axhline(0, color='black', linestyle='--', alpha=0.3)
    axes[0].grid(True, alpha=0.3)
    
    # Bottom: Two-body observables
    idx = n_qubits
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            axes[1].plot(outputs[:, idx], label=f"⟨Z{i}Z{j}⟩", linewidth=2)
            idx += 1
    axes[1].set_xlabel("Interpolation Step (t=0 → t=1)")
    axes[1].set_ylabel("Expectation Value")
    axes[1].set_title("Two-Body Observables Along Interpolation Path")
    axes[1].legend(loc='upper right')
    axes[1].axhline(0, color='black', linestyle='--', alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    
    # Add input annotations
    fig.text(0.02, 0.98, f"Start: {input_a.cpu().numpy()}", fontsize=9, transform=fig.transFigure)
    fig.text(0.02, 0.95, f"End:   {input_b.cpu().numpy()}", fontsize=9, transform=fig.transFigure)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()

def visualize_output_surface(n_qubits=4, fixed_dims=(0, 0), save_path="surface.png"):
    """3D surface showing output as function of 2 input dimensions."""
    # Vary first two inputs, fix others
    n_points = 30
    x = np.linspace(-np.pi, np.pi, n_points)
    y = np.linspace(-np.pi, np.pi, n_points)
    X, Y = np.meshgrid(x, y)
    
    Z_single = np.zeros((n_points, n_points, n_qubits))
    
    for i in range(n_points):
        for j in range(n_points):
            inputs = torch.tensor([X[i, j], Y[i, j], fixed_dims[0], fixed_dims[1]], dtype=torch.float32)
            out = create_qfm_circuit(inputs, n_qubits).cpu().numpy()
            Z_single[i, j, :] = out[:n_qubits]
    
    fig = plt.figure(figsize=(16, 4))
    
    for q in range(n_qubits):
        ax = fig.add_subplot(1, n_qubits, q + 1, projection='3d')
        ax.plot_surface(X, Y, Z_single[:, :, q], cmap='viridis', alpha=0.8)
        ax.set_xlabel("Input 0")
        ax.set_ylabel("Input 1")
        ax.set_zlabel(f"⟨Z{q}⟩")
        ax.set_title(f"Qubit {q} Response")
    
    plt.suptitle("Output Surface: ⟨Z⟩ vs Input[0] and Input[1]", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()

def main():
    setup_mps_environment()
    print("="*60)
    print("Phase 3: Sensitivity Analysis")
    print("="*60)
    
    n_qubits = 4
    output_dir = "results/phase3_sensitivity"
    
    # Test inputs
    test_cases = {
        "zero": torch.zeros(n_qubits),
        "uniform": torch.ones(n_qubits) * 0.5,
        "random": torch.randn(n_qubits),
        "parity_01": torch.tensor([0.0, np.pi, 0.0, np.pi]),
    }
    
    # 1. Jacobian Analysis
    print("\n--- Jacobian Analysis ---")
    for name, inputs in test_cases.items():
        print(f"Computing Jacobian for: {name}")
        jacobian = compute_jacobian(inputs.cpu(), n_qubits)
        visualize_jacobian(jacobian, f"{output_dir}/jacobian_{name}.png", f"Jacobian: {name}")
    
    # 2. Interpolation Analysis
    print("\n--- Interpolation Analysis ---")
    # Class 0 → Class 1 (parity)
    input_a = torch.tensor([0.0, 0.0, 0.0, 0.0])
    input_b = torch.tensor([np.pi, np.pi, np.pi, np.pi])
    visualize_interpolation(input_a, input_b, n_qubits, f"{output_dir}/interp_zero_to_pi.png")
    
    # Parity 0 → Parity 1
    input_a = torch.tensor([0.0, 0.0, 0.0, 0.0])  # Parity 0
    input_b = torch.tensor([np.pi, 0.0, 0.0, 0.0])  # Parity 1
    visualize_interpolation(input_a, input_b, n_qubits, f"{output_dir}/interp_parity_flip.png")
    
    # 3. Output Surface
    print("\n--- Output Surface ---")
    visualize_output_surface(n_qubits, fixed_dims=(0, 0), save_path=f"{output_dir}/surface_2d.png")
    
    print("\n" + "="*60)
    print("Phase 3 Complete!")
    print(f"Check {output_dir}/ for results")
    print("="*60)

if __name__ == "__main__":
    main()
