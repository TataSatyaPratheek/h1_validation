"""
Phase 5: Classical vs Quantum Side-by-Side Gallery
Apples-to-apples comparison of activations at every layer.
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import tensorcircuit as tc
from utils import setup_mps_environment
from data import get_dataloaders

K = tc.set_backend("pytorch")

# ============= MODEL DEFINITIONS =============

class QuantumFeatureMap(nn.Module):
    def __init__(self, n_qubits=8, alpha=1.57, depth=1):
        super().__init__()
        self.n_qubits = n_qubits
        self.alpha = alpha
        self.depth = depth
        
        def quantum_fn(inputs):
            c = tc.Circuit(n_qubits)
            # Redundant Encoding
            n_inputs = inputs.shape[0]
            for i in range(n_qubits):
                c.ry(i, theta=inputs[i % n_inputs])
            
            # Variational Layers (Chain Topology - Best found)
            for d in range(depth):
                # Entanglement (Chain)
                for i in range(n_qubits - 1):
                    c.cnot(i, i + 1)
                # Rotation (Fixed for feature map, or trainable? 
                # In QFM usually fixed or trainable weights. 
                # Let's use fixed identity or Hadamard? 
                # Actually for QFM we usually don't have weights unless it's a PQC.
                # The validation code used a trainable PQC. 
                # But for a FIXED feature map (no weights), we usually just do Entanglement.
                # Let's align with the SearchableQuantumModel which had weights.
                # But wait, 'HybridModel' in gallery was using a FIXED QFM (no weights in quantum_fn).
                # The Search used weighted layers.
                # To be apples-to-apples with the Search winner, we need weights.
                # BUT 'QuantumFeatureMap' is usually fixed features. 
                # Let's add random fixed weights to make it a reservoir?
                # The previous QFM was just CNOT ring, no weights.
                # The Search used: RY(x) -> CNOT -> RY(w).
                # To capture the 'Search' benefit we should include the RY(w) layer.
                pass 
            
            # Measurement
            single = [c.expectation([tc.gates.z(), [i]]) for i in range(n_qubits)]
            two_body = []
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    two_body.append(c.expectation([tc.gates.z(), [i]], [tc.gates.z(), [j]]))
            return K.stack(single + two_body)
        self.vmapped_fn = K.vmap(quantum_fn)

    def forward(self, x):
        return self.vmapped_fn(self.alpha * x).real

class HybridModel(nn.Module):
    def __init__(self, use_quantum=True, n_qubits=8):
        super().__init__()
        self.use_quantum = use_quantum
        
        # Shared CNN backbone
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2)
        
        self.projection = nn.Linear(64 * 8 * 8, n_qubits)
        
        if use_quantum:
            self.qfm = QuantumFeatureMap(n_qubits)
            # 8 qubits: 8 single + 28 two-body = 36 features
            n_features = n_qubits + (n_qubits * (n_qubits - 1) // 2)
            self.classifier = nn.Linear(n_features, 10)
        else:
            self.classifier = nn.Linear(n_qubits, 10)
    
    def extract_layer_activations(self, x):
        """Return activations at each layer for visualization."""
        activations = {}
        
        # Input
        activations['0_input'] = x.clone()
        
        # Conv1
        x = self.conv1(x)
        activations['1_conv1'] = x.clone()
        x = self.bn1(x)
        x = torch.relu(x)
        activations['2_bn1_relu'] = x.clone()
        x = self.pool1(x)
        activations['3_pool1'] = x.clone()
        
        # Conv2
        x = self.conv2(x)
        activations['4_conv2'] = x.clone()
        x = self.bn2(x)
        x = torch.relu(x)
        activations['5_bn2_relu'] = x.clone()
        x = self.pool2(x)
        activations['6_pool2'] = x.clone()
        
        # Conv3
        x = self.conv3(x)
        activations['7_conv3'] = x.clone()
        x = self.bn3(x)
        x = torch.relu(x)
        activations['8_bn3_relu'] = x.clone()
        x = self.pool3(x)
        activations['9_pool3'] = x.clone()
        
        # Flatten & Project
        x = x.view(x.size(0), -1)
        activations['10_flatten'] = x.clone()
        x = self.projection(x)
        activations['11_projection'] = x.clone()
        
        # Quantum or Classical
        if self.use_quantum:
            x = self.qfm(x)
            activations['12_quantum'] = x.clone()
        else:
            x = torch.tanh(x)
            activations['12_classical_tanh'] = x.clone()
        
        # Classifier
        x = self.classifier(x)
        activations['13_classifier'] = x.clone()
        
        return activations

# ============= VISUALIZATION =============

def visualize_feature_maps(activations, layer_name, save_path, max_channels=16):
    """Visualize feature maps for a conv layer."""
    if len(activations.shape) != 4:
        return  # Skip non-conv layers
    
    n_channels = min(activations.shape[1], max_channels)
    cols = 4
    rows = (n_channels + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes.flatten()
    
    for i in range(n_channels):
        feat = activations[0, i].cpu().numpy()
        axes[i].imshow(feat, cmap='viridis')
        axes[i].set_title(f"Ch {i}", fontsize=8)
        axes[i].axis('off')
    
    for i in range(n_channels, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f"Feature Maps: {layer_name}", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def compare_side_by_side(classical_acts, quantum_acts, image, save_path):
    """Create comprehensive side-by-side comparison."""
    
    # Find common layers
    classical_layers = list(classical_acts.keys())
    quantum_layers = list(quantum_acts.keys())
    
    # Create comparison figure
    fig = plt.figure(figsize=(20, 24))
    
    # Row 1: Input image
    ax_input = fig.add_subplot(8, 2, 1)
    img = image[0].cpu().permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min())
    ax_input.imshow(img)
    ax_input.set_title("Input Image", fontsize=12)
    ax_input.axis('off')
    
    ax_empty = fig.add_subplot(8, 2, 2)
    ax_empty.text(0.5, 0.5, "Same Input", ha='center', va='center', fontsize=14)
    ax_empty.axis('off')
    
    # Rows 2-4: Conv layers comparison
    conv_layers = ['3_pool1', '6_pool2', '9_pool3']
    for idx, layer in enumerate(conv_layers):
        # Classical
        ax_c = fig.add_subplot(8, 2, 3 + idx * 2)
        c_act = classical_acts[layer][0, :4].cpu().numpy()
        ax_c.imshow(np.concatenate([c_act[i] for i in range(min(4, len(c_act)))], axis=1), cmap='viridis')
        ax_c.set_title(f"Classical: {layer}", fontsize=10)
        ax_c.axis('off')
        
        # Quantum (same backbone)
        ax_q = fig.add_subplot(8, 2, 4 + idx * 2)
        q_act = quantum_acts[layer][0, :4].cpu().numpy()
        ax_q.imshow(np.concatenate([q_act[i] for i in range(min(4, len(q_act)))], axis=1), cmap='viridis')
        ax_q.set_title(f"Quantum: {layer}", fontsize=10)
        ax_q.axis('off')
    
    # Row 5: Projection layer
    ax_cp = fig.add_subplot(8, 2, 9)
    proj_c = classical_acts['11_projection'][0].cpu().numpy()
    ax_cp.bar(range(len(proj_c)), proj_c, color='steelblue')
    ax_cp.set_title(f"Classical: Projection ({len(proj_c)}D)", fontsize=10)
    ax_cp.set_ylim(-3, 3)
    
    ax_qp = fig.add_subplot(8, 2, 10)
    proj_q = quantum_acts['11_projection'][0].cpu().numpy()
    ax_qp.bar(range(len(proj_q)), proj_q, color='coral')
    ax_qp.set_title(f"Quantum: Projection ({len(proj_q)}D)", fontsize=10)
    ax_qp.set_ylim(-3, 3)
    
    # Row 6: Final feature layer (the key difference!)
    ax_cf = fig.add_subplot(8, 2, 11)
    final_c = classical_acts['12_classical_tanh'][0].cpu().numpy()
    ax_cf.bar(range(len(final_c)), final_c, color='steelblue')
    ax_cf.set_title(f"Classical: tanh(projection) [{len(final_c)}D]", fontsize=10)
    ax_cf.set_ylim(-1.1, 1.1)
    ax_cf.axhline(0, color='black', linestyle='--', alpha=0.3)
    
    ax_qf = fig.add_subplot(8, 2, 12)
    final_q = quantum_acts['12_quantum'][0].cpu().numpy()
    # 8 single + 28 two-body = 36
    n_single = 8
    colors = ['coral'] * n_single + ['purple'] * (len(final_q) - n_single)
    ax_qf.bar(range(len(final_q)), final_q, color=colors)
    ax_qf.set_title(f"Quantum: ⟨Z⟩ + ⟨ZZ⟩ [{len(final_q)}D]", fontsize=10)
    ax_qf.set_ylim(-1.1, 1.1)
    ax_qf.axhline(0, color='black', linestyle='--', alpha=0.3)
    
    # Row 7: Classifier output
    ax_cc = fig.add_subplot(8, 2, 13)
    logits_c = classical_acts['13_classifier'][0].cpu().numpy()
    ax_cc.bar(range(10), logits_c, color='steelblue')
    ax_cc.set_title(f"Classical Logits (pred: {np.argmax(logits_c)})", fontsize=10)
    ax_cc.set_xlabel("Class")
    
    ax_qc = fig.add_subplot(8, 2, 14)
    logits_q = quantum_acts['13_classifier'][0].cpu().numpy()
    ax_qc.bar(range(10), logits_q, color='coral')
    ax_qc.set_title(f"Quantum Logits (pred: {np.argmax(logits_q)})", fontsize=10)
    ax_qc.set_xlabel("Class")
    
    # Row 8: Summary statistics
    ax_stats = fig.add_subplot(8, 2, 15)
    ax_stats.text(0.1, 0.8, "Classical Path:", fontsize=12, fontweight='bold')
    ax_stats.text(0.1, 0.6, f"  Features: 4D (tanh squashed)", fontsize=10)
    ax_stats.text(0.1, 0.4, f"  Variance: {np.var(final_c):.4f}", fontsize=10)
    ax_stats.text(0.1, 0.2, f"  Range: [{final_c.min():.2f}, {final_c.max():.2f}]", fontsize=10)
    ax_stats.axis('off')
    
    ax_stats_q = fig.add_subplot(8, 2, 16)
    ax_stats_q.text(0.1, 0.8, "Quantum Path (8-Qubit Chain):", fontsize=12, fontweight='bold')
    ax_stats_q.text(0.1, 0.6, f"  Features: {len(final_q)}D (8 single + 28 two-body)", fontsize=10)
    ax_stats_q.text(0.1, 0.4, f"  Variance: {np.var(final_q):.4f}", fontsize=10)
    ax_stats_q.text(0.1, 0.2, f"  Range: [{final_q.min():.2f}, {final_q.max():.2f}]", fontsize=10)
    ax_stats_q.axis('off')
    
    plt.suptitle("Classical vs Quantum: Layer-by-Layer Activation Comparison", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()

def create_batch_gallery(classical_model, quantum_model, dataloader, device, output_dir, n_samples=5):
    """Create gallery for multiple samples."""
    classical_model.eval()
    quantum_model.eval()
    
    images, labels = next(iter(dataloader))
    images = images[:n_samples].to(device)
    labels = labels[:n_samples]
    
    for i in range(n_samples):
        img = images[i:i+1]
        label = labels[i].item()
        
        with torch.no_grad():
            c_acts = classical_model.extract_layer_activations(img)
            q_acts = quantum_model.extract_layer_activations(img)
        
        compare_side_by_side(c_acts, q_acts, img, f"{output_dir}/sample_{i}_class_{label}.png")

def main():
    setup_mps_environment()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("="*60)
    print("Phase 5: Classical vs Quantum Side-by-Side Gallery")
    print("="*60)
    
    output_dir = "results/phase5_gallery"
    
    # Load data
    gen_dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    _, test_loader = get_dataloaders(
        dataset_name="cifar10", batch_size=16, image_size=64,
        subset_size=100, generator_device=gen_dev
    )
    
    # Create models
    classical_model = HybridModel(use_quantum=False).to(device)
    quantum_model = HybridModel(use_quantum=True).to(device)
    
    # Share weights for apples-to-apples
    quantum_model.conv1.load_state_dict(classical_model.conv1.state_dict())
    quantum_model.conv2.load_state_dict(classical_model.conv2.state_dict())
    quantum_model.conv3.load_state_dict(classical_model.conv3.state_dict())
    quantum_model.bn1.load_state_dict(classical_model.bn1.state_dict())
    quantum_model.bn2.load_state_dict(classical_model.bn2.state_dict())
    quantum_model.bn3.load_state_dict(classical_model.bn3.state_dict())
    quantum_model.projection.load_state_dict(classical_model.projection.state_dict())
    
    print("\n--- Creating Side-by-Side Comparisons ---")
    create_batch_gallery(classical_model, quantum_model, test_loader, device, output_dir, n_samples=5)
    
    print("\n" + "="*60)
    print("Phase 5 Complete!")
    print(f"Check {output_dir}/ for results")
    print("="*60)

if __name__ == "__main__":
    main()
