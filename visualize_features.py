"""
Welch Labs Style Visualizations for Project Antigravity
Compares Classical CNN vs Quantum Feature Maps visually.

Visualizations:
1. Feature Space Scatter (t-SNE) - Shows clustering
2. Decision Boundary Contour - Shows non-linearity
3. Activation Heatmap - Shows Barren Plateau / Information Black Hole
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import tensorcircuit as tc
from utils import setup_mps_environment
from data import get_dataloaders

K = tc.set_backend("pytorch")

# ============= MODEL DEFINITIONS =============

class QuantumFeatureMap(nn.Module):
    def __init__(self, n_qubits=4, alpha=1.57):
        super().__init__()
        self.n_qubits = n_qubits
        self.alpha = alpha
        
        def quantum_fn(inputs):
            c = tc.Circuit(n_qubits)
            for i in range(n_qubits):
                c.ry(i, theta=inputs[i])
            for i in range(n_qubits):
                c.cnot(i, (i + 1) % n_qubits)
            single = [c.expectation([tc.gates.z(), [i]]) for i in range(n_qubits)]
            two_body = []
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    two_body.append(c.expectation([tc.gates.z(), [i]], [tc.gates.z(), [j]]))
            return K.stack(single + two_body)
        self.vmapped_fn = K.vmap(quantum_fn)

    def forward(self, x):
        theta = self.alpha * x
        return self.vmapped_fn(theta).real

class HybridModel(nn.Module):
    def __init__(self, use_quantum=True, n_qubits=4, num_classes=10):
        super().__init__()
        self.use_quantum = use_quantum
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.projection = nn.Linear(64 * 8 * 8, n_qubits)
        
        if use_quantum:
            self.qfm = QuantumFeatureMap(n_qubits)
            self.classifier = nn.Linear(10, num_classes)  # 4 single + 6 two-body
        else:
            self.classifier = nn.Linear(n_qubits, num_classes)

    def forward(self, x):
        f = self.feature_extractor(x)
        f = f.view(f.size(0), -1)
        z = self.projection(f)
        if self.use_quantum:
            z = self.qfm(z)
        else:
            z = torch.tanh(z)
        return self.classifier(z)
    
    def forward_features(self, x):
        """Return features BEFORE the final classifier (for visualization)."""
        f = self.feature_extractor(x)
        f = f.view(f.size(0), -1)
        z = self.projection(f)
        if self.use_quantum:
            z = self.qfm(z)
        else:
            z = torch.tanh(z)
        return z

# ============= VISUALIZATION 1: FEATURE SPACE =============

def visualize_feature_space(model, dataloader, device, title="Feature Space", save_path=None):
    """t-SNE visualization of the feature space before the linear classifier."""
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            feats = model.forward_features(inputs)
            features.append(feats.cpu().numpy())
            labels.append(targets.cpu().numpy())

    features = np.concatenate(features)
    labels = np.concatenate(labels)

    # Use t-SNE for 2D projection
    print(f"Running t-SNE on {len(features)} samples...")
    reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
    features_2d = reducer.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10', alpha=0.6, s=20)
    plt.colorbar(scatter, label='Class')
    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close()

# ============= VISUALIZATION 2: DECISION BOUNDARY (PARITY) =============

def visualize_decision_boundary_parity(model, device, title="Decision Boundary", save_path=None):
    """Decision boundary for 2D slice of 4D parity task."""
    model.eval()
    
    # Generate grid for first 2 dimensions, fix last 2
    x_range = np.linspace(0, 1, 50)
    y_range = np.linspace(0, 1, 50)
    xx, yy = np.meshgrid(x_range, y_range)
    
    # Create 4D inputs: (x, y, 0, 0)
    grid = np.zeros((xx.size, 4))
    grid[:, 0] = xx.ravel()
    grid[:, 1] = yy.ravel()
    grid_tensor = torch.tensor(grid, dtype=torch.float32).to(device)

    with torch.no_grad():
        preds = model(grid_tensor)
        probs = torch.softmax(preds, dim=1)[:, 1].cpu().numpy().reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(xx, yy, probs, levels=20, cmap="RdBu", alpha=0.8)
    plt.colorbar(contour, label="P(Class 1)")
    plt.title(title)
    plt.xlabel("Bit 0")
    plt.ylabel("Bit 1")
    
    # Mark the corners
    plt.scatter([0, 1, 0, 1], [0, 1, 1, 0], c='black', s=100, marker='x', label='Parity=0,0')
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close()

# ============= VISUALIZATION 3: ACTIVATION HEATMAP =============

def visualize_activation_heatmap(model, dataloader, device, n_samples=50, title="Feature Activations", save_path=None):
    """Heatmap showing activation patterns across samples and features."""
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            feats = model.forward_features(inputs)
            features.append(feats.cpu().numpy())
            labels.append(targets.cpu().numpy())
            if len(np.concatenate(features)) >= n_samples:
                break

    features = np.concatenate(features)[:n_samples]
    labels = np.concatenate(labels)[:n_samples]
    
    # Sort by label for cleaner visualization
    sorted_idx = np.argsort(labels)
    features = features[sorted_idx]
    labels = labels[sorted_idx]

    plt.figure(figsize=(12, 8))
    im = plt.imshow(features, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(im, label="Feature Value")
    plt.xlabel("Feature Index (Qubit Observable)")
    plt.ylabel("Sample Index (sorted by class)")
    plt.title(title)
    
    # Add class boundaries
    unique_labels, counts = np.unique(labels, return_counts=True)
    cumsum = np.cumsum(counts)
    for c in cumsum[:-1]:
        plt.axhline(y=c, color='red', linestyle='--', alpha=0.5)

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close()
    
    # Print variance stats
    print(f"Feature Variance (per column): {np.var(features, axis=0)}")
    print(f"Total Variance: {np.var(features):.8f}")

# ============= MAIN =============

def main():
    setup_mps_environment()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running Visualizations on: {device}")
    
    gen_dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    train_loader, test_loader = get_dataloaders(
        dataset_name="cifar10", batch_size=32, image_size=64, 
        subset_size=500, generator_device=gen_dev
    )
    
    # === CLASSICAL MODEL ===
    print("\n--- Classical Model Visualizations ---")
    model_classical = HybridModel(use_quantum=False).to(device)
    
    visualize_feature_space(model_classical, test_loader, device, 
                            title="Classical CNN Feature Space (Before Training)",
                            save_path="results/viz_classical_tsne.png")
    
    visualize_activation_heatmap(model_classical, test_loader, device,
                                  title="Classical CNN Activations",
                                  save_path="results/viz_classical_heatmap.png")
    
    # === QUANTUM MODEL ===
    print("\n--- Quantum Model Visualizations ---")
    model_quantum = HybridModel(use_quantum=True).to(device)
    
    visualize_feature_space(model_quantum, test_loader, device, 
                            title="Quantum Feature Map Space (Before Training)",
                            save_path="results/viz_quantum_tsne.png")
    
    visualize_activation_heatmap(model_quantum, test_loader, device,
                                  title="Quantum Feature Map Activations",
                                  save_path="results/viz_quantum_heatmap.png")
    
    print("\n--- All Visualizations Complete ---")
    print("Check results/ folder for: viz_classical_tsne.png, viz_quantum_tsne.png, etc.")

if __name__ == "__main__":
    main()
