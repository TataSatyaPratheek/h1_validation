"""
Phase 5: Spatial-Aware QFM for CIFAR-10
=======================================
Preserves spatial structure using patch-based processing.
Each 4×4 patch processed by independent QFM.
"""
import torch
import torch.nn as nn
import numpy as np
import tensorcircuit as tc
import json
import time

tc.set_backend("pytorch")
tc.set_dtype("complex64")
K = tc.backend


def create_patch_qfm(n_qubits: int = 16, depth: int = 2):
    """
    QFM for a single patch.
    n_qubits = 16 for 4×4 patch (1 qubit per pixel, grayscale)
    """
    def qfm_fn(patch):
        c = tc.Circuit(n_qubits)
        
        # Encode each pixel as angle
        for i in range(n_qubits):
            c.ry(i, theta=patch[i])
        
        # 2D-local entanglement (4×4 grid)
        for _ in range(depth):
            # Horizontal connections
            for row in range(4):
                for col in range(3):
                    i = row * 4 + col
                    j = row * 4 + col + 1
                    c.cnot(i, j)
            # Vertical connections
            for row in range(3):
                for col in range(4):
                    i = row * 4 + col
                    j = (row + 1) * 4 + col
                    c.cnot(i, j)
        
        # Measure single and two-body observables
        single = [c.expectation([tc.gates.z(), [i]]) for i in range(n_qubits)]
        # Only measure local two-body (adjacent pixels)
        two_body = []
        # Horizontal pairs
        for row in range(4):
            for col in range(3):
                i, j = row * 4 + col, row * 4 + col + 1
                two_body.append(c.expectation([tc.gates.z(), [i]], [tc.gates.z(), [j]]))
        # Vertical pairs
        for row in range(3):
            for col in range(4):
                i, j = row * 4 + col, (row + 1) * 4 + col
                two_body.append(c.expectation([tc.gates.z(), [i]], [tc.gates.z(), [j]]))
        
        return K.stack(single + two_body)
    
    return qfm_fn


class SpatialQFM(nn.Module):
    """
    Spatial-aware QFM that processes image patches.
    """
    def __init__(self, patch_size: int = 4, depth: int = 2, num_classes: int = 10):
        super().__init__()
        self.patch_size = patch_size
        
        # Create patch QFM (16 qubits for 4×4 patch)
        n_qubits = patch_size * patch_size
        self.patch_qfm = K.vmap(create_patch_qfm(n_qubits, depth), vectorized_argnums=0)
        
        # Features per patch: 16 single + 24 two-body (local only)
        self.n_features_per_patch = n_qubits + (3 * 4 + 4 * 3)  # horiz + vert pairs
        
        # Number of patches: (32/4) × (32/4) = 64 patches
        n_patches = (32 // patch_size) ** 2
        
        # Pooling and classification
        self.pool = nn.AdaptiveAvgPool1d(16)  # Pool 64 patches to 16
        self.classifier = nn.Sequential(
            nn.Linear(16 * self.n_features_per_patch, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def extract_patches(self, x):
        """
        Extract 4×4 patches from image.
        x: (B, 3, 32, 32) -> patches: (B, 64, 16)
        """
        B, C, H, W = x.shape
        ps = self.patch_size
        
        # Convert to grayscale for simplicity
        x_gray = x.mean(dim=1)  # (B, 32, 32)
        
        # Unfold into patches
        patches = x_gray.unfold(1, ps, ps).unfold(2, ps, ps)  # (B, 8, 8, 4, 4)
        patches = patches.contiguous().view(B, -1, ps * ps)  # (B, 64, 16)
        
        # Scale to [0, π] for angle encoding
        patches = (patches + 1) * np.pi / 2  # From [-1, 1] to [0, π]
        
        return patches
    
    def forward(self, x):
        B = x.shape[0]
        
        # Extract patches: (B, 64, 16)
        patches = self.extract_patches(x)
        
        # Process each patch through QFM
        # Reshape for batch processing: (B*64, 16)
        n_patches = patches.shape[1]
        patches_flat = patches.view(-1, patches.shape[-1])
        
        # Apply QFM to all patches
        features = self.patch_qfm(patches_flat).real  # (B*64, n_features)
        
        # Reshape back: (B, 64, n_features)
        features = features.view(B, n_patches, -1)
        
        # Pool across patches: (B, 16, n_features)
        features = features.transpose(1, 2)  # (B, n_features, 64)
        features = self.pool(features)  # (B, n_features, 16)
        features = features.transpose(1, 2)  # (B, 16, n_features)
        
        # Flatten and classify
        features = features.flatten(1)  # (B, 16 * n_features)
        return self.classifier(features)


class MLPBaseline(nn.Module):
    """MLP baseline with comparable capacity"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)


def load_cifar(n_samples=200):
    """Load CIFAR-10 subset (images, not flattened)"""
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    
    X = torch.stack([dataset[i][0] for i in indices])  # (N, 3, 32, 32)
    y = torch.tensor([dataset[i][1] for i in indices])
    
    return X, y


def train_and_eval(model, X, y, epochs=20, name="Model"):
    """Train and evaluate model"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Split
    n = len(X)
    train_idx = np.random.choice(n, int(0.8 * n), replace=False)
    val_idx = [i for i in range(n) if i not in train_idx]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    print(f"\nTraining {name}...")
    start = time.time()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(X_val).argmax(1) == y_val).float().mean().item() * 100
            print(f"  Epoch {epoch+1}: Loss={loss.item():.3f}, Val Acc={acc:.1f}%")
    
    model.eval()
    with torch.no_grad():
        final_acc = (model(X_val).argmax(1) == y_val).float().mean().item() * 100
    
    elapsed = time.time() - start
    print(f"Final: {final_acc:.1f}% ({elapsed:.1f}s)")
    
    return final_acc, elapsed


if __name__ == "__main__":
    print("="*60)
    print("PHASE 5: SPATIAL-AWARE QFM FOR CIFAR-10")
    print("="*60)
    print("Strategy: Process 4×4 patches with 2D-local entanglement")
    print("Preserves spatial structure via local CNOT connections")
    print()
    
    # Load data (keep as images, not flattened)
    print("Loading CIFAR-10 subset (200 samples)...")
    X, y = load_cifar(n_samples=200)
    print(f"Shape: {X.shape}")
    print()
    
    results = []
    
    # MLP Baseline
    mlp = MLPBaseline()
    acc_mlp, time_mlp = train_and_eval(mlp, X, y, epochs=30, name="MLP (128-64)")
    results.append({"model": "MLP", "accuracy": acc_mlp, "time": time_mlp})
    
    # Spatial QFM
    spatial_qfm = SpatialQFM(patch_size=4, depth=2)
    acc_qfm, time_qfm = train_and_eval(spatial_qfm, X, y, epochs=30, name="Spatial QFM (4×4 patches)")
    results.append({"model": "Spatial QFM", "accuracy": acc_qfm, "time": time_qfm})
    
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    for r in results:
        print(f"{r['model']:<15}: {r['accuracy']:.1f}% ({r['time']:.1f}s)")
    
    adv = acc_qfm - acc_mlp
    print(f"\nQuantum Advantage: {adv:+.1f}%")
    print("SPATIAL QFM WINS!" if adv > 0 else "MLP WINS!")
    
    with open('results/spatial_qfm_results.json', 'w') as f:
        json.dump(results, f, indent=2)
