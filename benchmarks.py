"""
Antigravity Benchmarks Suite
Synthetic datasets designed to test specific hypothesis about Quantum Feature Map capabilities.

Hypothesis: QFMs excel at:
1. Global Correlations (Parity/XOR) vs Local Features
2. Fourier-based boundaries (Periodic/Modular)
3. Topological Separation (Spirals/Circles)
"""
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def generate_parity_data(n_samples=1000, n_bits=4):
    """
    Task: XOR of all bits. 
    Class 0: Even distinct bits set
    Class 1: Odd distinct bits set
    Difficult for linear models (requires N-th order interaction).
    """
    X = torch.randint(0, 2, (n_samples, n_bits)).float()
    # Label is sum modulo 2
    Y = (X.sum(dim=1) % 2).long()
    
    # Scale to [0, pi] for quantum encoding
    X_scaled = X * np.pi
    
    return TensorDataset(X_scaled, Y)

def generate_modulo_sum_data(n_samples=1000, n_features=4, modulus=3):
    """
    Task: (Sum of inputs) mod k.
    Tests ability to model periodic functions.
    Quantum Ry gates (rotations) should be naturally periodic.
    """
    # Inputs in range [0, k]
    X = torch.rand(n_samples, n_features) * modulus * 2
    
    # Sum and modulo
    sum_vals = X.sum(dim=1)
    Y = (sum_vals.long() % modulus).long()
    
    # For binary classification, let's just take class 0 vs others
    # Or keep it multi-class. Let's stick to binary for now:
    # Class 0 if sum mod 3 == 0, else Class 1
    Y = (Y == 0).long() 
    
    # Scale for encoding. 
    # If Ry(theta) has period 4pi (bloch sphere 2pi), 
    # we want input range to map nicely. 
    X_scaled = X 
    
    return TensorDataset(X_scaled, Y)

def generate_spiral_data(n_samples=1000):
    """
    Task: Two interwined spirals.
    Tests topological connectedness separation.
    Hard for linear, easy for Kernel methods / Deep MLPs.
    """
    n = n_samples // 2
    dim = 2 # 2D spiral
    
    def spiral(delta_t, label):
        r = torch.linspace(1, 15, n)  # radius
        t = torch.linspace(0, 4*np.pi, n) + delta_t  # theta
        
        x = r * torch.cos(t)
        y = r * torch.sin(t)
        
        data = torch.stack([x, y], dim=1)
        # Normalize to [-1, 1] range roughly
        data = data / 15.0 
        
        # Scale to [0, pi] for quantum
        data = (data + 1) * (np.pi / 2) 
        
        labels = torch.ones(n).long() * label
        return data, labels

    X0, Y0 = spiral(0, 0)
    X1, Y1 = spiral(np.pi, 1)
    
    X = torch.cat([X0, X1])
    Y = torch.cat([Y0, Y1])
    
    # Shuffle
    idx = torch.randperm(n_samples)
    X = X[idx]
    Y = Y[idx]
    
    # Pad to 4 features (since our circuits are 4 qubits)
    # We add 2 dummy noise features
    X_padded = torch.zeros(n_samples, 4)
    X_padded[:, :2] = X
    # Noise features
    X_padded[:, 2:] = torch.rand(n_samples, 2) * np.pi 
    
    return TensorDataset(X_padded, Y)

def generate_hidden_correlation(n_samples=1000, n_features=4):
    """
    Task: Class determined by correlation of feature 0 and feature 3 (0-indexed).
    Features 1 and 2 are noise.
    Tests ability to ignore noise and find non-local correlations.
    Like XOR but with noise channels in between.
    """
    X = torch.randint(0, 2, (n_samples, n_features)).float()
    
    # Label = X[0] XOR X[3]
    Y = ((X[:, 0] + X[:, 3]) % 2).long()
    
    # Scale
    X_scaled = X * np.pi
    
    return TensorDataset(X_scaled, Y)

def generate_circle_data(n_samples=1000):
    """
    Task: Points inside circle radius R vs outside.
    Geometry: Decision boundary is a hypersphere.
    """
    X = (torch.rand(n_samples, 4) * 2) - 1 # [-1, 1]
    
    # Radius squared of first 2 dims
    r2 = X[:, 0]**2 + X[:, 1]**2
    Y = (r2 < 0.6).long() # Radius ~0.77
    
    # Scale [-1, 1] to [0, pi]
    X_scaled = (X + 1) * (np.pi / 2)
    
    return TensorDataset(X_scaled, Y)

BENCHMARKS = {
    "parity_4bit": generate_parity_data,
    "modulo_sum": generate_modulo_sum_data,
    "spiral_2d_padded": generate_spiral_data,
    "hidden_correlation_gap": generate_hidden_correlation,
    "circle_2d_padded": generate_circle_data
}

def get_benchmark_dataloader(name, n_samples=500, batch_size=32, generator_device=None):
    if name not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {name}")
    
    ds = BENCHMARKS[name](n_samples=n_samples)
    
    # Split train/test
    train_size = int(0.8 * len(ds))
    test_size = len(ds) - train_size
    
    # Manual shuffle and split to avoid MPS generator/device issues
    # We use CPU for the splitting indices to be safe, data tensors are moved later
    indices = torch.randperm(len(ds)).tolist()
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_ds = torch.utils.data.Subset(ds, train_indices)
    test_ds = torch.utils.data.Subset(ds, test_indices)
    
    # Create valid loader with explicit generator for MPS
    generator = None
    if generator_device == 'mps':
        generator = torch.Generator(device='mps')
        
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, generator=generator)
    
    return train_loader, test_loader

if __name__ == "__main__":
    print("Generating Benchmark Examples...")
    for name in BENCHMARKS:
        ds = BENCHMARKS[name](n_samples=5)
        print(f"\n{name}:")
        print(f"  Input shape: {ds[0][0].shape}")
        print(f"  Example X: {ds[0][0]}")
        print(f"  Example Y: {ds[0][1]}")
