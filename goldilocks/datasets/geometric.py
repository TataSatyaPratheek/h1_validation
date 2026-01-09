"""
Goldilocks Geometric Datasets
=============================
Synthetic datasets with radial/symmetry structure where quantum circuits excel.
"""
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_swiss_roll, make_moons, make_circles


def generate_circle(n_samples: int = 1000, noise: float = 0.1, pad_to: int = 4):
    """
    Circle (radial separation) dataset.
    Inner circle = class 0, outer ring = class 1.
    
    Quantum advantage: Chain topology creates "frequency doubling" for radial boundaries.
    """
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5)
    X = X.astype(np.float32)
    y = y.astype(np.int64)
    
    # Pad to desired dimension for quantum encoding
    if pad_to > 2:
        padding = np.zeros((n_samples, pad_to - 2), dtype=np.float32)
        X = np.hstack([X, padding])
    
    # Scale to [0, π]
    X = (X - X.min()) / (X.max() - X.min()) * np.pi
    
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))


def generate_concentric_rings(n_samples: int = 1000, n_rings: int = 3, noise: float = 0.1, pad_to: int = 4):
    """
    Concentric rings dataset (multi-class).
    
    Quantum advantage: Multiple radial boundaries test the frequency resolution.
    """
    samples_per_ring = n_samples // n_rings
    X_list, y_list = [], []
    
    for ring_idx in range(n_rings):
        radius = 0.5 + ring_idx * 0.5
        theta = np.random.uniform(0, 2 * np.pi, samples_per_ring)
        r = radius + np.random.normal(0, noise, samples_per_ring)
        
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        X_list.append(np.stack([x, y], axis=1))
        y_list.append(np.full(samples_per_ring, ring_idx))
    
    X = np.vstack(X_list).astype(np.float32)
    y = np.hstack(y_list).astype(np.int64)
    
    # Pad to desired dimension
    if pad_to > 2:
        padding = np.zeros((len(X), pad_to - 2), dtype=np.float32)
        X = np.hstack([X, padding])
    
    # Scale to [0, π]
    X = (X - X.min()) / (X.max() - X.min()) * np.pi
    
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))


def generate_swiss_roll(n_samples: int = 1000, noise: float = 0.5):
    """
    Swiss Roll manifold dataset.
    3D manifold that tests manifold unfolding capability.
    
    Quantum advantage: Unknown - this is exploratory.
    """
    X, t = make_swiss_roll(n_samples=n_samples, noise=noise)
    X = X.astype(np.float32)
    
    # Binary classification based on position on roll
    y = (t > np.median(t)).astype(np.int64)
    
    # Scale to [0, π]
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8) * np.pi
    
    # Pad to 4D for quantum encoding
    padding = np.zeros((n_samples, 1), dtype=np.float32)
    X = np.hstack([X, padding])
    
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))


def generate_spiral(n_samples: int = 1000, noise: float = 0.2, pad_to: int = 4):
    """
    Two-arm spiral dataset.
    Classic non-linear classification problem.
    """
    n_per_class = n_samples // 2
    
    # Spiral 1
    theta1 = np.sqrt(np.random.rand(n_per_class)) * 4 * np.pi
    r1 = theta1 / (4 * np.pi) + np.random.randn(n_per_class) * noise
    x1 = r1 * np.cos(theta1)
    y1 = r1 * np.sin(theta1)
    
    # Spiral 2 (rotated 180 degrees)
    theta2 = np.sqrt(np.random.rand(n_per_class)) * 4 * np.pi
    r2 = theta2 / (4 * np.pi) + np.random.randn(n_per_class) * noise
    x2 = -r2 * np.cos(theta2)
    y2 = -r2 * np.sin(theta2)
    
    X = np.vstack([
        np.stack([x1, y1], axis=1),
        np.stack([x2, y2], axis=1)
    ]).astype(np.float32)
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)]).astype(np.int64)
    
    # Pad
    if pad_to > 2:
        padding = np.zeros((len(X), pad_to - 2), dtype=np.float32)
        X = np.hstack([X, padding])
    
    # Scale
    X = (X - X.min()) / (X.max() - X.min() + 1e-8) * np.pi
    
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))


def get_geometric_dataloader(task: str, batch_size: int = 32, **kwargs):
    """Get a DataLoader for a geometric task."""
    generators = {
        "circle": generate_circle,
        "concentric_rings": generate_concentric_rings,
        "swiss_roll": generate_swiss_roll,
        "spiral": generate_spiral,
    }
    
    if task not in generators:
        raise ValueError(f"Unknown task: {task}. Available: {list(generators.keys())}")
    
    dataset = generators[task](**kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Registry for benchmark runner
GEOMETRIC_TASKS = {
    "circle_2d": {"task": "circle", "n_samples": 1000, "noise": 0.1, "pad_to": 4},
    "rings_3class": {"task": "concentric_rings", "n_samples": 1200, "n_rings": 3, "noise": 0.1, "pad_to": 4},
    "swiss_roll": {"task": "swiss_roll", "n_samples": 1000, "noise": 0.5},
    "spiral_2d": {"task": "spiral", "n_samples": 1000, "noise": 0.2, "pad_to": 4},
}
