"""
Goldilocks Boolean Datasets
===========================
Synthetic datasets with parity/XOR structure where quantum circuits excel.
"""
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


def generate_parity(n_bits: int = 4, n_samples: int = 1000):
    """
    N-bit Parity (XOR) dataset.
    Label = XOR of all input bits.
    
    Quantum advantage: Two-body ⟨ZᵢZⱼ⟩ measurements directly compute correlations.
    """
    X = np.random.randint(0, 2, size=(n_samples, n_bits)).astype(np.float32)
    y = (X.sum(axis=1) % 2).astype(np.int64)  # XOR = sum mod 2
    
    # Scale to [0, π] for angle encoding
    X_scaled = X * np.pi
    
    return TensorDataset(torch.from_numpy(X_scaled), torch.from_numpy(y))


def generate_majority(n_bits: int = 5, n_samples: int = 1000):
    """
    Majority voting dataset.
    Label = 1 if more than half of bits are 1.
    
    Quantum advantage: Threshold functions benefit from correlation terms.
    """
    X = np.random.randint(0, 2, size=(n_samples, n_bits)).astype(np.float32)
    y = (X.sum(axis=1) > n_bits / 2).astype(np.int64)
    
    X_scaled = X * np.pi
    return TensorDataset(torch.from_numpy(X_scaled), torch.from_numpy(y))


def generate_hidden_parity(n_bits: int = 4, n_distractors: int = 4, n_samples: int = 1000):
    """
    Hidden parity with distractor features.
    Only the first n_bits determine the label (XOR), rest are noise.
    
    Quantum advantage: QFM can filter noise through correlation structure.
    """
    total_bits = n_bits + n_distractors
    X = np.random.randint(0, 2, size=(n_samples, total_bits)).astype(np.float32)
    y = (X[:, :n_bits].sum(axis=1) % 2).astype(np.int64)  # XOR of first n_bits only
    
    X_scaled = X * np.pi
    return TensorDataset(torch.from_numpy(X_scaled), torch.from_numpy(y))


def get_boolean_dataloader(task: str, batch_size: int = 32, **kwargs):
    """Get a DataLoader for a boolean task."""
    generators = {
        "parity": generate_parity,
        "majority": generate_majority,
        "hidden_parity": generate_hidden_parity,
    }
    
    if task not in generators:
        raise ValueError(f"Unknown task: {task}. Available: {list(generators.keys())}")
    
    dataset = generators[task](**kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Registry for benchmark runner
BOOLEAN_TASKS = {
    "parity_4bit": {"task": "parity", "n_bits": 4, "n_samples": 1000},
    "parity_8bit": {"task": "parity", "n_bits": 8, "n_samples": 1000},
    "majority_5bit": {"task": "majority", "n_bits": 5, "n_samples": 1000},
    "hidden_parity_4_4": {"task": "hidden_parity", "n_bits": 4, "n_distractors": 4, "n_samples": 1000},
}
