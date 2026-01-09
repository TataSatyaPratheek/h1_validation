"""
Goldilocks Molecular Datasets
=============================
Real-world molecular property prediction datasets from MoleculeNet.
These are scientifically significant and may exhibit quantum advantage.

Data Source: MoleculeNet (https://moleculenet.org/)
Access via: pip install deepchem

Molecular fingerprints are high-dimensional binary vectors (1024 bits)
representing molecular substructures. The classification depends on
complex correlations between fingerprint bits.
"""
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import warnings


def load_tox21(max_samples: int = None, fingerprint_bits: int = 1024):
    """
    Load Tox21 toxicity dataset.
    
    Task: Predict 12 different toxicity endpoints (we use first one for simplicity).
    Samples: ~8,000 molecules
    Features: ECFP4 molecular fingerprints (1024 bits)
    
    Quantum advantage hypothesis: Molecular properties depend on electron correlations,
    which may be naturally captured by two-body quantum observables.
    
    Requires: pip install deepchem
    """
    try:
        import deepchem as dc
        from deepchem.molnet import load_tox21
    except ImportError:
        raise ImportError(
            "Tox21 requires deepchem. Install with: pip install deepchem"
        )
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Load with ECFP fingerprints
        featurizer = dc.feat.CircularFingerprint(size=fingerprint_bits, radius=2)
        tasks, datasets, transformers = load_tox21(featurizer=featurizer)
        
        train, valid, test = datasets
        
        # Combine for simplicity (we'll do our own split)
        X = np.vstack([train.X, valid.X, test.X]).astype(np.float32)
        y = np.vstack([train.y, valid.y, test.y])
        
        # Use first task (NR-AR), handle missing values
        y = y[:, 0]
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask].astype(np.int64)
        
        if max_samples and len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X, y = X[indices], y[indices]
        
        # Scale to [0, π] - fingerprints are already binary 0/1
        X = X * np.pi
        
        return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))


def load_bace(max_samples: int = None, fingerprint_bits: int = 1024):
    """
    Load BACE binding affinity dataset.
    
    Task: Predict binding to BACE1 enzyme (binary classification)
    Samples: ~1,500 molecules
    Features: ECFP4 fingerprints
    
    Requires: pip install deepchem
    """
    try:
        import deepchem as dc
        from deepchem.molnet import load_bace_classification
    except ImportError:
        raise ImportError(
            "BACE requires deepchem. Install with: pip install deepchem"
        )
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        featurizer = dc.feat.CircularFingerprint(size=fingerprint_bits, radius=2)
        tasks, datasets, transformers = load_bace_classification(featurizer=featurizer)
        
        train, valid, test = datasets
        
        X = np.vstack([train.X, valid.X, test.X]).astype(np.float32)
        y = np.hstack([train.y.flatten(), valid.y.flatten(), test.y.flatten()]).astype(np.int64)
        
        if max_samples and len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X, y = X[indices], y[indices]
        
        X = X * np.pi
        
        return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))


def load_bbbp(max_samples: int = None, fingerprint_bits: int = 1024):
    """
    Load BBBP (Blood-Brain Barrier Penetration) dataset.
    
    Task: Predict if molecule can cross blood-brain barrier (binary)
    Samples: ~2,000 molecules
    Features: ECFP4 fingerprints
    
    Requires: pip install deepchem
    """
    try:
        import deepchem as dc
        from deepchem.molnet import load_bbbp
    except ImportError:
        raise ImportError(
            "BBBP requires deepchem. Install with: pip install deepchem"
        )
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        featurizer = dc.feat.CircularFingerprint(size=fingerprint_bits, radius=2)
        tasks, datasets, transformers = load_bbbp(featurizer=featurizer)
        
        train, valid, test = datasets
        
        X = np.vstack([train.X, valid.X, test.X]).astype(np.float32)
        y = np.hstack([train.y.flatten(), valid.y.flatten(), test.y.flatten()]).astype(np.int64)
        
        if max_samples and len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X, y = X[indices], y[indices]
        
        X = X * np.pi
        
        return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))


def get_molecular_dataloader(task: str, batch_size: int = 32, **kwargs):
    """Get a DataLoader for a molecular task."""
    loaders = {
        "tox21": load_tox21,
        "bace": load_bace,
        "bbbp": load_bbbp,
    }
    
    if task not in loaders:
        raise ValueError(f"Unknown task: {task}. Available: {list(loaders.keys())}")
    
    dataset = loaders[task](**kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Registry for benchmark runner
MOLECULAR_TASKS = {
    "tox21_1k": {"task": "tox21", "max_samples": 1000, "fingerprint_bits": 64},  # Reduced for quantum
    "bace_full": {"task": "bace", "max_samples": None, "fingerprint_bits": 64},
    "bbbp_1k": {"task": "bbbp", "max_samples": 1000, "fingerprint_bits": 64},
}
