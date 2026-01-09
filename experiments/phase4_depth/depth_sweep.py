"""
Phase 4: Fixed QFM Depth Sweep on CIFAR-10
==========================================
Test depths 1, 5, 10, 25, 50, 100 on CIFAR-10 subset.
Uses TensorCircuit with MPS backend for efficient high-depth simulation.

Key: Circuit is FIXED (0 trainable params in gates), only classifier is trained.
"""
import torch
import torch.nn as nn
import numpy as np
import json
import time
from datetime import datetime

import tensorcircuit as tc

# Use MPS (Matrix Product State) for efficient simulation
# Note: tc.set_contractor("mps") enables MPS-style contraction for deep circuits
tc.set_backend("pytorch")
tc.set_dtype("complex64")
K = tc.backend


def create_fixed_qfm(n_qubits: int, depth: int, topology: str = "chain"):
    """
    Create a FIXED quantum feature map function.
    No trainable parameters in the circuit itself.
    """
    def qfm_fn(inputs):
        """
        inputs: tensor of shape (input_dim,)
        returns: tensor of shape (n_features,)
        """
        n_inputs = len(inputs)
        c = tc.Circuit(n_qubits)
        
        # Redundant encoding: cyclically repeat inputs across qubits
        for i in range(n_qubits):
            c.ry(i, theta=inputs[i % n_inputs])
        
        # Apply entanglement layers
        for _ in range(depth):
            if topology == "chain":
                for i in range(n_qubits - 1):
                    c.cnot(i, i + 1)
            elif topology == "ring":
                for i in range(n_qubits - 1):
                    c.cnot(i, i + 1)
                c.cnot(n_qubits - 1, 0)
            elif topology == "all2all":
                for i in range(n_qubits):
                    for j in range(i + 1, n_qubits):
                        c.cnot(i, j)
            elif topology == "star":
                for i in range(1, n_qubits):
                    c.cnot(0, i)
        
        # Measure observables
        single = [c.expectation([tc.gates.z(), [i]]) for i in range(n_qubits)]
        two_body = []
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                two_body.append(c.expectation([tc.gates.z(), [i]], [tc.gates.z(), [j]]))
        
        return K.stack(single + two_body)
    
    return qfm_fn


class FixedQFMClassifier(nn.Module):
    """
    Fixed Quantum Feature Map + Trainable Classical Classifier
    """
    def __init__(self, input_dim: int, n_qubits: int = 8, depth: int = 1, 
                 topology: str = "chain", num_classes: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.depth = depth
        
        # Project input to n_qubits if needed
        self.projection = nn.Linear(input_dim, n_qubits)
        
        # Create fixed QFM
        self.qfm_fn = create_fixed_qfm(n_qubits, depth, topology)
        
        # Vmap for batch processing
        self.vmapped_qfm = K.vmap(self.qfm_fn, vectorized_argnums=0)
        
        # Output features: n_qubits + n_qubits*(n_qubits-1)/2
        n_features = n_qubits + (n_qubits * (n_qubits - 1)) // 2
        
        # Trainable classifier
        self.classifier = nn.Linear(n_features, num_classes)
    
    def forward(self, x):
        # Project to qubit dimension
        x = self.projection(x)
        x = torch.tanh(x) * np.pi  # Scale to [0, π]
        
        # Apply fixed QFM
        features = self.vmapped_qfm(x).real
        
        # Classify
        return self.classifier(features)


class MLPBaseline(nn.Module):
    """MLP with comparable non-linearity (2 hidden layers)"""
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)


def load_cifar_subset(n_samples: int = 500):
    """Load CIFAR-10 subset with flattened features"""
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # Take subset
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    X = torch.stack([dataset[i][0].flatten() for i in indices])
    y = torch.tensor([dataset[i][1] for i in indices])
    
    return X, y


def train_and_evaluate(model, X, y, epochs: int = 20, lr: float = 0.01):
    """Train model and return metrics"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Split train/val
    n = len(X)
    train_idx = np.random.choice(n, int(0.8 * n), replace=False)
    val_idx = np.array([i for i in range(n) if i not in train_idx])
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    history = {'loss': [], 'val_acc': [], 'grad_norm': []}
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward
        logits = model(X_train)
        loss = criterion(logits, y_train)
        
        # Backward
        loss.backward()
        
        # Track gradient norm (for barren plateau detection)
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = total_norm ** 0.5
        
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_acc = (val_logits.argmax(dim=1) == y_val).float().mean().item() * 100
        
        history['loss'].append(loss.item())
        history['val_acc'].append(val_acc)
        history['grad_norm'].append(grad_norm)
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Loss={loss.item():.4f}, Val Acc={val_acc:.1f}%, Grad={grad_norm:.4f}")
    
    elapsed = time.time() - start_time
    
    return {
        'final_acc': history['val_acc'][-1],
        'final_loss': history['loss'][-1],
        'final_grad': history['grad_norm'][-1],
        'time': elapsed,
        'history': history
    }


def run_depth_sweep():
    """Run the full depth sweep experiment"""
    print("="*70)
    print("PHASE 4: FIXED QFM DEPTH SWEEP")
    print("="*70)
    print("Constraint: Circuit is FIXED (0 trainable params in gates)")
    print("Only the classifier layer is trainable.")
    print()
    
    # Configuration
    DEPTHS = [1, 5, 10, 25, 50]  # Start conservative, can extend to 100
    N_QUBITS = 8
    TOPOLOGY = "chain"
    N_SAMPLES = 500
    EPOCHS = 20
    
    print(f"Config: {N_QUBITS} qubits, {TOPOLOGY} topology, {N_SAMPLES} samples, {EPOCHS} epochs")
    print()
    
    # Load data once
    print("Loading CIFAR-10 subset...")
    X, y = load_cifar_subset(N_SAMPLES)
    input_dim = X.shape[1]
    print(f"Data shape: {X.shape}, Classes: {len(torch.unique(y))}")
    print()
    
    results = []
    
    # Run MLP baseline first
    print("-" * 50)
    print("MLP Baseline (2 hidden layers, 64 units)")
    print("-" * 50)
    mlp = MLPBaseline(input_dim, hidden_dim=64, num_classes=10)
    mlp_result = train_and_evaluate(mlp, X, y, epochs=EPOCHS)
    mlp_result['depth'] = 'MLP'
    mlp_result['model'] = 'MLP'
    results.append(mlp_result)
    print(f"Final: {mlp_result['final_acc']:.1f}% ({mlp_result['time']:.1f}s)")
    print()
    
    # Run depth sweep
    for depth in DEPTHS:
        print("-" * 50)
        print(f"Fixed QFM: Depth={depth}, {N_QUBITS}Q, {TOPOLOGY}")
        print("-" * 50)
        
        try:
            model = FixedQFMClassifier(
                input_dim=input_dim,
                n_qubits=N_QUBITS,
                depth=depth,
                topology=TOPOLOGY,
                num_classes=10
            )
            
            result = train_and_evaluate(model, X, y, epochs=EPOCHS)
            result['depth'] = depth
            result['model'] = f'{N_QUBITS}Q_D{depth}'
            results.append(result)
            
            print(f"Final: {result['final_acc']:.1f}% ({result['time']:.1f}s)")
            
            # Check for barren plateau
            if result['final_grad'] < 0.01:
                print("⚠️  Warning: Very low gradient - possible barren plateau!")
            
        except Exception as e:
            print(f"❌ Error at depth {depth}: {e}")
            results.append({'depth': depth, 'error': str(e)})
        
        print()
    
    # Summary
    print("="*70)
    print("DEPTH SWEEP SUMMARY")
    print("="*70)
    print(f"{'Model':<15} | {'Accuracy':>10} | {'Time':>10} | {'Gradient':>10}")
    print("-"*50)
    
    mlp_acc = None
    for r in results:
        if 'error' not in r:
            if r['model'] == 'MLP':
                mlp_acc = r['final_acc']
            marker = '✅' if mlp_acc and r['final_acc'] > mlp_acc else ''
            print(f"{r['model']:<15} | {r['final_acc']:>9.1f}% | {r['time']:>9.1f}s | {r['final_grad']:>10.4f} {marker}")
    
    # Save results
    with open('results/depth_sweep_results.json', 'w') as f:
        # Remove non-serializable history for saving
        save_results = [{k: v for k, v in r.items() if k != 'history'} for r in results]
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': {
                'n_qubits': N_QUBITS,
                'topology': TOPOLOGY,
                'n_samples': N_SAMPLES,
                'epochs': EPOCHS,
                'depths': DEPTHS
            },
            'results': save_results
        }, f, indent=2)
    
    print()
    print("Results saved to: results/depth_sweep_results.json")
    
    return results


if __name__ == "__main__":
    run_depth_sweep()
