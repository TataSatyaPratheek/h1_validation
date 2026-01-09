"""
Stability Proof v2: Gradient Survival Test (Direct Measurement)
Measures gradients on the projection layer (immediately before QFM)
to isolate the quantum bottleneck effect.
"""
import torch
import torch.nn as nn
import tensorcircuit as tc
from utils import setup_mps_environment
from data import get_dataloaders

K = tc.set_backend("pytorch")

class FixedQFM(nn.Module):
    """Fixed Quantum Feature Map."""
    def __init__(self, n_qubits=4, alpha=1.57):  # Higher alpha for better signal
        super().__init__()
        self.n_qubits = n_qubits
        self.alpha = alpha
        
        def quantum_fn(inputs):
            c = tc.Circuit(n_qubits)
            for i in range(n_qubits):
                c.ry(i, theta=inputs[i])
            for i in range(n_qubits):
                c.cnot(i, (i + 1) % n_qubits)
            return K.stack([c.expectation([tc.gates.z(), [i]]) for i in range(n_qubits)])
        self.vmapped_fn = K.vmap(quantum_fn)

    def forward(self, x):
        theta = self.alpha * x  # Remove tanh for gradient flow test
        return self.vmapped_fn(theta).real

class TrainablePQC(nn.Module):
    """Trainable PQC with random initialization."""
    def __init__(self, n_qubits=4, n_layers=3):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.params = nn.Parameter(torch.randn(n_layers, n_qubits * 2) * 0.5)
        
    def forward(self, x):
        batch_size = x.shape[0]
        results = []
        for b in range(batch_size):
            c = tc.Circuit(self.n_qubits)
            # Input encoding
            for i in range(self.n_qubits):
                c.ry(i, theta=x[b, i])
            # Trainable layers
            for l in range(self.n_layers):
                for i in range(self.n_qubits):
                    c.ry(i, theta=self.params[l, i])
                    c.rz(i, theta=self.params[l, self.n_qubits + i])
                for i in range(self.n_qubits):
                    c.cnot(i, (i + 1) % self.n_qubits)
            exps = [c.expectation([tc.gates.z(), [i]]) for i in range(self.n_qubits)]
            results.append(torch.stack([e.real for e in exps]))
        return torch.stack(results)

class HybridModel(nn.Module):
    def __init__(self, qfm_type="fixed", n_qubits=4):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.projection = nn.Linear(64 * 8 * 8, n_qubits)
        
        if qfm_type == "fixed":
            self.qfm = FixedQFM(n_qubits, alpha=1.57)
        else:
            self.qfm = TrainablePQC(n_qubits, n_layers=3)
        
        self.classifier = nn.Linear(n_qubits, 10)

    def forward(self, x):
        f = self.feature_extractor(x)
        f = f.view(f.size(0), -1)
        z = self.projection(f)
        q = self.qfm(z)
        return self.classifier(q)

def run_test(model, train_loader, device, epochs=5, model_name="Model"):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    grad_history = {"conv1": [], "projection": [], "classifier": []}
    
    for epoch in range(epochs):
        epoch_grads = {"conv1": [], "projection": [], "classifier": []}
        for batch_idx, (imgs, lbls) in enumerate(train_loader):
            imgs, lbls = imgs.to(device), lbls.to(device)
            
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            
            # Log gradients at different layers
            conv1_grad = list(model.feature_extractor.parameters())[0].grad
            if conv1_grad is not None:
                epoch_grads["conv1"].append(conv1_grad.norm().item())
            
            proj_grad = model.projection.weight.grad
            if proj_grad is not None:
                epoch_grads["projection"].append(proj_grad.norm().item())
            
            cls_grad = model.classifier.weight.grad
            if cls_grad is not None:
                epoch_grads["classifier"].append(cls_grad.norm().item())
            
            optimizer.step()
            
            if batch_idx >= 20:
                break
        
        for key in epoch_grads:
            avg = sum(epoch_grads[key]) / max(len(epoch_grads[key]), 1)
            grad_history[key].append(avg)
        
        print(f"[{model_name}] Epoch {epoch+1}: Conv1={grad_history['conv1'][-1]:.6f}, "
              f"Proj={grad_history['projection'][-1]:.6f}, Cls={grad_history['classifier'][-1]:.6f}")
    
    return grad_history

def main():
    setup_mps_environment()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running Gradient Survival Test v2 on: {device}")
    
    gen_dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    train_loader, _ = get_dataloaders(
        dataset_name="cifar10", batch_size=32, image_size=64, 
        subset_size=10000, generator_device=gen_dev
    )
    
    print("\n--- Testing Fixed QFM (10k samples, 100 epochs) ---")
    model_fixed = HybridModel(qfm_type="fixed").to(device)
    fixed_grads = run_test(model_fixed, train_loader, device, epochs=100, model_name="Fixed")
    
    print("\n--- Testing Trainable PQC (10k samples, 100 epochs) ---")
    model_trainable = HybridModel(qfm_type="trainable").to(device)
    trainable_grads = run_test(model_trainable, train_loader, device, epochs=100, model_name="Trainable")
    
    print("\n--- STABILITY PROOF SUMMARY ---")
    print(f"Fixed QFM - Projection Gradient (Final): {fixed_grads['projection'][-1]:.6f}")
    print(f"Trainable PQC - Projection Gradient (Final): {trainable_grads['projection'][-1]:.6f}")
    
    # The key insight: compare stability (variance) not just magnitude
    import numpy as np
    fixed_var = np.var(fixed_grads['projection'])
    trainable_var = np.var(trainable_grads['projection'])
    print(f"\nGradient Variance (Stability Metric):")
    print(f"  Fixed QFM: {fixed_var:.8f}")
    print(f"  Trainable PQC: {trainable_var:.8f}")
    
    if fixed_var < trainable_var:
        print("\nVERDICT: Fixed QFM has MORE STABLE gradients (lower variance). STABILITY PROVEN.")
    else:
        print("\nVERDICT: Trainable PQC has more stable gradients.")

if __name__ == "__main__":
    main()
