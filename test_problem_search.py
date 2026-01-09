"""
Problem Class Search Runner
Systematically test Quantum vs Classical models on the benchmark suite.
"""
import torch
import torch.nn as nn
import time
import json
import numpy as np
import tensorcircuit as tc
from utils import setup_mps_environment
from benchmarks import BENCHMARKS, get_benchmark_dataloader

K = tc.set_backend("pytorch")

# ============= MODELS =============

class QuantumModel(nn.Module):
    def __init__(self, n_qubits=4, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        
        def quantum_fn(inputs, weights):
            c = tc.Circuit(n_qubits)
            # Encoding
            for i in range(n_qubits):
                c.ry(i, theta=inputs[i])
            # Trainable layers
            for l in range(n_layers):
                for i in range(n_qubits):
                    c.cnot(i, (i + 1) % n_qubits)
                for i in range(n_qubits):
                    c.ry(i, theta=weights[l, i])
            
            # Measure all Z for dense info
            return K.stack([c.expectation([tc.gates.z(), [i]]) for i in range(n_qubits)])
            
        self.vmapped_fn = K.vmap(quantum_fn, vectorized_argnums=0)
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits))
        self.classifier = nn.Linear(n_qubits, 2)

    def forward(self, x):
        features = self.vmapped_fn(x, self.weights).real
        return self.classifier(features)

class ClassicalModel(nn.Module):
    def __init__(self, n_inputs=4, hidden_dim=16):
        super().__init__()
        # MLP matching parameter count somewhat or just capacity
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        return self.net(x)

class ClassicalKernelModel(nn.Module):
    def __init__(self, n_inputs=4, degree=2):
        super().__init__()
        self.degree = degree
        # Manual polynomial expansion approx
        self.classifier = nn.LazyLinear(2) 

    def forward(self, x):
        # Create poly features: x, x^2, ...
        features = [x]
        for d in range(2, self.degree + 1):
            features.append(x.pow(d))
        # Interaction terms? Just simple powers for now + dense layer
        # Or just use a wider MLP as "approx kernel"
        flat = torch.cat(features, dim=1)
        return self.classifier(flat)

# ============= TRAINING ENGINE =============

def train_and_eval(model, train_loader, test_loader, epochs=30, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_acc': [], 'test_acc': []}
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Train
        model.train()
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        train_acc = correct / total
        
        # Eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        test_acc = correct / total
        
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
    
    duration = time.time() - start_time
    return max(history['test_acc']), duration

# ============= MAIN RUNNER =============

def main():
    setup_mps_environment()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running benchmark search on {device}...")
    
    results = {}
    
    for task_name in BENCHMARKS:
        print(f"\n--- Benchmarking Task: {task_name} ---")
        # Prepare device string for generator
        gen_dev = 'mps' if device.type == 'mps' else 'cpu'
        train_dl, test_dl = get_benchmark_dataloader(task_name, n_samples=500, batch_size=32, generator_device=gen_dev)
        
        # 1. Quantum Model
        q_model = QuantumModel(n_qubits=4, n_layers=2).to(device)
        q_acc, q_time = train_and_eval(q_model, train_dl, test_dl, epochs=20, device=device)
        print(f"  Quantum:   Acc={q_acc:.4f} (Time={q_time:.2f}s)")
        
        # 2. Classical MLP (Linear + Non-Linearity)
        c_model = ClassicalModel(n_inputs=4, hidden_dim=8).to(device) # Small MLP
        c_acc, c_time = train_and_eval(c_model, train_dl, test_dl, epochs=20, device=device)
        print(f"  Classical: Acc={c_acc:.4f} (Time={c_time:.2f}s)")
        
        results[task_name] = {
            "Quantum": q_acc,
            "Classical": c_acc,
            "Advantage": q_acc - c_acc
        }
    
    print("\n" + "="*60)
    print("SEARCH RESULTS: Quantum Advantage Candidates")
    print("="*60)
    print(f"{'Task':<25} | {'Quantum':<8} | {'Classical':<8} | {'Advantage'}")
    print("-" * 60)
    
    saved_candidates = []
    
    for task, metrics in results.items():
        adv = metrics["Advantage"]
        marker = "**" if adv > 0.1 else ""
        print(f"{task:<25} | {metrics['Quantum']:.4f}   | {metrics['Classical']:.4f}   | {adv:+.4f} {marker}")
        
        if adv > 0.05:
            saved_candidates.append(task)
            
    # Save results
    with open("results/problem_class_search.json", "w") as f:
        json.dump(results, f, indent=2)
    
    if saved_candidates:
        print(f"\nPotential Goldilocks Problems found: {saved_candidates}")
    else:
        print("\nNo significant advantage found yet. Try harder tasks.")

if __name__ == "__main__":
    main()
