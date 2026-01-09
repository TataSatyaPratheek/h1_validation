"""
Architecture Search
Systematic sweep of Qubits, Depth, and Topology on the Circle Task.
"""
import torch
import torch.nn as nn
import json
import itertools
import numpy as np
import tensorcircuit as tc
from utils import setup_mps_environment
from benchmarks import generate_circle_data

K = tc.set_backend("pytorch")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ============= FLEXIBLE CIRCUIT FACTORY =============

def create_circuit(n_qubits, depth, topology, inputs, weights):
    """
    Creates a circuit with specified configuration.
    inputs: [batch, n_qubits] (or subset if n_features < n_qubits)
    weights: [depth, n_qubits]
    """
    c = tc.Circuit(n_qubits)
    
    # Encoding Layer (Ry)
    # If inputdim < n_qubits, repeat inputs
    # inputs are usually [batch, 4]. 
    # If n_qubits=8, we map input[0]->q0, input[1]->q1... input[0]->q4...
    n_inputs = inputs.shape[-1]
    
    for i in range(n_qubits):
        # Redundant encoding scheme
        val = inputs[i % n_inputs]
        c.ry(i, theta=val)
    
    # Variational Layers
    for l in range(depth):
        # Entanglement Layer
        if topology == "ring":
            for i in range(n_qubits):
                c.cnot(i, (i + 1) % n_qubits)
        elif topology == "chain":
            for i in range(n_qubits - 1):
                c.cnot(i, i + 1)
        elif topology == "all2all": # Expensive!
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    c.cnot(i, j)
        elif topology == "pairwise_block": # Even/Odd blocks (like Matchgate)
             for i in range(0, n_qubits - 1, 2):
                 c.cnot(i, i + 1)
             for i in range(1, n_qubits - 1, 2):
                 c.cnot(i, i + 1)
        
        # Rotation Layer
        for i in range(n_qubits):
            c.ry(i, theta=weights[l, i])
            
    return K.stack([c.expectation([tc.gates.z(), [i]]) for i in range(n_qubits)])

class SearchableQuantumModel(nn.Module):
    def __init__(self, n_qubits=4, depth=2, topology="ring", n_features=4):
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        self.topology = topology
        
        # We define a localized function for vmap to pick up
        def quantum_fn(inputs, weights):
            return create_circuit(n_qubits, depth, topology, inputs, weights)
            
        self.vmapped_fn = K.vmap(quantum_fn, vectorized_argnums=0)
        self.weights = nn.Parameter(torch.randn(depth, n_qubits))
        self.classifier = nn.Linear(n_qubits, 2)

    def forward(self, x):
        features = self.vmapped_fn(x, self.weights).real
        return self.classifier(features)

# ============= TRAINING LOOP =============

def train_and_eval(model, data, epochs=30):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    X, Y = data.tensors
    X, Y = X.to(device), Y.to(device)
    
    best_acc = 0.0
    
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, Y)
        loss.backward()
        optimizer.step()
        
        # Simple accuracy check
        pred = out.argmax(dim=1)
        acc = (pred == Y).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            
    return best_acc

def main():
    setup_mps_environment()
    print("="*60)
    print("Architecture Search: Circle Task")
    print("="*60)
    
    # Config Search Space
    QUBITS = [4, 8] # Redundant encoding on 8
    DEPTHS = [1, 2, 4]
    TOPOLOGIES = ["chain", "ring", "all2all"]
    
    # Dataset
    data = generate_circle_data(n_samples=200) # Small for speed
    
    results = []
    
    print(f"{'Qubits':<6} | {'Depth':<6} | {'Topology':<10} | {'Acc':<6}")
    print("-" * 40)
    
    for q, d, t in itertools.product(QUBITS, DEPTHS, TOPOLOGIES):
        try:
            model = SearchableQuantumModel(n_qubits=q, depth=d, topology=t, n_features=4).to(device)
            acc = train_and_eval(model, data, epochs=25)
            
            print(f"{q:<6} | {d:<6} | {t:<10} | {acc:.4f}")
            
            results.append({
                "qubits": q,
                "depth": d,
                "topology": t,
                "accuracy": acc
            })
        except Exception as e:
            print(f"{q:<6} | {d:<6} | {t:<10} | FAILED: {e}")
    
    # Save results
    with open("results/architecture_search.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\nSearch Complete.")
    best = max(results, key=lambda x: x['accuracy'])
    print(f"Best Config: {best}")

if __name__ == "__main__":
    main()
