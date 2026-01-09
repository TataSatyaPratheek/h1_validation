"""
Comprehensive 8-Qubit Architecture Search
Sweep Depth and Topology for 8-qubit models across ALL benchmark tasks.
"""
import torch
import torch.nn as nn
import json
import itertools
import numpy as np
import tensorcircuit as tc
from utils import setup_mps_environment
from benchmarks import BENCHMARKS, get_benchmark_dataloader

K = tc.set_backend("pytorch")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ============= CIRCUIT FACTORY (Reused) =============

def create_circuit(n_qubits, depth, topology, inputs, weights):
    c = tc.Circuit(n_qubits)
    n_inputs = inputs.shape[-1]
    
    # Redundant Encoding (Cyclic)
    for i in range(n_qubits):
        val = inputs[i % n_inputs]
        c.ry(i, theta=val)
    
    # Variational Layers
    for l in range(depth):
        # Entanglement
        if topology == "ring":
            for i in range(n_qubits):
                c.cnot(i, (i + 1) % n_qubits)
        elif topology == "chain":
            for i in range(n_qubits - 1):
                c.cnot(i, i + 1)
        elif topology == "all2all":
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    c.cnot(i, j)
        
        # Roation
        for i in range(n_qubits):
            c.ry(i, theta=weights[l, i])
            
    # Measure all Z
    return K.stack([c.expectation([tc.gates.z(), [i]]) for i in range(n_qubits)])

class SearchableQuantumModel(nn.Module):
    def __init__(self, n_qubits=8, depth=2, topology="chain", n_features=4):
        super().__init__()
        self.n_qubits = n_qubits
        
        def quantum_fn(inputs, weights):
            return create_circuit(n_qubits, depth, topology, inputs, weights)
            
        self.vmapped_fn = K.vmap(quantum_fn, vectorized_argnums=0)
        self.weights = nn.Parameter(torch.randn(depth, n_qubits))
        self.classifier = nn.Linear(n_qubits, 2)

    def forward(self, x):
        features = self.vmapped_fn(x, self.weights).real
        return self.classifier(features)

# ============= TRAINING LOOP =============

def train_and_eval(model, train_loader, test_loader, epochs=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Train
    model.train()
    for _ in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
    
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
            
    return correct / total

def main():
    setup_mps_environment()
    print("="*60)
    print("Comprehensive 8-Qubit Architecture Search")
    print("="*60)
    
    # Search Space
    QUBITS = 8
    DEPTHS = [1, 2, 4]
    TOPOLOGIES = ["chain", "ring", "all2all"]
    
    # Tasks
    TASKS = list(BENCHMARKS.keys())
    
    all_results = {}
    
    for task in TASKS:
        print(f"\n--- Searching Task: {task} ---")
        # Prepare device string for generator
        gen_dev = 'mps' if device.type == 'mps' else 'cpu'
        train_dl, test_dl = get_benchmark_dataloader(task, n_samples=300, batch_size=32, generator_device=gen_dev)
        
        task_results = []
        print(f"{'Depth':<6} | {'Topology':<10} | {'Acc':<6}")
        print("-" * 30)
        
        for d, t in itertools.product(DEPTHS, TOPOLOGIES):
            try:
                model = SearchableQuantumModel(n_qubits=QUBITS, depth=d, topology=t).to(device)
                acc = train_and_eval(model, train_dl, test_dl, epochs=15) # Quick sweep
                
                print(f"{d:<6} | {t:<10} | {acc:.4f}")
                task_results.append({
                    "depth": d,
                    "topology": t,
                    "accuracy": acc
                })
            except Exception as e:
                print(f"{d:<6} | {t:<10} | FAILED: {e}")
        
        best = max(task_results, key=lambda x: x['accuracy'])
        print(f"  -> Best for {task}: {best}")
        all_results[task] = task_results

    # Save
    with open("results/architecture_search_all_8qubit.json", "w") as f:
        json.dump(all_results, f, indent=2)
        
    print("\n" + "="*60)
    print("Comparison Summary (Best 8-Qubit vs Classical Baseline)")
    print("-" * 60)
    
    # Load classical baselines if available (from previous run)
    try:
        with open("results/problem_class_search.json", "r") as f:
            baselines = json.load(f)
    except:
        baselines = {}
        
    for task in TASKS:
        best_q = max(all_results[task], key=lambda x: x['accuracy'])['accuracy']
        class_acc = baselines.get(task, {}).get("Classical", 0.0)
        adv = best_q - class_acc
        marker = "**" if adv > 0.05 else ""
        print(f"{task:<25} | Q(8): {best_q:.4f} | C: {class_acc:.4f} | Adv: {adv:+.4f} {marker}")

if __name__ == "__main__":
    main()
