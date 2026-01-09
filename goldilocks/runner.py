"""
Goldilocks Benchmark Runner
===========================
Automated benchmark runner for surface exploration across all datasets.
Compares 8Q Chain (quantum) vs Classical baseline.
"""
import torch
import torch.nn as nn
import numpy as np
import json
import time
from datetime import datetime
from typing import Dict, List, Any

# Import datasets
from goldilocks.datasets import ALL_TASKS, get_boolean_dataloader, get_geometric_dataloader, get_molecular_dataloader

# Import model (will need path adjustment after full reorganization)
import sys
sys.path.insert(0, '.')
from antigravity.models.quantum import QuantumFeatureMap


class SimpleQuantumClassifier(nn.Module):
    """Simple quantum classifier for Goldilocks benchmarks."""
    
    def __init__(self, input_dim: int, n_qubits: int = 8, depth: int = 1, 
                 topology: str = "chain", num_classes: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.n_qubits = n_qubits
        
        # Project to n_qubits if input is larger
        self.projection = nn.Linear(input_dim, n_qubits) if input_dim != n_qubits else nn.Identity()
        
        # Quantum Feature Map
        self.qfm = QuantumFeatureMap(n_qubits=n_qubits, depth=depth, topology=topology)
        
        # Output features: n_qubits single + n_qubits*(n_qubits-1)/2 two-body
        n_features = n_qubits + (n_qubits * (n_qubits - 1)) // 2
        
        # Classifier
        self.classifier = nn.Linear(n_features, num_classes)
    
    def forward(self, x):
        x = self.projection(x)
        x = self.qfm(x)
        return self.classifier(x)


class SimpleClassicalClassifier(nn.Module):
    """Simple classical baseline for Goldilocks benchmarks."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 32, num_classes: int = 2):
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


def run_benchmark(task_name: str, task_config: Dict, epochs: int = 50, 
                  device: str = "cpu") -> Dict[str, Any]:
    """
    Run a single benchmark comparing quantum vs classical.
    
    Returns dict with accuracy scores and timing.
    """
    print(f"\n{'='*60}")
    print(f"Benchmark: {task_name}")
    print(f"{'='*60}")
    
    # Get dataloader based on category
    category = task_name.split("/")[0]
    task_key = task_config["task"]
    
    if category == "boolean":
        dataset = get_boolean_dataloader(task_key, batch_size=32, **{k: v for k, v in task_config.items() if k != "task"})
    elif category == "geometric":
        dataset = get_geometric_dataloader(task_key, batch_size=32, **{k: v for k, v in task_config.items() if k != "task"})
    elif category == "molecular":
        dataset = get_molecular_dataloader(task_key, batch_size=32, **{k: v for k, v in task_config.items() if k != "task"})
    else:
        raise ValueError(f"Unknown category: {category}")
    
    # Get input dimension from first batch
    sample_x, sample_y = next(iter(dataset))
    input_dim = sample_x.shape[1]
    num_classes = len(torch.unique(sample_y))
    
    print(f"Input dim: {input_dim}, Classes: {num_classes}, Samples: {len(dataset.dataset)}")
    
    results = {"task": task_name, "input_dim": input_dim, "num_classes": num_classes}
    
    # Train and evaluate each model
    for model_name, model_fn in [
        ("8Q_Chain", lambda: SimpleQuantumClassifier(input_dim, n_qubits=8, topology="chain", num_classes=num_classes)),
        ("4Q_Ring", lambda: SimpleQuantumClassifier(input_dim, n_qubits=4, topology="ring", num_classes=num_classes)),
        ("Classical", lambda: SimpleClassicalClassifier(input_dim, num_classes=num_classes)),
    ]:
        print(f"\n  Training {model_name}...")
        
        model = model_fn().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        start_time = time.time()
        
        for epoch in range(epochs):
            model.train()
            for x, y in dataset:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in dataset:
                x, y = x.to(device), y.to(device)
                out = model(x)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += len(y)
        
        accuracy = 100 * correct / total
        elapsed = time.time() - start_time
        
        results[model_name] = {"accuracy": accuracy, "time": elapsed}
        print(f"    {model_name}: {accuracy:.1f}% ({elapsed:.1f}s)")
    
    # Compute quantum advantage
    results["advantage_8Q"] = results["8Q_Chain"]["accuracy"] - results["Classical"]["accuracy"]
    results["advantage_4Q"] = results["4Q_Ring"]["accuracy"] - results["Classical"]["accuracy"]
    
    return results


def run_all_benchmarks(epochs: int = 50, output_file: str = "goldilocks_results.json"):
    """Run benchmarks on all Goldilocks tasks."""
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Running Goldilocks Benchmark Suite on {device}")
    print(f"Tasks: {len(ALL_TASKS)}")
    
    all_results = []
    
    for task_name, task_config in ALL_TASKS.items():
        try:
            result = run_benchmark(task_name, task_config, epochs=epochs, device=device)
            all_results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results.append({"task": task_name, "error": str(e)})
    
    # Save results
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "device": device,
            "epochs": epochs,
            "results": all_results
        }, f, indent=2)
    
    print(f"\n\nResults saved to {output_file}")
    
    # Print summary table
    print("\n" + "="*80)
    print("GOLDILOCKS BENCHMARK SUMMARY")
    print("="*80)
    print(f"{'Task':<30} | {'8Q Chain':>10} | {'4Q Ring':>10} | {'Classical':>10} | {'Advantage':>10}")
    print("-"*80)
    
    for r in all_results:
        if "error" not in r:
            adv = r.get("advantage_8Q", 0)
            marker = "✅" if adv > 5 else ("⚠️" if adv > 0 else "❌")
            print(f"{r['task']:<30} | {r['8Q_Chain']['accuracy']:>9.1f}% | {r['4Q_Ring']['accuracy']:>9.1f}% | {r['Classical']['accuracy']:>9.1f}% | {adv:>+9.1f}% {marker}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Goldilocks Benchmark Runner")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs per task")
    parser.add_argument("--output", type=str, default="goldilocks_results.json", help="Output file")
    parser.add_argument("--task", type=str, default=None, help="Run single task (e.g., 'boolean/parity_4bit')")
    
    args = parser.parse_args()
    
    if args.task:
        if args.task in ALL_TASKS:
            run_benchmark(args.task, ALL_TASKS[args.task], epochs=args.epochs)
        else:
            print(f"Unknown task: {args.task}")
            print(f"Available: {list(ALL_TASKS.keys())}")
    else:
        run_all_benchmarks(epochs=args.epochs, output_file=args.output)
