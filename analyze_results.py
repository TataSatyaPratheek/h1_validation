import re
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from scipy import stats

def main():
    # Load all logs
    logs = [
        ("results_cifar10_8q_chain.json", "8-Qubit Chain (Redundant)"),
        ("results_cifar10_4q_ring.json", "4-Qubit Ring (Baseline)"),
        ("results_cifar10_classical.json", "Classical Baseline"),
    ]
    
    data = {}
    for filename, label in logs:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data[label] = json.load(f)
        else:
            print(f"Warning: {filename} not found.")

    if not data:
        print("No data found to analyze.")
        return

    # Plot Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['green', 'orange', 'steelblue']
    
    # 1. Validation Accuracy
    for i, (label, history) in enumerate(data.items()):
        acc = history['val_acc']
        epochs = range(1, len(acc) + 1)
        ax1.plot(epochs, acc, label=f"{label} (Max: {max(acc):.2f}%)", color=colors[i], linewidth=2)
    
    ax1.set_title("Validation Accuracy (100 Epochs)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Training Loss
    for i, (label, history) in enumerate(data.items()):
        loss = history['train_loss']
        epochs = range(1, len(loss) + 1)
        ax2.plot(epochs, loss, label=label, color=colors[i], linewidth=2)
    
    ax2.set_title("Training Loss Convergence")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Cross Entropy Loss")
    ax2.set_yscale('log') # Log scale
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("impact_analysis_100epochs.png", dpi=150)
    print("Saved: impact_analysis_100epochs.png")
    
    # Print Summary Table
    print("\n" + "="*60)
    print(f"{'Model':<30} | {'Max Acc':<8} | {'Final Loss':<10}")
    print("-" * 60)
    for label, history in data.items():
        print(f"{label:<30} | {max(history['val_acc']):.2f}%   | {history['train_loss'][-1]:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
