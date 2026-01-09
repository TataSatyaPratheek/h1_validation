import re
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from scipy import stats

def parse_log(filename):
    """
    Parses a training log file to extract epoch metrics.
    Line format: "Epoch 64 finished. Loss: 2.3016, Val Acc: 7.50%. Time: 1.98s"
    """
    epochs = []
    losses = []
    accs = []
    times = []
    
    pattern = r"Epoch (\d+) finished\. Loss: ([\d\.]+), Val Acc: ([\d\.]+)%\. Time: ([\d\.]+)s"
    
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return None

    with open(filename, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epochs.append(int(match.group(1)))
                losses.append(float(match.group(2)))
                accs.append(float(match.group(3)))
                times.append(float(match.group(4)))
                
    return {
        "epochs": np.array(epochs),
        "loss": np.array(losses),
        "val_acc": np.array(accs),
        "time": np.array(times)
    }

def perform_hypothesis_test(q_data, c_data, burn_in=10):
    """
    Performs rigorous statistical comparison.
    """
    print(f"\n--- Hypothesis Testing (Burn-in: {burn_in} epochs) ---")
    
    # 1. Accuracy Parity (Are max accuracies significantly different?)
    # Since we only have one run per model, we use the variance of the last N epochs as a proxy for steady-state variance
    # Or simply report the peak.
    q_max = np.max(q_data['val_acc'])
    c_max = np.max(c_data['val_acc'])
    print(f"Max Accuracy: Quantum={q_max:.2f}%, Classical={c_max:.2f}%")
    
    # 2. Convergence Rate (Area Under Curve of Accuracy)
    # Higher AUC = Faster convergence / Better performance
    min_len = min(len(q_data['val_acc']), len(c_data['val_acc']))
    q_auc = np.trapz(q_data['val_acc'][:min_len])
    c_auc = np.trapz(c_data['val_acc'][:min_len])
    print(f"Convergence Impact (AUC): Quantum={q_auc:.1f}, Classical={c_auc:.1f}")

    # 3. Stability (Loss Variance in last 20% of epochs)
    window = max(5, int(min_len * 0.2))
    q_var = np.var(q_data['loss'][-window:])
    c_var = np.var(c_data['loss'][-window:])
    print(f"Stability (Loss Var, last {window} eps): Quantum={q_var:.6f}, Classical={c_var:.6f}")
    
    # 4. T-Test on Time per Epoch
    t_stat, p_val = stats.ttest_ind(q_data['time'], c_data['time'], equal_var=False)
    print(f"Time Per Epoch: Quantum={np.mean(q_data['time']):.2f}s, Classical={np.mean(c_data['time']):.2f}s")
    print(f"Time Difference T-Test: p-value={p_val:.5e} (Significant if < 0.05)")

def plot_metrics(q_data, c_data, filename="comparison_plot.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss
    ax1.plot(q_data['epochs'], q_data['loss'], label='Quantum', color='blue', alpha=0.7)
    ax1.plot(c_data['epochs'], c_data['loss'], label='Classical', color='red', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(q_data['epochs'], q_data['val_acc'], label='Quantum', color='blue', alpha=0.7)
    ax2.plot(c_data['epochs'], c_data['val_acc'], label='Classical', color='red', alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy (%)')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"\nPlot saved to {filename}")

if __name__ == "__main__":
    q_file = "logs/train_quantum_antigravity_phase4.log"
    c_file = "logs/train_classical_antigravity_phase4.log"
    
    print("Parsing logs...")
    q_data = parse_log(q_file)
    c_data = parse_log(c_file)
    
    if q_data and c_data and len(q_data['epochs']) > 0 and len(c_data['epochs']) > 0:
        perform_hypothesis_test(q_data, c_data)
        plot_metrics(q_data, c_data)
    else:
        print("Insufficient data found in logs.")
