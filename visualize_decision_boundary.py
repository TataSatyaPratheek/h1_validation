"""
Decision Boundary Visualization for Parity Task
Shows the non-linear topology of Quantum vs Classical decision surfaces.
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import tensorcircuit as tc
from utils import setup_mps_environment

K = tc.set_backend("pytorch")

class ParityQFM(nn.Module):
    def __init__(self, n_qubits=4, alpha=3.14159):
        super().__init__()
        self.n_qubits = n_qubits
        self.alpha = alpha
        
        def quantum_fn(inputs):
            c = tc.Circuit(n_qubits)
            for i in range(n_qubits):
                c.ry(i, theta=inputs[i])
            for i in range(n_qubits):
                c.cnot(i, (i + 1) % n_qubits)
            single = [c.expectation([tc.gates.z(), [i]]) for i in range(n_qubits)]
            two_body = []
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    two_body.append(c.expectation([tc.gates.z(), [i]], [tc.gates.z(), [j]]))
            # Full N-body parity
            full_parity = [c.expectation(*[(tc.gates.z(), [i]) for i in range(n_qubits)])]
            return K.stack(single + two_body + full_parity)
        self.vmapped_fn = K.vmap(quantum_fn)

    def forward(self, x):
        theta = self.alpha * x
        return self.vmapped_fn(theta).real

class QuantumParityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.qfm = ParityQFM()
        self.head = nn.Linear(11, 2)
    def forward(self, x):
        return self.head(self.qfm(x))

class ClassicalLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(4, 2)
    def forward(self, x):
        return self.head(x)

def train_model(model, bits, labels, epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        out = model(bits)
        loss = criterion(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

def plot_decision_boundary(model, title, ax, device):
    """Plot decision boundary for 2D slice (bits 0,1) with bits 2,3 = 0."""
    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 1, 50)
    xx, yy = np.meshgrid(x, y)
    
    grid = np.zeros((xx.size, 4))
    grid[:, 0] = xx.ravel()
    grid[:, 1] = yy.ravel()
    grid_tensor = torch.tensor(grid, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        preds = model(grid_tensor)
        probs = torch.softmax(preds, dim=1)[:, 1].cpu().numpy().reshape(xx.shape)
    
    contour = ax.contourf(xx, yy, probs, levels=20, cmap='RdBu', alpha=0.8)
    
    # Mark true parity labels
    corners = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    labels = [0, 1, 1, 0]  # XOR pattern for bits 0,1
    colors = ['blue' if l == 0 else 'red' for l in labels]
    ax.scatter(corners[:, 0], corners[:, 1], c=colors, s=200, edgecolors='black', zorder=5)
    
    ax.set_xlabel("Bit 0")
    ax.set_ylabel("Bit 1")
    ax.set_title(title)
    return contour

def main():
    setup_mps_environment()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running Decision Boundary on: {device}")
    
    # Generate parity data
    bits = torch.randint(0, 2, (1000, 4)).float().to(device)
    labels = (torch.sum(bits, dim=1).long() % 2).to(device)
    
    # Train both models
    print("Training Classical Linear...")
    model_classical = ClassicalLinearModel().to(device)
    model_classical = train_model(model_classical, bits, labels, epochs=200)
    
    print("Training Quantum QFM...")
    model_quantum = QuantumParityModel().to(device)
    model_quantum = train_model(model_quantum, bits, labels, epochs=200)
    
    # Calculate accuracies
    with torch.no_grad():
        classical_acc = (model_classical(bits).argmax(1) == labels).float().mean()
        quantum_acc = (model_quantum(bits).argmax(1) == labels).float().mean()
    print(f"Classical Accuracy: {classical_acc:.2%}")
    print(f"Quantum Accuracy: {quantum_acc:.2%}")
    
    # Plot side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    plot_decision_boundary(model_classical, 
                           f"Classical Linear (Acc: {classical_acc:.0%})\nCannot solve XOR", 
                           ax1, device)
    contour = plot_decision_boundary(model_quantum, 
                           f"Quantum QFM + Linear (Acc: {quantum_acc:.0%})\nNon-linear Correlation Filter", 
                           ax2, device)
    
    fig.colorbar(contour, ax=[ax1, ax2], label="P(Parity=1)")
    fig.suptitle("Topology of Decision: Classical vs Quantum on Parity/XOR Task", fontsize=14)
    
    plt.subplots_adjust(top=0.88, bottom=0.1, left=0.08, right=0.92)
    plt.savefig("results/decision_boundary_parity.png", dpi=150, bbox_inches='tight')
    print("Saved: results/decision_boundary_parity.png")
    
    # Parameter count comparison
    print("\n--- Parameter Count ---")
    classical_params = sum(p.numel() for p in model_classical.parameters())
    quantum_params = sum(p.numel() for p in model_quantum.parameters())
    print(f"Classical Linear: {classical_params} parameters")
    print(f"Quantum QFM + Linear: {quantum_params} trainable parameters (QFM is FIXED)")

if __name__ == "__main__":
    main()
