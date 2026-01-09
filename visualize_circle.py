"""
Visualize the Circle Task Advantage
Why does Quantum (0.82) beat Classical (0.69) on the Radial Separation task?
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import tensorcircuit as tc
from utils import setup_mps_environment
from benchmarks import generate_circle_data

K = tc.set_backend("pytorch")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ============= MODELS (Same as search) =============

class QuantumModel(nn.Module):
    def __init__(self, n_qubits=4, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        
        def quantum_fn(inputs, weights):
            c = tc.Circuit(n_qubits)
            for i in range(n_qubits):
                c.ry(i, theta=inputs[i])
            for l in range(n_layers):
                for i in range(n_qubits):
                    c.cnot(i, (i + 1) % n_qubits)
                for i in range(n_qubits):
                    c.ry(i, theta=weights[l, i])
            return K.stack([c.expectation([tc.gates.z(), [i]]) for i in range(n_qubits)])
            
        self.vmapped_fn = K.vmap(quantum_fn, vectorized_argnums=0)
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits))
        self.classifier = nn.Linear(n_qubits, 2)

    def forward(self, x):
        features = self.vmapped_fn(x, self.weights).real
        return self.classifier(features)

class ClassicalModel(nn.Module):
    def __init__(self, n_inputs=4, hidden_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        return self.net(x)

def train_model(model, data, epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    X, Y = data.tensors
    X, Y = X.to(device), Y.to(device)
    
    losses = []
    
    model.train()
    for i in range(epochs):
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, Y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
    return losses

def plot_decision_boundary(model, X_train, Y_train, title, save_path):
    """Plot 2D slice of decision boundary (first 2 features)."""
    model.eval()
    
    # Grid range [0, pi]
    x_range = np.linspace(0, np.pi, 100)
    y_range = np.linspace(0, np.pi, 100)
    XX, YY = np.meshgrid(x_range, y_range)
    
    # Flatten and pad with mean of other features
    grid = torch.zeros(100*100, 4).to(device)
    grid[:, 0] = torch.tensor(XX.flatten(), dtype=torch.float32)
    grid[:, 1] = torch.tensor(YY.flatten(), dtype=torch.float32)
    # Fill noise dims with mean value (approx pi/2)
    grid[:, 2:] = np.pi / 2
    
    with torch.no_grad():
        preds = model(grid).argmax(dim=1).cpu().numpy().reshape(100, 100)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(XX, YY, preds, alpha=0.3, cmap='bwr')
    
    # Scatter points (only show points where noise dims are close to mean? No, too sparse)
    # Just show first 200 points
    X_plot = X_train[:200, :2].cpu().numpy()
    Y_plot = Y_train[:200].cpu().numpy()
    
    ax.scatter(X_plot[Y_plot==0, 0], X_plot[Y_plot==0, 1], c='blue', s=20, label='Inside')
    ax.scatter(X_plot[Y_plot==1, 0], X_plot[Y_plot==1, 1], c='red', s=20, label='Outside')
    
    ax.set_title(title)
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def main():
    setup_mps_environment()
    print("="*60)
    print("Visualizing Circle Task Advantage")
    print("="*60)
    
    # Generate data
    dataset = generate_circle_data(n_samples=500)
    
    # Train Quantum
    print("\nTraining Quantum Model...")
    q_model = QuantumModel().to(device)
    q_losses = train_model(q_model, dataset)
    print(f"Final Loss: {q_losses[-1]:.4f}")
    
    # Train Classical
    print("\nTraining Classical Model...")
    c_model = ClassicalModel().to(device)
    c_losses = train_model(c_model, dataset)
    print(f"Final Loss: {c_losses[-1]:.4f}")
    
    # Plot Boundaries
    print("\nplotting...")
    X, Y = dataset.tensors
    plot_decision_boundary(q_model, X, Y, "Quantum Decision Boundary (Circle)", "results/viz_circle_quantum.png")
    plot_decision_boundary(c_model, X, Y, "Classical Decision Boundary (Circle)", "results/viz_circle_classical.png")
    
    print("\nSaved visualizations to results/viz_circle_*.png")

if __name__ == "__main__":
    main()
