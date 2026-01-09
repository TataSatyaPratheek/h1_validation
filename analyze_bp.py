import torch
import torch.nn as nn
from model import HybridModel
from data import get_dataloaders
from utils import setup_mps_environment
import numpy as np
import matplotlib.pyplot as plt

def analyze_bp():
    setup_mps_environment()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Analyzing Barren Plateau on device: {device}")

    # Load Model with Circuit 14
    model = HybridModel(n_qubits=4, use_quantum=True, alpha=0.5).to(device)
    model.eval()

    # Get some data
    gen_dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    train_loader, _ = get_dataloaders(dataset_name="cifar10", batch_size=100, image_size=64, subset_size=100, generator_device=gen_dev)
    images, labels = next(iter(train_loader))
    images = images.to(device)
    labels = labels.to(device)

    # 1. Measure Feature Variance
    # We want to see the distribution of the quantum features (a_i)
    # Pipeline: feature_extractor -> projection -> quantum_map
    with torch.no_grad():
        features = model.feature_extractor(images)
        z_tilde = model.projection(features)
        q_features = model.quantum_map(z_tilde) # (Batch, N_qubits)
    
    q_features_np = q_features.cpu().numpy()
    feat_var = np.var(q_features_np, axis=0)
    feat_mean = np.mean(q_features_np, axis=0)
    
    print("\n--- Feature Statistics ---")
    print(f"Mean of Expectation Values (per qubit): {feat_mean}")
    print(f"Variance of Expectation Values (per qubit): {feat_var}")
    print(f"Total Mean Variance: {np.mean(feat_var):.6f}")

    # 2. Gradient Magnitude Analysis
    model.train()
    model.zero_grad()
    outputs = model(images)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)
    loss.backward()

    # Gradient of the projection layer (the weights immediately BEFORE the quantum map)
    proj_grad = model.projection.weight.grad
    grad_norm = torch.norm(proj_grad).item()
    grad_std = torch.std(proj_grad).item()

    print("\n--- Gradient Statistics (Projection Layer) ---")
    print(f"Gradient Norm: {grad_norm:.6f}")
    print(f"Gradient Std Dev: {grad_std:.6f}")

    # 3. Alpha Sensitivity
    print("\n--- Alpha Sensitivity Analysis ---")
    alphas = [0.1, 0.3, 0.5, 0.7, 1.0, 2.0, 3.14]
    avg_vars = []
    
    for a in alphas:
        model.alpha = a
        model.quantum_map.alpha = a
        with torch.no_grad():
            q_feat = model.quantum_map(z_tilde)
            avg_vars.append(np.mean(np.var(q_feat.cpu().numpy(), axis=0)))
        print(f"Alpha={a:.2f} -> Avg Feature Variance: {avg_vars[-1]:.6f}")

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(alphas, avg_vars, marker='o')
    plt.xlabel('Alpha (Scaling Factor)')
    plt.ylabel('Average Feature Variance')
    plt.title('Circuit 14: Expressed Variance vs Alpha')
    plt.grid(True)
    plt.savefig('results/alpha_sensitivity.png')
    print("\nAlpha sensitivity plot saved to results/alpha_sensitivity.png")

    # Conclusion Logic
    if np.mean(feat_var) < 1e-4:
        print("\nDIAGNOSIS: Global Barren Plateau (Fixed Map).")
        print("The highly expressive Circuit 14 is mapping all inputs to nearly identical expectation values.")
    elif grad_norm < 1e-5:
        print("\nDIAGNOSIS: Gradient Bottleneck.")
        print("Features have variance, but the partial derivatives through the fixed map are vanishing.")
    else:
        print("\nDIAGNOSIS: Sufficient signal, potentially slow convergence or other factor.")

if __name__ == "__main__":
    analyze_bp()
