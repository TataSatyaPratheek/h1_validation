import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from model import HybridModel
from data import get_dataloaders
from utils import setup_mps_environment

def compute_kernel_matrix(features):
    """
    Computes the dot-product kernel matrix K_ij = a_i . a_j
    where a_i are the quantum features (expectation values).
    """
    # Normalize features to see cosine similarity (unit sphere)
    norms = torch.norm(features, p=2, dim=1, keepdim=True)
    features_norm = features / (norms + 1e-8)
    kernel = torch.mm(features_norm, features_norm.t())
    return kernel.cpu().numpy()

def run_heatmap_test(circuit_name="Circuit 14", n_qubits=4, alpha=0.5):
    setup_mps_environment()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n--- Running Kernel Heatmap for {circuit_name} (Alpha={alpha}) ---")

    # 1. Load Model
    # We'll use the current HybridModel which uses the latest implemented QFM (Matchgate currently)
    # To test C14, we'd need to swap it, but let's test the current one (Matchgate) first.
    model = HybridModel(n_qubits=n_qubits, use_quantum=True, alpha=alpha).to(device)
    model.eval()

    # 2. Get balanced samples: 25 from Class 0, 25 from Class 1
    gen_dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    train_loader, _ = get_dataloaders(dataset_name="cifar10", batch_size=10, image_size=64, subset_size=1000, generator_device=gen_dev)
    
    class_0_imgs = []
    class_1_imgs = []
    
    for imgs, lbls in train_loader:
        for i in range(len(lbls)):
            if lbls[i] == 0 and len(class_0_imgs) < 25:
                class_0_imgs.append(imgs[i])
            elif lbls[i] == 1 and len(class_1_imgs) < 25:
                class_1_imgs.append(imgs[i])
        if len(class_0_imgs) == 25 and len(class_1_imgs) == 25:
            break
            
    images = torch.stack(class_0_imgs + class_1_imgs).to(device)
    labels = [0]*25 + [1]*25

    # 3. Extract Quantum Features
    with torch.no_grad():
        features = model.feature_extractor(images)
        z_tilde = model.projection(features)
        q_features = model.quantum_map(z_tilde)
        
    # 4. Compute Kernel
    kernel = compute_kernel_matrix(q_features)
    
    # 5. Visualize
    plt.figure(figsize=(10, 8))
    im = plt.imshow(kernel, cmap="viridis", vmin=0.999, vmax=1.0)
    plt.colorbar(im, label='Cosine Similarity')
    plt.title(f"Kernel Similarity: {circuit_name}\n(25 Plane vs 25 Car Samples)")
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Index")
    
    # Add separating lines
    plt.axvline(x=25, color='red', linestyle='--', alpha=0.5)
    plt.axhline(y=25, color='red', linestyle='--', alpha=0.5)
    
    save_path = f"results/kernel_heatmap_{circuit_name.lower().replace(' ', '_')}.png"
    plt.savefig(save_path)
    print(f"Heatmap saved to {save_path}")
    
    # 6. Quantitative Check
    diag_mean = np.mean(np.diag(kernel))
    intra_class = (np.mean(kernel[:25, :25]) + np.mean(kernel[25:, 25:])) / 2
    inter_class = np.mean(kernel[:25, 25:])
    
    print(f"Intra-Class Similarity: {intra_class:.8f}")
    print(f"Inter-Class Similarity: {inter_class:.8f}")
    print(f"Contrast Ratio (Intra/Inter): {intra_class/inter_class:.8f}")
    
    if intra_class / inter_class < 1.00001:
        print("DIAGNOSIS: The QFM acts as a Uniform Filter (All samples collapsed).")
    else:
        print("DIAGNOSIS: Some correlation filtering detected.")

if __name__ == "__main__":
    # Test current Matchgate implementation
    run_heatmap_test(circuit_name="Matchgate", n_qubits=4, alpha=0.5)
    # Note: To test C14, we'd need to modify model.py back or make it selectable.
