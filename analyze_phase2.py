import torch
import torch.nn as nn
from model import HybridModel
from utils import setup_mps_environment
import numpy as np

def analyze_weights():
    setup_mps_environment()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Analyzing Weights on device: {device}")

    # Initialize model (weights will be random unless we load saved ones, 
    # but let's check the scaling/magnitude of a trained model if we had saved it.
    # Since we didn't save the checkpoint, I'll check the current model's gradients 
    # after one backward pass to see which features are "wanted" by the loss.
    
    model = HybridModel(n_qubits=4, use_quantum=True, alpha=0.5).to(device)
    
    # We need to simulate a training step to see the gradient distribution
    from data import get_dataloaders
    gen_dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    train_loader, _ = get_dataloaders(dataset_name="cifar10", batch_size=100, image_size=64, subset_size=100, generator_device=gen_dev)
    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)
    
    model.train()
    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    loss.backward()
    
    # Analyze gradients of the classifier (Linear(10, 10))
    # weights are [10, 10] (num_classes, num_features)
    grads = model.classifier.weight.grad.abs().cpu().numpy()
    
    # Feature map: 0-3 are single-body, 4-9 are two-body
    single_body_grad = np.mean(grads[:, :4])
    two_body_grad = np.mean(grads[:, 4:])
    
    print("\n--- Phase 2: Gradient Magnitude Analysis ---")
    print(f"Avg Absolute Gradient (Single-Body Z_i): {single_body_grad:.8f}")
    print(f"Avg Absolute Gradient (Two-Body Z_i Z_j): {two_body_grad:.8f}")
    
    ratio = two_body_grad / (single_body_grad + 1e-12)
    print(f"Interaction/Single-Body Ratio: {ratio:.4f}")
    
    if ratio > 0.5:
        print("DIAGNOSIS: Evidence of Correlation Extraction. The model is actively using entanglement features.")
    else:
        print("DIAGNOSIS: The model is ignoring interactions. The engine is running but the signal isn't useful.")

if __name__ == "__main__":
    analyze_weights()
