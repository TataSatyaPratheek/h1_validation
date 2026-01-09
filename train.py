import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time
import json
from model import HybridModel
from data import get_dataloaders
from utils import setup_mps_environment, get_generator

# 1. Setup Environment (Paper Phase 1)
setup_mps_environment()

def train(args):
    # Device Management
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")

    # Data (Section 2 Domain Application / Phase 2 Pipeline)
    # Using CIFAR10 as proxy for ImageNet subset for accessibility
    gen_dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    train_loader, val_loader = get_dataloaders(
        dataset_name=args.dataset, 
        batch_size=args.batch_size, 
        image_size=64,
        generator_device=gen_dev,
        subset_size=args.subset_size
    )

    # Model (Section 3 Hybrid Architecture)
    # Phase 3: Parity is a 2-class problem
    num_classes = 2 if args.dataset == "parity" else 10
    model = HybridModel(num_classes=num_classes, n_qubits=4, use_quantum=args.use_quantum, alpha=args.alpha).to(device)
    
    # Loss & Optimizer (Section 4 Differentiability)
    # "L(pi, y) denotes the cross-entropy loss." (Page 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"Starting Strict Training Pipeline (Quantum={args.use_quantum}, Alpha={args.alpha})...")
    
    history = {'train_loss': [], 'val_acc': []}
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        history['train_loss'].append(epoch_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1} finished. Loss: {epoch_loss:.4f}, Val Acc: {val_acc:.2f}%. Time: {time.time()-start_time:.2f}s")
    
    # Save history for Analysis
    model_type = "quantum" if args.use_quantum else "classical"
    filename = f"results_{args.dataset}_{model_type}.json"
    with open(filename, "w") as f:
        json.dump(history, f, indent=4)
        
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strict Implementation of Hybrid Quantum-Classical Model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--use_quantum", action="store_true", help="Enable Quantum Feature Map (Section 3.2)")
    parser.add_argument("--n_qubits", type=int, default=4, help="Number of qubits in the latent space")
    parser.add_argument("--alpha", type=float, default=0.5, help="Angle scaling factor (Section 5.1 Stability)")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "imagenet", "fake", "parity"])
    parser.add_argument("--subset_size", type=int, default=1000, help="Limit dataset size for rapid hypothesis testing")
    
    args = parser.parse_args()
    train(args)
