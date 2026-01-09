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
    # Model configuration
    parser.add_argument("--use_quantum", type=str, default="true", 
                        choices=["true", "false"], help="Use quantum layer?")
    parser.add_argument("--alpha", type=float, default=0.5, help="Input scaling factor (0.5 for stability)")
    parser.add_argument("--n_qubits", type=int, default=8, help="Number of qubits per block (4 or 8)")
    parser.add_argument("--depth", type=int, default=1, help="Circuit depth")
    parser.add_argument("--topology", type=str, default="chain", choices=["ring", "chain", "all2all"], help="Entanglement topology")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "imagenet", "fake", "parity"], help="Dataset to use")

    # Debugging
    parser.add_argument("--subset_size", type=int, default=None, help="Use small subset for debugging")
    parser.add_argument("--log_file", type=str, default="training_log.json", help="Path to save JSON logs")

    args = parser.parse_args()
    
    # Global Device Setup for printing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        
    use_quantum = (args.use_quantum.lower() == "true")
    
    print("="*60)
    print(f"Project Antigravity: Validation Run")
    print(f"Device: {device}")
    print(f"Model: {'Quantum Hybrid' if use_quantum else 'Classical Baseline'}")
    print(f"  Configuration: {args.n_qubits} Qubits, Depth {args.depth}, Topology '{args.topology}'")
    print(f"  Alpha: {args.alpha}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    if args.subset_size:
        print(f"DEBUG MODE: Subset size {args.subset_size}")
    print("="*60)

    # 1. Data
    train_loader, test_loader = get_dataloaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        subset_size=args.subset_size,
        generator_device=('mps' if device.type == 'mps' else None)
    )

    # 2. Model
    num_classes = 2 if args.dataset == "parity" else 10
    model = HybridModel(
        num_classes=num_classes, 
        n_qubits=args.n_qubits,
        depth=args.depth,
        topology=args.topology,
        use_quantum=use_quantum,
        alpha=args.alpha
    ).to(device)
    
    # 3. Training Loop
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
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
        
        epoch_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        history['train_loss'].append(epoch_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.2f}% | Time: {time.time()-start_time:.2f}s")
    
    # Save history
    with open(args.log_file, "w") as f:
        json.dump(history, f, indent=4)
    print(f"Training complete. Results saved to {args.log_file}")
