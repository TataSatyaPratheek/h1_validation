import torch
import tensorcircuit as tc
from model import QuantumFeatureMap, HybridModel

def test_quantum_map_forward():
    print("Testing QuantumFeatureMap Forward Pass...")
    n_qubits = 4
    batch_size = 5
    model = QuantumFeatureMap(n_qubits)
    
    x = torch.randn(batch_size, n_qubits)
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    assert y.shape == (batch_size, n_qubits)
    assert not torch.isnan(y).any()
    print("Forward pass successful!\n")

def test_hybrid_model_forward():
    print("Testing HybridModel Forward Pass...")
    n_qubits = 4
    num_classes = 2
    batch_size = 3
    # Fake image 64x64
    x = torch.randn(batch_size, 3, 64, 64)
    
    # Quantum
    model_q = HybridModel(num_classes=num_classes, n_qubits=n_qubits, use_quantum=True)
    y_q = model_q(x)
    print(f"Quantum Model Output shape: {y_q.shape}")
    assert y_q.shape == (batch_size, num_classes)
    
    # Classical
    model_c = HybridModel(num_classes=num_classes, n_qubits=n_qubits, use_quantum=False)
    y_c = model_c(x)
    print(f"Classical Model Output shape: {y_c.shape}")
    assert y_c.shape == (batch_size, num_classes)
    
    print("HybridModel tests successful!\n")

def test_grads():
    print("Testing Gradient Flow (Autograd)...")
    n_qubits = 2
    model = QuantumFeatureMap(n_qubits)
    x = torch.randn(2, n_qubits, requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()
    
    print(f"Input grads: {x.grad}")
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    print("Gradient flow successful!\n")

if __name__ == "__main__":
    tc.set_backend("pytorch")
    test_quantum_map_forward()
    test_grads()
    test_hybrid_model_forward()
