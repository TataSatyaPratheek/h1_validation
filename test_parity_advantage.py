import torch
import torch.nn as nn
from model import QuantumFeatureMap
from parity_data import ParityDataset
from torch.utils.data import DataLoader
from utils import setup_mps_environment

def run_parity_benchmark():
    setup_mps_environment()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running Parity Benchmark on: {device}")

    # 1. Data: Simple 4-bit strings (not images)
    bits = torch.randint(0, 2, (1000, 4)).float().to(device)
    labels = (torch.sum(bits, dim=1).long() % 2).to(device)
    
    val_bits = torch.randint(0, 2, (200, 4)).float().to(device)
    val_labels = (torch.sum(val_bits, dim=1).long() % 2).to(device)

    # 2. Model A: Purely Linear Classical
    model_a = nn.Linear(4, 2).to(device)
    optimizer_a = torch.optim.Adam(model_a.parameters(), lr=0.01)
    
    # 3. Model B: QFM + Linear Head
    # We need to project 4 bits to 10 features for the Matchgate QFM
    # but the Matchgate QFM I wrote expects 10 inputs and gives 10 outputs?
    # Wait, my QuantumFeatureMap expects 'n_qubits' features (it uses them for Ry, etc.)
    # Let me check model.py again.
    
    # QFM for Matchgates uses 10 inputs (4 Ry + 6 Matchgate params).
    # Since we only have 4 bits, we'll pad with zeros or use a simple Ring QFM for this test.
    # Actually, let's use a simpler 4-qubit parity-specific QFM if needed.
    # But let's stick to our current QuantumFeatureMap and pad.
    
    import tensorcircuit as tc
    K = tc.set_backend("pytorch")

    class ParityQFM(nn.Module):
        def __init__(self, n_qubits=4, alpha=1.57):
            super().__init__()
            self.n_qubits = n_qubits
            self.alpha = alpha
            
            def quantum_fn(inputs):
                c = tc.Circuit(n_qubits)
                # 1. Rotations
                for i in range(n_qubits):
                    c.ry(i, theta=inputs[i])
                
                # 2. Entanglement (CNOT Ring)
                for i in range(n_qubits):
                    c.cnot(i, (i + 1) % n_qubits)
                
                # 3. Readout: Single-Body + Two-Body + FULL PARITY (N-body)
                single_body = [c.expectation([tc.gates.z(), [i]]) for i in range(n_qubits)]
                two_body = []
                for i in range(n_qubits):
                    for j in range(i + 1, n_qubits):
                        two_body.append(c.expectation([tc.gates.z(), [i]], [tc.gates.z(), [j]]))
                
                # Full Parity: Z0 Z1 Z2 Z3
                full_parity = [c.expectation(*[(tc.gates.z(), [i]) for i in range(n_qubits)])]
                
                return K.stack(single_body + two_body + full_parity)

            self.vmapped_fn = K.vmap(quantum_fn)

        def forward(self, x):
            # theta = 3.14 * x maps: bit 0 -> 0 (Ry(0)|0> = |0>), bit 1 -> 3.14 (Ry(pi)|0> = |1>)
            theta = 3.14159 * x
            return self.vmapped_fn(theta).real

    class QuantumParityModel(nn.Module):
        def __init__(self, n_qubits=4, alpha=3.14159):
            super().__init__()
            self.qfm = ParityQFM(n_qubits=n_qubits, alpha=alpha)
            # 4 single + 6 two-body + 1 four-body = 11 features
            self.head = nn.Linear(11, 2)
        def forward(self, x):
            q_feats = self.qfm(x)
            return self.head(q_feats)

    model_b = QuantumParityModel(alpha=1.57).to(device)
    optimizer_b = torch.optim.Adam(model_b.parameters(), lr=0.01)
    
    criterion = nn.CrossEntropyLoss()

    print("\n--- Training Model A (Classical Linear) ---")
    for epoch in range(200):
        out = model_a(bits)
        loss = criterion(out, labels)
        optimizer_a.zero_grad()
        loss.backward()
        optimizer_a.step()
        if epoch % 40 == 0:
            with torch.no_grad():
                val_out = model_a(val_bits)
                acc = (val_out.argmax(1) == val_labels).float().mean()
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Val Acc: {acc:.4f}")

    print("\n--- Training Model B (QFM + Linear) ---")
    for epoch in range(200):
        out = model_b(bits)
        loss = criterion(out, labels)
        optimizer_b.zero_grad()
        loss.backward()
        optimizer_b.step()
        if epoch % 40 == 0:
            with torch.no_grad():
                val_out = model_b(val_bits)
                acc = (val_out.argmax(1) == val_labels).float().mean()
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Val Acc: {acc:.4f}")

    # Final verdict
    with torch.no_grad():
        acc_a = (model_a(val_bits).argmax(1) == val_labels).float().mean()
        acc_b = (model_b(val_bits).argmax(1) == val_labels).float().mean()
    
    print("\n--- Final Results ---")
    print(f"Classical Linear Acc: {acc_a:.4f} (Target: ~0.50)")
    print(f"Quantum QFM Acc: {acc_b:.4f} (Target: >0.90)")

if __name__ == "__main__":
    run_parity_benchmark()
