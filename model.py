import torch
import torch.nn as nn
import tensorcircuit as tc
import numpy as np

# Set tensorcircuit backend to PyTorch (Paper: "Hybrid classical-quantum machine learning")
K = tc.set_backend("pytorch")
tc.set_dtype("complex64")

def matchgate_matrix(theta, phi1, phi2):
    """
    Constructs the unitary matrix for the Matchgate M(theta, phi1, phi2).
    As reconstructed from Eq (2) in 'Quantum Agents for Algorithmic Discovery'.
    """
    # Use torch directly for matrix construction to avoid backend implementation gaps
    theta = theta.to(torch.complex64)
    phi1 = phi1.to(torch.complex64)
    phi2 = phi2.to(torch.complex64)
    
    c = torch.cos(theta / 2.0)
    s = torch.sin(theta / 2.0)
    
    e_i_phi1 = torch.exp(1.0j * phi1)
    e_mi_phi1 = torch.exp(-1.0j * phi1)
    e_i_phi2 = torch.exp(1.0j * phi2)
    e_i_diff = torch.exp(1.0j * (phi2 - phi1))
    
    # row0: [1, 0, 0, 0]
    row0 = torch.stack([torch.tensor(1.0, dtype=torch.complex64, device=theta.device),
                        torch.tensor(0.0, dtype=torch.complex64, device=theta.device),
                        torch.tensor(0.0, dtype=torch.complex64, device=theta.device),
                        torch.tensor(0.0, dtype=torch.complex64, device=theta.device)])
    
    # row1: [0, cos, -e^{i phi1} sin, 0]
    row1 = torch.stack([torch.tensor(0.0, dtype=torch.complex64, device=theta.device),
                        c, -1.0 * e_i_phi1 * s,
                        torch.tensor(0.0, dtype=torch.complex64, device=theta.device)])
    
    # row2: [0, e^{-i phi1} sin, e^{i(phi2-phi1)} cos, 0]
    row2 = torch.stack([torch.tensor(0.0, dtype=torch.complex64, device=theta.device),
                        e_mi_phi1 * s, e_i_diff * c,
                        torch.tensor(0.0, dtype=torch.complex64, device=theta.device)])
    
    # row3: [0, 0, 0, e^{i phi2}]
    row3 = torch.stack([torch.tensor(0.0, dtype=torch.complex64, device=theta.device),
                        torch.tensor(0.0, dtype=torch.complex64, device=theta.device),
                        torch.tensor(0.0, dtype=torch.complex64, device=theta.device),
                        e_i_phi2])
    
    return torch.stack([row0, row1, row2, row3])

# Helper to apply the matchgate to a circuit
def apply_matchgate(c, i, j, theta, phi1, phi2):
    matrix = matchgate_matrix(theta, phi1, phi2)
    # Use any() to apply a custom unitary matrix
    c.any(i, j, unitary=matrix)

class QuantumFeatureMap(nn.Module):
    """
    Matchgate-based Feature Map (Phase 5).
    Uses Ry rotations and parameterized Matchgates as entanglers.
    Total Parameters (4 Qubits): 4 (Ry) + 2 * 3 (Matchgates) = 10
    """
    def __init__(self, n_qubits, alpha=0.5):
        super().__init__()
        self.n_qubits = n_qubits
        self.alpha = alpha
        
        def quantum_fn(inputs):
            c = tc.Circuit(self.n_qubits)
            
            # 1. Rotation Layer (Features 0-3)
            for i in range(self.n_qubits):
                c.ry(i, theta=inputs[i])
            
            # 2. Matchgate Entanglers
            # Matchgate(0, 1) uses features 4, 5, 6
            apply_matchgate(c, 0, 1, theta=inputs[4], phi1=inputs[5], phi2=inputs[6])
            
            # Matchgate(2, 3) uses features 7, 8, 9
            if self.n_qubits >= 4:
                apply_matchgate(c, 2, 3, theta=inputs[7], phi1=inputs[8], phi2=inputs[9])
            
            # Phase 2.1: Interaction Observables (Readout Layer Tweak)
            # Output 4 Single-Body: <Z_i>
            single_body = []
            for i in range(self.n_qubits):
                single_body.append(c.expectation([tc.gates.z(), [i]]))
            
            # Output 6 Two-Body: <Z_i Z_j>
            two_body = []
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    two_body.append(c.expectation([tc.gates.z(), [i]], [tc.gates.z(), [j]]))
            
            return K.stack(single_body + two_body)

        self.vmapped_quantum_fn = K.vmap(quantum_fn)

    def forward(self, x):
        theta = self.alpha * torch.tanh(x)
        return self.vmapped_quantum_fn(theta).real

class HybridModel(nn.Module):
    """
    Hybrid Architecture Formulation with Circuit 14 Expressivity.
    """
    def __init__(self, num_classes=10, n_qubits=4, use_quantum=True, alpha=0.5):
        super(HybridModel, self).__init__()
        self.use_quantum = use_quantum
        self.n_qubits = n_qubits
        self.alpha = alpha
        
        # Section 3.1: Classical Pre-processing
        # Step 1-5: Conv layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2), # 64 -> 32
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # 32 -> 16
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # 16 -> 8
        )
        # 64 channels * 8 * 8 = 4096
        self.flat_dim = 64 * 8 * 8
        
        # Phase 4: Wide Architecture
        # Split 4096 features into 16 chunks of 256, 
        # project each to 10 quantum features.
        self.num_blocks = 16
        self.q_input_dim_per_block = 10
        self.projections = nn.ModuleList([
            nn.Linear(self.flat_dim // self.num_blocks, self.q_input_dim_per_block)
            for _ in range(self.num_blocks)
        ])
        
        if self.use_quantum:
            self.quantum_maps = nn.ModuleList([
                QuantumFeatureMap(n_qubits, alpha=alpha)
                for _ in range(self.num_blocks)
            ])
        
        # Section 3.3: Classical Decoding
        # Concatenated features: 16 blocks * 10 features = 160
        self.classifier = nn.Linear(self.num_blocks * 10, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1) # Flatten
        
        # Chunk features
        chunk_size = features.size(1) // self.num_blocks
        out_features = []
        
        for i in range(self.num_blocks):
            chunk = features[:, i*chunk_size : (i+1)*chunk_size]
            z_tilde = self.projections[i](chunk)
            
            if self.use_quantum:
                q_out = self.quantum_maps[i](z_tilde)
                out_features.append(q_out)
            else:
                # Classical baseline: tanh(z_tilde) to match QFM range
                out_features.append(torch.tanh(z_tilde))
        
        combined = torch.cat(out_features, dim=1)
        return self.classifier(combined)
