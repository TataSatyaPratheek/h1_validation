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
    Configurable Quantum Feature Map.
    Defaults to the "Redundant Chain" (8 qubits, depth 1) found in Phase 2.
    """
    def __init__(self, n_qubits=8, depth=1, topology="chain", alpha=0.5):
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        self.topology = topology
        self.alpha = alpha
        
        def quantum_fn(inputs):
            c = tc.Circuit(n_qubits)
            
            # Redundant Encoding (Cyclic)
            # Input dimension is inferred from tensor shape in vmap
            n_inputs = inputs.shape[0]
            for i in range(n_qubits):
                c.ry(i, theta=inputs[i % n_inputs])
            
            # Variational/Entanglement Layers
            for d in range(depth):
                if topology == "chain":
                    for i in range(n_qubits - 1):
                        c.cnot(i, i + 1)
                elif topology == "ring":
                    for i in range(n_qubits):
                        c.cnot(i, (i + 1) % n_qubits)
                elif topology == "all2all":
                    for i in range(n_qubits):
                        for j in range(i + 1, n_qubits):
                            c.cnot(i, j)
                
                # Optional: Fixed rotation layer to add non-linearity depth?
                # In search we used weights. Here for a fixed map we can use fixed Identity
                # or random fixed weights. To preserve the exact logic of "SearchableQuantumModel"
                # without training weights, we should probably stick to just entanglement
                # OR add a fixed random layer.
                # Let's stick to pure entanglement for now as "Feature Map".
                pass
            
            # Measurement: <Z> and <ZZ>
            single = [c.expectation([tc.gates.z(), [i]]) for i in range(n_qubits)]
            two_body = []
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    two_body.append(c.expectation([tc.gates.z(), [i]], [tc.gates.z(), [j]]))
            return K.stack(single + two_body)

        self.vmapped_quantum_fn = K.vmap(quantum_fn)

    def forward(self, x):
        # Scale inputs (tanh squashing usually good for rotation angles)
        theta = self.alpha * torch.tanh(x)
        return self.vmapped_quantum_fn(theta).real

class HybridModel(nn.Module):
    """
    Hybrid Architecture Formulation with Circuit 14 Expressivity.
    """
class HybridModel(nn.Module):
    """
    Hybrid Architecture Formulation with Circuit 14 Expressivity.
    """
    def __init__(self, num_classes=10, n_qubits=8, depth=1, topology="chain", use_quantum=True, alpha=0.5):
        super(HybridModel, self).__init__()
        self.use_quantum = use_quantum
        self.n_qubits = n_qubits
        self.depth = depth
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
        # Split 4096 features into 16 chunks of 256
        self.num_blocks = 16 
        # Project each chunk to n_qubits input dimension (redundant encoding handled inside QFM if needed)
        # Actually QFM expects n_qubits inputs or less. 
        # If we project 256 -> n_qubits, we feed dense info.
        self.q_input_dim_per_block = n_qubits 
        
        self.projections = nn.ModuleList([
            nn.Linear(self.flat_dim // self.num_blocks, self.q_input_dim_per_block)
            for _ in range(self.num_blocks)
        ])
        
        if self.use_quantum:
            self.quantum_maps = nn.ModuleList([
                QuantumFeatureMap(n_qubits=n_qubits, depth=depth, topology=topology, alpha=alpha)
                for _ in range(self.num_blocks)
            ])
            # Calculate feature dimension: n single + n(n-1)/2 two-body
            self.q_out_dim = n_qubits + (n_qubits * (n_qubits - 1) // 2)
        else:
            self.q_out_dim = n_qubits # Baseline tanh dimension matches projection size
        
        # Section 3.3: Classical Decoding
        # Concatenated features
        self.classifier = nn.Linear(self.num_blocks * self.q_out_dim, num_classes)

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
