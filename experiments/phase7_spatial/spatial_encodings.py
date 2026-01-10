"""
SPATIAL-PRESERVING QFM ENCODINGS FOR CIFAR-10
==============================================
Testing 3 approaches that preserve spatial structure:
A. Patch-Based: 4×4 patches → QFM per patch → Pool
B. Grid Topology: 2D CNOT connections matching pixel adjacency
C. Hierarchical: Multi-scale pooling like CNN
"""
import numpy as np
import tensorcircuit as tc
from torchvision import datasets, transforms
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import time
import warnings
warnings.filterwarnings('ignore')

tc.set_backend('numpy')
tc.set_dtype('complex64')

print('='*70)
print('SPATIAL-PRESERVING QFM ENCODINGS FOR CIFAR-10')
print('='*70)

# Load CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
indices = np.random.choice(len(dataset), 300, replace=False)

X_images = torch.stack([dataset[i][0] for i in indices])  # (300, 3, 32, 32)
y = np.array([dataset[i][1] for i in indices])

# Convert to grayscale for simplicity
X_gray = X_images.mean(dim=1).numpy()  # (300, 32, 32)

print(f'Data: {X_gray.shape}, Classes: {len(np.unique(y))}')

# Split
X_train, X_test = X_gray[:240], X_gray[240:]
y_train, y_test = y[:240], y[240:]

results = []


# =============================================================================
# APPROACH A: PATCH-BASED ENCODING
# =============================================================================
def patch_based_qfm(images, patch_size=4, n_qubits=16, bond_dim=50, max_body=5):
    """
    Split image into patches, process each with QFM, pool features
    """
    n_patches_h = images.shape[1] // patch_size  # 32/4 = 8
    n_patches_w = images.shape[2] // patch_size  # 8
    
    def single_patch_qfm(patch):
        """QFM for one patch (4×4 = 16 pixels)"""
        c = tc.MPSCircuit(n_qubits)
        c.set_split_rules({'max_singular_values': bond_dim})
        
        # Encode patch pixels (flattened)
        patch_flat = patch.flatten()
        for i in range(n_qubits):
            val = (patch_flat[i % len(patch_flat)] + 1) / 2 * np.pi  # Normalize to [0, π]
            c.ry(i, theta=float(val))
        
        # 2D Grid entanglement (4×4 qubits)
        grid_size = int(np.sqrt(n_qubits))
        for row in range(grid_size):
            for col in range(grid_size - 1):
                i = row * grid_size + col
                c.cnot(i, i + 1)
        for row in range(grid_size - 1):
            for col in range(grid_size):
                i = row * grid_size + col
                j = i + grid_size
                c.cnot(i, j)
        
        features = []
        # 1-body (4 samples)
        for i in [0, 5, 10, 15]:
            features.append(float(c.expectation_ps(z=[i]).real))
        # 2-body adjacent (4)
        for i in [0, 4, 8, 12]:
            if i+1 < n_qubits:
                features.append(float(c.expectation_ps(z=[i, i+1]).real))
        # 4-body (corners)
        if max_body >= 4:
            features.append(float(c.expectation_ps(z=[0, 3, 12, 15]).real))
        
        return np.array(features)
    
    all_features = []
    for img in images:
        patch_features = []
        for ph in range(n_patches_h):
            for pw in range(n_patches_w):
                patch = img[ph*patch_size:(ph+1)*patch_size, 
                           pw*patch_size:(pw+1)*patch_size]
                patch_features.append(single_patch_qfm(patch))
        # Pool: mean across patches
        pooled = np.mean(patch_features, axis=0)
        all_features.append(pooled)
    return np.array(all_features)


# =============================================================================
# APPROACH B: GRID TOPOLOGY (Full image, 2D entanglement)
# =============================================================================
def grid_topology_qfm(images, grid_qubits=10, bond_dim=50, max_body=5):
    """
    Map downsampled image to 2D qubit grid with spatial CNOT connections
    """
    from skimage.transform import resize
    
    def single_image_qfm(img):
        # Downsample to grid_qubits × grid_qubits
        img_small = resize(img, (grid_qubits, grid_qubits), anti_aliasing=True)
        n_total = grid_qubits * grid_qubits
        
        c = tc.MPSCircuit(n_total)
        c.set_split_rules({'max_singular_values': bond_dim})
        
        # Encode pixels
        for i in range(n_total):
            row, col = divmod(i, grid_qubits)
            val = (img_small[row, col] + 1) / 2 * np.pi
            c.ry(i, theta=float(val))
        
        # 2D Grid entanglement
        for row in range(grid_qubits):
            for col in range(grid_qubits - 1):
                i = row * grid_qubits + col
                c.cnot(i, i + 1)
        for row in range(grid_qubits - 1):
            for col in range(grid_qubits):
                i = row * grid_qubits + col
                j = i + grid_qubits
                c.cnot(i, j)
        
        features = []
        # Sample 1-body (corners + center)
        sample_pts = [0, grid_qubits-1, n_total//2, n_total-grid_qubits, n_total-1]
        for i in sample_pts[:min(5, n_total)]:
            features.append(float(c.expectation_ps(z=[i]).real))
        
        # 2-body: horizontal adjacent (first row)
        for i in range(min(5, grid_qubits-1)):
            features.append(float(c.expectation_ps(z=[i, i+1]).real))
        
        # 2-body: vertical adjacent (first col)
        for i in range(min(5, grid_qubits-1)):
            j = i * grid_qubits
            features.append(float(c.expectation_ps(z=[j, j+grid_qubits]).real))
        
        # 5-body cross pattern (center)
        if max_body >= 5 and n_total > 20:
            mid = n_total // 2
            cross = [mid, mid-1, mid+1, mid-grid_qubits, mid+grid_qubits]
            cross = [x for x in cross if 0 <= x < n_total]
            if len(cross) >= 5:
                features.append(float(c.expectation_ps(z=cross[:5]).real))
        
        return np.array(features)
    
    return np.array([single_image_qfm(img) for img in images])


# =============================================================================
# APPROACH C: HIERARCHICAL POOLING
# =============================================================================
def hierarchical_qfm(images, bond_dim=50, max_body=5):
    """
    Hierarchical: 16 patches (8×8) → QFM → Pool to 4 → QFM → Classifier
    """
    def layer1_qfm(patch):
        """First layer: Process 8×8 patch with 16 qubits"""
        n_qubits = 16
        c = tc.MPSCircuit(n_qubits)
        c.set_split_rules({'max_singular_values': bond_dim})
        
        # Downsample 8×8 to 4×4 (16 pixels)
        from skimage.transform import resize
        patch_small = resize(patch, (4, 4), anti_aliasing=True)
        
        for i in range(n_qubits):
            row, col = divmod(i, 4)
            val = (patch_small[row, col] + 1) / 2 * np.pi
            c.ry(i, theta=float(val))
        
        # 2D Grid entanglement
        for row in range(4):
            for col in range(3):
                i = row * 4 + col
                c.cnot(i, i + 1)
        for row in range(3):
            for col in range(4):
                i = row * 4 + col
                c.cnot(i, i + 4)
        
        # Features: corners + center
        features = []
        for i in [0, 3, 5, 10, 12, 15]:
            features.append(float(c.expectation_ps(z=[i]).real))
        return np.array(features)
    
    def layer2_qfm(patch_features):
        """Second layer: Combine 16 patch features"""
        n_qubits = min(len(patch_features), 16)
        c = tc.MPSCircuit(n_qubits)
        c.set_split_rules({'max_singular_values': bond_dim})
        
        for i in range(n_qubits):
            val = (patch_features[i % len(patch_features)] + 1) / 2 * np.pi
            c.ry(i, theta=float(val))
        
        for i in range(n_qubits - 1):
            c.cnot(i, i + 1)
        
        features = []
        for i in range(min(8, n_qubits)):
            features.append(float(c.expectation_ps(z=[i]).real))
        return np.array(features)
    
    all_features = []
    for img in images:
        # Layer 1: Split into 16 patches (each 8×8)
        patch_size = 8
        l1_features = []
        for ph in range(4):
            for pw in range(4):
                patch = img[ph*patch_size:(ph+1)*patch_size, 
                           pw*patch_size:(pw+1)*patch_size]
                l1_features.append(layer1_qfm(patch))
        
        # Pool Layer 1: Flatten all patch features
        l1_pooled = np.concatenate(l1_features)
        
        # Layer 2: Process pooled features
        l2_features = layer2_qfm(l1_pooled)
        
        all_features.append(np.concatenate([l1_pooled[:16], l2_features]))
    
    return np.array(all_features)


# =============================================================================
# RUN TESTS
# =============================================================================

print('\n' + '='*70)
print('APPROACH A: PATCH-BASED ENCODING (4×4 patches, 16Q per patch)')
print('='*70)
start = time.time()
F_train = patch_based_qfm(X_train, patch_size=4, n_qubits=16, max_body=5)
F_test = patch_based_qfm(X_test, patch_size=4, n_qubits=16, max_body=5)
elapsed = time.time() - start

lr = LogisticRegression(max_iter=1000)
lr.fit(F_train, y_train)
acc_patch = lr.score(F_test, y_test) * 100

print(f'Features: {F_train.shape[1]}, Accuracy: {acc_patch:.1f}%, Time: {elapsed:.1f}s')
results.append(('Patch-Based', acc_patch))


print('\n' + '='*70)
print('APPROACH B: GRID TOPOLOGY (8×8 qubits, 2D entanglement)')
print('='*70)
start = time.time()
F_train = grid_topology_qfm(X_train, grid_qubits=8, max_body=5)
F_test = grid_topology_qfm(X_test, grid_qubits=8, max_body=5)
elapsed = time.time() - start

lr = LogisticRegression(max_iter=1000)
lr.fit(F_train, y_train)
acc_grid = lr.score(F_test, y_test) * 100

print(f'Features: {F_train.shape[1]}, Accuracy: {acc_grid:.1f}%, Time: {elapsed:.1f}s')
results.append(('Grid Topology', acc_grid))


print('\n' + '='*70)
print('APPROACH C: HIERARCHICAL POOLING (16 patches → Pool → QFM)')
print('='*70)
start = time.time()
F_train = hierarchical_qfm(X_train, max_body=5)
F_test = hierarchical_qfm(X_test, max_body=5)
elapsed = time.time() - start

lr = LogisticRegression(max_iter=1000)
lr.fit(F_train, y_train)
acc_hier = lr.score(F_test, y_test) * 100

print(f'Features: {F_train.shape[1]}, Accuracy: {acc_hier:.1f}%, Time: {elapsed:.1f}s')
results.append(('Hierarchical', acc_hier))


# =============================================================================
# CLASSICAL BASELINE
# =============================================================================
print('\n' + '='*70)
print('CLASSICAL BASELINE (Flattened grayscale → RF)')
print('='*70)
X_train_flat = X_train.reshape(len(X_train), -1)
X_test_flat = X_test.reshape(len(X_test), -1)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train_flat, y_train)
acc_rf = rf.score(X_test_flat, y_test) * 100
print(f'RF Accuracy: {acc_rf:.1f}%')
results.append(('Classical RF', acc_rf))


# =============================================================================
# SUMMARY
# =============================================================================
print('\n' + '='*70)
print('SUMMARY: SPATIAL-PRESERVING QFM ENCODINGS')
print('='*70)

best_qfm = max([r[1] for r in results if r[0] != 'Classical RF'])

for name, acc in results:
    marker = '✅' if acc >= acc_rf else ''
    print(f'{name:<20}: {acc:>5.1f}% {marker}')

print(f'\nBest QFM: {best_qfm:.1f}%')
print(f'Classical: {acc_rf:.1f}%')
print(f'Advantage: {best_qfm - acc_rf:+.1f}%')

if best_qfm > acc_rf:
    print('\n✅ QUANTUM ADVANTAGE WITH SPATIAL ENCODING!')
else:
    print('\n❌ Spatial encoding helps but still no advantage over RF')
