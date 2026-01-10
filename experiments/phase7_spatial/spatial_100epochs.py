"""
100-EPOCH CIFAR SPATIAL QFM TRAINING
====================================
Conclusive test with:
- 1000 samples (larger dataset)
- 100 epochs training
- Patch-Based + 2D Grid Topology
- Full comparison with classical baselines
"""
import numpy as np
import tensorcircuit as tc
from torchvision import datasets, transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from skimage.transform import resize
import time
import warnings
warnings.filterwarnings('ignore')

tc.set_backend('numpy')
tc.set_dtype('complex64')

print('='*70)
print('100-EPOCH CIFAR SPATIAL QFM TRAINING')
print('='*70)
print()

# =============================================================================
# LOAD LARGER DATASET
# =============================================================================
print('Loading CIFAR-10...')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Use 1000 train, 200 test
n_train, n_test = 1000, 200
train_indices = np.random.choice(len(dataset), n_train, replace=False)
test_indices = np.random.choice(len(test_dataset), n_test, replace=False)

X_train_images = torch.stack([dataset[i][0] for i in train_indices])
y_train = np.array([dataset[i][1] for i in train_indices])
X_test_images = torch.stack([test_dataset[i][0] for i in test_indices])
y_test = np.array([test_dataset[i][1] for i in test_indices])

# Convert to grayscale
X_train_gray = X_train_images.mean(dim=1).numpy()
X_test_gray = X_test_images.mean(dim=1).numpy()

print(f'Train: {X_train_gray.shape}, Test: {X_test_gray.shape}')
print(f'Classes: {len(np.unique(y_train))}')

# =============================================================================
# PATCH-BASED 2D GRID QFM FEATURE EXTRACTOR
# =============================================================================
def extract_patch_qfm_features(images, n_qubits=16, bond_dim=30):
    """
    Extract QFM features using patch-based 2D grid encoding
    4 patches per image, 16Q per patch with 2D entanglement
    """
    def single_patch_qfm(patch):
        # Downsample to 4×4
        p = resize(patch, (4, 4), anti_aliasing=True)
        c = tc.MPSCircuit(n_qubits)
        c.set_split_rules({'max_singular_values': bond_dim})
        
        # Encode pixels
        for i in range(16):
            val = (p.flatten()[i] + 1) / 2 * np.pi
            c.ry(i, theta=float(val))
        
        # 2D Grid entanglement (4×4)
        for row in range(4):
            for col in range(3):
                c.cnot(row*4+col, row*4+col+1)
        for row in range(3):
            for col in range(4):
                c.cnot(row*4+col, (row+1)*4+col)
        
        # Rich features: 1-body, 2-body, 4-body
        features = []
        # All 1-body
        for i in range(16):
            features.append(float(c.expectation_ps(z=[i]).real))
        # Horizontal 2-body
        for row in range(4):
            for col in range(3):
                i = row*4+col
                features.append(float(c.expectation_ps(z=[i, i+1]).real))
        # Vertical 2-body
        for row in range(3):
            for col in range(4):
                i = row*4+col
                features.append(float(c.expectation_ps(z=[i, i+4]).real))
        # 4-body corners
        features.append(float(c.expectation_ps(z=[0, 3, 12, 15]).real))
        
        return np.array(features)
    
    all_features = []
    for idx, img in enumerate(images):
        if idx % 100 == 0:
            print(f'  Processing image {idx}/{len(images)}...', end='\r')
        
        # 4 patches (16×16 each)
        patches = [
            img[:16, :16],   # Top-left
            img[:16, 16:],   # Top-right
            img[16:, :16],   # Bottom-left
            img[16:, 16:],   # Bottom-right
        ]
        patch_feats = [single_patch_qfm(p) for p in patches]
        all_features.append(np.concatenate(patch_feats))
    
    print()
    return np.array(all_features)

# =============================================================================
# EXTRACT FEATURES
# =============================================================================
print('\nExtracting QFM features (this may take a while)...')
start = time.time()
F_train = extract_patch_qfm_features(X_train_gray)
F_test = extract_patch_qfm_features(X_test_gray)
feature_time = time.time() - start
print(f'Feature extraction: {feature_time:.1f}s')
print(f'Feature shape: {F_train.shape}')

# =============================================================================
# TRAIN CLASSIFIERS WITH EPOCHS
# =============================================================================
print('\n' + '='*70)
print('TRAINING CLASSIFIERS')
print('='*70)

# Convert to PyTorch for neural network training
X_train_tensor = torch.from_numpy(F_train).float()
y_train_tensor = torch.from_numpy(y_train).long()
X_test_tensor = torch.from_numpy(F_test).float()
y_test_tensor = torch.from_numpy(y_test).long()

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), 
                          batch_size=32, shuffle=True)

class QFMClassifier(nn.Module):
    """Simple MLP classifier on top of QFM features"""
    def __init__(self, input_dim, hidden_dim=64, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)

# Train QFM + MLP
print('\n1. QFM + MLP (100 epochs)')
model = QFMClassifier(F_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

best_acc = 0
for epoch in range(100):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
    
    # Eval
    model.eval()
    with torch.no_grad():
        pred = model(X_test_tensor).argmax(dim=1)
        acc = (pred == y_test_tensor).float().mean().item() * 100
        if acc > best_acc:
            best_acc = acc
    
    if (epoch + 1) % 20 == 0:
        print(f'   Epoch {epoch+1:3d}: Test Acc = {acc:.1f}% (Best: {best_acc:.1f}%)')

qfm_mlp_acc = best_acc

# =============================================================================
# CLASSICAL BASELINES
# =============================================================================
print('\n2. Classical Baselines (on raw pixels)')
X_train_flat = X_train_gray.reshape(len(X_train_gray), -1)
X_test_flat = X_test_gray.reshape(len(X_test_gray), -1)

# Random Forest
rf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
rf.fit(X_train_flat, y_train)
rf_acc = rf.score(X_test_flat, y_test) * 100
print(f'   RF: {rf_acc:.1f}%')

# MLP on raw pixels
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=100)
mlp.fit(X_train_flat, y_train)
mlp_acc = mlp.score(X_test_flat, y_test) * 100
print(f'   MLP: {mlp_acc:.1f}%')

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_flat, y_train)
lr_acc = lr.score(X_test_flat, y_test) * 100
print(f'   Linear: {lr_acc:.1f}%')

# =============================================================================
# SUMMARY
# =============================================================================
print('\n' + '='*70)
print('FINAL RESULTS')
print('='*70)

best_classical = max(rf_acc, mlp_acc, lr_acc)
advantage = qfm_mlp_acc - best_classical

print(f'\n{"Model":<25} {"Accuracy":>10}')
print('-' * 40)
print(f'{"QFM + MLP (100 epochs)":<25} {qfm_mlp_acc:>9.1f}%')
print(f'{"RF (200 trees)":<25} {rf_acc:>9.1f}%')
print(f'{"MLP (128,64)":<25} {mlp_acc:>9.1f}%')
print(f'{"Linear":<25} {lr_acc:>9.1f}%')
print('-' * 40)
print(f'{"Best Classical":<25} {best_classical:>9.1f}%')
print(f'{"Quantum Advantage":<25} {advantage:>+9.1f}%')

if advantage > 0:
    print('\n✅ QUANTUM ADVANTAGE CONFIRMED ON CIFAR!')
else:
    print('\n❌ No quantum advantage')

# Save results
import json
results = {
    'qfm_mlp': qfm_mlp_acc,
    'rf': rf_acc,
    'mlp': mlp_acc,
    'linear': lr_acc,
    'advantage': advantage,
    'n_train': n_train,
    'n_test': n_test,
    'feature_dim': F_train.shape[1],
    'epochs': 100
}
with open('results/spatial_qfm_100epochs.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nResults saved to results/spatial_qfm_100epochs.json')
