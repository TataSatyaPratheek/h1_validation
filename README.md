# Project Antigravity: Quantum Feature Map Validation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorCircuit](https://img.shields.io/badge/quantum-TensorCircuit-purple.svg)](https://github.com/tencent-quantum-lab/tensorcircuit)
[![PyTorch](https://img.shields.io/badge/deep_learning-PyTorch-orange.svg)](https://pytorch.org/)

## Overview

This project rigorously validates the **Quantum Correlation Filter** hypothesis: Fixed Quantum Feature Maps (QFMs) are specialized high-dimensional correlation filters, not general-purpose feature extractors.

### Key Findings

| Task | 8Q Chain | 4Q Ring | Classical | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Parity/XOR** | **100%** | 71% | 53% | ✅ Quantum Advantage |
| **Circle (Radial)** | **98%** | 69% | 69% | ✅ Quantum Advantage |
| **CIFAR-10 (100 epochs)** | 40% | 42% | **46%** | ❌ Classical Wins |

### Core Discovery: The "Goldilocks Zone"

**Quantum advantage is problem-specific, not universal.**
- ✅ **Excels at**: Parity/XOR, Radial Separation (Circle), Low-D correlation tasks
- ❌ **Fails at**: High-dimensional natural images (CIFAR-10), Generic classification

**New Architecture**: 8-Qubit Redundant Chain (4 inputs → 8 qubits via cyclic encoding)

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/project-antigravity.git
cd project-antigravity

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch torchvision tensorcircuit numpy matplotlib seaborn scikit-learn
```

## Quick Start

### 1. Parity Task (Quantum Advantage Demo)
```bash
python test_parity_advantage.py
```
Expected: Quantum 100% vs Classical ~50%

### 2. CIFAR-10 Training
```bash
# Quantum Model (Wide Architecture)
python train.py --epochs 100 --dataset cifar10 --subset_size 1000 --use_quantum

# Classical Baseline
python train.py --epochs 100 --dataset cifar10 --subset_size 1000
```

### 3. Gradient Survival Test
```bash
python test_gradient_survival.py
```

### 4. Visualizations
```bash
python visualize_features.py          # t-SNE + Heatmaps
python visualize_decision_boundary.py  # XOR topology
```

## Project Structure

```
project-antigravity/
├── model.py                 # Quantum & Classical architectures
├── train.py                 # Training pipeline
├── data.py                  # CIFAR-10 & Parity datasets
├── utils.py                 # MPS optimization patches
│
├── test_parity_advantage.py # Proof 1: Interaction
├── test_gradient_survival.py# Proof 3: Gradient Flow
├── visualize_*.py           # Welch Labs-style plots
│
├── results/                 # Generated plots & metrics
├── logs/                    # Training logs
└── research_summary.md      # Full scientific report
```

## Architecture

### Quantum Hybrid Model
```
CNN(64×64) → Flatten(4096) → [Projection(10) → QFM(4 qubits)]×16 → Linear(10)
```

### Quantum Circuit: 8-Qubit Redundant Chain (Best)
```
|0⟩ ─ Ry(θ₀) ─●─────────────── ⟨Z⟩
|0⟩ ─ Ry(θ₁) ─X─●───────────── ⟨Z⟩
|0⟩ ─ Ry(θ₂) ───X─●─────────── ⟨Z⟩
|0⟩ ─ Ry(θ₃) ─────X─●───────── ⟨Z⟩
|0⟩ ─ Ry(θ₀) ───────X─●─────── ⟨Z⟩  (redundant encoding)
|0⟩ ─ Ry(θ₁) ─────────X─●───── ⟨Z⟩
|0⟩ ─ Ry(θ₂) ───────────X─●─── ⟨Z⟩
|0⟩ ─ Ry(θ₃) ─────────────X─── ⟨Z⟩
```

**Observables**: 8 single-body ⟨Zᵢ⟩ + 28 two-body ⟨ZᵢZⱼ⟩ = **36 features**

## Results

### Decision Boundary (Parity Task)
![Decision Boundary](results/decision_boundary_parity.png)

### Phase 4 Training (Wide Architecture)
![Phase 4](results/comparison_plot_phase4.png)

## MPS Optimization (Apple Silicon)

This project includes patches for running on Apple M1/M2/M3:
- `float64 → float32` auto-casting for TensorCircuit
- MPS-aware DataLoader generators

## Citation

If you use this work, please cite:
```bibtex
@misc{antigravity2026,
  title={Project Antigravity: Quantum Feature Maps as Correlation Filters},
  author={Your Name},
  year={2026},
  howpublished={GitHub}
}
```

## License

MIT License - See [LICENSE](LICENSE) for details.
