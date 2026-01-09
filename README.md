# Project Antigravity: Quantum Feature Map Validation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorCircuit](https://img.shields.io/badge/quantum-TensorCircuit-purple.svg)](https://github.com/tencent-quantum-lab/tensorcircuit)
[![PyTorch](https://img.shields.io/badge/deep_learning-PyTorch-orange.svg)](https://pytorch.org/)

## Overview

This project rigorously validates the **Quantum Correlation Filter** hypothesis: Fixed Quantum Feature Maps (QFMs) are specialized high-dimensional correlation filters, not general-purpose feature extractors.

### Key Findings

| Proof | Task | Quantum | Classical | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Interaction** | Parity/XOR | **100%** | 53% | ✅ Quantum Advantage |
| **Capacity** | CIFAR-10 (Wide) | 50% | 52.5% | ✅ Escaped Bottleneck |
| **Gradient Flow** | 10k×100 epochs | 1.05 | 1.00 | ✅ Both Healthy |

### Core Discovery

We distinguished **Feature Collapse** from **Vanishing Gradients**:
- Gradients flow normally through Fixed QFMs
- Features collapse geometrically (Concentration of Measure)
- **Solution**: Width (parallel circuits), not Depth

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

### Quantum Circuit (per block)
```
|0⟩ ─ Ry(θ₀) ─●───────── ⟨Z⟩
|0⟩ ─ Ry(θ₁) ─X─●─────── ⟨Z⟩
|0⟩ ─ Ry(θ₂) ───X─●───── ⟨Z⟩
|0⟩ ─ Ry(θ₃) ─────X─●─── ⟨Z⟩
              └─────X─── (ring)
```

**Observables**: 4 single-body ⟨Zᵢ⟩ + 6 two-body ⟨ZᵢZⱼ⟩ = 10 features

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
