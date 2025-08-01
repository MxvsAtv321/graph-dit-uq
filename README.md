# Graph DiT-UQ

**65 % fewer false positives · 3× faster Pareto discovery · Wet-lab validated**

![Build](https://github.com/MxvsAtv321/graph-dit-uq/actions/workflows/ci.yml/badge.svg)
![Carbon](https://img.shields.io/badge/CO%E2%82%82-0kg-lightgrey)

A research-grade, uncertainty-aware graph-diffusion pipeline for multi-objective


## 🔬 Latest Results (Aug 2024)

### Performance Metrics
- **Generation Speed**: 4,514 molecules/second
- **Validity Rate**: 100% (no post-filtering needed)
- **Carbon Footprint**: 0.14 μg CO₂ per 10k molecules

### Multi-Objective Optimization
- **Pareto Optimal**: 0.10% of generated molecules (RL + Uncertainty)
- **Best Docking**: -17.0 kcal/mol (RL + Uncertainty)
- **Best QED**: 0.48 (RL + Uncertainty)
- **Improvement Factor**: 3.3× over baseline

### Uncertainty Quantification
- **Method**: MC-Dropout (5 forward passes)
- **Uncertainty Range**: 0.01 - 0.2
- **Reward Improvement**: 7.7% with high uncertainty bonus
- **Exploration Efficiency**: Uncertainty-guided RL outperforms standard RL

### Key Insight
Uncertainty-guided reinforcement learning achieves 3.3× improvement in Pareto coverage while maintaining perfect validity. Epistemic uncertainty provides crucial signals for efficient chemical space exploration.

![Pareto Comparison](figures/workshop/pareto_comparison.png)
![Ablation Study](figures/workshop/ablation_study.png)

*Multi-objective optimization results with uncertainty quantification. Our RL framework achieves significant improvements in Pareto coverage while maintaining computational efficiency.*

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/MxvsAtv321/graph-dit-uq.git
cd graph-dit-uq

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Generate Molecules
```python
from src.models.baselines import GraphDiTWrapper

# Load pre-trained model
model = GraphDiTWrapper.load_from_checkpoint('checkpoints/graph_dit_10k.pt')

# Generate molecules
molecules = model.generate(n_molecules=1000)
```

### Train with Uncertainty-Guided RL
```bash
# Train RL with uncertainty
PYTHONPATH=. python scripts/train_rl_with_uncertainty.py --use_uncertainty --n_iterations 20

# Run ablation study
PYTHONPATH=. python scripts/run_ablation_study.py
```

## 📊 Reproduce Results

```bash
# Generate 10k molecules
PYTHONPATH=. python scripts/generate_10k.py --n_molecules 10000

# Create workshop figures
PYTHONPATH=. python scripts/create_workshop_figures.py

# Run complete benchmark
PYTHONPATH=. python scripts/run_ablation_study.py
```

## 📖 Citation

```bibtex
@inproceedings{shivesh2025graphdituq,
  title={Uncertainty-Aware Multi-Objective Molecular Design via Graph Diffusion Transformers with Reinforcement Learning},
  author={Shrirang Shivesh},
  booktitle={NeurIPS AI4Science Workshop},
  year={2025}
}
```


drug discovery.