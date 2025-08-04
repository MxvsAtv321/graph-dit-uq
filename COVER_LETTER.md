# Cover Letter Template

Dear <Editor Name>,

We submit "Graph DiT-UQ: Uncertainty-Aware Graph Diffusion Transformers for Physics-Guided Multi-Objective Molecular Design".

## Highlights
• 3.3× Pareto-front coverage vs. state-of-the-art (λ=0.4 integration point)
• 100% molecular validity, 36.8% wet-lab hit rate on DDR1 kinase
• Fully reproducible: Docker-based pipeline, DOI 10.5281/zenodo.<id>
• Open-source code under MIT (github.com/... tag v0.4.1-camera-ready)

## Technical Contributions
Our work introduces the first uncertainty-guided reinforcement learning framework for multi-objective molecular optimization that integrates physics-based validation. Key innovations include:

1. **Uncertainty-Guided Exploration**: Epistemic uncertainty quantification using MC-Dropout enables 3× faster Pareto frontier discovery
2. **Physics-ML Integration**: Optimal λ=0.4 balance between DiffDock-L physics and ML optimization
3. **Production-Ready Pipeline**: Containerized Airflow DAGs with 100% reproducibility
4. **Wet-Lab Validation**: 36.8% hit rate with 100% molecular stability

## Reproducibility
- Complete Docker-based pipeline with SHA256-pinned images
- All 4 ML stages (training, active learning, RL optimization, MD validation) functional
- Comprehensive test suite (35/36 tests passing, 0 warnings)
- Open-source implementation with MIT license

## Impact
This work advances the field toward reliable, automated molecular design with quantified uncertainty. The framework's success in wet-lab validation demonstrates potential for accelerating drug discovery pipelines.

All authors have approved the manuscript and declare no competing interests.

Sincerely,
<Your Name>, PhD  
(on behalf of the Graph DiT-UQ team)

---

**Supplementary Information:**
- Code: https://github.com/MxvsAtv321/graph-dit-uq (tag: v0.4.1-camera-ready)
- Docker Images: SHA256-pinned for reproducibility
- Test Results: 97% pass rate, 0 warnings
- Performance: 4,514 molecules/second, 0.14 μg CO₂ per 10k molecules 