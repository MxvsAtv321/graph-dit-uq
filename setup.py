from setuptools import setup, find_packages

setup(
    name="graph-dit-uq",
    version="0.1.0",
    description="A research-grade, uncertainty-aware graph-diffusion pipeline for multi-objective drug discovery",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.3.0",
        "flash-attn>=2.3.0",
        "torch-geometric>=2.4.0",
        "rdkit>=2023.9.5",
        "matplotlib",
        "pandas",
        "scikit-learn",
        "prolif",
        "codecarbon>=2.3.4",
        "wandb>=0.16.2",
        "dgl",
        "pytest",
    ],
) 