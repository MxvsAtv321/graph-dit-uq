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
        "rdkit-pypi",
        "torch_geometric",
        "matplotlib",
        "pandas",
        "scikit-learn",
        "prolif",
        "codecarbon",
        "wandb",
        "dgl",
    ],
) 