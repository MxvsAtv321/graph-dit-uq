import torch
from torch_geometric.datasets import QM9
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def prepare_qm9_subset(fraction=0.05, seed=42):
    """Download and prepare QM9 subset for quick experiments"""
    data_dir = Path("data/raw")
    processed_path = Path("data/qm9_subset.pt")

    # Create directories
    data_dir.mkdir(parents=True, exist_ok=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    if processed_path.exists():
        logger.info(f"Loading cached QM9 subset from {processed_path}")
        return torch.load(processed_path, weights_only=False)

    logger.info("Downloading full QM9 dataset...")
    # Download full QM9
    dataset = QM9(root=data_dir)

    # Take fraction for speed
    torch.manual_seed(seed)
    indices = torch.randperm(len(dataset))[: int(len(dataset) * fraction)]
    subset = [dataset[i] for i in indices]

    # Save processed
    torch.save(subset, processed_path, _use_new_zipfile_serialization=False)
    logger.info(f"Saved {len(subset)} molecules to {processed_path}")
    return subset


def get_qm9_statistics(dataset):
    """Get basic statistics about the QM9 dataset"""
    if not dataset:
        return {}

    num_atoms = [data.num_nodes for data in dataset]
    num_edges = [data.num_edges for data in dataset]

    stats = {
        "num_molecules": len(dataset),
        "avg_atoms": sum(num_atoms) / len(num_atoms),
        "max_atoms": max(num_atoms),
        "min_atoms": min(num_atoms),
        "avg_edges": sum(num_edges) / len(num_edges),
        "max_edges": max(num_edges),
        "min_edges": min(num_edges),
    }

    return stats


if __name__ == "__main__":
    # Test the loader
    logging.basicConfig(level=logging.INFO)
    dataset = prepare_qm9_subset(fraction=0.05)
    stats = get_qm9_statistics(dataset)
    print(f"Dataset statistics: {stats}")
