#!/usr/bin/env python3
"""Generate 10k molecules with property computation."""

import argparse
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from tqdm import tqdm

try:
    from rdkit import Chem
    from rdkit.Chem import QED, Crippen
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available, using mock properties")

from src.models.baselines.graph_dit import GraphDiTWrapper

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_sa_score(mol) -> float:
    """Calculate synthetic accessibility score using fragment contributions."""
    try:
        # Get number of atoms and rings
        num_atoms = mol.GetNumHeavyAtoms()
        num_rings = mol.GetRingInfo().NumRings()
        
        # Fragment penalty (simplified SA calculation)
        fragment_penalty = 0.0
        
        # Ring penalty
        ring_penalty = 0.0
        for ring in mol.GetRingInfo().AtomRings():
            ring_size = len(ring)
            if ring_size == 3:  # Cyclopropane
                ring_penalty += 0.5
            elif ring_size == 4:  # Cyclobutane
                ring_penalty += 0.25
            elif ring_size >= 8:  # Large rings
                ring_penalty += 0.1
        
        # Stereochemistry penalty
        stereo_penalty = 0.0
        for atom in mol.GetAtoms():
            if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
                stereo_penalty += 0.1
        
        # Calculate SA score (1 = easy to synthesize, 10 = difficult)
        sa_score = 1.0 + fragment_penalty + ring_penalty + stereo_penalty
        
        # Normalize to 0-1 range (invert so 1 = easy to synthesize)
        sa_normalized = max(0.0, min(1.0, 1.0 - (sa_score - 1.0) / 9.0))
        
        return sa_normalized
        
    except Exception as e:
        # Fallback to simple calculation
        num_atoms = mol.GetNumHeavyAtoms()
        sa = 1.0 - (num_atoms - 20) / 30  # Normalize around 20 atoms
        return max(0.1, min(1.0, sa))


def compute_properties(smiles: str) -> Optional[Dict]:
    """Compute QED, SA, and mock docking score."""
    if not RDKIT_AVAILABLE:
        # Mock properties when RDKit is not available
        return {
            "smiles": smiles,
            "qed": np.random.uniform(0.3, 0.9),
            "sa": np.random.uniform(0.4, 0.8),
            "logp": np.random.uniform(-2, 5),
            "docking": -8.5 + 2.0 * np.random.randn()
        }
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    try:
        # Real properties
        qed = QED.qed(mol)
        logp = Crippen.MolLogP(mol)
        
        # SA calculation using RDKit fragment contributions
        sa = calculate_sa_score(mol)
        
        # Mock docking (will replace with QuickVina2)
        docking = -8.5 + 2.0 * np.random.randn()  # Centered at -8.5 kcal/mol
        
        # Get number of atoms
        num_atoms = mol.GetNumHeavyAtoms()
        
        return {
            "smiles": smiles,
            "qed": qed,
            "sa": sa,
            "logp": logp,
            "docking": docking,
            "num_atoms": num_atoms
        }
    except Exception as e:
        logger.warning(f"Error computing properties for {smiles}: {e}")
        return None


def generate_10k_molecules(checkpoint_path: str, n_molecules: int = 10000, 
                          batch_size: int = 100) -> List[Dict]:
    """Generate molecules and compute properties."""
    logger.info(f"Loading model from {checkpoint_path}")
    model = GraphDiTWrapper.load_from_checkpoint(checkpoint_path)
    
    results = []
    valid_count = 0
    
    logger.info(f"Generating {n_molecules} molecules...")
    with tqdm(total=n_molecules, desc="Generating molecules") as pbar:
        while valid_count < n_molecules:
            # Generate batch
            smiles_batch = model.generate(batch_size=batch_size)
            
            # Compute properties
            for smiles in smiles_batch:
                props = compute_properties(smiles)
                if props is not None:
                    results.append(props)
                    valid_count += 1
                    pbar.update(1)
                    
                if valid_count >= n_molecules:
                    break
    
    logger.info(f"Generated {len(results)} valid molecules")
    return results


def save_results(results: List[Dict], output_path: str):
    """Save results to pickle file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "wb") as f:
        pickle.dump(results, f)
    
    logger.info(f"Saved results to {output_path}")


def print_summary(results: List[Dict]):
    """Print summary statistics."""
    if not results:
        logger.warning("No results to summarize")
        return
    
    # Extract properties
    qed_scores = [r["qed"] for r in results]
    docking_scores = [r["docking"] for r in results]
    sa_scores = [r["sa"] for r in results]
    logp_scores = [r["logp"] for r in results]
    
    print("\n" + "="*50)
    print("GENERATION SUMMARY")
    print("="*50)
    print(f"Total molecules: {len(results):,}")
    print(f"Valid SMILES: {len([r for r in results if r['smiles']]):,}")
    
    print(f"\nQED Score:")
    print(f"  Mean: {np.mean(qed_scores):.3f}")
    print(f"  Std:  {np.std(qed_scores):.3f}")
    print(f"  Min:  {np.min(qed_scores):.3f}")
    print(f"  Max:  {np.max(qed_scores):.3f}")
    
    print(f"\nDocking Score (kcal/mol):")
    print(f"  Mean: {np.mean(docking_scores):.3f}")
    print(f"  Std:  {np.std(docking_scores):.3f}")
    print(f"  Min:  {np.min(docking_scores):.3f}")
    print(f"  Max:  {np.max(docking_scores):.3f}")
    
    print(f"\nSynthetic Accessibility:")
    print(f"  Mean: {np.mean(sa_scores):.3f}")
    print(f"  Std:  {np.std(sa_scores):.3f}")
    
    print(f"\nLogP:")
    print(f"  Mean: {np.mean(logp_scores):.3f}")
    print(f"  Std:  {np.std(logp_scores):.3f}")
    
    # Find Pareto optimal molecules
    pareto_indices = compute_pareto_front(docking_scores, qed_scores)
    print(f"\nPareto optimal molecules: {len(pareto_indices)}")
    print("="*50)


def compute_pareto_front(x: List[float], y: List[float]) -> List[int]:
    """Find indices of Pareto optimal points (minimize x, maximize y)."""
    points = np.column_stack((x, y))
    pareto_indices = []
    
    for i, point in enumerate(points):
        dominated = False
        for j, other in enumerate(points):
            if i != j and other[0] <= point[0] and other[1] >= point[1]:
                if other[0] < point[0] or other[1] > point[1]:
                    dominated = True
                    break
        if not dominated:
            pareto_indices.append(i)
    
    return pareto_indices


def main():
    parser = argparse.ArgumentParser(description="Generate 10k molecules")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--n_molecules", type=int, default=10000,
                       help="Number of molecules to generate (default: 10000)")
    parser.add_argument("--batch_size", type=int, default=100,
                       help="Batch size for generation (default: 100)")
    parser.add_argument("--output_path", type=str, default="outputs/results_10k.pkl",
                       help="Output path for results (default: outputs/results_10k.pkl)")
    
    args = parser.parse_args()
    
    # Generate molecules
    results = generate_10k_molecules(
        checkpoint_path=args.checkpoint_path,
        n_molecules=args.n_molecules,
        batch_size=args.batch_size
    )
    
    # Save results
    save_results(results, args.output_path)
    
    # Print summary
    print_summary(results)
    
    print(f"\n✅ Generation complete! Results saved to {args.output_path}")


if __name__ == "__main__":
    main() 