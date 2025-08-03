#!/usr/bin/env python3
"""
Select top ligands from Stage 3 results for MD validation.
"""

import pandas as pd
import argparse
import os

def select_top_ligands(input_file, output_file, n_ligands=20, criteria='physics_reward'):
    """
    Select top N ligands based on specified criteria.
    
    Args:
        input_file: Path to stage3_results.parquet
        output_file: Path to save selected ligands
        n_ligands: Number of top ligands to select
        criteria: Selection criteria ('physics_reward', 'diffdock_confidence', 'quickvina_score')
    """
    
    print(f"🔬 Selecting top {n_ligands} ligands for MD validation...")
    
    # Read Stage 3 results
    df = pd.read_parquet(input_file)
    print(f"📊 Total molecules: {len(df)}")
    
    # Sort by criteria
    if criteria == 'physics_reward':
        df_sorted = df.sort_values('physics_reward', ascending=False)
        print(f"🎯 Sorting by physics reward (range: {df['physics_reward'].min():.4f} - {df['physics_reward'].max():.4f})")
    elif criteria == 'diffdock_confidence':
        df_sorted = df.sort_values('diffdock_confidence', ascending=False)
        print(f"🎯 Sorting by DiffDock confidence (range: {df['diffdock_confidence'].min():.4f} - {df['diffdock_confidence'].max():.4f})")
    elif criteria == 'quickvina_score':
        df_sorted = df.sort_values('quickvina_score', ascending=True)  # Lower is better
        print(f"🎯 Sorting by QuickVina score (range: {df['quickvina_score'].min():.4f} - {df['quickvina_score'].max():.4f})")
    else:
        raise ValueError(f"Unknown criteria: {criteria}")
    
    # Select top N ligands
    top_ligands = df_sorted.head(n_ligands).copy()
    
    # Add metadata
    top_ligands['selection_criteria'] = criteria
    top_ligands['selection_rank'] = range(1, n_ligands + 1)
    top_ligands['md_validation_ready'] = True
    
    # Save selected ligands
    top_ligands.to_csv(output_file, index=False)
    
    print(f"✅ Selected {len(top_ligands)} ligands")
    print(f"📁 Saved to: {output_file}")
    
    # Print summary
    print(f"\n🏆 TOP 5 LIGANDS:")
    for i, (idx, row) in enumerate(top_ligands.head(5).iterrows(), 1):
        print(f"{i}. SMILES: {row['smiles']}")
        print(f"   Physics Reward: {row['physics_reward']:.4f}")
        print(f"   DiffDock Conf: {row['diffdock_confidence']:.4f}")
        print(f"   QuickVina: {row['quickvina_score']:.4f}")
        print(f"   QED: {row['qed']:.4f}")
        print()
    
    return top_ligands

def main():
    parser = argparse.ArgumentParser(description="Select top ligands for MD validation")
    parser.add_argument("--input", default="data/stage3_results.parquet", 
                       help="Input Stage 3 results file")
    parser.add_argument("--output", default="ablation/top20_ddr1.csv", 
                       help="Output CSV file for selected ligands")
    parser.add_argument("--n", type=int, default=20, 
                       help="Number of ligands to select")
    parser.add_argument("--criteria", default="physics_reward", 
                       choices=['physics_reward', 'diffdock_confidence', 'quickvina_score'],
                       help="Selection criteria")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    select_top_ligands(args.input, args.output, args.n, args.criteria)

if __name__ == "__main__":
    main() 