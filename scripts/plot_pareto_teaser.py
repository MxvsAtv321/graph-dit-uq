#!/usr/bin/env python3
"""Generate early Pareto plot showing multi-objective trade-offs."""

import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_results(results_path: str) -> list:
    """Load results from pickle file."""
    with open(results_path, "rb") as f:
        results = pickle.load(f)
    logger.info(f"Loaded {len(results)} molecules from {results_path}")
    return results


def compute_pareto_front(x: list, y: list) -> list:
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
    
    # Sort by x coordinate for nice plotting
    pareto_indices.sort(key=lambda i: x[i])
    return pareto_indices


def plot_pareto_teaser(results_path: str = "outputs/results_10k.pkl", 
                      output_path: str = "screenshots/early_pareto.png"):
    """Generate early Pareto plot showing multi-objective trade-offs."""
    
    # Load results
    results = load_results(results_path)
    
    # Extract properties
    qed = [r["qed"] for r in results]
    docking = [r["docking"] for r in results]
    sa = [r["sa"] for r in results]
    
    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Main Pareto plot: Docking vs QED
    scatter1 = ax1.scatter(docking, qed, alpha=0.6, s=20, c=qed, cmap='viridis')
    
    # Find Pareto front
    pareto_indices = compute_pareto_front(docking, qed)
    pareto_x = [docking[i] for i in pareto_indices]
    pareto_y = [qed[i] for i in pareto_indices]
    
    # Plot Pareto front
    ax1.plot(pareto_x, pareto_y, 'r-', linewidth=3, label='Pareto Front')
    ax1.scatter(pareto_x, pareto_y, color='red', s=100, edgecolors='darkred', 
               linewidth=2, zorder=5, label='Pareto Optimal')
    
    # Annotations
    ax1.set_xlabel('Docking Score (kcal/mol)', fontsize=14)
    ax1.set_ylabel('QED Score', fontsize=14)
    ax1.set_title('Docking vs QED: Pareto Front', fontsize=16, weight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add text box
    textstr = f'Molecules: {len(results):,}\nPareto Optimal: {len(pareto_indices)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    # 2. QED distribution
    ax2.hist(qed, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(np.mean(qed), color='red', linestyle='--', label=f'Mean: {np.mean(qed):.3f}')
    ax2.set_xlabel('QED Score', fontsize=14)
    ax2.set_ylabel('Count', fontsize=14)
    ax2.set_title('QED Distribution', fontsize=16, weight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Docking score distribution
    ax3.hist(docking, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    ax3.axvline(np.mean(docking), color='red', linestyle='--', label=f'Mean: {np.mean(docking):.3f}')
    ax3.set_xlabel('Docking Score (kcal/mol)', fontsize=14)
    ax3.set_ylabel('Count', fontsize=14)
    ax3.set_title('Docking Score Distribution', fontsize=16, weight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. SA vs QED scatter
    scatter4 = ax4.scatter(sa, qed, alpha=0.6, s=20, c=docking, cmap='plasma')
    ax4.set_xlabel('Synthetic Accessibility', fontsize=14)
    ax4.set_ylabel('QED Score', fontsize=14)
    ax4.set_title('SA vs QED (colored by docking)', fontsize=16, weight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter4, ax=ax4)
    cbar.set_label('Docking Score (kcal/mol)', fontsize=12)
    
    # Overall title
    fig.suptitle('Graph DiT-UQ: Early Results - 10k Generated Molecules', 
                fontsize=20, weight='bold', y=0.98)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved Pareto plot to {output_path}")
    
    # Print statistics
    print("\n" + "="*60)
    print("PARETO ANALYSIS")
    print("="*60)
    print(f"Total molecules: {len(results):,}")
    print(f"Pareto optimal: {len(pareto_indices):,} ({len(pareto_indices)/len(results)*100:.1f}%)")
    print(f"Best QED: {max(qed):.3f}")
    print(f"Best docking: {min(docking):.3f}")
    print(f"Pareto QED range: {min(pareto_y):.3f} - {max(pareto_y):.3f}")
    print(f"Pareto docking range: {min(pareto_x):.3f} - {max(pareto_x):.3f}")
    print("="*60)


def update_readme_with_plot():
    """Update README with plot information."""
    readme_path = Path("README.md")
    if not readme_path.exists():
        logger.warning("README.md not found, skipping update")
        return
    
    # Read current README
    with open(readme_path, "r") as f:
        content = f.read()
    
    # Add plot section if not present
    if "## Early Results" not in content:
        plot_section = """

## Early Results

![Pareto Front](screenshots/early_pareto.png)

*Multi-objective optimization results from 10k generated molecules. The red line shows the Pareto front, representing the optimal trade-off between docking score and QED.*

"""
        # Insert before the last line
        lines = content.split('\n')
        lines.insert(-1, plot_section)
        content = '\n'.join(lines)
        
        # Write back
        with open(readme_path, "w") as f:
            f.write(content)
        
        logger.info("Updated README.md with plot section")


def main():
    parser = argparse.ArgumentParser(description="Generate Pareto plot")
    parser.add_argument("--results_path", type=str, default="outputs/results_10k.pkl",
                       help="Path to results pickle file")
    parser.add_argument("--output_path", type=str, default="screenshots/early_pareto.png",
                       help="Output path for plot")
    parser.add_argument("--update_readme", action="store_true",
                       help="Update README with plot section")
    
    args = parser.parse_args()
    
    # Generate plot
    plot_pareto_teaser(args.results_path, args.output_path)
    
    # Update README if requested
    if args.update_readme:
        update_readme_with_plot()
    
    print(f"✅ Pareto plot generated: {args.output_path}")


if __name__ == "__main__":
    main() 