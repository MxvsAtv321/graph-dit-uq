#!/usr/bin/env python3
"""
Render hypervolume vs lambda figure for λ-sweep ablation study
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def render_hypervolume_figure(csv_path, output_path):
    """Render hypervolume vs lambda figure"""
    # Load data
    df = pd.read_csv(csv_path)
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Plot hypervolume vs lambda
    plt.plot(df['lambda'], df['mean_physics_reward'], marker='o', linewidth=2, markersize=8)
    
    # Customize
    plt.xlabel('λ (physics weight)', fontsize=12)
    plt.ylabel('Mean Physics Reward', fontsize=12)
    plt.title('Physics Weight Ablation Study', fontsize=14, fontweight='bold')
    
    # Grid and styling
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Figure saved to: {output_path}")
    
    # Print summary
    print(f"📊 Data points: {len(df)}")
    print(f"📈 Lambda range: {df['lambda'].min()} - {df['lambda'].max()}")
    print(f"📊 Physics reward range: {df['mean_physics_reward'].min():.4f} - {df['mean_physics_reward'].max():.4f}")

def main():
    parser = argparse.ArgumentParser(description='Render hypervolume vs lambda figure')
    parser.add_argument('--csv', required=True, help='Path to lambda_sweep_summary.csv')
    parser.add_argument('--out', required=True, help='Output PNG path')
    
    args = parser.parse_args()
    
    try:
        render_hypervolume_figure(args.csv, args.out)
    except Exception as e:
        print(f"❌ Error rendering figure: {e}")
        exit(1)

if __name__ == "__main__":
    main() 