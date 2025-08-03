#!/usr/bin/env python3
"""
Render hypervolume vs lambda plot for publication.
Enhanced with color-blind friendly palette and proper formatting.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np

def render_hypervolume_plot(csv_path, output_path):
    """Render publication-ready hypervolume vs lambda plot."""
    
    # Set up publication-quality styling
    plt.style.use('default')
    sns.set_palette("tab10")  # Color-blind friendly
    
    # Read data
    df = pd.read_csv(csv_path)
    
    # Create figure with proper sizing
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=300)
    
    # Plot data
    ax.plot(df['lambda'], df['mean_physics_reward'], 
            marker='o', linewidth=2, markersize=8, 
            color='#1f77b4', label='Physics Reward')
    
    # Add error bars if available
    if 'std_physics_reward' in df.columns:
        ax.errorbar(df['lambda'], df['mean_physics_reward'], 
                   yerr=df['std_physics_reward'], 
                   fmt='none', capsize=5, capthick=2, 
                   color='#1f77b4', alpha=0.7)
    
    # Formatting
    ax.set_xlabel('λ_diffdock', fontsize=14, fontweight='bold')
    ax.set_ylabel('Physics Reward', fontsize=14, fontweight='bold')
    ax.set_title('Hypervolume vs λ_diffdock\nPhysics-ML Integration Performance', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Set proper x-axis ticks
    ax.set_xticks([0.0, 0.2, 0.4, 0.6])
    ax.set_xticklabels(['0.0', '0.2', '0.4', '0.6'], fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add legend
    ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    
    # Add data point annotations
    for i, row in df.iterrows():
        ax.annotate(f"{row['mean_physics_reward']:.3f}", 
                   (row['lambda'], row['mean_physics_reward']),
                   textcoords="offset points", 
                   xytext=(0,10), ha='center', fontsize=10)
    
    # Add shaded confidence interval if multiple runs
    if len(df) > 1:
        # Calculate confidence interval
        y_mean = df['mean_physics_reward'].values
        y_std = df.get('std_physics_reward', pd.Series([0.01] * len(df))).values
        
        ax.fill_between(df['lambda'], 
                       y_mean - y_std, 
                       y_mean + y_std, 
                       alpha=0.2, color='#1f77b4')
    
    # Tight layout
    plt.tight_layout()
    
    # Save with high quality
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"✅ Figure saved to: {output_path}")
    print(f"📊 Data points: {len(df)}")
    print(f"📈 Lambda range: {df['lambda'].min():.1f} - {df['lambda'].max():.1f}")
    print(f"📊 Physics reward range: {df['mean_physics_reward'].min():.4f} - {df['mean_physics_reward'].max():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render hypervolume vs lambda plot")
    parser.add_argument("--csv", required=True, help="Input CSV file path")
    parser.add_argument("--out", required=True, help="Output PNG file path")
    
    args = parser.parse_args()
    render_hypervolume_plot(args.csv, args.out) 