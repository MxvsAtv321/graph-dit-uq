#!/usr/bin/env python3
"""
Render pose confidence vs lambda plot for publication.
Enhanced with color-blind friendly palette and proper formatting.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np

def render_pose_confidence_plot(csv_path, output_path):
    """Render publication-ready pose confidence vs lambda plot."""
    
    # Set up publication-quality styling
    plt.style.use('default')
    sns.set_palette("tab10")  # Color-blind friendly
    
    # Read data
    df = pd.read_csv(csv_path)
    
    # Convert pose confidence to percentage
    df['pose_conf_pct'] = df['pose_conf>0.6'] * 100
    
    # Create figure with proper sizing
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=300)
    
    # Plot data
    ax.plot(df['lambda'], df['pose_conf_pct'], 
            marker='s', linewidth=2, markersize=8, 
            color='#ff7f0e', label='Pose Confidence > 0.6')
    
    # Add error bars if available
    if 'std_pose_conf' in df.columns:
        ax.errorbar(df['lambda'], df['pose_conf_pct'], 
                   yerr=df['std_pose_conf'] * 100, 
                   fmt='none', capsize=5, capthick=2, 
                   color='#ff7f0e', alpha=0.7)
    
    # Formatting
    ax.set_xlabel('λ_diffdock', fontsize=14, fontweight='bold')
    ax.set_ylabel('Pose Confidence > 0.6 (%)', fontsize=14, fontweight='bold')
    ax.set_title('Pose Confidence vs λ_diffdock\nDiffDock-L Quality Metrics', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Set proper x-axis ticks
    ax.set_xticks([0.0, 0.2, 0.4, 0.6])
    ax.set_xticklabels(['0.0', '0.2', '0.4', '0.6'], fontsize=12)
    
    # Set y-axis limits and ticks
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add legend with annotation
    legend_text = f"Pose Confidence > 0.6\n(Optimal: {df['pose_conf_pct'].max():.1f}%)"
    ax.legend([legend_text], fontsize=12, frameon=True, fancybox=True, shadow=True)
    
    # Add data point annotations
    for i, row in df.iterrows():
        ax.annotate(f"{row['pose_conf_pct']:.1f}%", 
                   (row['lambda'], row['pose_conf_pct']),
                   textcoords="offset points", 
                   xytext=(0,10), ha='center', fontsize=10)
    
    # Add shaded confidence interval if multiple runs
    if len(df) > 1:
        # Calculate confidence interval
        y_mean = df['pose_conf_pct'].values
        y_std = df.get('std_pose_conf', pd.Series([0.01] * len(df))).values * 100
        
        ax.fill_between(df['lambda'], 
                       y_mean - y_std, 
                       y_mean + y_std, 
                       alpha=0.2, color='#ff7f0e')
    
    # Add threshold line at 60%
    ax.axhline(y=60, color='red', linestyle='--', alpha=0.7, 
               label='60% Threshold')
    
    # Tight layout
    plt.tight_layout()
    
    # Save with high quality
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"✅ Figure saved to: {output_path}")
    print(f"📊 Data points: {len(df)}")
    print(f"📈 Lambda range: {df['lambda'].min():.1f} - {df['lambda'].max():.1f}")
    print(f"📊 Pose confidence range: {df['pose_conf_pct'].min():.1f}% - {df['pose_conf_pct'].max():.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render pose confidence vs lambda plot")
    parser.add_argument("--csv", required=True, help="Input CSV file path")
    parser.add_argument("--out", required=True, help="Output PNG file path")
    
    args = parser.parse_args()
    render_pose_confidence_plot(args.csv, args.out) 