#!/usr/bin/env python3
"""Run ablation study on uncertainty-guided RL."""

import subprocess
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def run_ablation_study():
    """Run complete ablation study on uncertainty guidance"""
    
    experiments = [
        {
            'name': 'baseline_no_rl',
            'description': 'Pure Graph DiT without RL',
            'command': None  # Use existing results
        },
        {
            'name': 'rl_without_uncertainty',
            'description': 'RL fine-tuning without uncertainty bonus',
            'command': [
                'python', 'scripts/train_rl_with_uncertainty.py',
                '--n_iterations', '20',
                '--lambda_qed', '0.3',
                '--lambda_docking', '0.5',
                '--lambda_sa', '0.2'
            ]
        },
        {
            'name': 'rl_with_uncertainty_low',
            'description': 'RL with low uncertainty bonus (0.05)',
            'command': [
                'python', 'scripts/train_rl_with_uncertainty.py',
                '--use_uncertainty',
                '--uncertainty_bonus', '0.05',
                '--n_iterations', '20'
            ]
        },
        {
            'name': 'rl_with_uncertainty_medium',
            'description': 'RL with medium uncertainty bonus (0.1)',
            'command': [
                'python', 'scripts/train_rl_with_uncertainty.py',
                '--use_uncertainty',
                '--uncertainty_bonus', '0.1',
                '--n_iterations', '20'
            ]
        },
        {
            'name': 'rl_with_uncertainty_high',
            'description': 'RL with high uncertainty bonus (0.2)',
            'command': [
                'python', 'scripts/train_rl_with_uncertainty.py',
                '--use_uncertainty',
                '--uncertainty_bonus', '0.2',
                '--n_iterations', '20'
            ]
        }
    ]
    
    results = {}
    
    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"Running: {exp['name']}")
        print(f"Description: {exp['description']}")
        print('='*60)
        
        if exp['command']:
            # Run experiment
            subprocess.run(exp['command'])
            
            # Load results
            result_files = list(Path('outputs').glob(f"rl_results_*{exp['name']}*.json"))
            if result_files:
                with open(result_files[-1], 'r') as f:
                    results[exp['name']] = json.load(f)
        else:
            # Use baseline results
            results[exp['name']] = {
                'final_evaluation': {
                    'pareto_percentage': 0.0003,  # 0.03%
                    'validity_rate': 1.0,
                    'mean_qed': 0.434,
                    'mean_docking': -8.468
                }
            }
    
    # Create comparison plot
    create_ablation_comparison_plot(results)
    
    # Print summary table
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)
    print(f"{'Experiment':<30} {'Pareto %':<10} {'Improvement':<15} {'Validity':<10}")
    print("-"*80)
    
    baseline_pareto = results['baseline_no_rl']['final_evaluation']['pareto_percentage']
    
    for name, result in results.items():
        pareto = result['final_evaluation']['pareto_percentage']
        improvement = pareto / baseline_pareto
        validity = result['final_evaluation']['validity_rate']
        
        print(f"{name:<30} {pareto*100:>8.3f}% {improvement:>12.1f}x {validity:>9.1%}")
    
    return results

def create_ablation_comparison_plot(results):
    """Create comparison plot for ablation study"""
    
    # Extract data
    experiments = list(results.keys())
    pareto_coverage = [results[e]['final_evaluation']['pareto_percentage'] * 100 for e in experiments]
    validity_rates = [results[e]['final_evaluation']['validity_rate'] * 100 for e in experiments]
    mean_qed = [results[e]['final_evaluation']['mean_qed'] for e in experiments]
    mean_docking = [results[e]['final_evaluation']['mean_docking'] for e in experiments]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel A: Pareto coverage comparison
    ax = axes[0, 0]
    colors = ['gray', 'blue', 'green', 'orange', 'red']
    bars = ax.bar(range(len(experiments)), pareto_coverage, color=colors, edgecolor='black')
    ax.set_xticks(range(len(experiments)))
    ax.set_xticklabels([e.replace('_', '\n') for e in experiments], rotation=45, ha='right')
    ax.set_ylabel('Pareto Coverage (%)', fontsize=12)
    ax.set_title('Impact of Uncertainty on Pareto Coverage', fontsize=14)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}%', ha='center', va='bottom')
    
    # Panel B: Validity rates
    ax = axes[0, 1]
    bars = ax.bar(range(len(experiments)), validity_rates, color=colors, edgecolor='black')
    ax.set_xticks(range(len(experiments)))
    ax.set_xticklabels([e.replace('_', '\n') for e in experiments], rotation=45, ha='right')
    ax.set_ylabel('Validity Rate (%)', fontsize=12)
    ax.set_title('Validity Rate Comparison', fontsize=14)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # Panel C: Multi-objective trade-offs
    ax = axes[1, 0]
    scatter_colors = ['gray', 'blue', 'green', 'orange', 'red']
    for i, (name, color) in enumerate(zip(experiments, scatter_colors)):
        ax.scatter(mean_docking[i], mean_qed[i], s=200, c=color, 
                  edgecolors='black', linewidth=2, alpha=0.7, label=name.replace('_', ' '))
    
    ax.set_xlabel('Mean Docking Score (kcal/mol)', fontsize=12)
    ax.set_ylabel('Mean QED Score', fontsize=12)
    ax.set_title('Multi-Objective Performance', fontsize=14)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Panel D: Improvement factors
    ax = axes[1, 1]
    baseline = pareto_coverage[0]
    improvements = [p / baseline for p in pareto_coverage]
    
    bars = ax.bar(range(len(experiments)), improvements, 
                   color=colors, edgecolor='black')
    ax.set_xticks(range(len(experiments)))
    ax.set_xticklabels([e.replace('_', '\n') for e in experiments], rotation=45, ha='right')
    ax.set_ylabel('Improvement Factor', fontsize=12)
    ax.set_title('Improvement Over Baseline', fontsize=14)
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}x', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save figure
    fig.savefig('figures/publication/figure_4_ablation_study.pdf', dpi=300, bbox_inches='tight')
    fig.savefig('figures/publication/figure_4_ablation_study.png', dpi=300, bbox_inches='tight')
    
    print("\n✅ Ablation study plot saved to figures/publication/")
    
    return fig

if __name__ == "__main__":
    # Create figures directory
    Path('figures/publication').mkdir(parents=True, exist_ok=True)
    
    # Run ablation study
    results = run_ablation_study()
    
    print("\n✅ Ablation study complete!") 