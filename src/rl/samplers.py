"""
Preference samplers for multi-objective optimization.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Optional
import warnings


class MLPSPreferenceSampler:
    """
    MLPS preference vector generator with exploration/exploitation balance.
    """
    
    def __init__(self, n_objectives: int = 3, warmup_samples: int = 100):
        self.n_objectives = n_objectives
        self.warmup_samples = warmup_samples
        self.iteration = 0
        self.pareto_coverage = np.zeros((10, 10, 10))  # Discretized coverage
        
    def sample_preference(self, mode: str = 'adaptive') -> np.ndarray:
        """
        Generate preference vector with curriculum learning support.
        
        Args:
            mode: Sampling mode ('adaptive', 'curriculum', 'uniform')
            
        Returns:
            Preference vector (normalized)
        """
        self.iteration += 1
        
        if self.iteration < self.warmup_samples:
            # Uniform sampling during warmup
            pref = np.random.dirichlet(np.ones(self.n_objectives))
        else:
            if mode == 'adaptive':
                # Thompson sampling over Pareto regions
                uncertainty = 1.0 / (1.0 + self.pareto_coverage)
                region_probs = uncertainty.flatten() / uncertainty.sum()
                selected_region = np.random.choice(len(region_probs), p=region_probs)
                
                # Convert region to preference center
                i, j, k = np.unravel_index(selected_region, self.pareto_coverage.shape)
                center = np.array([i, j, k]) / 9.0  # Normalize to [0,1]
                
                # Add noise for exploration
                noise_scale = 0.1 * np.exp(-self.iteration / 1000)  # Decay
                pref = np.random.dirichlet(center * 10 + noise_scale)
                
            elif mode == 'curriculum':
                # Progressive difficulty: start with single objectives
                stage = min(self.iteration // 500, self.n_objectives - 1)
                weights = np.zeros(self.n_objectives)
                active_objectives = np.random.choice(self.n_objectives, stage + 1, replace=False)
                weights[active_objectives] = 1.0
                pref = weights / weights.sum()
                
            elif mode == 'uniform':
                # Uniform sampling throughout
                pref = np.random.dirichlet(np.ones(self.n_objectives))
                
            else:
                raise ValueError(f"Unknown sampling mode: {mode}")
        
        # Ensure normalization
        pref = pref / np.sum(pref)
        return pref
    
    def update_coverage(self, achieved_objectives: np.ndarray):
        """
        Update Pareto coverage for adaptive sampling.
        
        Args:
            achieved_objectives: Achieved objective values [qed, docking, sa]
        """
        # Discretize achieved objectives to grid
        indices = (achieved_objectives * 9).astype(int).clip(0, 9)
        if len(indices) == 3:  # Ensure we have 3 objectives
            self.pareto_coverage[tuple(indices)] += 1
    
    def get_coverage_stats(self) -> Dict[str, Any]:
        """
        Get coverage statistics.
        
        Returns:
            Coverage statistics dictionary
        """
        total_coverage = np.sum(self.pareto_coverage)
        max_coverage = np.max(self.pareto_coverage)
        min_coverage = np.min(self.pareto_coverage)
        
        return {
            'total_coverage': total_coverage,
            'max_coverage': max_coverage,
            'min_coverage': min_coverage,
            'coverage_ratio': min_coverage / max_coverage if max_coverage > 0 else 0.0,
            'iteration': self.iteration
        }


class PreferenceSampler:
    """
    Wrapper class for preference sampling with replay buffer integration.
    """
    
    def __init__(self, n_objectives: int = 3, warmup_samples: int = 100, mode: str = 'adaptive'):
        self.sampler = MLPSPreferenceSampler(n_objectives, warmup_samples)
        self.mode = mode
        self.replay_buffer = []
        
    def sample(self) -> np.ndarray:
        """
        Sample a preference vector.
        
        Returns:
            Preference vector
        """
        return self.sampler.sample_preference(self.mode)
    
    def update(self, achieved_objectives: np.ndarray, metadata: Optional[Dict[str, Any]] = None):
        """
        Update sampler with achieved objectives.
        
        Args:
            achieved_objectives: Achieved objective values
            metadata: Additional metadata for replay buffer
        """
        self.sampler.update_coverage(achieved_objectives)
        
        # Store in replay buffer
        replay_entry = {
            'objectives': achieved_objectives,
            'iteration': self.sampler.iteration,
            'mode': self.mode
        }
        if metadata:
            replay_entry.update(metadata)
        
        self.replay_buffer.append(replay_entry)
        
        # Keep only last 1000 entries
        if len(self.replay_buffer) > 1000:
            self.replay_buffer = self.replay_buffer[-1000:]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get sampler statistics.
        
        Returns:
            Statistics dictionary
        """
        coverage_stats = self.sampler.get_coverage_stats()
        coverage_stats['replay_buffer_size'] = len(self.replay_buffer)
        coverage_stats['mode'] = self.mode
        return coverage_stats 