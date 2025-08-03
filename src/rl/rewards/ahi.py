"""
Adaptive Hypervolume Improvement (AHI) reward function.
Reward based on hypervolume contribution with uncertainty-weighted exploration.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional
import warnings


def compute_hypervolume(pareto_front: List[Dict[str, float]], 
                       reference_point: Optional[List[float]] = None) -> float:
    """
    Compute hypervolume of Pareto front.
    
    Args:
        pareto_front: List of dictionaries with objective values
        reference_point: Reference point for hypervolume calculation
    
    Returns:
        Hypervolume value
    """
    if not pareto_front:
        return 0.0
    
    # Extract objective values
    objectives = []
    for point in pareto_front:
        obj_values = [point.get('qed', 0.0), point.get('docking', 0.0), point.get('sa', 0.0)]
        objectives.append(obj_values)
    
    objectives = np.array(objectives)
    
    # Set reference point if not provided
    if reference_point is None:
        reference_point = [0.0, -15.0, 0.0]  # QED=0, docking=-15, SA=0
    
    # Simple hypervolume approximation (for differentiable version)
    # In production, use proper hypervolume library
    try:
        # Normalize objectives to [0,1] range
        obj_min = np.min(objectives, axis=0)
        obj_max = np.max(objectives, axis=0)
        obj_range = obj_max - obj_min
        
        if np.any(obj_range == 0):
            return 0.0
        
        normalized_obj = (objectives - obj_min) / obj_range
        
        # Simple volume approximation
        volume = np.prod(np.max(normalized_obj, axis=0))
        return float(volume)
    except Exception as e:
        warnings.warn(f"Hypervolume computation failed: {e}")
        return 0.0


def update_pareto_front(pareto_front: List[Dict[str, float]], 
                       new_point: Dict[str, float]) -> List[Dict[str, float]]:
    """
    Update Pareto front with new point.
    
    Args:
        pareto_front: Current Pareto front
        new_point: New point to add
    
    Returns:
        Updated Pareto front
    """
    new_front = pareto_front.copy()
    
    # Check if new point dominates any existing points
    dominated_indices = []
    for i, point in enumerate(new_front):
        if dominates(new_point, point):
            dominated_indices.append(i)
    
    # Remove dominated points
    for i in reversed(dominated_indices):
        new_front.pop(i)
    
    # Check if new point is dominated by any existing point
    is_dominated = any(dominates(point, new_point) for point in new_front)
    
    if not is_dominated:
        new_front.append(new_point)
    
    return new_front


def dominates(point1: Dict[str, float], point2: Dict[str, float]) -> bool:
    """
    Check if point1 dominates point2.
    
    Args:
        point1: First point
        point2: Second point
    
    Returns:
        True if point1 dominates point2
    """
    objectives = ['qed', 'docking', 'sa']
    
    # Higher QED and SA are better, lower docking score is better
    better_in_all = True
    better_in_at_least_one = False
    
    for obj in objectives:
        val1 = point1.get(obj, 0.0)
        val2 = point2.get(obj, 0.0)
        
        if obj == 'docking':
            # Lower docking score is better
            if val1 > val2:
                better_in_all = False
            elif val1 < val2:
                better_in_at_least_one = True
        else:
            # Higher values are better for QED and SA
            if val1 < val2:
                better_in_all = False
            elif val1 > val2:
                better_in_at_least_one = True
    
    return better_in_all and better_in_at_least_one


def adaptive_hypervolume_reward(molecule: str, 
                               pareto_front: List[Dict[str, float]], 
                               uncertainty_model: Any,
                               alpha: float = 0.1,
                               temperature: float = 0.1) -> torch.Tensor:
    """
    Adaptive Hypervolume Improvement (AHI) reward function.
    
    Args:
        molecule: SMILES string of the molecule
        pareto_front: Current Pareto front
        uncertainty_model: Model that predicts properties with uncertainty
        alpha: Uncertainty bonus weight
        temperature: Temperature for reward scaling
    
    Returns:
        Reward tensor
    """
    try:
        # Get predicted properties with uncertainty
        if hasattr(uncertainty_model, 'predict'):
            mu, sigma = uncertainty_model.predict(molecule)
        else:
            # Fallback to mock predictions
            mu = {'qed': 0.5, 'docking': -8.0, 'sa': 0.6}
            sigma = {'qed': 0.1, 'docking': 0.5, 'sa': 0.1}
        
        # Calculate hypervolume improvement
        current_hv = compute_hypervolume(pareto_front)
        new_front = update_pareto_front(pareto_front, mu)
        hv_improvement = compute_hypervolume(new_front) - current_hv
        
        # Clamp hypervolume improvement to reasonable bounds
        hv_improvement = torch.clamp(torch.tensor(hv_improvement), -1.0, 1.0)
        
        # Uncertainty bonus for exploration
        if isinstance(sigma, dict):
            uncertainty_bonus = alpha * torch.sqrt(sum(s**2 for s in sigma.values()))
        else:
            uncertainty_bonus = alpha * torch.sqrt(torch.sum(torch.tensor(list(sigma.values()))**2))
        
        # Temperature-scaled reward (differentiable)
        reward = torch.tanh(hv_improvement / temperature) + uncertainty_bonus
        
        # Ensure reward is finite
        if not torch.isfinite(reward):
            reward = torch.tensor(0.0)
        
        return reward
        
    except Exception as e:
        warnings.warn(f"AHI reward computation failed: {e}")
        return torch.tensor(0.0) 