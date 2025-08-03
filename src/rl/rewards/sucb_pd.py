"""
Scalarized UCB with Pareto Dominance (SUCB-PD) reward function.
Upper confidence bound with Pareto dominance detection.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional
import warnings


def predict_with_uncertainty(molecule: str, objective: str) -> tuple:
    """
    Predict objective value with uncertainty.
    
    Args:
        molecule: SMILES string
        objective: Objective name ('qed', 'docking', 'sa')
    
    Returns:
        Tuple of (mean, uncertainty)
    """
    # Mock predictions for demonstration
    # In production, this would call the actual prediction models
    
    if objective == 'qed':
        return 0.6, 0.1  # QED prediction with uncertainty
    elif objective == 'docking':
        return -8.5, 0.5  # Docking score with uncertainty
    elif objective == 'sa':
        return 0.7, 0.15  # SA score with uncertainty
    else:
        return 0.5, 0.2  # Default


def dominates(objectives1: List[float], objectives2: List[float]) -> bool:
    """
    Check if objectives1 dominates objectives2.
    
    Args:
        objectives1: First set of objectives [qed, docking, sa]
        objectives2: Second set of objectives [qed, docking, sa]
    
    Returns:
        True if objectives1 dominates objectives2
    """
    if len(objectives1) != 3 or len(objectives2) != 3:
        return False
    
    qed1, docking1, sa1 = objectives1
    qed2, docking2, sa2 = objectives2
    
    # Higher QED and SA are better, lower docking score is better
    better_in_all = True
    better_in_at_least_one = False
    
    # QED (higher is better)
    if qed1 < qed2:
        better_in_all = False
    elif qed1 > qed2:
        better_in_at_least_one = True
    
    # Docking (lower is better)
    if docking1 > docking2:
        better_in_all = False
    elif docking1 < docking2:
        better_in_at_least_one = True
    
    # SA (higher is better)
    if sa1 < sa2:
        better_in_all = False
    elif sa1 > sa2:
        better_in_at_least_one = True
    
    return better_in_all and better_in_at_least_one


def sucb_pareto_dominance_reward(molecule: str, 
                                archive: List[Dict[str, Any]],
                                ucb_weights: Optional[List[float]] = None,
                                kappa: float = 2.0) -> torch.Tensor:
    """
    Scalarized UCB with Pareto Dominance (SUCB-PD) reward function.
    
    Args:
        molecule: SMILES string of the molecule
        archive: Archive of previously evaluated molecules
        ucb_weights: Weights for UCB scalarization
        kappa: UCB exploration parameter
    
    Returns:
        Reward tensor
    """
    try:
        # Default weights if not provided
        if ucb_weights is None:
            ucb_weights = [0.4, 0.4, 0.2]  # QED, docking, SA weights
        
        # Normalize weights
        ucb_weights = np.array(ucb_weights)
        ucb_weights = ucb_weights / np.sum(ucb_weights)
        
        # Get predictions with uncertainty for all objectives
        mu_vec, sigma_vec = [], []
        objectives = ['qed', 'docking', 'sa']
        
        for objective in objectives:
            mu, sigma = predict_with_uncertainty(molecule, objective)
            mu_vec.append(mu)
            sigma_vec.append(sigma)
        
        mu_vec = np.array(mu_vec)
        sigma_vec = np.array(sigma_vec)
        
        # Scalarized UCB
        weighted_mean = np.dot(ucb_weights, mu_vec)
        weighted_ucb = weighted_mean + kappa * np.sqrt(np.dot(ucb_weights**2, sigma_vec**2))
        
        # Pareto dominance bonus
        dominance_count = 0
        if archive:
            for mol_data in archive:
                if 'objectives' in mol_data:
                    if dominates(mu_vec, mol_data['objectives']):
                        dominance_count += 1
        
        # Log-scaled dominance bonus
        dominance_bonus = np.log1p(dominance_count)
        
        # Smooth combination using softplus
        reward = torch.nn.functional.softplus(torch.tensor(weighted_ucb + dominance_bonus))
        
        # Ensure reward is finite and reasonable
        if not torch.isfinite(reward):
            reward = torch.tensor(0.0)
        
        # Cap reward to prevent extreme values
        reward = torch.clamp(reward, 0.0, 10.0)
        
        return reward
        
    except Exception as e:
        warnings.warn(f"SUCB-PD reward computation failed: {e}")
        return torch.tensor(0.0) 