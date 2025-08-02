"""
Latent Pareto Energy Function (LPEF) reward function.
Energy-based reward using learned Pareto manifold representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
import warnings


class LatentEnergyModel(nn.Module):
    """
    Simple latent energy model for demonstration.
    In production, this would be a more sophisticated VAE or energy-based model.
    """
    
    def __init__(self, latent_dim: int = 64, n_objectives: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_objectives = n_objectives
        
        # Encoder (SMILES to latent)
        self.encoder = nn.Sequential(
            nn.Linear(2048, 512),  # Assuming ECFP6 fingerprint
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2)  # Mean and logvar
        )
        
        # Energy function
        self.energy_net = nn.Sequential(
            nn.Linear(latent_dim + n_objectives, 256),  # latent + preference
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Objective decoder
        self.objective_decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_objectives)
        )
        
        # Ensemble for uncertainty estimation
        self.ensemble_size = 3
        self.energy_ensemble = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim + n_objectives, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            ) for _ in range(self.ensemble_size)
        ])
    
    def encode(self, smiles: str) -> torch.Tensor:
        """
        Encode SMILES to latent representation.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Latent representation
        """
        # Mock fingerprint for demonstration
        # In production, use RDKit ECFP6
        fingerprint = torch.randn(2048)
        
        # Encode
        encoded = self.encoder(fingerprint)
        mu, logvar = torch.chunk(encoded, 2, dim=-1)
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return z
    
    def energy(self, z: torch.Tensor, preference: torch.Tensor) -> torch.Tensor:
        """
        Compute energy for latent point and preference vector.
        
        Args:
            z: Latent representation
            preference: Preference vector
            
        Returns:
            Energy value
        """
        # Concatenate latent and preference
        input_tensor = torch.cat([z, preference], dim=-1)
        energy = self.energy_net(input_tensor)
        return energy
    
    def log_prob(self, z: torch.Tensor, preference: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability (negative energy).
        
        Args:
            z: Latent representation
            preference: Preference vector
            
        Returns:
            Log probability
        """
        energy = self.energy(z, preference)
        return -energy
    
    def decode_objectives(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to objective values.
        
        Args:
            z: Latent representation
            
        Returns:
            Objective values
        """
        return self.objective_decoder(z)


def latent_pareto_energy_reward(molecule: str, 
                               preference_vector: List[float],
                               energy_model: Optional[LatentEnergyModel] = None,
                               beta: float = 0.5) -> torch.Tensor:
    """
    Latent Pareto Energy Function (LPEF) reward function.
    
    Args:
        molecule: SMILES string of the molecule
        preference_vector: Preference vector for objectives
        energy_model: Latent energy model
        beta: Uncertainty weight
    
    Returns:
        Reward tensor
    """
    try:
        # Create energy model if not provided
        if energy_model is None:
            energy_model = LatentEnergyModel()
        
        # Normalize preference vector
        preference = torch.tensor(preference_vector, dtype=torch.float32)
        preference = F.softmax(preference, dim=0)
        
        # Encode molecule to latent space
        z = energy_model.encode(molecule)
        
        # Compute energy on Pareto manifold
        pareto_energy = -energy_model.log_prob(z, preference)
        
        # Multi-objective alignment term
        objectives = energy_model.decode_objectives(z)
        alignment = torch.dot(objectives, preference)
        
        # Uncertainty from ensemble disagreement
        ensemble_energies = []
        for ensemble_member in energy_model.energy_ensemble:
            input_tensor = torch.cat([z, preference], dim=-1)
            energy = ensemble_member(input_tensor)
            ensemble_energies.append(energy)
        
        ensemble_energies = torch.stack(ensemble_energies)
        uncertainty = torch.std(ensemble_energies)
        
        # Differentiable combination
        reward = torch.sigmoid(alignment - pareto_energy + beta * uncertainty)
        
        # Ensure reward is finite
        if not torch.isfinite(reward):
            reward = torch.tensor(0.5)  # Neutral reward
        
        return reward
        
    except Exception as e:
        warnings.warn(f"LPEF reward computation failed: {e}")
        return torch.tensor(0.5)  # Neutral reward 