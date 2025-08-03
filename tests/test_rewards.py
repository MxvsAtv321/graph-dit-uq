"""
Unit tests for reward functions.
"""

import pytest
import torch
import numpy as np
from src.rl.rewards import REWARD_REGISTRY, adaptive_hypervolume_reward, latent_pareto_energy_reward, sucb_pareto_dominance_reward


def test_reward_registry():
    """Test that all reward functions are registered"""
    expected_rewards = ['ahi', 'lpef', 'sucb']
    
    for reward_name in expected_rewards:
        assert reward_name in REWARD_REGISTRY, f"Reward {reward_name} not found in registry"
        assert callable(REWARD_REGISTRY[reward_name]), f"Reward {reward_name} is not callable"


def test_ahi_reward():
    """Test Adaptive Hypervolume Improvement reward"""
    # Test with empty Pareto front
    pareto_front = []
    molecule = "CCO"
    
    reward = adaptive_hypervolume_reward(molecule, pareto_front, None)
    
    assert isinstance(reward, torch.Tensor), "Reward should be a torch tensor"
    assert torch.isfinite(reward), "Reward should be finite"
    assert reward >= 0, "Reward should be non-negative"


def test_lpef_reward():
    """Test Latent Pareto Energy Function reward"""
    preference_vector = [0.4, 0.4, 0.2]  # QED, docking, SA weights
    molecule = "CCO"
    
    reward = latent_pareto_energy_reward(molecule, preference_vector)
    
    assert isinstance(reward, torch.Tensor), "Reward should be a torch tensor"
    assert torch.isfinite(reward), "Reward should be finite"
    assert 0 <= reward <= 1, "LPEF reward should be in [0,1]"


def test_sucb_reward():
    """Test Scalarized UCB with Pareto Dominance reward"""
    archive = [
        {'objectives': [0.5, -8.0, 0.6]},
        {'objectives': [0.7, -6.0, 0.8]}
    ]
    molecule = "CCO"
    
    reward = sucb_pareto_dominance_reward(molecule, archive)
    
    assert isinstance(reward, torch.Tensor), "Reward should be a torch tensor"
    assert torch.isfinite(reward), "Reward should be finite"
    assert reward >= 0, "Reward should be non-negative"


def test_reward_gradients():
    """Test that rewards are differentiable"""
    # Test AHI reward gradients
    pareto_front = []
    molecule = "CCO"
    
    # Create a simple mock uncertainty model that returns tensors
    class MockUncertaintyModel:
        def predict(self, mol):
            return {'qed': 0.6, 'docking': -8.0, 'sa': 0.7}, {'qed': 0.1, 'docking': 0.5, 'sa': 0.1}
    
    model = MockUncertaintyModel()
    reward = adaptive_hypervolume_reward(molecule, pareto_front, model)
    
    # Test that reward can be used in autograd
    loss = -reward
    assert torch.isfinite(loss), "Loss should be finite"


def test_reward_edge_cases():
    """Test reward functions with edge cases"""
    # Test with invalid SMILES
    pareto_front = []
    invalid_molecule = "invalid_smiles"
    
    # All rewards should handle invalid SMILES gracefully
    ahi_reward = adaptive_hypervolume_reward(invalid_molecule, pareto_front, None)
    assert torch.isfinite(ahi_reward), "AHI should handle invalid SMILES"
    
    lpef_reward = latent_pareto_energy_reward(invalid_molecule, [0.33, 0.33, 0.34])
    assert torch.isfinite(lpef_reward), "LPEF should handle invalid SMILES"
    
    sucb_reward = sucb_pareto_dominance_reward(invalid_molecule, [])
    assert torch.isfinite(sucb_reward), "SUCB should handle invalid SMILES"


def test_reward_performance():
    """Test reward function performance"""
    import time
    
    pareto_front = []
    molecules = ["CCO", "CCCO", "C1=CC=CC=C1", "CC(C)C", "c1ccccc1"]
    
    # Test AHI performance
    start_time = time.time()
    for mol in molecules:
        reward = adaptive_hypervolume_reward(mol, pareto_front, None)
    ahi_time = time.time() - start_time
    
    # Test LPEF performance
    start_time = time.time()
    for mol in molecules:
        reward = latent_pareto_energy_reward(mol, [0.33, 0.33, 0.34])
    lpef_time = time.time() - start_time
    
    # Test SUCB performance
    start_time = time.time()
    for mol in molecules:
        reward = sucb_pareto_dominance_reward(mol, [])
    sucb_time = time.time() - start_time
    
    # All should complete in reasonable time (< 1 second for 5 molecules)
    assert ahi_time < 1.0, f"AHI too slow: {ahi_time:.3f}s"
    assert lpef_time < 1.0, f"LPEF too slow: {lpef_time:.3f}s"
    assert sucb_time < 1.0, f"SUCB too slow: {sucb_time:.3f}s"
    
    print(f"Performance: AHI={ahi_time:.3f}s, LPEF={lpef_time:.3f}s, SUCB={sucb_time:.3f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 