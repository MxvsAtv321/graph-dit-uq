"""
Reward function registry for multi-objective molecular optimization.
"""

from .ahi import adaptive_hypervolume_reward
from .lpef import latent_pareto_energy_reward
from .sucb_pd import sucb_pareto_dominance_reward

REWARD_REGISTRY = {
    "ahi": adaptive_hypervolume_reward,
    "lpef": latent_pareto_energy_reward,
    "sucb": sucb_pareto_dominance_reward,
}

__all__ = [
    "REWARD_REGISTRY",
    "adaptive_hypervolume_reward",
    "latent_pareto_energy_reward",
    "sucb_pareto_dominance_reward",
]
