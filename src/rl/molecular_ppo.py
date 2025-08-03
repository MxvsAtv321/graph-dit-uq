#!/usr/bin/env python3
"""PPO for uncertainty-guided molecular optimization."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import logging

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
import random

logger = logging.getLogger(__name__)

# Set random seeds for reproducibility


def set_seeds(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Set seeds globally
set_seeds(42)


class MolecularPPO:
    """PPO for uncertainty-guided molecular optimization"""

    def __init__(self, generator, config):
        self.generator = generator
        self.config = config

        # PPO hyperparameters
        self.lr = config.get("lr", 3e-4)
        self.epsilon = config.get("epsilon", 0.2)
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.value_loss_coef = config.get("value_loss_coef", 0.5)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)

        # Uncertainty parameters
        self.uncertainty_bonus_weight = config.get("uncertainty_bonus", 0.1)
        self.uncertainty_threshold = config.get("uncertainty_threshold", 0.1)

        # Multi-objective weights
        self.lambda_qed = config.get("lambda_qed", 0.3)
        self.lambda_docking = config.get("lambda_docking", 0.5)
        self.lambda_sa = config.get("lambda_sa", 0.2)

        # Value network
        self.value_net = self._build_value_network()
        self.optimizer = torch.optim.Adam(
            list(self.generator.parameters()) + list(self.value_net.parameters()),
            lr=self.lr,
        )

        # Tracking
        self.episode_rewards = deque(maxlen=100)
        self.pareto_molecules = []

    def _build_value_network(self):
        """Build value function approximator"""
        hidden_dim = getattr(self.generator, "hidden_dim", 256)
        return nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def compute_multi_objective_reward(self, molecule_data):
        """Compute reward balancing multiple objectives"""
        # Extract properties
        qed = molecule_data["qed"]
        docking = molecule_data["docking_score"]
        sa = molecule_data["sa_score"]
        uncertainty = molecule_data.get("uncertainty", 0.0)

        # Normalize to [0, 1] range
        qed_norm = np.clip(qed, 0, 1)  # Already in [0, 1]
        docking_norm = np.clip((-docking - 5) / 10, 0, 1)  # -15 to -5 → 0 to 1
        sa_norm = np.clip((5 - sa) / 4, 0, 1)  # 5 to 1 → 0 to 1

        # Base multi-objective reward
        base_reward = (
            self.lambda_qed * qed_norm
            + self.lambda_docking * docking_norm
            + self.lambda_sa * sa_norm
        )

        # Uncertainty exploration bonus
        if uncertainty > self.uncertainty_threshold:
            exploration_bonus = self.uncertainty_bonus_weight * np.sqrt(uncertainty)
        else:
            exploration_bonus = 0.0

        # Pareto bonus - extra reward if molecule is Pareto optimal
        is_pareto = self._check_pareto_optimal(molecule_data)
        pareto_bonus = 0.5 if is_pareto else 0.0

        total_reward = base_reward + exploration_bonus + pareto_bonus

        # Track if Pareto optimal
        if is_pareto:
            self.pareto_molecules.append(molecule_data)

        return total_reward, {
            "base_reward": base_reward,
            "exploration_bonus": exploration_bonus,
            "pareto_bonus": pareto_bonus,
            "qed_norm": qed_norm,
            "docking_norm": docking_norm,
            "sa_norm": sa_norm,
        }

    def _check_pareto_optimal(self, molecule):
        """Check if molecule is Pareto optimal among current batch"""
        # Simple check - can be optimized later
        for other in self.pareto_molecules[-100:]:  # Check last 100
            if (
                other["qed"] >= molecule["qed"]
                and other["docking_score"] <= molecule["docking_score"]
                and other["sa_score"] <= molecule["sa_score"]
            ):
                if (
                    other["qed"] > molecule["qed"]
                    or other["docking_score"] < molecule["docking_score"]
                    or other["sa_score"] < molecule["sa_score"]
                ):
                    return False
        return True

    def collect_trajectories(self, n_steps=2048):
        """Collect trajectories for PPO update"""
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []
        infos = []

        for _ in range(n_steps):
            # Generate molecule
            with torch.no_grad():
                state = self.generator.get_state()
                action, log_prob = self.generator.sample_action()
                value = self.value_net(state)

            # Execute action (generate molecule)
            molecule_smiles = self.generator.decode_action(action)

            # Compute properties and reward
            molecule_data = self.evaluate_molecule(molecule_smiles)
            reward, reward_info = self.compute_multi_objective_reward(molecule_data)

            # Store trajectory
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            dones.append(False)  # No episode termination in molecular generation
            infos.append(reward_info)

        # Convert to tensors
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        values = torch.stack(values).squeeze()
        log_probs = torch.stack(log_probs)
        dones = torch.tensor(dones, dtype=torch.float32)

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "values": values,
            "log_probs": log_probs,
            "dones": dones,
            "infos": infos,
        }

    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_advantage = (
                delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            )

        returns = advantages + values
        return advantages, returns

    def update(self, trajectories, n_epochs=4):
        """PPO update with clipped objective"""
        states = trajectories["states"]
        actions = trajectories["actions"]
        old_log_probs = trajectories["log_probs"]

        # Compute advantages
        advantages, returns = self.compute_gae(
            trajectories["rewards"], trajectories["values"], trajectories["dones"]
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO epochs
        for epoch in range(n_epochs):
            # Shuffle data
            indices = torch.randperm(len(states))

            for start in range(0, len(states), self.config["batch_size"]):
                end = start + self.config["batch_size"]
                batch_indices = indices[start:end]

                # Get batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Forward pass
                new_log_probs = self.generator.get_log_prob(batch_states, batch_actions)
                entropy = self.generator.get_entropy(batch_states)
                values = self.value_net(batch_states).squeeze()

                # PPO loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
                    * batch_advantages
                )

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, batch_returns)
                entropy_loss = -entropy.mean()

                total_loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # Update
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.generator.parameters())
                    + list(self.value_net.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                # Log to wandb
                if WANDB_AVAILABLE and wandb.run:
                    wandb.log(
                        {
                            "ppo/policy_loss": policy_loss.item(),
                            "ppo/value_loss": value_loss.item(),
                            "ppo/entropy_loss": entropy_loss.item(),
                            "ppo/total_loss": total_loss.item(),
                            "ppo/mean_ratio": ratio.mean().item(),
                        }
                    )

    def evaluate_molecule(self, smiles):
        """Evaluate molecule properties including uncertainty"""
        try:
            from rdkit import Chem
            from rdkit.Chem import QED, Crippen

            RDKIT_AVAILABLE = True
        except ImportError:
            RDKIT_AVAILABLE = False

        if not RDKIT_AVAILABLE:
            # Mock evaluation for testing
            return {
                "smiles": smiles,
                "valid": True,
                "qed": np.random.uniform(0.3, 0.9),
                "sa_score": np.random.uniform(1.0, 5.0),
                "docking_score": -8.5 + 2.0 * np.random.randn(),
                "uncertainty": np.random.uniform(0.01, 0.2),
            }

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                "smiles": smiles,
                "valid": False,
                "qed": 0.0,
                "sa_score": 10.0,
                "docking_score": 0.0,
                "uncertainty": 1.0,
            }

        # Compute properties
        qed = QED.qed(mol)
        Crippen.MolLogP(mol)

        # SA calculation (simplified)
        num_atoms = mol.GetNumHeavyAtoms()
        sa_score = 1.0 - (num_atoms - 15) / 25  # Normalize around 15 atoms
        sa_score = max(0.1, min(1.0, sa_score))

        # Mock docking score
        docking_score = -8.5 + 2.0 * np.random.randn()

        # Uncertainty via MC-dropout
        uncertainty = self._compute_uncertainty(smiles)

        return {
            "smiles": smiles,
            "valid": True,
            "qed": qed,
            "sa_score": sa_score,
            "docking_score": docking_score,
            "uncertainty": uncertainty,
        }

    def _compute_uncertainty(self, smiles, n_samples=5):
        """Compute epistemic uncertainty using MC-dropout"""
        # Simplified uncertainty computation
        # In practice, would use MC-dropout on the generator
        return np.random.uniform(0.01, 0.2)  # Mock uncertainty


class RLGraphDiT(nn.Module):
    """Graph DiT wrapped for RL training"""

    def __init__(self, checkpoint_path, device="cuda"):
        super().__init__()

        # Load pretrained Graph DiT
        from src.models.baselines.graph_dit import GraphDiTWrapper

        self.base_model = GraphDiTWrapper.load_from_checkpoint(checkpoint_path)
        self.device = device

        # RL-specific components
        self.hidden_dim = 256  # Default hidden dimension
        self.vocab_size = 100  # Mock vocabulary size

        # State representation
        self.current_state = None

    def get_state(self):
        """Get current state representation"""
        if self.current_state is None:
            # Initialize with start token
            self.current_state = torch.zeros(1, self.hidden_dim).to(self.device)
        return self.current_state

    def sample_action(self):
        """Sample action (next token) from policy"""
        self.get_state()

        # Mock action sampling - use smaller range for valid molecules
        action_logits = torch.randn(1, 100).to(self.device)  # 100 possible actions
        dist = torch.distributions.Categorical(logits=action_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob

    def decode_action(self, action):
        """Decode action to partial molecule"""
        # Mock molecule generation - create valid SMILES
        action_val = action.item()

        # Create valid SMILES based on action value
        if action_val < 20:
            smiles = "CCO"  # Ethanol
        elif action_val < 40:
            smiles = "CCCO"  # Propanol
        elif action_val < 60:
            smiles = "CCCC"  # Butane
        elif action_val < 80:
            smiles = "C1=CC=CC=C1"  # Benzene
        else:
            smiles = "CC(C)C"  # Isobutane

        return smiles

    def get_log_prob(self, states, actions):
        """Get log probabilities for state-action pairs"""
        logits = torch.randn(len(states), 100).to(self.device)  # 100 possible actions
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions)

    def get_entropy(self, states):
        """Get policy entropy for exploration"""
        logits = torch.randn(len(states), 100).to(self.device)  # 100 possible actions
        dist = torch.distributions.Categorical(logits=logits)
        return dist.entropy()

    def encode_molecule(self, smiles):
        """Encode SMILES to latent representation"""
        return torch.randn(1, self.hidden_dim).to(self.device)

    def predict_properties(self, mol_repr):
        """Predict molecular properties from representation"""
        return torch.randn(1, 3).to(self.device)  # QED, SA, LogP
