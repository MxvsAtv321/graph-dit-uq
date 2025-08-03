"""GraphDiT baseline model wrapper."""

from typing import Dict, Any, List
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from .base import BaseModelWrapper


class GraphDiTModel(nn.Module):
    """Simple Graph DiT model for demonstration."""

    def __init__(self, hidden_dim=256, num_layers=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(128, hidden_dim)

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=8,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, 128)

    def forward(self, x):
        # x: [batch_size, seq_len, 128]
        x = self.input_proj(x)

        for layer in self.layers:
            x = layer(x)

        x = self.output_proj(x)
        return x


class GraphDiTWrapper(BaseModelWrapper):
    """Wrapper for GraphDiT model."""

    def __init__(self, hidden_dim=256, num_layers=8, dropout=0.1):
        super().__init__("GraphDiT")
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.model = None
        self.optimizer = None
        self.criterion = None

    def _load_model(self) -> None:
        """Load GraphDiT model."""
        if self.model is None:
            self.model = GraphDiTModel(
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout,
            ).to(self.device)

    def _prepare_data(self, dataset: str) -> Any:
        """Prepare QM9 subset data."""
        # TODO: Implement actual data loading
        # For now, create dummy data for smoke testing
        batch_size = 32
        data = {
            "x": torch.randn(batch_size, 128).to(self.device),
            "y": torch.randn(batch_size, 128).to(self.device),  # Match output size
        }
        return data

    def _train_epoch(self, data: Any) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()

        # Simulate training
        optimizer.zero_grad()
        output = self.model(data["x"])
        loss = criterion(output, data["y"])
        loss.backward()
        optimizer.step()

        # Calculate dummy metrics
        validity = 0.98  # Simulated validity score
        loss_value = loss.item()

        return {"loss": loss_value, "validity": validity}

    def train(
        self,
        dataset,
        epochs=1,
        batch_size=128,
        lr=1e-4,
        log_every_n_steps=50,
        log_wandb=False,
    ) -> Dict[str, float]:
        """Train the model with proper logging."""
        self._load_model()

        # Setup optimizer and criterion
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # Prepare data (simplified for now)
        data = self._prepare_data(dataset)

        # Training loop
        total_steps = 0
        for epoch in range(epochs):
            self.model.train()

            # Simulate multiple batches
            for step in range(100):  # Simulate 100 steps per epoch
                self.optimizer.zero_grad()

                # Forward pass
                output = self.model(data["x"])
                loss = self.criterion(output, data["y"])

                # Backward pass
                loss.backward()
                self.optimizer.step()

                total_steps += 1

                # Logging
                if log_wandb and total_steps % log_every_n_steps == 0:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "step": total_steps,
                            "loss": loss.item(),
                            "learning_rate": lr,
                        }
                    )

        # Calculate final metrics
        self.model.eval()
        with torch.no_grad():
            output = self.model(data["x"])
            final_loss = self.criterion(output, data["y"]).item()

        metrics = {
            "loss": final_loss,
            "validity": 0.98,  # Simulated
            "runtime": 0.0,  # Will be set by caller
            "model": self.model_name,
        }

        return metrics

    def generate(self, batch_size=100) -> List[str]:
        """Generate molecules (simplified for now)."""
        self._load_model()
        self.model.eval()

        # Generate dummy SMILES for demonstration
        dummy_smiles = [
            "CCO",
            "CCCO",
            "CCCC",
            "C1=CC=CC=C1",
            "CC(C)C",
            "CCOC",
            "CCCCCC",
            "C1CCCCC1",
            "CC(C)(C)C",
            "CCOCC",
        ]

        # Return batch_size number of SMILES
        return dummy_smiles[:batch_size]

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        self._load_model()
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": (
                self.optimizer.state_dict() if self.optimizer else None
            ),
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "model_name": self.model_name,
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_from_checkpoint(cls, path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")

        # Create wrapper
        wrapper = cls(
            hidden_dim=checkpoint["hidden_dim"],
            num_layers=checkpoint["num_layers"],
            dropout=checkpoint["dropout"],
        )

        # Load model
        wrapper._load_model()
        wrapper.model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer if available
        if checkpoint["optimizer_state_dict"]:
            wrapper.optimizer = optim.Adam(wrapper.model.parameters())
            wrapper.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return wrapper
