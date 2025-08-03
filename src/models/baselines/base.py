"""Base wrapper class for baseline models."""

import time
from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
import wandb


class BaseModelWrapper(ABC):
    """Base class for all baseline model wrappers."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def _load_model(self) -> None:
        """Load the specific model implementation."""
        pass

    @abstractmethod
    def _prepare_data(self, dataset: str) -> Any:
        """Prepare dataset for training."""
        pass

    @abstractmethod
    def _train_epoch(self, data: Any) -> Dict[str, float]:
        """Train for one epoch and return metrics."""
        pass

    def train_one_epoch(
        self, dataset: str = "qm9_subset", epochs: int = 1, num_workers: int = 2
    ) -> Dict[str, float]:
        """Standardized training interface for smoke testing."""
        start_time = time.time()

        # Load model if not already loaded
        if self.model is None:
            self._load_model()

        # Prepare data
        data = self._prepare_data(dataset)

        # Train
        metrics = {}
        for epoch in range(epochs):
            epoch_metrics = self._train_epoch(data)
            metrics.update(epoch_metrics)

        # Calculate runtime
        runtime = time.time() - start_time
        metrics["runtime"] = runtime
        metrics["model"] = self.model_name

        return metrics

    def log_to_wandb(self, metrics: Dict[str, float], project: str = "graph-dit-uq"):
        """Log metrics to Weights & Biases."""
        if wandb.run is None:
            wandb.init(project=project, name=f"{self.model_name}-smoke-test")

        wandb.log(metrics)
