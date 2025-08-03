#!/usr/bin/env python3
"""Graph DiT fine-tuning script with W&B logging and carbon tracking."""

import argparse
import logging
import wandb
from pathlib import Path
from codecarbon import EmissionsTracker

from src.models.baselines.graph_dit import GraphDiTWrapper
from src.data.qm9_loader import prepare_qm9_subset, get_qm9_statistics

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def finetune_graph_dit(
    fraction=0.05,
    hidden_dim=256,
    num_layers=8,
    dropout=0.1,
    epochs=1,
    batch_size=128,
    lr=1e-4,
    log_every_n_steps=50,
    checkpoint_dir="checkpoints",
):
    """Fine-tune Graph DiT model on QM9 subset."""

    # Initialize tracking
    wandb.init(
        project="GraphDiT-UQ",
        name="graph-dit-finetune",
        config={
            "fraction": fraction,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "log_every_n_steps": log_every_n_steps,
        },
    )

    tracker = EmissionsTracker()
    tracker.start()

    try:
        # Load data
        logger.info("Loading QM9 subset...")
        dataset = prepare_qm9_subset(fraction=fraction)
        stats = get_qm9_statistics(dataset)
        logger.info(f"Dataset statistics: {stats}")

        # Log dataset info
        wandb.log({"dataset_size": len(dataset), **stats})

        # Initialize model
        logger.info("Initializing Graph DiT model...")
        model = GraphDiTWrapper(
            hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout
        )

        # Train
        logger.info(f"Starting training for {epochs} epochs...")
        metrics = model.train(
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            log_every_n_steps=log_every_n_steps,
            log_wandb=True,
        )

        # Save checkpoint
        checkpoint_path = Path(checkpoint_dir) / "graph_dit_10k.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_checkpoint(str(checkpoint_path))
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Log final metrics
        wandb.log(
            {
                "final_validity": metrics["validity"],
                "final_loss": metrics["loss"],
                "training_time": metrics.get("runtime", 0),
            }
        )

        return str(checkpoint_path)

    finally:
        # Log carbon emissions
        emissions = tracker.stop()
        wandb.log({"carbon_kg": emissions})
        logger.info(f"Carbon emissions: {emissions} kg CO2")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Graph DiT on QM9")
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.05,
        help="Fraction of QM9 to use (default: 0.05)",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=256, help="Hidden dimension (default: 256)"
    )
    parser.add_argument(
        "--num_layers", type=int, default=8, help="Number of layers (default: 8)"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout rate (default: 0.1)"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of epochs (default: 1)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size (default: 128)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=50,
        help="Log every N steps (default: 50)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Checkpoint directory (default: checkpoints)",
    )

    args = parser.parse_args()

    checkpoint_path = finetune_graph_dit(
        fraction=args.fraction,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        log_every_n_steps=args.log_every_n_steps,
        checkpoint_dir=args.checkpoint_dir,
    )

    print(f"✅ Fine-tuning complete! Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
