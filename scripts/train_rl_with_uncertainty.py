#!/usr/bin/env python3
"""Train RL with uncertainty-guided exploration."""

import argparse
import torch
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import logging

from src.rl.molecular_ppo import MolecularPPO, RLGraphDiT

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import random

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


def train_rl_with_ablation(args):
    """Train RL with ablation study on uncertainty"""

    # Initialize generator
    logger.info(f"Loading generator from {args.checkpoint_path}")
    generator = RLGraphDiT(args.checkpoint_path, device="cpu")  # Use CPU for now

    # Initialize PPO
    ppo_config = {
        "lr": args.lr,
        "batch_size": args.batch_size,
        "uncertainty_bonus": args.uncertainty_bonus if args.use_uncertainty else 0.0,
        "lambda_qed": args.lambda_qed,
        "lambda_docking": args.lambda_docking,
        "lambda_sa": args.lambda_sa,
    }

    logger.info("Initializing PPO with config:")
    for key, value in ppo_config.items():
        logger.info(f"  {key}: {value}")

    ppo = MolecularPPO(generator, ppo_config)

    # Training metrics
    metrics = {
        "rewards": [],
        "pareto_coverage": [],
        "valid_molecules": [],
        "unique_molecules": set(),
        "best_molecules": [],
    }

    # Training loop
    for iteration in range(args.n_iterations):
        logger.info(f"\n{'='*60}")
        logger.info(f"Iteration {iteration + 1}/{args.n_iterations}")
        logger.info("=" * 60)

        # Collect trajectories
        logger.info("Collecting trajectories...")
        trajectories = ppo.collect_trajectories(n_steps=args.n_steps)

        # Update policy
        logger.info("Updating policy...")
        ppo.update(trajectories, n_epochs=args.n_epochs)

        # Evaluate progress
        if (iteration + 1) % args.eval_frequency == 0:
            logger.info("Evaluating progress...")
            eval_metrics = evaluate_rl_progress(ppo, n_molecules=100)

            # Log metrics
            logger.info(f"Mean Reward: {np.mean(trajectories['rewards'].numpy()):.3f}")
            logger.info(f"Pareto Coverage: {eval_metrics['pareto_percentage']:.2%}")
            logger.info(f"Validity Rate: {eval_metrics['validity_rate']:.2%}")
            logger.info(f"Mean QED: {eval_metrics['mean_qed']:.3f}")
            logger.info(f"Mean Docking: {eval_metrics['mean_docking']:.2f}")

            # Update metrics
            metrics["rewards"].append(np.mean(trajectories["rewards"].numpy()))
            metrics["pareto_coverage"].append(eval_metrics["pareto_percentage"])

            # Save best molecules
            if eval_metrics["best_molecules"]:
                metrics["best_molecules"].extend(eval_metrics["best_molecules"])

        # Save checkpoint
        if (iteration + 1) % args.save_frequency == 0:
            save_path = Path(
                f"checkpoints/rl_iter_{iteration+1}_{'with' if args.use_uncertainty else 'without'}_uncertainty.pt"
            )
            torch.save(
                {
                    "iteration": iteration,
                    "generator_state": generator.state_dict(),
                    "optimizer_state": ppo.optimizer.state_dict(),
                    "metrics": metrics,
                },
                save_path,
            )
            logger.info(f"Checkpoint saved to {save_path}")

    # Final evaluation
    logger.info("\n" + "=" * 60)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 60)

    final_eval = evaluate_rl_progress(ppo, n_molecules=1000)

    # Save results
    results = {
        "config": vars(args),
        "metrics": {
            "rewards": [float(r) for r in metrics["rewards"]],  # Convert numpy to float
            "pareto_coverage": [
                float(p) for p in metrics["pareto_coverage"]
            ],  # Convert numpy to float
            "valid_molecules": metrics["valid_molecules"],
            "unique_molecules": list(
                metrics["unique_molecules"]
            ),  # Convert set to list
            "best_molecules": metrics["best_molecules"],
        },
        "final_evaluation": {
            "validity_rate": float(final_eval["validity_rate"]),
            "pareto_percentage": float(final_eval["pareto_percentage"]),
            "n_pareto": int(final_eval["n_pareto"]),
            "mean_qed": float(final_eval["mean_qed"]),
            "mean_docking": float(final_eval["mean_docking"]),
            "mean_sa": float(final_eval["mean_sa"]),
        },
        "run_name": f"RL_{'with' if args.use_uncertainty else 'without'}_uncertainty_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    }

    output_path = Path(f"outputs/rl_results_{results['run_name']}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    logger.info("\nTraining Complete!")
    logger.info(f"Final Pareto Coverage: {final_eval['pareto_percentage']:.2%}")
    logger.info(f"Improvement Factor: {final_eval['pareto_percentage'] / 0.03:.1f}x")
    logger.info(f"Best Molecules Generated: {len(metrics['best_molecules'])}")
    logger.info(f"Results saved to {output_path}")

    return results


def evaluate_rl_progress(ppo, n_molecules=1000):
    """Evaluate current RL policy"""
    molecules = []
    properties = []

    # Generate molecules
    logger.info(f"Generating {n_molecules} molecules for evaluation...")
    for i in range(n_molecules):
        if i % 100 == 0:
            logger.info(f"Generated {i}/{n_molecules} molecules...")

        # Generate molecule
        ppo.generator.eval()
        with torch.no_grad():
            smiles = ppo.generator.decode_action(
                torch.tensor([i % 10])
            )  # Mock generation

        if smiles:
            mol_data = ppo.evaluate_molecule(smiles)
            molecules.append(smiles)
            properties.append(mol_data)

    # Compute metrics
    valid_molecules = [p for p in properties if p["valid"]]
    validity_rate = len(valid_molecules) / len(molecules) if molecules else 0

    # Pareto analysis
    if valid_molecules:
        pareto_metrics = compute_pareto_metrics(valid_molecules)

        # Get best molecules
        pareto_indices = pareto_metrics["pareto_indices"]
        best_molecules = [valid_molecules[i] for i in pareto_indices[:10]]
    else:
        pareto_metrics = {"pareto_percentage": 0, "n_pareto": 0}
        best_molecules = []

    return {
        "validity_rate": validity_rate,
        "pareto_percentage": pareto_metrics["pareto_percentage"],
        "n_pareto": pareto_metrics["n_pareto"],
        "mean_qed": (
            np.mean([m["qed"] for m in valid_molecules]) if valid_molecules else 0
        ),
        "mean_docking": (
            np.mean([m["docking_score"] for m in valid_molecules])
            if valid_molecules
            else 0
        ),
        "mean_sa": (
            np.mean([m["sa_score"] for m in valid_molecules]) if valid_molecules else 0
        ),
        "best_molecules": best_molecules,
    }


def compute_pareto_metrics(molecules):
    """Compute Pareto optimality metrics"""
    if not molecules:
        return {"pareto_percentage": 0, "n_pareto": 0, "pareto_indices": []}

    # Extract properties
    [m["qed"] for m in molecules]
    [m["docking_score"] for m in molecules]
    [m["sa_score"] for m in molecules]

    # Find Pareto optimal points
    pareto_indices = []
    for i, molecule in enumerate(molecules):
        dominated = False
        for j, other in enumerate(molecules):
            if i != j:
                # Check if other dominates this molecule
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
                        dominated = True
                        break
        if not dominated:
            pareto_indices.append(i)

    pareto_percentage = len(pareto_indices) / len(molecules)

    return {
        "pareto_percentage": pareto_percentage,
        "n_pareto": len(pareto_indices),
        "pareto_indices": pareto_indices,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument(
        "--checkpoint_path", type=str, default="checkpoints/graph_dit_10k.pt"
    )

    # RL hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--n_epochs", type=int, default=4)
    parser.add_argument("--n_iterations", type=int, default=50)  # Shorter for testing

    # Uncertainty
    parser.add_argument(
        "--use_uncertainty",
        action="store_true",
        help="Use uncertainty-guided exploration",
    )
    parser.add_argument("--uncertainty_bonus", type=float, default=0.1)

    # Multi-objective weights
    parser.add_argument("--lambda_qed", type=float, default=0.3)
    parser.add_argument("--lambda_docking", type=float, default=0.5)
    parser.add_argument("--lambda_sa", type=float, default=0.2)

    # Training
    parser.add_argument("--eval_frequency", type=int, default=10)
    parser.add_argument("--save_frequency", type=int, default=20)

    args = parser.parse_args()

    # Run training
    train_rl_with_ablation(args)
