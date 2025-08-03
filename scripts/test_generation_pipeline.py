#!/usr/bin/env python3
"""Test the entire generation pipeline end-to-end."""

import logging
from pathlib import Path
import tempfile

from src.data.qm9_loader import prepare_qm9_subset
from src.models.baselines.graph_dit import GraphDiTWrapper
from scripts.generate_10k import generate_10k_molecules, save_results
from scripts.plot_pareto_teaser import plot_pareto_teaser

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_pipeline():
    """Test the entire pipeline with small numbers."""
    logger.info("🧪 Testing generation pipeline...")

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # 1. Test data loading
        logger.info("1. Testing data loading...")
        try:
            dataset = prepare_qm9_subset(fraction=0.001)  # Very small subset
            logger.info(f"✅ Data loading: {len(dataset)} molecules")
        except Exception as e:
            logger.error(f"❌ Data loading failed: {e}")
            return False

        # 2. Test model training
        logger.info("2. Testing model training...")
        try:
            model = GraphDiTWrapper(hidden_dim=64, num_layers=2)  # Small model
            metrics = model.train(dataset=dataset, epochs=1, batch_size=32)
            logger.info(f"✅ Model training: loss={metrics['loss']:.4f}")
        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")
            return False

        # 3. Test checkpoint saving/loading
        logger.info("3. Testing checkpoint operations...")
        try:
            checkpoint_path = temp_path / "test_checkpoint.pt"
            model.save_checkpoint(str(checkpoint_path))
            loaded_model = GraphDiTWrapper.load_from_checkpoint(str(checkpoint_path))
            logger.info("✅ Checkpoint save/load successful")
        except Exception as e:
            logger.error(f"❌ Checkpoint operations failed: {e}")
            return False

        # 4. Test molecule generation
        logger.info("4. Testing molecule generation...")
        try:
            results = generate_10k_molecules(
                checkpoint_path=str(checkpoint_path),
                n_molecules=10,  # Small test
                batch_size=5,
            )
            logger.info(f"✅ Generation: {len(results)} molecules")
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            return False

        # 5. Test results saving
        logger.info("5. Testing results saving...")
        try:
            results_path = temp_path / "test_results.pkl"
            save_results(results, str(results_path))
            logger.info("✅ Results saving successful")
        except Exception as e:
            logger.error(f"❌ Results saving failed: {e}")
            return False

        # 6. Test plotting
        logger.info("6. Testing plotting...")
        try:
            plot_path = temp_path / "test_plot.png"
            plot_pareto_teaser(str(results_path), str(plot_path))
            logger.info("✅ Plotting successful")
        except Exception as e:
            logger.error(f"❌ Plotting failed: {e}")
            return False

    logger.info("🎉 All pipeline tests passed!")
    return True


def main():
    """Run the pipeline test."""
    success = test_pipeline()

    if success:
        print("\n" + "=" * 50)
        print("✅ PIPELINE TEST SUCCESSFUL")
        print("=" * 50)
        print("All components working correctly:")
        print("- Data loading ✓")
        print("- Model training ✓")
        print("- Checkpoint operations ✓")
        print("- Molecule generation ✓")
        print("- Results saving ✓")
        print("- Plotting ✓")
        print("\nReady for production use!")
    else:
        print("\n" + "=" * 50)
        print("❌ PIPELINE TEST FAILED")
        print("=" * 50)
        print("Check the logs above for details.")
        exit(1)


if __name__ == "__main__":
    main()
