"""Smoke tests for baseline models."""

import pytest
import time
from src.models.baselines import GraphDiTWrapper, ADiTWrapper, DMolWrapper, MolXLWrapper


@pytest.mark.parametrize(
    "model_class", [GraphDiTWrapper, ADiTWrapper, DMolWrapper, MolXLWrapper]
)
def test_baseline_smoke(model_class):
    """One-epoch smoke test on QM9 subset."""
    print(f"\nTesting {model_class.__name__}...")

    # Initialize model
    model = model_class()

    # Run training
    start_time = time.time()
    metrics = model.train_one_epoch(dataset="qm9_subset", epochs=1, num_workers=2)
    runtime = time.time() - start_time

    # Assertions
    assert metrics["validity"] >= 0.95, f"Validity {metrics['validity']} < 0.95"
    assert runtime < 240, f"Runtime {runtime}s > 240s (4 minutes)"
    assert "loss" in metrics, "Loss metric missing"
    assert "model" in metrics, "Model name missing"

    print(
        f"✅ {model_class.__name__}: validity={metrics['validity']:.3f}, runtime={runtime:.1f}s"
    )


def test_all_baselines_sequential():
    """Test all baselines sequentially to ensure no conflicts."""
    models = [GraphDiTWrapper(), ADiTWrapper(), DMolWrapper(), MolXLWrapper()]

    results = {}
    for model in models:
        print(f"\nTesting {model.model_name}...")
        start_time = time.time()

        metrics = model.train_one_epoch(dataset="qm9_subset", epochs=1)
        runtime = time.time() - start_time

        results[model.model_name] = {
            "validity": metrics["validity"],
            "runtime": runtime,
            "loss": metrics["loss"],
        }

        print(
            f"✅ {model.model_name}: validity={metrics['validity']:.3f}, runtime={runtime:.1f}s"
        )

    # Summary assertions
    total_runtime = sum(r["runtime"] for r in results.values())
    assert total_runtime < 600, f"Total runtime {total_runtime}s > 600s (10 minutes)"

    avg_validity = sum(r["validity"] for r in results.values()) / len(results)
    assert avg_validity >= 0.96, f"Average validity {avg_validity:.3f} < 0.96"

    print(
        f"\n📊 Summary: avg_validity={avg_validity:.3f}, total_runtime={total_runtime:.1f}s"
    )


def test_gpu_memory_usage():
    """Test GPU memory usage doesn't exceed reasonable limits."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Test with largest model (MolXL)
    model = MolXLWrapper()

    # Get initial memory
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated()

    # Run training
    metrics = model.train_one_epoch(dataset="qm9_subset", epochs=1)

    # Get peak memory
    peak_memory = torch.cuda.max_memory_allocated()
    memory_used = peak_memory - initial_memory

    # Assert reasonable memory usage (8GB limit)
    max_memory_gb = 8 * 1024**3  # 8GB in bytes
    assert (
        memory_used < max_memory_gb
    ), f"Memory usage {memory_used/1024**3:.1f}GB > 8GB"

    print(f"✅ GPU memory usage: {memory_used/1024**3:.1f}GB")


if __name__ == "__main__":
    # Run smoke tests
    pytest.main([__file__, "-v", "-s"])
