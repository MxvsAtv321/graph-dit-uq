#!/usr/bin/env python3
"""
Launch λ-sweep ablation study for Stage 3 physics-ML integration
"""

import subprocess
import time
import json
from datetime import datetime
import os


def launch_lambda_sweep():
    """Launch the λ-sweep ablation study"""

    print("🔬 LAUNCHING λ-SWEEP ABLATION STUDY")
    print("=" * 50)

    # Configuration
    lambda_values = [0.0, 0.2, 0.4, 0.6]
    iterations_per_sweep = 50  # Reduced for faster completion
    run_id_prefix = f"lambda_sweep_{datetime.now().strftime('%Y%m%d_%H%M')}"

    print("📊 Configuration:")
    print(f"  λ values: {lambda_values}")
    print(f"  Iterations per sweep: {iterations_per_sweep}")
    print(f"  Run ID prefix: {run_id_prefix}")
    print(f"  Total iterations: {len(lambda_values) * iterations_per_sweep}")

    # Launch each λ value
    for i, lambda_val in enumerate(lambda_values):
        print(f"\n🚀 Launching λ = {lambda_val} ({i+1}/{len(lambda_values)})")

        run_id = f"{run_id_prefix}_lambda_{lambda_val}"

        # Trigger the DAG
        cmd = [
            "docker",
            "compose",
            "exec",
            "-T",
            "airflow-worker",
            "airflow",
            "dags",
            "trigger",
            "dit_uq_stage3",
            "--run-id",
            run_id,
            "--conf",
            f'{{"iters":{iterations_per_sweep},"lambda_diffdock":{lambda_val}}}',
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ Successfully triggered λ = {lambda_val}")
            print(f"   Run ID: {run_id}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to trigger λ = {lambda_val}: {e}")
            continue

        # Small delay between launches
        if i < len(lambda_values) - 1:
            print("⏳ Waiting 10 seconds before next launch...")
            time.sleep(10)

    print("\n🎉 λ-SWEEP LAUNCHED!")
    print("📊 Monitoring commands:")
    print("  # Check all runs:")
    print(
        f"  docker compose exec airflow-worker airflow dags list-runs --dag-id dit_uq_stage3 | grep '{run_id_prefix}'"
    )
    print("  # Monitor progress:")
    print(
        f"  watch -n 30 'docker compose exec airflow-worker airflow dags list-runs --dag-id dit_uq_stage3 | grep \"{run_id_prefix}\" | grep success | wc -l'"
    )

    # Save sweep configuration
    sweep_config = {
        "timestamp": datetime.now().isoformat(),
        "run_id_prefix": run_id_prefix,
        "lambda_values": lambda_values,
        "iterations_per_sweep": iterations_per_sweep,
        "total_iterations": len(lambda_values) * iterations_per_sweep,
        "status": "launched",
    }

    config_file = f"ablation/raw/{run_id_prefix}_config.json"
    os.makedirs(os.path.dirname(config_file), exist_ok=True)

    with open(config_file, "w") as f:
        json.dump(sweep_config, f, indent=2)

    print(f"📄 Sweep configuration saved to: {config_file}")
    print(
        f"\n🎯 Expected completion time: ~{len(lambda_values) * iterations_per_sweep * 0.2:.0f} minutes"
    )
    print("📊 Results will be available in: ablation/raw/")


if __name__ == "__main__":
    launch_lambda_sweep()
