#!/usr/bin/env python3
"""
Quick test: Run 5 iterations of Stage 3 pipeline.
"""

import subprocess
import time
import json
import os
from datetime import datetime
import pandas as pd


def run_iteration(iteration_num, run_id_prefix="test_5"):
    """Run a single iteration of the Stage 3 pipeline."""

    run_id = f"{run_id_prefix}_iter_{iteration_num:02d}"

    print(f"🚀 Starting iteration {iteration_num}/5 with run ID: {run_id}")

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
        '{"iters":1,"lambda_diffdock":0.4}',
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ Triggered iteration {iteration_num}")
        return run_id, True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to trigger iteration {iteration_num}: {e}")
        return run_id, False


def wait_for_completion(run_id, max_wait=120):
    """Wait for a DAG run to complete."""

    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            cmd = [
                "docker",
                "compose",
                "exec",
                "-T",
                "airflow-worker",
                "airflow",
                "dags",
                "list-runs",
                "--dag-id",
                "dit_uq_stage3",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            # Check if our run is complete
            if run_id in result.stdout and "success" in result.stdout:
                print(f"✅ Iteration {run_id} completed successfully")
                return True
            elif run_id in result.stdout and "failed" in result.stdout:
                print(f"❌ Iteration {run_id} failed")
                return False
            else:
                print(f"⏳ Waiting for {run_id} to complete...")
                time.sleep(5)

        except subprocess.CalledProcessError as e:
            print(f"Error checking status: {e}")
            time.sleep(5)

    print(f"⏰ Timeout waiting for {run_id}")
    return False


def copy_results(run_id, iteration_num):
    """Copy results from a completed iteration."""

    source = "data/stage3_results.parquet"
    dest = f"data/stage3_results_{run_id}.parquet"

    if os.path.exists(source):
        try:
            df = pd.read_parquet(source)
            df["iteration"] = iteration_num
            df["run_id"] = run_id
            df["iteration_timestamp"] = datetime.now()

            df.to_parquet(dest, index=False)
            print(f"📁 Saved results for iteration {iteration_num} to {dest}")
            return dest
        except Exception as e:
            print(f"❌ Error copying results for iteration {iteration_num}: {e}")
            return None
    else:
        print(f"❌ No results file found for iteration {iteration_num}")
        return None


def run_5_iterations():
    """Run 5 iterations of the Stage 3 pipeline."""

    start_time = datetime.now()
    run_id_prefix = f"test_5_{start_time.strftime('%Y%m%d_%H%M')}"

    print(f"🎯 Starting 5-iteration test with prefix: {run_id_prefix}")
    print(f"⏰ Start time: {start_time}")

    successful_iterations = []
    failed_iterations = []
    result_files = []

    for i in range(1, 6):
        print(f"\n{'='*50}")
        print(f"ITERATION {i}/5")
        print(f"{'='*50}")

        run_id, triggered = run_iteration(i, run_id_prefix)

        if triggered:
            completed = wait_for_completion(run_id)

            if completed:
                result_file = copy_results(run_id, i)
                if result_file:
                    successful_iterations.append(i)
                    result_files.append(result_file)
                else:
                    failed_iterations.append(i)
            else:
                failed_iterations.append(i)
        else:
            failed_iterations.append(i)

        time.sleep(2)

    end_time = datetime.now()
    duration = end_time - start_time

    summary = {
        "run_id_prefix": run_id_prefix,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration.total_seconds(),
        "total_iterations": 5,
        "successful_iterations": len(successful_iterations),
        "failed_iterations": len(failed_iterations),
        "success_rate": len(successful_iterations) / 5,
        "result_files": result_files,
    }

    summary_file = f"data/test_5_summary_{start_time.strftime('%Y%m%d_%H%M')}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print("🎉 5-ITERATION TEST COMPLETE")
    print(f"{'='*50}")
    print(f"⏰ Duration: {duration}")
    print(
        f"✅ Successful: {len(successful_iterations)}/5 ({summary['success_rate']:.1%})"
    )
    print(f"❌ Failed: {len(failed_iterations)}/5")
    print(f"📁 Results saved to: {summary_file}")

    return summary


if __name__ == "__main__":
    summary = run_5_iterations()
    print("\n✅ Test complete! Summary saved to: data/test_5_summary_*.json")
