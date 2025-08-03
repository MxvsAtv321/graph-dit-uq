#!/usr/bin/env python3
"""
Merge multiple Stage 3 iterations into a single comprehensive dataset.
"""

import pandas as pd
import glob
import os
import json
from datetime import datetime


def merge_iteration_results(run_prefix="sandbox_iter", output_file=None):
    """
    Merge results from multiple Stage 3 iterations.

    Args:
        run_prefix: Prefix for the run IDs to merge
        output_file: Output file path (optional)
    """

    # Find all stage3_results files
    data_dir = "data"
    pattern = os.path.join(data_dir, "stage3_results.parquet")

    # Get all parquet files in data directory
    all_files = glob.glob(os.path.join(data_dir, "*.parquet"))

    print(f"Found {len(all_files)} parquet files in {data_dir}")

    # Read and combine all results
    all_dfs = []
    iteration_count = 0

    for file in all_files:
        try:
            df = pd.read_parquet(file)
            if "physics_reward" in df.columns:  # This is a Stage 3 result
                # Add iteration number based on file modification time
                df["iteration"] = iteration_count + 1
                df["source_file"] = os.path.basename(file)
                all_dfs.append(df)
                iteration_count += 1
                print(f"Added {len(df)} molecules from {os.path.basename(file)}")
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not all_dfs:
        print("No valid Stage 3 result files found!")
        return None

    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Add metadata
    combined_df["merge_timestamp"] = datetime.now()
    combined_df["total_iterations"] = len(all_dfs)

    # Calculate cumulative statistics
    combined_df["cumulative_molecules"] = range(1, len(combined_df) + 1)

    # Sort by iteration and physics reward
    combined_df = combined_df.sort_values(
        ["iteration", "physics_reward"], ascending=[True, False]
    )

    # Generate output filename
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(data_dir, f"stage3_combined_{timestamp}.parquet")

    # Save combined results
    combined_df.to_parquet(output_file, index=False)

    # Generate summary statistics
    summary = {
        "total_iterations": len(all_dfs),
        "total_molecules": len(combined_df),
        "mean_physics_reward": combined_df["physics_reward"].mean(),
        "mean_diffdock_confidence": combined_df["diffdock_confidence"].mean(),
        "mean_quickvina_score": combined_df["quickvina_score"].mean(),
        "mean_qed": combined_df["qed"].mean(),
        "mean_sa_score": combined_df["sa_score"].mean(),
        "output_file": output_file,
        "merge_timestamp": datetime.now().isoformat(),
    }

    # Save summary
    summary_file = output_file.replace(".parquet", "_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== MERGE SUMMARY ===")
    print(f"Total iterations: {summary['total_iterations']}")
    print(f"Total molecules: {summary['total_molecules']}")
    print(f"Mean physics reward: {summary['mean_physics_reward']:.3f}")
    print(f"Mean DiffDock confidence: {summary['mean_diffdock_confidence']:.3f}")
    print(f"Mean QuickVina score: {summary['mean_quickvina_score']:.3f}")
    print(f"Mean QED: {summary['mean_qed']:.3f}")
    print(f"Mean SA score: {summary['mean_sa_score']:.3f}")
    print(f"Output file: {output_file}")
    print(f"Summary file: {summary_file}")

    return output_file


if __name__ == "__main__":
    # Merge all iterations
    output_file = merge_iteration_results()

    if output_file:
        print(f"\n✅ Successfully merged iterations to: {output_file}")
    else:
        print("\n❌ Failed to merge iterations")
