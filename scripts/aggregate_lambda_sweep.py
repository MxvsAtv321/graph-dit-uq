#!/usr/bin/env python3
"""
Aggregate and analyze λ-sweep ablation study results
"""

import pandas as pd
import json
from datetime import datetime
import os
import glob


def aggregate_lambda_sweep_results():
    """Aggregate and analyze λ-sweep results"""

    print("🔬 AGGREGATING λ-SWEEP RESULTS")
    print("=" * 50)

    # Find the latest λ-sweep results
    data_dir = "data"
    lambda_sweep_files = glob.glob(
        os.path.join(data_dir, "stage3_results_lambda_sweep_*.parquet")
    )

    if not lambda_sweep_files:
        print("❌ No λ-sweep result files found")
        return None

    print(f"📊 Found {len(lambda_sweep_files)} λ-sweep result files")

    # Load and combine results
    all_results = []
    lambda_values = []

    for file in lambda_sweep_files:
        try:
            df = pd.read_parquet(file)
            # Extract lambda value from filename
            if "lambda_0.0" in file:
                lambda_val = 0.0
            elif "lambda_0.2" in file:
                lambda_val = 0.2
            elif "lambda_0.4" in file:
                lambda_val = 0.4
            elif "lambda_0.6" in file:
                lambda_val = 0.6
            else:
                lambda_val = 0.4  # default

            df["lambda_diffdock"] = lambda_val
            df["source_file"] = os.path.basename(file)
            all_results.append(df)
            lambda_values.append(lambda_val)

            print(f"✅ Loaded {len(df)} molecules from λ = {lambda_val}")

        except Exception as e:
            print(f"❌ Error loading {file}: {e}")
            continue

    if not all_results:
        print("❌ No valid results found")
        return None

    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    print(f"\n📊 Combined dataset: {len(combined_df)} total molecules")
    print(f"📋 Lambda values tested: {sorted(set(lambda_values))}")

    # Analysis by lambda value
    print("\n📈 ANALYSIS BY λ VALUE:")
    print("-" * 40)

    summary_stats = []

    for lambda_val in sorted(set(lambda_values)):
        lambda_data = combined_df[combined_df["lambda_diffdock"] == lambda_val]

        if len(lambda_data) == 0:
            continue

        print(f"\nλ = {lambda_val}:")
        print(f"  Molecules: {len(lambda_data)}")

        if "physics_reward" in lambda_data.columns:
            print(f"  Mean physics reward: {lambda_data['physics_reward'].mean():.4f}")
            print(f"  Max physics reward: {lambda_data['physics_reward'].max():.4f}")

        if "diffdock_confidence" in lambda_data.columns:
            high_conf_pct = (lambda_data["diffdock_confidence"] > 0.6).mean() * 100
            print(f"  High-confidence poses (>0.6): {high_conf_pct:.1f}%")

        if "quickvina_score" in lambda_data.columns:
            print(
                f"  Mean QuickVina score: {lambda_data['quickvina_score'].mean():.4f}"
            )
            print(f"  Best QuickVina score: {lambda_data['quickvina_score'].min():.4f}")

        if "qed" in lambda_data.columns:
            drug_like_pct = (lambda_data["qed"] > 0.5).mean() * 100
            print(f"  Drug-like molecules (QED>0.5): {drug_like_pct:.1f}%")

        # Store summary stats
        summary_stats.append(
            {
                "lambda_diffdock": lambda_val,
                "molecules": len(lambda_data),
                "mean_physics_reward": (
                    float(lambda_data["physics_reward"].mean())
                    if "physics_reward" in lambda_data.columns
                    else None
                ),
                "max_physics_reward": (
                    float(lambda_data["physics_reward"].max())
                    if "physics_reward" in lambda_data.columns
                    else None
                ),
                "high_conf_pct": (
                    float((lambda_data["diffdock_confidence"] > 0.6).mean() * 100)
                    if "diffdock_confidence" in lambda_data.columns
                    else None
                ),
                "mean_quickvina": (
                    float(lambda_data["quickvina_score"].mean())
                    if "quickvina_score" in lambda_data.columns
                    else None
                ),
                "best_quickvina": (
                    float(lambda_data["quickvina_score"].min())
                    if "quickvina_score" in lambda_data.columns
                    else None
                ),
                "drug_like_pct": (
                    float((lambda_data["qed"] > 0.5).mean() * 100)
                    if "qed" in lambda_data.columns
                    else None
                ),
            }
        )

    # Create summary table
    summary_df = pd.DataFrame(summary_stats)

    # Save results
    output_dir = "ablation"
    os.makedirs(output_dir, exist_ok=True)

    # Save combined results
    combined_file = os.path.join(output_dir, "lambda_sweep_combined.parquet")
    combined_df.to_parquet(combined_file, index=False)
    print(f"\n📄 Combined results saved to: {combined_file}")

    # Save summary table
    summary_file = os.path.join(output_dir, "lambda_sweep_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"📄 Summary table saved to: {summary_file}")

    # Generate ablation report
    ablation_report = {
        "timestamp": datetime.now().isoformat(),
        "lambda_values_tested": sorted(set(lambda_values)),
        "total_molecules": len(combined_df),
        "summary_stats": summary_stats,
        "files_processed": [os.path.basename(f) for f in lambda_sweep_files],
        "output_files": {
            "combined_results": combined_file,
            "summary_table": summary_file,
        },
    }

    report_file = os.path.join(output_dir, "lambda_sweep_report.json")
    with open(report_file, "w") as f:
        json.dump(ablation_report, f, indent=2)

    print(f"📄 Ablation report saved to: {report_file}")

    # Print summary table
    print("\n📊 λ-SWEEP SUMMARY TABLE:")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    # Find best performing lambda
    if "mean_physics_reward" in summary_df.columns:
        best_lambda = summary_df.loc[summary_df["mean_physics_reward"].idxmax()]
        print("\n🏆 BEST PERFORMING λ:")
        print(f"  λ = {best_lambda['lambda_diffdock']}")
        print(f"  Mean physics reward: {best_lambda['mean_physics_reward']:.4f}")
        print(f"  Molecules: {best_lambda['molecules']}")

    print("\n✅ λ-SWEEP ANALYSIS COMPLETE!")
    print(f"📊 Results available in: {output_dir}/")

    return ablation_report


if __name__ == "__main__":
    aggregate_lambda_sweep_results()
