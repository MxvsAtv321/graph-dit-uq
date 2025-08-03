#!/usr/bin/env python3
"""
Render pose confidence vs lambda figure for λ-sweep ablation study
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def render_pose_confidence_figure(csv_path, output_path):
    """Render pose confidence vs lambda figure"""
    # Load data
    df = pd.read_csv(csv_path)

    # Create figure
    plt.figure(figsize=(8, 6))

    # Plot pose confidence vs lambda (convert to percentage)
    plt.plot(
        df["lambda"], df["pose_conf>0.6"] * 100, marker="s", linewidth=2, markersize=8
    )

    # Customize
    plt.xlabel("λ (physics weight)", fontsize=12)
    plt.ylabel("% Poses with Confidence > 0.6", fontsize=12)
    plt.title("Pose Reliability vs Physics Weight", fontsize=14, fontweight="bold")

    # Grid and styling
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Figure saved to: {output_path}")

    # Print summary
    print(f"📊 Data points: {len(df)}")
    print(f"📈 Lambda range: {df['lambda'].min()} - {df['lambda'].max()}")
    print(
        f"📊 Pose confidence range: {df['pose_conf>0.6'].min()*100:.1f}% - {df['pose_conf>0.6'].max()*100:.1f}%"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Render pose confidence vs lambda figure"
    )
    parser.add_argument("--csv", required=True, help="Path to lambda_sweep_summary.csv")
    parser.add_argument("--out", required=True, help="Output PNG path")

    args = parser.parse_args()

    try:
        render_pose_confidence_figure(args.csv, args.out)
    except Exception as e:
        print(f"❌ Error rendering figure: {e}")
        exit(1)


if __name__ == "__main__":
    main()
