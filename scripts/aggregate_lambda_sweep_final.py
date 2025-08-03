import glob
import pandas as pd
import os

records = []
for f in glob.glob("ablation/raw/*.parquet"):
    df = pd.read_parquet(f)
    # Extract lambda from filename
    filename = os.path.basename(f)
    if "lambda04" in filename:
        lam = 0.4
    elif "lambda02" in filename:
        lam = 0.2
    elif "lambda00" in filename:
        lam = 0.0
    elif "lambda06" in filename:
        lam = 0.6
    else:
        lam = 0.4  # default

    # Calculate metrics using available columns
    if "physics_reward" in df.columns:
        mean_physics = df["physics_reward"].mean()
        max_physics = df["physics_reward"].max()
    else:
        mean_physics = 0.0
        max_physics = 0.0

    if "diffdock_confidence" in df.columns:
        conf = (df["diffdock_confidence"] > 0.6).mean()
    else:
        conf = 0.0

    if "quickvina_score" in df.columns:
        mean_quickvina = df["quickvina_score"].mean()
        best_quickvina = df["quickvina_score"].min()
    else:
        mean_quickvina = 0.0
        best_quickvina = 0.0

    if "qed" in df.columns:
        drug_like_pct = (df["qed"] > 0.5).mean()
    else:
        drug_like_pct = 0.0

    records.append(
        {
            "lambda": lam,
            "mean_physics_reward": mean_physics,
            "max_physics_reward": max_physics,
            "pose_conf>0.6": conf,
            "mean_quickvina": mean_quickvina,
            "best_quickvina": best_quickvina,
            "drug_like_pct": drug_like_pct,
            "molecules": len(df),
        }
    )
    print(
        f"✅ Processed {filename}: λ={lam}, Physics={mean_physics:.4f}, Conf={conf:.3f}, Molecules={len(df)}"
    )

if not records:
    print("❌ No valid records found")
    exit(1)

tbl = pd.DataFrame(records).sort_values("lambda")
tbl.to_csv("ablation/lambda_sweep_summary.csv", index=False)
print("\n📊 λ-SWEEP SUMMARY TABLE:")
print("=" * 80)
print(tbl.to_string(index=False))

print("\n✅ CSV saved to: ablation/lambda_sweep_summary.csv")
print(f"📊 Total records: {len(records)}")
print(f"🎯 Lambda values tested: {sorted([r['lambda'] for r in records])}")

# Check pass/fail gates
if len(records) > 0:
    best_physics = max(records, key=lambda x: x["mean_physics_reward"])
    best_lambda = best_physics["lambda"]

    print("\n🏆 BEST PERFORMING λ:")
    print(f"  λ = {best_lambda}")
    print(f"  Mean physics reward = {best_physics['mean_physics_reward']:.4f}")
    print(f"  Max physics reward = {best_physics['max_physics_reward']:.4f}")
    print(f"  Pose confidence >0.6 = {best_physics['pose_conf>0.6']:.1%}")
    print(f"  Drug-like molecules = {best_physics['drug_like_pct']:.1%}")
    print(f"  Molecules generated = {best_physics['molecules']}")

    # Check if we have baseline comparison
    baseline = next((r for r in records if r["lambda"] == 0.0), None)
    if baseline and best_lambda != 0.0:
        improvement = (
            (best_physics["mean_physics_reward"] - baseline["mean_physics_reward"])
            / baseline["mean_physics_reward"]
        ) * 100
        print(f"  Improvement vs baseline (λ=0.0): {improvement:.1f}%")

        if improvement >= 20:
            print("  ✅ PASS: Physics reward improves by ≥20%")
        else:
            print("  ⚠️  WARNING: Physics reward improvement <20%")

    if best_physics["pose_conf>0.6"] >= 0.3:
        print("  ✅ PASS: Pose confidence ≥30%")
    else:
        print("  ⚠️  WARNING: Pose confidence <30%")

print("\n🎉 λ-SWEEP ANALYSIS COMPLETE!")
print("📄 Results available in: ablation/lambda_sweep_summary.csv")
