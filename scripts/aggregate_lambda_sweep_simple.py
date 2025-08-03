import glob
import pandas as pd
import os

records = []
for f in glob.glob("ablation/raw/*.parquet"):
    df = pd.read_parquet(f)
    lam = (
        float(os.path.basename(f).split("_")[0].replace("stage3_results_lambda", ""))
        / 10
    )
    hv = df["hypervolume"].iloc[-1]
    conf = (df["diffdock_confidence"] > 0.6).mean()
    records.append({"lambda": lam, "hypervolume": hv, "pose_conf>0.6": conf})

tbl = pd.DataFrame(records).sort_values("lambda")
tbl.to_csv("ablation/lambda_sweep_summary.csv", index=False)
print("📊 λ-SWEEP SUMMARY TABLE:")
print("=" * 50)
print(tbl.to_string(index=False))

print("\n✅ CSV saved to: ablation/lambda_sweep_summary.csv")
print(f"📊 Total records: {len(records)}")
print(f"🎯 Lambda values tested: {sorted([r['lambda'] for r in records])}")

# Check pass/fail gates
if len(records) > 0:
    baseline_hv = records[0]["hypervolume"] if records[0]["lambda"] == 0.0 else None
    best_hv = max([r["hypervolume"] for r in records])
    best_lambda = max(records, key=lambda x: x["hypervolume"])["lambda"]

    print("\n🏆 BEST PERFORMING λ:")
    print(f"  λ = {best_lambda}")
    print(f"  Hypervolume = {best_hv:.4f}")

    if baseline_hv:
        improvement = ((best_hv - baseline_hv) / baseline_hv) * 100
        print(f"  Improvement vs baseline: {improvement:.1f}%")

        if improvement >= 20:
            print("  ✅ PASS: Hypervolume improves by ≥20%")
        else:
            print("  ⚠️  WARNING: Hypervolume improvement <20%")

    best_conf = max([r["pose_conf>0.6"] for r in records])
    if best_conf >= 0.3:
        print("  ✅ PASS: Pose confidence ≥30%")
    else:
        print("  ⚠️  WARNING: Pose confidence <30%")

print("\n🎉 λ-SWEEP ANALYSIS COMPLETE!")
