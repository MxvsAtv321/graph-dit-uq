import glob
import pandas as pd
import os

records = []
for f in glob.glob("ablation/raw/*.parquet"):
    df = pd.read_parquet(f)
    # Extract lambda from filename - handle the actual filename format
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

    # Calculate metrics
    if "hypervolume" in df.columns:
        hv = df["hypervolume"].iloc[-1]
    else:
        hv = 0.0  # default if column doesn't exist

    if "diffdock_confidence" in df.columns:
        conf = (df["diffdock_confidence"] > 0.6).mean()
    else:
        conf = 0.0  # default if column doesn't exist

    records.append({"lambda": lam, "hypervolume": hv, "pose_conf>0.6": conf})
    print(f"✅ Processed {filename}: λ={lam}, HV={hv:.4f}, Conf={conf:.3f}")

if not records:
    print("❌ No valid records found")
    exit(1)

tbl = pd.DataFrame(records).sort_values("lambda")
tbl.to_csv("ablation/lambda_sweep_summary.csv", index=False)
print("\n📊 λ-SWEEP SUMMARY TABLE:")
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
