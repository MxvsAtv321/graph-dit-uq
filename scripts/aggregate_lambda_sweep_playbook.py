import glob
import pandas as pd
import os
import matplotlib.pyplot as plt

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
print(tbl)

# ── Figure 1: Hyper-volume vs λ ──────────────────────────
plt.figure()
plt.plot(tbl["lambda"], tbl["hypervolume"], marker="o")
plt.xlabel("λ (physics weight)")
plt.ylabel("Final hyper-volume")
plt.title("Physics weight ablation")
plt.savefig("ablation/fig_hv_vs_lambda.png", dpi=300, bbox_inches="tight")

# ── Figure 2: Pose confidence vs λ ───────────────────────
plt.figure()
plt.plot(tbl["lambda"], tbl["pose_conf>0.6"] * 100, marker="s")
plt.xlabel("λ")
plt.ylabel("% poses conf > 0.6")
plt.title("Pose reliability vs physics weight")
plt.savefig("ablation/fig_pose_conf_vs_lambda.png", dpi=300, bbox_inches="tight")

print("\nArtefacts written to ablation/ directory")
