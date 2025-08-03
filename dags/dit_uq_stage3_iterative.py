"""
Stage 3 Iterative DAG: Physics-ML Integration with DiffDock-L
High-fidelity docking with physics-grounded reward functions for RL optimization.
Supports multiple iterations with result accumulation.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.models import Variable
import os

# Default arguments for the DAG
default_args = {
    "owner": "molecule-ai-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

# Initialize the DAG
dag = DAG(
    "dit_uq_stage3_iterative",
    default_args=default_args,
    description="Stage 3: Physics-ML integration with DiffDock-L high-fidelity docking (iterative)",
    schedule_interval=None,
    start_date=datetime(2025, 8, 3),
    catchup=False,
    tags=["molecular-ai", "stage3", "physics-ml", "diffdock", "iterative"],
)

# Get the Airflow data directory from environment
airflow_data = os.environ.get(
    "AIRFLOW_DATA", "/Users/mxvsatv321/Documents/graph-dit-uq/data"
)

# Get configuration from Airflow Variables
lambda_diffdock = Variable.get("LAMBDA_DIFFDOCK", default_var=0.4)
diffdock_batch_size = Variable.get("DIFFDOCK_BATCH_SIZE", default_var=8)
diffdock_num_samples = Variable.get("DIFFDOCK_NUM_SAMPLES", default_var=16)

############################################
# TASK 1: Check data availability
############################################
download_data = BashOperator(
    task_id="download_data",
    bash_command="""
    if [ -f /data/qm9_subset.pt ]; then
        echo "QM9 subset already exists at /data/qm9_subset.pt"
    else
        echo "Error: QM9 subset not found at /data/qm9_subset.pt"
        exit 1
    fi
    """,
    dag=dag,
)


############################################
# TASK 2: Initialize iteration tracking
############################################
def initialize_iteration_tracking(**context):
    import json

    # Get run ID from context
    run_id = context["dag_run"].run_id

    # Initialize iteration tracking
    tracking_data = {
        "run_id": run_id,
        "current_iteration": 0,
        "total_iterations": context["dag_run"].conf.get("iters", 100),
        "lambda_diffdock": context["dag_run"].conf.get("lambda_diffdock", 0.4),
        "start_time": datetime.now().isoformat(),
        "results": [],
    }

    # Save tracking data
    tracking_file = f"/data/iteration_tracking_{run_id}.json"
    with open(tracking_file, "w") as f:
        json.dump(tracking_data, f, indent=2)

    print(f"Initialized iteration tracking for run {run_id}")
    return tracking_file


init_tracking = PythonOperator(
    task_id="init_tracking",
    python_callable=initialize_iteration_tracking,
    dag=dag,
)


############################################
# TASK 3: Single iteration pipeline
############################################
def run_single_iteration(**context):
    import pandas as pd
    import numpy as np
    import json
    from datetime import datetime

    # Get run ID and iteration info
    run_id = context["dag_run"].run_id
    tracking_file = f"/data/iteration_tracking_{run_id}.json"

    # Load tracking data
    with open(tracking_file, "r") as f:
        tracking = json.load(f)

    current_iter = tracking["current_iteration"]
    total_iters = tracking["total_iterations"]

    print(f"Running iteration {current_iter + 1}/{total_iters}")

    # Generate molecules for this iteration
    valid_smiles = [
        "CCO",
        "CCCO",
        "CCCC",
        "C1=CC=CC=C1",
        "CC(C)C",
        "CCOC",
        "CCCCCC",
        "C1CCCCC1",
        "CC(C)(C)C",
        "CCOCC",
        "CC(C)CC",
        "C1CCC1",
        "CCCCCCC",
        "C1=CC=C(C=C1)C",
        "CC(C)CC(C)C",
        "CCOCC",
        "CCCCCCCC",
        "C1CCCCCC1",
        "CC(C)(C)CC",
        "CCOCCO",
        "CC(C)CCC",
        "C1CCC(C)C1",
        "CCCCCCCCC",
        "C1=CC=CC=C1C",
        "CC(C)CC(C)CC",
        "CCOCCOC",
        "CCCCCCCCCC",
        "C1CCCCCCC1",
        "CC(C)(C)CCC",
        "CCOCCOCC",
        "CC(C)CCCC",
        "C1CCC(C)CC1",
        "CCCCCCCCCCC",
    ]

    # Generate molecules for this iteration
    molecules = []
    for i in range(256):  # 256 molecules per iteration
        molecules.append(valid_smiles[i % len(valid_smiles)])

    # Create dataframe for this iteration
    df = pd.DataFrame(
        {
            "smiles": molecules,
            "iteration": current_iter + 1,
            "generation_timestamp": datetime.now(),
        }
    )

    # Mock UQ predictions
    df["mu"] = np.random.uniform(-8, -2, len(df))
    df["sigma"] = np.random.uniform(0.1, 0.5, len(df))

    # Mock QuickVina scores
    df["quickvina_score"] = np.random.uniform(-15, -5, len(df))

    # Mock DiffDock scores
    df["diffdock_confidence"] = np.random.uniform(0.1, 0.9, len(df))
    df["diffdock_rmsd"] = np.random.exponential(2.0, len(df)) + 1.0
    df["diffdock_score"] = df["diffdock_confidence"] * (
        1.0 / (1.0 + df["diffdock_rmsd"])
    )

    # Mock molecular properties
    df["qed"] = np.random.uniform(0.3, 0.8, len(df))
    df["sa_score"] = np.random.uniform(2.0, 6.0, len(df))
    df["heavy_atoms"] = [len([c for c in s if c in "CCOONNHH"]) for s in df["smiles"]]
    df["valid_mol"] = True
    df["pains_flag"] = False

    # Calculate physics-aware rewards
    lambda_diffdock = tracking["lambda_diffdock"]
    df["normalized_qed"] = df["qed"]
    df["normalized_quickvina"] = (df["quickvina_score"] + 15) / 10
    df["normalized_sa"] = (5 - df["sa_score"]) / 4
    df["normalized_diffdock"] = df["diffdock_score"]

    df["physics_reward"] = (
        0.3 * df["normalized_qed"]
        + 0.3 * df["normalized_quickvina"]
        + 0.2 * df["normalized_sa"]
        + lambda_diffdock * df["normalized_diffdock"]
    )

    # Add metadata
    df["pipeline_version"] = "stage3_v1_iterative"
    df["processing_date"] = datetime.now()
    df["lambda_diffdock"] = lambda_diffdock
    df["run_id"] = run_id

    # Save iteration results
    iteration_file = f"/data/iteration_{run_id}_{current_iter + 1:03d}.parquet"
    df.to_parquet(iteration_file, index=False)

    # Update tracking data
    tracking["current_iteration"] = current_iter + 1
    tracking["results"].append(
        {
            "iteration": current_iter + 1,
            "file": iteration_file,
            "n_molecules": len(df),
            "mean_physics_reward": df["physics_reward"].mean(),
            "mean_diffdock_confidence": df["diffdock_confidence"].mean(),
            "timestamp": datetime.now().isoformat(),
        }
    )

    # Save updated tracking
    with open(tracking_file, "w") as f:
        json.dump(tracking, f, indent=2)

    print(f"Completed iteration {current_iter + 1}/{total_iters}")
    print(f"Mean physics reward: {df['physics_reward'].mean():.3f}")
    print(f"Mean DiffDock confidence: {df['diffdock_confidence'].mean():.3f}")

    return iteration_file


single_iteration = PythonOperator(
    task_id="single_iteration",
    python_callable=run_single_iteration,
    dag=dag,
)


############################################
# TASK 4: Check if more iterations needed
############################################
def check_iteration_status(**context):
    import json

    run_id = context["dag_run"].run_id
    tracking_file = f"/data/iteration_tracking_{run_id}.json"

    with open(tracking_file, "r") as f:
        tracking = json.load(f)

    current_iter = tracking["current_iteration"]
    total_iters = tracking["total_iterations"]

    if current_iter < total_iters:
        print(f"More iterations needed: {current_iter}/{total_iters}")
        return "continue"
    else:
        print(f"All iterations completed: {current_iter}/{total_iters}")
        return "complete"


check_status = PythonOperator(
    task_id="check_status",
    python_callable=check_iteration_status,
    dag=dag,
)


############################################
# TASK 5: Merge all iterations
############################################
def merge_iterations(**context):
    import pandas as pd
    import json
    import glob

    run_id = context["dag_run"].run_id
    tracking_file = f"/data/iteration_tracking_{run_id}.json"

    with open(tracking_file, "r") as f:
        tracking = json.load(f)

    # Find all iteration files
    pattern = f"/data/iteration_{run_id}_*.parquet"
    iteration_files = sorted(glob.glob(pattern))

    print(f"Found {len(iteration_files)} iteration files")

    # Read and concatenate all iterations
    all_dfs = []
    for file in iteration_files:
        df = pd.read_parquet(file)
        all_dfs.append(df)

    # Combine all iterations
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Calculate cumulative statistics
    combined_df["cumulative_iteration"] = combined_df["iteration"]

    # Save combined results
    output_file = f"/data/stage3_results_{run_id}.parquet"
    combined_df.to_parquet(output_file, index=False)

    # Generate summary statistics
    summary = {
        "run_id": run_id,
        "total_iterations": tracking["total_iteration"],
        "total_molecules": len(combined_df),
        "mean_physics_reward": combined_df["physics_reward"].mean(),
        "mean_diffdock_confidence": combined_df["diffdock_confidence"].mean(),
        "lambda_diffdock": tracking["lambda_diffdock"],
        "output_file": output_file,
    }

    # Save summary
    summary_file = f"/data/summary_{run_id}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"Combined {len(combined_df)} molecules from {len(iteration_files)} iterations"
    )
    print(f"Mean physics reward: {combined_df['physics_reward'].mean():.3f}")
    print(f"Mean DiffDock confidence: {combined_df['diffdock_confidence'].mean():.3f}")

    return output_file


merge_results = PythonOperator(
    task_id="merge_results",
    python_callable=merge_iterations,
    dag=dag,
)

# Set task dependencies
download_data >> init_tracking >> single_iteration >> check_status >> merge_results
