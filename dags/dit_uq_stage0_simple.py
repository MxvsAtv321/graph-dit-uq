from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
import os

# Default arguments for the DAG
default_args = {
    'owner': 'molecule-ai-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Initialize the DAG
dag = DAG(
    'dit_uq_stage0_simple',
    default_args=default_args,
    description='Simplified Stage 0 pipeline for Graph DiT-UQ molecular generation',
    schedule_interval=None,
    start_date=datetime(2025, 8, 3),
    catchup=False,
    tags=['molecular-ai', 'stage0', 'dit-uq'],
)

# Get the Airflow data directory from environment
airflow_data = os.environ.get('AIRFLOW_DATA', '/data')

# Task 1: Copy existing data (since qm9_subset.pt already exists)
download_data = BashOperator(
    task_id='download_data',
    bash_command="""
    mkdir -p /data && \
    if [ -f /data/qm9_subset.pt ]; then
        echo "QM9 subset already exists at /data/qm9_subset.pt"
    else
        echo "Error: qm9_subset.pt not found in /data/"
        exit 1
    fi
    """,
    dag=dag,
)

# Task 2: Generate molecules using PythonOperator
def generate_molecules_fn(**context):
    import torch
    import pandas as pd
    from datetime import datetime
    from pathlib import Path
    import random
    import string
    
    # Safety check: ensure required files exist
    checkpoint_path = Path('/data/checkpoints/graph_dit_10k.pt')
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Mock implementation for container execution
    class RLGraphDiT:
        def __init__(self, checkpoint_path):
            self.checkpoint_path = checkpoint_path
        
        def generate_molecules(self, n):
            # Generate mock SMILES for demonstration
            molecules = []
            for _ in range(n):
                length = random.randint(10, 50)
                smiles = ''.join(random.choices('CCCCOONNHHH()=[]', k=length))
                molecules.append(smiles)
            return molecules
    
    model = RLGraphDiT(checkpoint_path='/data/checkpoints/graph_dit_10k.pt')
    molecules = model.generate_molecules(n=256)
    
    df = pd.DataFrame({
        'smiles': molecules,
        'generation_timestamp': datetime.now()
    })
    df.to_parquet('/data/generated_molecules.parquet', index=False)
    print(f"Generated {len(molecules)} molecules")
    return f"Generated {len(molecules)} molecules"

generative_sample = PythonOperator(
    task_id='generative_sample',
    python_callable=generate_molecules_fn,
    dag=dag,
)

# Task 3: UQ prediction using PythonOperator
def uq_predict_fn(**context):
    import pandas as pd
    import numpy as np
    from pathlib import Path
    
    # Safety check: ensure input file exists
    input_file = Path('/data/generated_molecules.parquet')
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Read generated molecules
    df = pd.read_parquet('/data/generated_molecules.parquet')
    
    # Mock AutoGNNUQ API call (replace with actual endpoint)
    def get_uq_predictions(smiles_list):
        # Mock uncertainty predictions
        predictions = []
        for smiles in smiles_list:
            mu = np.random.randn() * 2 + 5  # Mock mean prediction
            sigma = np.abs(np.random.randn() * 0.5 + 0.1)  # Mock uncertainty
            predictions.append({'mu': mu, 'sigma': sigma})
        return predictions
    
    # Process in batches
    batch_size = 32
    all_predictions = []
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]['smiles'].tolist()
        predictions = get_uq_predictions(batch)
        all_predictions.extend(predictions)
    
    # Add predictions to dataframe
    df['mu'] = [p['mu'] for p in all_predictions]
    df['sigma'] = [p['sigma'] for p in all_predictions]
    
    # Save updated parquet
    df.to_parquet('/data/molecules_with_uq.parquet', index=False)
    print(f"Added UQ predictions for {len(df)} molecules")
    return f"Added UQ predictions for {len(df)} molecules"

uq_predict = PythonOperator(
    task_id='uq_predict',
    python_callable=uq_predict_fn,
    dag=dag,
)

# Task 4: Docking score calculation using PythonOperator
def docking_score_fn(**context):
    import pandas as pd
    from pathlib import Path
    
    # Safety check: ensure input file exists
    input_file = Path('/data/molecules_with_uq.parquet')
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Read molecules with UQ
    df = pd.read_parquet('/data/molecules_with_uq.parquet')
    
    # Mock docking scores
    import numpy as np
    df['docking_score'] = np.random.randn(len(df)) * 2 - 5  # Mock docking scores
    
    # Save updated parquet
    df.to_parquet('/data/molecules_with_docking.parquet', index=False)
    print(f"Added docking scores for {len(df)} molecules")
    return f"Added docking scores for {len(df)} molecules"

docking_score = PythonOperator(
    task_id='docking_score',
    python_callable=docking_score_fn,
    dag=dag,
)

# Task 5: Merge parquet files
def merge_parquet_files(**context):
    import pandas as pd
    import glob
    from datetime import datetime
    import os
    
    # Find all intermediate parquet files
    parquet_files = glob.glob('/data/molecules_*.parquet')
    
    # Read the most recent file (should have all columns)
    if parquet_files:
        df = pd.read_parquet('/data/molecules_with_docking.parquet')
        
        # Add metadata
        df['pipeline_version'] = 'stage0_v1'
        df['processing_date'] = datetime.now()
        
        # Save final merged file
        df.to_parquet('/data/stage0_results.parquet', index=False)
        
        # Clean up intermediate files
        for file in parquet_files:
            if file != '/data/stage0_results.parquet':
                os.remove(file)
        
        return f"Merged {len(df)} records into stage0_results.parquet"
    else:
        raise ValueError("No parquet files found to merge")

parquet_merge = PythonOperator(
    task_id='parquet_merge',
    python_callable=merge_parquet_files,
    dag=dag,
)

# Set task dependencies - tasks execute in order
download_data >> generative_sample >> uq_predict >> docking_score >> parquet_merge 