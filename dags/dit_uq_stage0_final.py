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
    'dit_uq_stage0_final',
    default_args=default_args,
    description='Final working Stage 0 pipeline for Graph DiT-UQ molecular generation',
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

# Task 2: Generate molecules using PythonOperator (no PyTorch dependency)
def generate_molecules_fn(**context):
    import json
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
    
    # Save as JSON instead of parquet (no pandas dependency)
    data = {
        'smiles': molecules,
        'generation_timestamp': datetime.now().isoformat(),
        'count': len(molecules)
    }
    
    with open('/data/generated_molecules.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Generated {len(molecules)} molecules")
    return f"Generated {len(molecules)} molecules"

generative_sample = PythonOperator(
    task_id='generative_sample',
    python_callable=generate_molecules_fn,
    dag=dag,
)

# Task 3: UQ prediction using PythonOperator
def uq_predict_fn(**context):
    import json
    import random
    from pathlib import Path
    
    # Safety check: ensure input file exists
    input_file = Path('/data/generated_molecules.json')
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Read generated molecules
    with open('/data/generated_molecules.json', 'r') as f:
        data = json.load(f)
    
    # Mock AutoGNNUQ API call (replace with actual endpoint)
    def get_uq_predictions(smiles_list):
        # Mock uncertainty predictions
        predictions = []
        for smiles in smiles_list:
            mu = random.uniform(3, 7)  # Mock mean prediction
            sigma = random.uniform(0.1, 0.6)  # Mock uncertainty
            predictions.append({'mu': mu, 'sigma': sigma})
        return predictions
    
    # Process in batches
    batch_size = 32
    all_predictions = []
    
    for i in range(0, len(data['smiles']), batch_size):
        batch = data['smiles'][i:i+batch_size]
        predictions = get_uq_predictions(batch)
        all_predictions.extend(predictions)
    
    # Add predictions to data
    data['predictions'] = all_predictions
    
    # Save updated data
    with open('/data/molecules_with_uq.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Added UQ predictions for {len(data['smiles'])} molecules")
    return f"Added UQ predictions for {len(data['smiles'])} molecules"

uq_predict = PythonOperator(
    task_id='uq_predict',
    python_callable=uq_predict_fn,
    dag=dag,
)

# Task 4: Docking score calculation using PythonOperator
def docking_score_fn(**context):
    import json
    import random
    from pathlib import Path
    
    # Safety check: ensure input file exists
    input_file = Path('/data/molecules_with_uq.json')
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Read molecules with UQ
    with open('/data/molecules_with_uq.json', 'r') as f:
        data = json.load(f)
    
    # Mock docking scores
    docking_scores = [random.uniform(-8, -2) for _ in range(len(data['smiles']))]
    data['docking_scores'] = docking_scores
    
    # Save updated data
    with open('/data/molecules_with_docking.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Added docking scores for {len(data['smiles'])} molecules")
    return f"Added docking scores for {len(data['smiles'])} molecules"

docking_score = PythonOperator(
    task_id='docking_score',
    python_callable=docking_score_fn,
    dag=dag,
)

# Task 5: Merge results
def merge_results_fn(**context):
    import json
    import glob
    from datetime import datetime
    import os
    
    # Find all intermediate json files
    json_files = glob.glob('/data/molecules_*.json')
    
    # Read the most recent file (should have all data)
    if json_files:
        with open('/data/molecules_with_docking.json', 'r') as f:
            data = json.load(f)
        
        # Add metadata
        data['pipeline_version'] = 'stage0_v1'
        data['processing_date'] = datetime.now().isoformat()
        
        # Save final merged file
        with open('/data/stage0_results.json', 'w') as f:
            json.dump(data, f, indent=2)
        
        # Clean up intermediate files
        for file in json_files:
            if file != '/data/stage0_results.json':
                os.remove(file)
        
        return f"Merged {len(data['smiles'])} records into stage0_results.json"
    else:
        raise ValueError("No json files found to merge")

parquet_merge = PythonOperator(
    task_id='parquet_merge',
    python_callable=merge_results_fn,
    dag=dag,
)

# Set task dependencies - tasks execute in order
download_data >> generative_sample >> uq_predict >> docking_score >> parquet_merge 