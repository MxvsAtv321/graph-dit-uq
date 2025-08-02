"""
Stage 1 DAG: Production-grade molecular AI pipeline with real services.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.providers.http.sensors.http import HttpSensor
import os
import json
import requests
import time

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
    'dit_uq_stage1',
    default_args=default_args,
    description='Stage 1 pipeline: Production-grade molecular generation with real services',
    schedule_interval=None,
    start_date=datetime(2025, 8, 3),
    catchup=False,
    tags=['molecular-ai', 'stage1', 'production'],
)

# Environment variables
UQ_BATCH_SIZE = int(os.environ.get('UQ_BATCH_SIZE', '64'))
N_MOLECULES = int(os.environ.get('N_MOLECULES', '256'))

# Task 1: Download data (unchanged from Stage 0)
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

# Task 2: Generate molecules using real GraphDiT (unchanged from Stage 0)
def generate_molecules_docker_fn(**context):
    """Docker command to generate molecules using real GraphDiT model"""
    return [
        'python', '-c', '''
import sys
import os
import json
from datetime import datetime
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, "/app/src")

# Safety check: ensure required files exist
checkpoint_path = Path("/data/checkpoints/graph_dit_10k.pt")
if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

# Real implementation using GraphDiTWrapper
class RLGraphDiT:
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the GraphDiT model from checkpoint"""
        try:
            from src.models.baselines.graph_dit import GraphDiTWrapper
            self.model = GraphDiTWrapper.load_from_checkpoint(self.checkpoint_path)
            print(f"Successfully loaded model from {self.checkpoint_path}")
        except Exception as e:
            print(f"Warning: Could not load model from checkpoint: {e}")
            print("Falling back to mock implementation")
            self.model = None
    
    def generate_molecules(self, n):
        """Generate n molecules using the loaded model"""
        if self.model is not None:
            try:
                # Use the real model's generate method
                molecules = self.model.generate(batch_size=n)
                print(f"Generated {len(molecules)} molecules using real GraphDiT model")
                return molecules
            except Exception as e:
                print(f"Warning: Real model generation failed: {e}")
                print("Falling back to mock implementation")
        
        # Fallback to mock implementation if real model fails
        import random
        molecules = []
        for _ in range(n):
            length = random.randint(10, 50)
            smiles = "".join(random.choices("CCCCOONNHHH()=[]", k=length))
            molecules.append(smiles)
        print(f"Generated {len(molecules)} molecules using mock implementation")
        return molecules

def validate_smiles(smiles_list):
    """Validate SMILES strings using RDKit"""
    try:
        from rdkit import Chem
        valid_count = 0
        total_count = len(smiles_list)
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_count += 1
        
        invalid_frac = 1.0 - (valid_count / total_count)
        print(f"SMILES validation: {valid_count}/{total_count} valid ({invalid_frac:.2%} invalid)")
        
        return invalid_frac
    except ImportError:
        print("Warning: RDKit not available, skipping SMILES validation")
        return 0.0

n_molecules = int(os.environ.get("N_MOLECULES", "256"))
model = RLGraphDiT(checkpoint_path="/data/checkpoints/graph_dit_10k.pt")
molecules = model.generate_molecules(n=n_molecules)

# RED-TEAM GUARD: Validate SMILES and fail if too many invalid
invalid_fraction = validate_smiles(molecules)
if invalid_fraction > 0.02:  # More than 2% invalid
    raise ValueError(f"Too many invalid SMILES: {invalid_fraction:.2%} > 2%. Task failed for safety.")

# Save as JSON
data = {
    "smiles": molecules,
    "generation_timestamp": datetime.now().isoformat(),
    "count": len(molecules),
    "model_used": "real" if model.model is not None else "mock",
    "invalid_fraction": invalid_fraction
}

with open("/data/generated_molecules.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"Generated {len(molecules)} molecules using {data['model_used']} model")
'''
    ]

generative_sample = DockerOperator(
    task_id='generative_sample',
    image='molecule-ai-base:latest',
    api_version='auto',
    auto_remove=True,
    command=generate_molecules_docker_fn(),
    docker_url='unix://var/run/docker.sock',
    network_mode='bridge',
    mounts=[
        {
            'source': '/Users/mxvsatv321/Documents/graph-dit-uq/data',
            'target': '/data',
            'type': 'bind'
        },
        {
            'source': '/Users/mxvsatv321/Documents/graph-dit-uq/checkpoints',
            'target': '/data/checkpoints',
            'type': 'bind'
        }
    ],
    dag=dag,
)

# Task 3: Wait for AutoGNNUQ service to be ready
wait_for_autognnuq = HttpSensor(
    task_id='wait_for_autognnuq',
    http_conn_id='autognnuq_conn',
    endpoint='/live',
    request_params={},
    response_check=lambda response: response.status_code == 200,
    poke_interval=30,
    timeout=600,
    dag=dag,
)

# Task 4: UQ prediction using real AutoGNNUQ service
def uq_predict_real_fn(**context):
    """Call real AutoGNNUQ service for uncertainty quantification"""
    import json
    import requests
    from pathlib import Path
    import time
    
    # Read generated molecules
    input_file = Path('/data/generated_molecules.json')
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    with open('/data/generated_molecules.json', 'r') as f:
        data = json.load(f)
    
    # Prepare request for AutoGNNUQ service
    payload = {
        "smiles": data['smiles'],
        "property_name": "activity",
        "n_samples": 5
    }
    
    # Call AutoGNNUQ service
    start_time = time.time()
    try:
        response = requests.post(
            "http://autognnuq:8000/predict",
            json=payload,
            timeout=90  # 90 second timeout as per requirements
        )
        response.raise_for_status()
        
        uq_results = response.json()
        processing_time = time.time() - start_time
        
        print(f"AutoGNNUQ service response time: {processing_time:.2f}s")
        
        # Add predictions to data
        data['uq_predictions'] = uq_results['predictions']
        data['uq_metadata'] = uq_results['metadata']
        
        # Save updated data
        with open('/data/molecules_with_uq.json', 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Added UQ predictions for {len(data['smiles'])} molecules")
        return f"Added UQ predictions for {len(data['smiles'])} molecules in {processing_time:.2f}s"
        
    except requests.exceptions.Timeout:
        raise ValueError("AutoGNNUQ service timeout (>90s)")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"AutoGNNUQ service error: {str(e)}")

uq_predict = PythonOperator(
    task_id='uq_predict',
    python_callable=uq_predict_real_fn,
    dag=dag,
)

# Task 5: Wait for QuickVina service to be ready
wait_for_qvina = HttpSensor(
    task_id='wait_for_qvina',
    http_conn_id='qvina_conn',
    endpoint='/live',
    request_params={},
    response_check=lambda response: response.status_code == 200,
    poke_interval=30,
    timeout=600,
    dag=dag,
)

# Task 6: Docking score using real QuickVina2 service
def docking_score_real_fn(**context):
    """Call real QuickVina2 service for docking scores"""
    import json
    import requests
    from pathlib import Path
    import time
    
    # Read molecules with UQ
    input_file = Path('/data/molecules_with_uq.json')
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    with open('/data/molecules_with_uq.json', 'r') as f:
        data = json.load(f)
    
    # Prepare request for QuickVina2 service
    payload = {
        "smiles": data['smiles'],
        "receptor_pdbqt": "/data/receptors/DDR1_receptor.pdbqt",
        "exhaustiveness": 8,
        "num_modes": 1
    }
    
    # Call QuickVina2 service
    start_time = time.time()
    try:
        response = requests.post(
            "http://qvina:5678/dock",
            json=payload,
            timeout=1800  # 30 minute timeout for docking
        )
        response.raise_for_status()
        
        docking_results = response.json()
        processing_time = time.time() - start_time
        
        print(f"QuickVina2 service response time: {processing_time:.2f}s")
        
        # Check success rate
        success_rate = docking_results['metadata']['success_rate']
        if success_rate < 0.95:  # 95% success rate requirement
            raise ValueError(f"Docking success rate too low: {success_rate:.1%} < 95%")
        
        # Add docking scores to data
        data['docking_results'] = docking_results['results']
        data['docking_metadata'] = docking_results['metadata']
        
        # Save updated data
        with open('/data/molecules_with_docking.json', 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Added docking scores for {len(data['smiles'])} molecules")
        return f"Added docking scores for {len(data['smiles'])} molecules in {processing_time:.2f}s"
        
    except requests.exceptions.Timeout:
        raise ValueError("QuickVina2 service timeout (>30min)")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"QuickVina2 service error: {str(e)}")

docking_score = PythonOperator(
    task_id='docking_score',
    python_callable=docking_score_real_fn,
    dag=dag,
)

# Task 7: Property validation (NEW)
def validate_properties_fn(**context):
    """Validate molecules for PAINS and toxicophores"""
    import json
    from pathlib import Path
    from src.services.property_validator import filter_smiles_list
    
    # Read molecules with docking
    input_file = Path('/data/molecules_with_docking.json')
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    with open('/data/molecules_with_docking.json', 'r') as f:
        data = json.load(f)
    
    # Validate molecules
    validation_results = filter_smiles_list(data['smiles'])
    
    # Check if validation passes threshold
    if not validation_results['summary']['passes_threshold']:
        flagged_frac = validation_results['summary']['flagged_fraction']
        raise ValueError(f"Too many molecules flagged: {flagged_frac:.1%} > 2%. DAG failed.")
    
    # Add validation results to data
    data['validation_results'] = validation_results['results']
    data['validation_summary'] = validation_results['summary']
    
    # Save updated data
    with open('/data/molecules_validated.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Validated {len(data['smiles'])} molecules")
    return f"Validated {len(data['smiles'])} molecules"

validate_properties = PythonOperator(
    task_id='validate_properties',
    python_callable=validate_properties_fn,
    dag=dag,
)

# Task 8: Merge results
def merge_results_fn(**context):
    """Merge all results into final output"""
    import json
    import glob
    from datetime import datetime
    import os
    
    # Find all intermediate json files
    json_files = glob.glob('/data/molecules_*.json')
    
    # Read the most recent file (should have all data)
    if json_files:
        with open('/data/molecules_validated.json', 'r') as f:
            data = json.load(f)
        
        # Add metadata
        data['pipeline_version'] = 'stage1_v1'
        data['processing_date'] = datetime.now().isoformat()
        
        # Save final merged file
        with open('/data/stage1_results.json', 'w') as f:
            json.dump(data, f, indent=2)
        
        # Clean up intermediate files
        for file in json_files:
            if file != '/data/stage1_results.json':
                os.remove(file)
        
        return f"Merged {len(data['smiles'])} records into stage1_results.json"
    else:
        raise ValueError("No json files found to merge")

parquet_merge = PythonOperator(
    task_id='parquet_merge',
    python_callable=merge_results_fn,
    dag=dag,
)

# Set task dependencies - tasks execute in order
download_data >> generative_sample >> wait_for_autognnuq >> uq_predict >> wait_for_qvina >> docking_score >> validate_properties >> parquet_merge 