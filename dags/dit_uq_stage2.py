"""
Stage 2 DAG: Active Learning with RL Fine-tuning
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.exceptions import AirflowFailException
from airflow.models import Variable
import os
import json
import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Any

# Default arguments for the DAG
default_args = {
    'owner': 'molecule-ai-team',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(minutes=60),  # Longer for RL training
}

# Initialize the DAG
dag = DAG(
    'dit_uq_stage2',
    default_args=default_args,
    description='Stage 2 pipeline: Active Learning with RL Fine-tuning',
    schedule_interval=None,
    start_date=datetime(2025, 8, 3),
    catchup=False,
    tags=['molecular-ai', 'stage2', 'active-learning', 'rl'],
    max_active_runs=1,
)

# Configuration
airflow_data = os.environ.get('AIRFLOW_DATA', '/opt/airflow/data')
n_iterations = int(os.environ.get('N_ITERATIONS', '1000'))
batch_size = int(os.environ.get('BATCH_SIZE', '256'))
reward_type = os.environ.get('REWARD_TYPE', 'ahi')
pref_mode = os.environ.get('PREF_MODE', 'adaptive')

# Get configuration from Airflow Variables
rl_checkpoint_path = Variable.get("RL_CHECKPOINT_PATH", "/data/checkpoints/graph_dit_10k.pt")
hard_negatives_path = Variable.get("HARD_NEGATIVES_PATH", "/data/reference/ddr1_hard_negatives.sdf")

# Task 1: Initialize RL environment and load checkpoint
def initialize_rl_environment(**context):
    """Initialize RL environment and load checkpoint"""
    import time
    start_time = time.time()
    
    # Check if checkpoint exists
    if not os.path.exists(rl_checkpoint_path):
        raise AirflowFailException(f"RL checkpoint not found: {rl_checkpoint_path}")
    
    # Initialize replay buffer and preference sampler
    from src.rl.samplers import PreferenceSampler
    
    sampler = PreferenceSampler(n_objectives=3, warmup_samples=100, mode=pref_mode)
    
    # Save initialization state
    init_state = {
        'checkpoint_path': rl_checkpoint_path,
        'reward_type': reward_type,
        'pref_mode': pref_mode,
        'n_iterations': n_iterations,
        'batch_size': batch_size,
        'initialization_time': datetime.utcnow().isoformat()
    }
    
    with open(f"{airflow_data}/stage2_init_state.json", 'w') as f:
        json.dump(init_state, f, indent=2)
    
    # Emit metrics
    duration = time.time() - start_time
    metrics = {
        'task_id': 'initialize_rl_environment',
        'duration_s': duration,
        'checkpoint_size_mb': os.path.getsize(rl_checkpoint_path) / (1024*1024),
        'reward_type': reward_type,
        'pref_mode': pref_mode
    }
    
    print(f"METRICS: {json.dumps(metrics)}")
    print(f"RL environment initialized with {reward_type} reward and {pref_mode} preference sampling")
    
    return metrics

initialize_rl = PythonOperator(
    task_id='initialize_rl_environment',
    python_callable=initialize_rl_environment,
    dag=dag,
)

# Task 2: RL Fine-tuning with active learning loop
rl_finetune = DockerOperator(
    task_id='rl_finetune',
    image='molecule-ai-base:latest',
    api_version='auto',
    auto_remove=True,
    command=['python', '-c', f'''
import sys
import os
import json
import time
import torch
import numpy as np
from datetime import datetime

# Add the src directory to Python path
sys.path.insert(0, "/app/src")

start_time = time.time()
run_id = os.environ.get('AIRFLOW_CTX_DAG_RUN_ID', 'unknown')

# Load configuration
with open("/data/stage2_init_state.json", "r") as f:
    config = json.load(f)

# Import RL components
from src.rl.rewards import REWARD_REGISTRY
from src.rl.samplers import PreferenceSampler
from src.models.baselines.graph_dit import GraphDiTWrapper

# Initialize components
reward_fn = REWARD_REGISTRY[config["reward_type"]]
sampler = PreferenceSampler(n_objectives=3, warmup_samples=100, mode=config["pref_mode"])

# Load model
model = GraphDiTWrapper.load_from_checkpoint(config["checkpoint_path"])
model.train()

# Active learning loop
pareto_front = []
archive = []
training_log = []

for iteration in range(config["n_iterations"]):
    iter_start = time.time()
    
    # Sample preference vector
    preference = sampler.sample()
    
    # Generate molecules
    molecules = model.generate(batch_size=config["batch_size"])
    
    # Evaluate molecules (mock for now - would call UQ and docking services)
    batch_results = []
    for mol in molecules:
        # Mock evaluation - replace with real service calls
        qed = np.random.uniform(0.3, 0.9)
        docking = np.random.uniform(-12.0, -5.0)
        sa = np.random.uniform(0.4, 0.8)
        
        objectives = [qed, docking, sa]
        
        # Compute reward
        if config["reward_type"] == "ahi":
            reward = reward_fn(mol, pareto_front, None)  # Mock uncertainty model
        elif config["reward_type"] == "lpef":
            reward = reward_fn(mol, preference, None)  # Mock energy model
        elif config["reward_type"] == "sucb":
            reward = reward_fn(mol, archive, None)  # Mock archive
        
        batch_results.append({{
            "smiles": mol,
            "objectives": objectives,
            "reward": float(reward),
            "preference": preference.tolist(),
            "iteration": iteration
        }})
    
    # Update Pareto front and archive
    for result in batch_results:
        # Update Pareto front (simplified)
        pareto_front.append({{
            "qed": result["objectives"][0],
            "docking": result["objectives"][1],
            "sa": result["objectives"][2]
        }})
        
        # Update archive
        archive.append({{
            "smiles": result["smiles"],
            "objectives": result["objectives"]
        }})
    
    # Update sampler with best result
    best_result = max(batch_results, key=lambda x: x["reward"])
    sampler.update(np.array(best_result["objectives"]), {{
        "reward": best_result["reward"],
        "iteration": iteration
    }})
    
    # Log training progress
    avg_reward = np.mean([r["reward"] for r in batch_results])
    training_log.append({{
        "iteration": iteration,
        "avg_reward": avg_reward,
        "best_reward": best_result["reward"],
        "pareto_front_size": len(pareto_front),
        "archive_size": len(archive),
        "duration_s": time.time() - iter_start
    }})
    
    # Save checkpoint every 100 iterations
    if iteration % 100 == 0:
        checkpoint_path = f"/data/stage2_checkpoint_{{run_id}}_iter_{{iteration}}.pt"
        torch.save({{
            "model_state_dict": model.state_dict(),
            "iteration": iteration,
            "pareto_front": pareto_front,
            "archive": archive,
            "sampler_stats": sampler.get_stats(),
            "config": config
        }}, checkpoint_path)
        print(f"Checkpoint saved: {{checkpoint_path}}")

# Save final results
final_results = {{
    "training_log": training_log,
    "pareto_front": pareto_front,
    "archive": archive,
    "sampler_stats": sampler.get_stats(),
    "config": config,
    "total_duration_s": time.time() - start_time
}}

output_file = f"/data/stage2_results_{{run_id}}.json"
with open(output_file, "w") as f:
    json.dump(final_results, f, indent=2)

# Emit final metrics
final_metrics = {{
    "task_id": "rl_finetune",
    "duration_s": time.time() - start_time,
    "n_iterations": config["n_iterations"],
    "final_avg_reward": np.mean([log["avg_reward"] for log in training_log[-10:]]),
    "pareto_front_size": len(pareto_front),
    "archive_size": len(archive),
    "reward_type": config["reward_type"],
    "pref_mode": config["pref_mode"]
}}

print(f"METRICS: {{json.dumps(final_metrics)}}")
print(f"RL fine-tuning completed: {{len(pareto_front)}} Pareto points, {{len(archive)}} archive entries")
print(f"Results saved to: {{output_file}}")
'''],
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
    environment={
        'AIRFLOW_CTX_DAG_RUN_ID': '{{ run_id }}',
        'CUDA_VISIBLE_DEVICES': '0'
    },
    dag=dag,
)

# Task 3: Evaluate and analyze results
def evaluate_results(**context):
    """Evaluate RL fine-tuning results"""
    import time
    import glob
    
    start_time = time.time()
    run_id = context['run_id']
    
    # Load results
    results_file = f"/data/stage2_results_{run_id}.json"
    if not os.path.exists(results_file):
        raise AirflowFailException(f"Results file not found: {results_file}")
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Analyze Pareto front
    pareto_front = results['pareto_front']
    training_log = results['training_log']
    
    # Calculate metrics
    if pareto_front:
        qed_values = [p['qed'] for p in pareto_front]
        docking_values = [p['docking'] for p in pareto_front]
        sa_values = [p['sa'] for p in pareto_front]
        
        pareto_metrics = {
            'pareto_size': len(pareto_front),
            'qed_range': [min(qed_values), max(qed_values)],
            'docking_range': [min(docking_values), max(docking_values)],
            'sa_range': [min(sa_values), max(sa_values)],
            'avg_qed': np.mean(qed_values),
            'avg_docking': np.mean(docking_values),
            'avg_sa': np.mean(sa_values)
        }
    else:
        pareto_metrics = {'pareto_size': 0}
    
    # Analyze training progress
    if training_log:
        final_rewards = [log['avg_reward'] for log in training_log[-10:]]
        training_metrics = {
            'final_avg_reward': np.mean(final_rewards),
            'reward_improvement': final_rewards[-1] - final_rewards[0] if len(final_rewards) > 1 else 0,
            'total_iterations': len(training_log)
        }
    else:
        training_metrics = {'total_iterations': 0}
    
    # Save analysis
    analysis = {
        'pareto_metrics': pareto_metrics,
        'training_metrics': training_metrics,
        'sampler_stats': results['sampler_stats'],
        'config': results['config'],
        'analysis_time': datetime.utcnow().isoformat()
    }
    
    analysis_file = f"/data/stage2_analysis_{run_id}.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Emit metrics
    duration = time.time() - start_time
    metrics = {
        'task_id': 'evaluate_results',
        'duration_s': duration,
        **pareto_metrics,
        **training_metrics
    }
    
    print(f"METRICS: {json.dumps(metrics)}")
    print(f"Analysis complete: {pareto_metrics['pareto_size']} Pareto points")
    print(f"Analysis saved to: {analysis_file}")
    
    return metrics

evaluate_results = PythonOperator(
    task_id='evaluate_results',
    python_callable=evaluate_results,
    dag=dag,
)

# Set task dependencies
initialize_rl >> rl_finetune >> evaluate_results 