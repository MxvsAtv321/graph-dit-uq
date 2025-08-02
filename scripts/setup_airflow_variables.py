#!/usr/bin/env python3
"""
Setup script for Airflow Variables used in the dit_uq_stage1 DAG.
This script creates the necessary Variables in Airflow for secure configuration.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def setup_airflow_variables():
    """Set up Airflow Variables for the Stage 1 pipeline"""
    
    # Import Airflow components
    try:
        from airflow.models import Variable
        from airflow import settings
    except ImportError:
        print("Error: Airflow not available. Make sure you're running this in an Airflow environment.")
        return False
    
    # Define the variables to set
    variables = {
        "S3_QM9_PATH": "s3://molecule-ai-stage0/qm9_subset.pt",
        "QM9_SHA256": "d41d8cd98f00b204e9800998ecf8427e",  # Placeholder - replace with actual hash
        "UQ_BATCH_SIZE": "64",
        "N_MOLECULES": "256",
        "DOCKING_BATCH_SIZE": "32",
        "DOCKING_MIN_SUCCESS_RATE": "0.80",
        "UQ_MIN_SUCCESS_RATE": "0.95",
        "VALIDATION_MAX_FLAGGED_PCT": "2.0",
        "GENERATION_MIN_VALIDITY_RATE": "0.98"
    }
    
    # Set up Airflow session
    session = settings.Session()
    
    try:
        for key, value in variables.items():
            # Check if variable already exists
            existing_var = session.query(Variable).filter(Variable.key == key).first()
            
            if existing_var:
                print(f"Updating existing variable: {key}")
                existing_var.val = value
            else:
                print(f"Creating new variable: {key}")
                new_var = Variable(key=key, val=value)
                session.add(new_var)
        
        session.commit()
        print("✅ Successfully set up all Airflow Variables")
        return True
        
    except Exception as e:
        print(f"❌ Error setting up Airflow Variables: {e}")
        session.rollback()
        return False
    finally:
        session.close()

def list_airflow_variables():
    """List all current Airflow Variables"""
    try:
        from airflow.models import Variable
        from airflow import settings
        
        session = settings.Session()
        variables = session.query(Variable).all()
        
        print("\n📋 Current Airflow Variables:")
        print("-" * 50)
        for var in variables:
            print(f"{var.key}: {var.val}")
        
        session.close()
        
    except Exception as e:
        print(f"Error listing variables: {e}")

if __name__ == "__main__":
    print("🚀 Setting up Airflow Variables for dit_uq_stage1 DAG...")
    
    success = setup_airflow_variables()
    
    if success:
        list_airflow_variables()
        print("\n✅ Setup complete! The dit_uq_stage1 DAG is now configured.")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")
        sys.exit(1) 