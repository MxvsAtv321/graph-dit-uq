#!/usr/bin/env python3
"""
Integration test script for the dit_uq_stage1 DAG.
This script tests the complete pipeline with real services.
"""

import os
import sys
import time
import json
import requests
import subprocess
from pathlib import Path
from datetime import datetime

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_docker_services():
    """Check if the required Docker services are running"""
    print("🔍 Checking Docker services...")
    
    services = {
        'autognnuq': 'http://localhost:8000/live',
        'qvina': 'http://localhost:5678/live'
    }
    
    all_healthy = True
    for service_name, health_url in services.items():
        try:
            response = requests.get(health_url, timeout=10)
            if response.status_code == 200:
                print(f"✅ {service_name}: Healthy")
            else:
                print(f"❌ {service_name}: Unhealthy (status {response.status_code})")
                all_healthy = False
        except Exception as e:
            print(f"❌ {service_name}: Not reachable ({e})")
            all_healthy = False
    
    return all_healthy

def test_autognnuq_service():
    """Test the AutoGNNUQ service with sample SMILES"""
    print("\n🧪 Testing AutoGNNUQ service...")
    
    test_smiles = ["CCO", "CCCO", "C1=CC=CC=C1"]
    
    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json={
                "smiles": test_smiles,
                "property_name": "activity",
                "n_samples": 5
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ AutoGNNUQ test passed: {len(result['predictions'])} predictions")
            return True
        else:
            print(f"❌ AutoGNNUQ test failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ AutoGNNUQ test failed: {e}")
        return False

def test_qvina_service():
    """Test the QuickVina2 service with sample SMILES"""
    print("\n🧪 Testing QuickVina2 service...")
    
    test_smiles = ["CCO", "CCCO", "C1=CC=CC=C1"]
    
    try:
        response = requests.post(
            "http://localhost:5678/dock",
            json={
                "smiles": test_smiles,
                "receptor_pdbqt": "/data/receptors/DDR1_receptor.pdbqt",
                "exhaustiveness": 8
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            success_rate = result['metadata']['success_rate']
            print(f"✅ QuickVina2 test passed: {success_rate:.1%} success rate")
            return True
        else:
            print(f"❌ QuickVina2 test failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ QuickVina2 test failed: {e}")
        return False

def test_dag_syntax():
    """Test the DAG syntax and import"""
    print("\n🧪 Testing DAG syntax...")
    
    try:
        # Test DAG import
        from dags.dit_uq_stage1 import dag
        
        # Check basic DAG properties
        assert dag.dag_id == 'dit_uq_stage1'
        assert len(dag.tasks) == 6  # download_data, generative_sample, uq_predict, docking_score, validate_properties, parquet_merge
        
        print("✅ DAG syntax test passed")
        return True
        
    except Exception as e:
        print(f"❌ DAG syntax test failed: {e}")
        return False

def test_airflow_variables():
    """Test Airflow Variables setup"""
    print("\n🧪 Testing Airflow Variables...")
    
    try:
        from airflow.models import Variable
        
        required_vars = [
            "S3_QM9_PATH",
            "UQ_BATCH_SIZE", 
            "N_MOLECULES",
            "DOCKING_BATCH_SIZE"
        ]
        
        missing_vars = []
        for var_name in required_vars:
            try:
                value = Variable.get(var_name)
                print(f"✅ {var_name}: {value}")
            except:
                missing_vars.append(var_name)
                print(f"❌ {var_name}: Missing")
        
        if missing_vars:
            print(f"⚠️  Missing variables: {missing_vars}")
            return False
        
        print("✅ Airflow Variables test passed")
        return True
        
    except Exception as e:
        print(f"❌ Airflow Variables test failed: {e}")
        return False

def test_file_permissions():
    """Test file permissions and paths"""
    print("\n🧪 Testing file permissions...")
    
    required_paths = [
        "/Users/mxvsatv321/Documents/graph-dit-uq/data",
        "/Users/mxvsatv321/Documents/graph-dit-uq/checkpoints",
        "/Users/mxvsatv321/Documents/graph-dit-uq/data/receptors"
    ]
    
    all_good = True
    for path in required_paths:
        if os.path.exists(path) and os.access(path, os.R_OK | os.W_OK):
            print(f"✅ {path}: Readable and writable")
        else:
            print(f"❌ {path}: Missing or not accessible")
            all_good = False
    
    # Check for required files
    required_files = [
        "/Users/mxvsatv321/Documents/graph-dit-uq/checkpoints/graph_dit_10k.pt",
        "/Users/mxvsatv321/Documents/graph-dit-uq/data/receptors/DDR1_receptor.pdbqt"
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}: Exists")
        else:
            print(f"⚠️  {file_path}: Missing (will use mock)")
    
    return all_good

def run_small_scale_test():
    """Run a small-scale test of the pipeline"""
    print("\n🧪 Running small-scale pipeline test...")
    
    # Set environment for small test
    test_env = os.environ.copy()
    test_env['N_MOLECULES'] = '8'  # Small test
    test_env['UQ_BATCH'] = '4'
    
    try:
        # Test individual tasks
        print("Testing download_data task...")
        result = subprocess.run([
            'docker', 'compose', 'exec', '-T', 'airflow-worker',
            'airflow', 'tasks', 'test', 'dit_uq_stage1', 'download_data', '2025-08-03'
        ], capture_output=True, text=True, env=test_env, timeout=60)
        
        if result.returncode == 0:
            print("✅ download_data task passed")
        else:
            print(f"❌ download_data task failed: {result.stderr}")
            return False
        
        print("Testing generative_sample task...")
        result = subprocess.run([
            'docker', 'compose', 'exec', '-T', 'airflow-worker',
            'airflow', 'tasks', 'test', 'dit_uq_stage1', 'generative_sample', '2025-08-03'
        ], capture_output=True, text=True, env=test_env, timeout=300)  # 5 minutes
        
        if result.returncode == 0:
            print("✅ generative_sample task passed")
        else:
            print(f"❌ generative_sample task failed: {result.stderr}")
            return False
        
        print("✅ Small-scale test completed successfully")
        return True
        
    except subprocess.TimeoutExpired:
        print("❌ Test timed out")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Run all integration tests"""
    print("🚀 Starting Stage 1 Integration Tests")
    print("=" * 50)
    
    start_time = time.time()
    
    # Run all tests
    tests = [
        ("Docker Services", check_docker_services),
        ("AutoGNNUQ Service", test_autognnuq_service),
        ("QuickVina2 Service", test_qvina_service),
        ("DAG Syntax", test_dag_syntax),
        ("Airflow Variables", test_airflow_variables),
        ("File Permissions", test_file_permissions),
        ("Small-Scale Pipeline", run_small_scale_test)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 INTEGRATION TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    duration = time.time() - start_time
    print(f"\nResults: {passed}/{total} tests passed")
    print(f"Duration: {duration:.1f} seconds")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Stage 1 pipeline is ready for production.")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please fix issues before deployment.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 