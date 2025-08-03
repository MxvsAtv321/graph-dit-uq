#!/usr/bin/env python3
"""
Local integration test script for the dit_uq_stage1 DAG.
This script tests components that can be validated outside of Airflow.
"""

import os
import sys
import time
import requests
from pathlib import Path


def check_docker_services():
    """Check if the required Docker services are running"""
    print("🔍 Checking Docker services...")

    services = {
        "autognnuq": "http://localhost:8000/live",
        "qvina": "http://localhost:5678/live",
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
            json={"smiles": test_smiles, "property_name": "activity", "n_samples": 5},
            timeout=30,
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ AutoGNNUQ test passed: {len(result['predictions'])} predictions")

            # Validate response structure
            for pred in result["predictions"]:
                assert "smiles" in pred
                assert "property" in pred
                assert "mu" in pred
                assert "sigma" in pred

            print("✅ Response structure validation passed")
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
                "exhaustiveness": 8,
            },
            timeout=60,
        )

        if response.status_code == 200:
            result = response.json()
            success_rate = result["metadata"]["success_rate"]
            print(f"✅ QuickVina2 test passed: {success_rate:.1%} success rate")

            # Validate response structure
            assert "results" in result
            assert "metadata" in result
            assert len(result["results"]) == len(test_smiles)

            for res in result["results"]:
                assert "smiles" in res
                assert "binding_affinity" in res
                assert "status" in res

            print("✅ Response structure validation passed")
            return True
        else:
            print(f"❌ QuickVina2 test failed: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ QuickVina2 test failed: {e}")
        return False


def test_dag_file_structure():
    """Test the DAG file structure and basic syntax"""
    print("\n🧪 Testing DAG file structure...")

    dag_file = Path("dags/dit_uq_stage1.py")

    if not dag_file.exists():
        print(f"❌ DAG file not found: {dag_file}")
        return False

    try:
        # Read and check basic structure
        with open(dag_file, "r") as f:
            content = f.read()

        # Check for required imports
        required_imports = [
            "from airflow import DAG",
            "from airflow.operators.python import PythonOperator",
            "from airflow.providers.docker.operators.docker import DockerOperator",
        ]

        for imp in required_imports:
            if imp not in content:
                print(f"❌ Missing import: {imp}")
                return False

        # Check for required tasks
        required_tasks = [
            "download_data",
            "generative_sample",
            "uq_predict",
            "docking_score",
            "validate_properties",
            "parquet_merge",
        ]

        for task in required_tasks:
            if f"task_id='{task}'" not in content:
                print(f"❌ Missing task: {task}")
                return False

        print("✅ DAG file structure validation passed")
        return True

    except Exception as e:
        print(f"❌ DAG file structure test failed: {e}")
        return False


def test_file_permissions():
    """Test file permissions and paths"""
    print("\n🧪 Testing file permissions...")

    required_paths = [
        "/Users/mxvsatv321/Documents/graph-dit-uq/data",
        "/Users/mxvsatv321/Documents/graph-dit-uq/checkpoints",
        "/Users/mxvsatv321/Documents/graph-dit-uq/data/receptors",
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
        "/Users/mxvsatv321/Documents/graph-dit-uq/data/receptors/DDR1_receptor.pdbqt",
    ]

    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}: Exists")
        else:
            print(f"⚠️  {file_path}: Missing (will use mock)")

    return all_good


def test_docker_compose():
    """Test Docker Compose configuration"""
    print("\n🧪 Testing Docker Compose configuration...")

    compose_file = Path("docker-compose.yaml")

    if not compose_file.exists():
        print(f"❌ Docker Compose file not found: {compose_file}")
        return False

    try:
        with open(compose_file, "r") as f:
            content = f.read()

        # Check for required services
        required_services = ["autognnuq", "qvina", "airflow-worker"]

        for service in required_services:
            if service not in content:
                print(f"❌ Missing service: {service}")
                return False

        # Check for required ports
        required_ports = ["8000:8000", "5678:5678"]

        for port in required_ports:
            if port not in content:
                print(f"❌ Missing port mapping: {port}")
                return False

        print("✅ Docker Compose configuration validation passed")
        return True

    except Exception as e:
        print(f"❌ Docker Compose test failed: {e}")
        return False


def test_environment_variables():
    """Test environment variable configuration"""
    print("\n🧪 Testing environment variables...")

    # Check if AIRFLOW_DATA is set
    airflow_data = os.environ.get("AIRFLOW_DATA")
    if airflow_data:
        print(f"✅ AIRFLOW_DATA: {airflow_data}")
    else:
        print("⚠️  AIRFLOW_DATA not set (will use default)")

    # Check if AIRFLOW_PROJ_DIR is set
    airflow_proj_dir = os.environ.get("AIRFLOW_PROJ_DIR")
    if airflow_proj_dir:
        print(f"✅ AIRFLOW_PROJ_DIR: {airflow_proj_dir}")
    else:
        print("⚠️  AIRFLOW_PROJ_DIR not set (will use default)")

    return True


def main():
    """Run all local integration tests"""
    print("🚀 Starting Stage 1 Local Integration Tests")
    print("=" * 50)

    start_time = time.time()

    # Run all tests
    tests = [
        ("Docker Services", check_docker_services),
        ("AutoGNNUQ Service", test_autognnuq_service),
        ("QuickVina2 Service", test_qvina_service),
        ("DAG File Structure", test_dag_file_structure),
        ("File Permissions", test_file_permissions),
        ("Docker Compose", test_docker_compose),
        ("Environment Variables", test_environment_variables),
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
    print("📊 LOCAL INTEGRATION TEST SUMMARY")
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
        print("\n🎉 ALL LOCAL TESTS PASSED! Stage 1 pipeline is ready for deployment.")
        print("\n📋 Next steps:")
        print("1. Start Airflow: docker compose up -d")
        print("2. Set up Airflow Variables: python scripts/setup_airflow_variables.py")
        print("3. Trigger the DAG: airflow dags trigger dit_uq_stage1")
        return True
    else:
        print(
            f"\n⚠️  {total - passed} tests failed. Please fix issues before deployment."
        )
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
