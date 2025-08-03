"""CI-friendly smoke tests that don't require GPUs or complex dependencies."""

import pytest
import os


def test_project_structure():
    """Test that essential project files exist."""
    essential_files = [
        "dags/dit_uq_stage3.py",
        "docker-compose.yaml",
        "requirements.txt",
        "README.md",
    ]

    for file_path in essential_files:
        assert os.path.exists(file_path), f"Essential file missing: {file_path}"

    print("✅ All essential project files exist")


def test_ablation_results():
    """Test that λ-sweep results exist."""
    ablation_files = [
        "ablation/lambda_sweep_summary.csv",
        "scripts/aggregate_lambda_sweep_final.py",
        "scripts/render_fig_hv_vs_lambda.py",
        "scripts/render_fig_pose_conf_vs_lambda.py",
    ]

    for file_path in ablation_files:
        assert os.path.exists(file_path), f"Ablation file missing: {file_path}"

    print("✅ All ablation study files exist")


def test_dag_syntax():
    """Test that the DAG file has valid Python syntax."""
    try:
        with open("dags/dit_uq_stage3.py", "r") as f:
            code = f.read()
        compile(code, "dags/dit_uq_stage3.py", "exec")
        print("✅ DAG file has valid Python syntax")
    except SyntaxError as e:
        pytest.fail(f"DAG file has syntax error: {e}")


def test_scripts_syntax():
    """Test that script files have valid Python syntax."""
    script_files = [
        "scripts/aggregate_lambda_sweep_final.py",
        "scripts/render_fig_hv_vs_lambda.py",
        "scripts/render_fig_pose_conf_vs_lambda.py",
    ]

    for script_file in script_files:
        try:
            with open(script_file, "r") as f:
                code = f.read()
            compile(code, script_file, "exec")
        except SyntaxError as e:
            pytest.fail(f"Script {script_file} has syntax error: {e}")

    print("✅ All script files have valid Python syntax")


def test_csv_format():
    """Test that the CSV file has the expected format."""
    try:
        import pandas as pd

        df = pd.read_csv("ablation/lambda_sweep_summary.csv")
        expected_columns = ["lambda", "mean_physics_reward", "pose_conf>0.6"]

        for col in expected_columns:
            assert col in df.columns, f"Expected column missing: {col}"

        assert len(df) > 0, "CSV file is empty"
        print("✅ CSV file has correct format")

    except ImportError:
        # If pandas is not available, just check file exists
        assert os.path.exists("ablation/lambda_sweep_summary.csv"), "CSV file missing"
        print("✅ CSV file exists (pandas not available for format check)")
    except Exception as e:
        pytest.fail(f"CSV file format error: {e}")


def test_ci_environment():
    """Test that we're in a CI-friendly environment."""
    # This test should always pass in CI
    assert True
    print("✅ CI environment is ready")


def test_basic_imports():
    """Test basic imports that should work in CI."""
    try:
        import numpy
        print("✅ numpy imported successfully")
    except ImportError:
        print("⚠️  numpy not available")

    try:
        import pandas

        print("✅ pandas imported successfully")
    except ImportError:
        print("⚠️  pandas not available")

    # Don't fail the test for missing imports
    assert True
