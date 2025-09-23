import hashlib
import pytest
import subprocess
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

from services.training.scripts.dvc_traning_management_script import ProductionDVCTrainingManager

# --- Fixtures ---

@pytest.fixture
def manager(tmp_path, monkeypatch):
    """
    Provides a fully mocked instance of ProductionDVCTrainingManager for testing.
    It sets necessary environment variables and mocks external systems like subprocess and S3.
    """
    # 1. Set all required environment variables for the test environment
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("DVC_S3_BUCKET", "my-test-bucket")
    monkeypatch.setenv("PIPELINE_VERSION", "v1.0")
    monkeypatch.setenv("COMMIT_SHA", "test_sha")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

    # 2. Mock external dependencies (subprocess, boto3)
    with patch('subprocess.run') as mock_run, \
         patch('boto3.client') as mock_boto:
        
        # Configure mocks
        mock_run.return_value = MagicMock(returncode=0)
        mock_s3_client = MagicMock()
        mock_boto.return_value = mock_s3_client
        
        # Instantiate the manager, which will use the mocked dependencies
        manager_instance = ProductionDVCTrainingManager()
        
        # Attach the mock S3 client to the instance for easy access in tests
        manager_instance.s3_client = mock_s3_client
        
        # Yield only the manager instance
        yield manager_instance

# --- Test Cases ---

def test_setup_dvc_environment(manager):
    """
    Verifies that the __init__ method correctly sets up instance attributes
    based on the mocked environment variables.
    """
    assert manager.s3_bucket == "my-test-bucket"
    assert manager.pipeline_version == "v1.0"
    # Assert that the S3 client was used to download config files
    assert manager.s3_client.download_file.call_count > 0

def test_download_pipeline_configuration(manager):
    """
    Tests that the logic correctly attempts to download pipeline files from S3.
    """
    workspace_path_str = str(manager.workspace)
    manager.s3_client.download_file.assert_any_call(
        "my-test-bucket", "pipeline-configs/v1.0/dvc.yaml", f"{workspace_path_str}/dvc.yaml"
    )
    manager.s3_client.download_file.assert_any_call(
        "my-test-bucket", "pipeline-configs/v1.0/params.yaml", f"{workspace_path_str}/params.yaml"
    )

def test_upload_pipeline_configuration(manager):
    """
    Tests the logic for uploading pipeline configuration files to S3.
    """
    # Simulate that the files exist in the (temporary) workspace
    with patch('pathlib.Path.exists', return_value=True):
        manager.upload_pipeline_configuration()

    workspace_path_str = str(manager.workspace)
    manager.s3_client.upload_file.assert_any_call(
        f"{workspace_path_str}/dvc.lock", "my-test-bucket", f"pipeline-configs/v1.0/{manager.commit_sha}/dvc.lock"
    )

def test_generate_training_fingerprint(manager):
    """
    Tests the creation of a unique training fingerprint.
    """
    mock_dvc_lock = "stages: ..."
    mock_params = "train_size: 0.8"
    m = mock_open()
    m.side_effect = [
        mock_open(read_data=mock_dvc_lock).return_value,
        mock_open(read_data=mock_params).return_value
    ]

    with patch('builtins.open', m), \
         patch('pathlib.Path.exists', return_value=True), \
         patch.object(manager, 'get_dvc_data_fingerprints', return_value={"status": "mock_status"}):
        
        fingerprint = manager.generate_training_fingerprint()

    assert fingerprint["pipeline_version"] == "v1.0"
    assert fingerprint["commit_sha"] == "test_sha"
    assert "training_id" in fingerprint
    assert fingerprint["dvc_lock_hash"] == hashlib.md5(mock_dvc_lock.encode()).hexdigest()

def test_run_training_with_dvc_success(manager):
    """
    Tests the successful execution of the DVC training pipeline.
    """
    mock_process = MagicMock()
    mock_process.stdout.readline.side_effect = ["Running stage 'data_ingestion'\n", ""]
    mock_process.wait.return_value = None
    mock_process.returncode = 0

    with patch('subprocess.Popen', return_value=mock_process) as mock_popen:
        events = list(manager.run_training_with_dvc())
    
    # Check that 'dvc repro' and 'dvc push' were called
    assert mock_popen.call_count == 2
    mock_popen.assert_any_call(
        ["dvc", "repro", "-f", "-v"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=manager.workspace, bufsize=1
    )
    mock_popen.assert_any_call(
        ["dvc", "push"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=manager.workspace, bufsize=1
    )

def test_run_training_with_dvc_failure(manager):
    """
    Tests that an exception is raised when the 'dvc repro' command fails.
    """
    mock_process = MagicMock()
    mock_process.stdout.readline.side_effect = ["DVC failed miserably\n", ""]
    mock_process.wait.return_value = None
    mock_process.returncode = 1  # Simulate failure

    with patch('subprocess.Popen', return_value=mock_process):
        with pytest.raises(Exception, match="DVC pipeline failed with exit code 1"):
            list(manager.run_training_with_dvc())

def test_reproduce_training(manager):
    """
    Tests the logic for reproducing a previous training run.
    """
    training_id = "train_test_123"
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="Success")
        result = manager.reproduce_training(training_id)
    
    manager.s3_client.download_file.assert_any_call(
        "my-test-bucket", f"experiments/{training_id}/dvc.lock", manager.workspace / "dvc.lock"
    )
    mock_run.assert_any_call(["dvc", "pull"], check=True, cwd=manager.workspace)
    assert result["status"] == "reproduction_successful"