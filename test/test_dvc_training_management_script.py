import hashlib
import pytest
import subprocess
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

from services.training.scripts.dvc_traning_management_script import ProductionDVCTrainingManager

# --- Fixtures ---

import hashlib
import pytest
import subprocess
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

from services.training.scripts.dvc_traning_management_script import ProductionDVCTrainingManager

@pytest.fixture
def manager(tmp_path, monkeypatch):
    """Provides a fully mocked instance of ProductionDVCTrainingManager."""
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("DVC_S3_BUCKET", "my-test-bucket")
    monkeypatch.setenv("PIPELINE_VERSION", "v1.0")
    monkeypatch.setenv("COMMIT_SHA", "test_sha") # <-- Added missing env var
    # ... other env vars

    with patch('subprocess.run') as mock_run, \
         patch('boto3.client') as mock_boto:
        
        mock_s3_client = MagicMock()
        mock_boto.return_value = mock_s3_client
        
        manager_instance = ProductionDVCTrainingManager()
        # Attach the mock to the instance for easy access
        manager_instance.s3_client = mock_s3_client 
        
        # Yield both for tests that need them
        yield manager_instance, mock_s3_client





# --- Test Cases ---

def test_setup_dvc_environment(manager):
    """
    Verifies that the __init__ method correctly sets up instance attributes
    based on the mocked environment variables.
    """
    manager_instance, _ = manager
    assert manager_instance.s3_bucket == "my-test-bucket"
    assert manager_instance.pipeline_version == "v1.0"
    # Assert that the S3 client was used to download config files
    assert manager_instance.s3_client.download_file.call_count > 0

def test_download_pipeline_configuration(manager):
    """
    Tests that the logic correctly attempts to download pipeline files from S3.
    """
    manager_instance, mock_s3_client = manager
    workspace_path_str = str(manager_instance.workspace)
    mock_s3_client.download_file.assert_any_call(
        "my-test-bucket", 
        f"pipeline-configs/v1.0/{manager_instance.commit_sha}/dvc.yaml", 
        f"{workspace_path_str}/dvc.yaml"
    )
    mock_s3_client.download_file.assert_any_call(
        "my-test-bucket", 
        f"pipeline-configs/v1.0/{manager_instance.commit_sha}/params.yaml", 
        f"{workspace_path_str}/params.yaml"
    )

def test_upload_pipeline_configuration(manager):
    manager_instance, mock_s3_client = manager
    with patch('pathlib.Path.exists', return_value=True):
        manager_instance.upload_pipeline_configuration()

    workspace_path_str = str(manager_instance.workspace)
    # The key now correctly includes the commit_sha
    expected_key = f"pipeline-configs/v1.0/{manager_instance.commit_sha}/dvc.lock"
    mock_s3_client.upload_file.assert_any_call(
        f"{workspace_path_str}/dvc.lock", "my-test-bucket", expected_key
    )

def test_generate_training_fingerprint(manager):
    """
    Tests the creation of a unique training fingerprint.
    """
    manager_instance, _ = manager
    mock_dvc_lock = "stages: ..."
    mock_params = "train_size: 0.8"
    m = mock_open()
    m.side_effect = [
        mock_open(read_data=mock_dvc_lock).return_value,
        mock_open(read_data=mock_params).return_value
    ]

    with patch('builtins.open', m), \
         patch('pathlib.Path.exists', return_value=True), \
         patch.object(manager_instance, 'get_dvc_data_fingerprints', return_value={"status": "mock_status"}):
        
        fingerprint = manager_instance.generate_training_fingerprint()

    assert fingerprint["pipeline_version"] == "v1.0"
    assert fingerprint["commit_sha"] == "test_sha"
    assert "training_id" in fingerprint
    assert fingerprint["dvc_lock_hash"] == hashlib.md5(mock_dvc_lock.encode()).hexdigest()

def test_run_training_with_dvc_success(manager):
    """
    Tests the successful execution of the DVC training pipeline.
    """
    manager_instance, _ = manager
    mock_process = MagicMock()
    mock_process.stdout.readline.side_effect = ["Running stage 'data_ingestion'\n", ""]
    mock_process.wait.return_value = None
    mock_process.returncode = 0

    with patch('subprocess.Popen', return_value=mock_process) as mock_popen:
        events = list(manager_instance.run_training_with_dvc())
    
    # Check that 'dvc repro' and 'dvc push' were called
    assert mock_popen.call_count == 2
    mock_popen.assert_any_call(
        ["dvc", "repro", "-f", "-v"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=manager_instance.workspace, bufsize=1
    )
    mock_popen.assert_any_call(
        ["dvc", "push"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=manager_instance.workspace, bufsize=1
    )

def test_run_training_with_dvc_failure(manager):
    """
    Tests that an exception is raised when the 'dvc repro' command fails.
    """
    manager_instance, _ = manager
    mock_process = MagicMock()
    mock_process.stdout.readline.side_effect = ["DVC failed miserably\n", ""]
    mock_process.wait.return_value = None
    mock_process.returncode = 1  # Simulate failure

    with patch('subprocess.Popen', return_value=mock_process):
        with pytest.raises(Exception, match="DVC pipeline failed with exit code 1"):
            list(manager_instance.run_training_with_dvc())

def test_reproduce_training(manager):
    """
    Tests the entire training reproduction process.

    Verifies that:
    1. The correct pipeline configuration files (dvc.lock, dvc.yaml, params.yaml) are downloaded from the experiment's S3 prefix.
    2. DVC commands `pull`, `checkout`, and `repro` are executed in the correct order.
    """
    manager_instance, mock_s3_client = manager
    training_id = "train_test_123"
    workspace_path_str = str(manager_instance.workspace)

    # Mock the subprocess.run command to simulate DVC calls
    with patch('subprocess.run') as mock_run:
        # Configure the mock to simulate successful execution
        mock_run.return_value = MagicMock(returncode=0, stdout="Success", stderr="")
        
        # --- Act ---
        # Call the method under test
        result = manager_instance.reproduce_training(training_id)

    # --- Assert ---
    # 1. Verify that the correct pipeline files were downloaded from S3
    base_prefix = f"experiments/{training_id}/"
    mock_s3_client.download_file.assert_any_call(
        "my-test-bucket", f"{base_prefix}dvc.lock", f"{workspace_path_str}/dvc.lock"
    )
    mock_s3_client.download_file.assert_any_call(
        "my-test-bucket", f"{base_prefix}dvc.yaml", f"{workspace_path_str}/dvc.yaml"
    )
    mock_s3_client.download_file.assert_any_call(
        "my-test-bucket", f"{base_prefix}params.yaml", f"{workspace_path_str}/params.yaml"
    )

    # 2. Verify that the DVC commands were executed
    expected_calls = [
        # The order of these first two calls is important
        call(['dvc', 'pull'], check=True, cwd=manager_instance.workspace),
        call(['dvc', 'checkout'], check=True, cwd=manager_instance.workspace),
        # The final 'repro' call executes the pipeline
        call(['dvc', 'repro', '-f'], capture_output=True, text=True, cwd=manager_instance.workspace)
    ]
    mock_run.assert_has_calls(expected_calls, any_order=False)

    # 3. Verify the structure of the successful return value
    assert result["status"] == "reproduction_successful"
    assert result["original_training_id"] == training_id
    assert "reproduction_timestamp" in result
    