import hashlib
import pytest
import subprocess
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

from services.training.scripts.dvc_traning_management_script import ProductionDVCTrainingManager

# --- Fixtures ---

@pytest.fixture
def mock_env(monkeypatch):
    """Mocks environment variables required by the manager."""
    monkeypatch.setenv("WORKSPACE_DIR", "/app")
    monkeypatch.setenv("DVC_S3_BUCKET", "my-test-bucket")
    monkeypatch.setenv("PIPELINE_VERSION", "v1.0")
    monkeypatch.setenv("COMMIT_SHA", "test_sha")


@pytest.fixture
def manager(tmp_path):
    """Provides an instance of ProductionDVCTrainingManager with mocked dependencies."""
    # Mock the Path object used inside the class's __init__
    with patch('services.training.scripts.dvc_traning_management_script.Path') as mock_path, \
         patch('subprocess.run'), \
         patch('boto3.client'):
        
        # Make the mocked Path object return our temporary test path
        mock_path.return_value = tmp_path
        
        manager_instance = ProductionDVCTrainingManager()
        yield manager_instance, MagicMock()

# --- Test Cases ---

def test_setup_dvc_environment(manager):
    """
    Verifies that the __init__ method correctly sets up the DVC environment.
    This is implicitly tested by the `manager` fixture, but this test makes it explicit.
    """
    assert manager.workspace == Path("/app")
    assert manager.s3_bucket == "my-test-bucket"
    assert manager.pipeline_version == "v1.0"
    # Assert that the s3 client was configured and used for downloads
    assert manager.s3_client.download_file.called

def test_download_pipeline_configuration(manager):
    """
    Tests the logic for downloading pipeline files from S3.
    """
    # The manager fixture already calls this. We can simply assert the calls were correct.
    manager.s3_client.download_file.assert_any_call(
        "my-test-bucket", "pipeline-configs/v1.0/dvc.yaml", "/app/dvc.yaml"
    )
    manager.s3_client.download_file.assert_any_call(
        "my-test-bucket", "pipeline-configs/v1.0/params.yaml", "/app/params.yaml"
    )
    manager.s3_client.download_file.assert_any_call(
        "my-test-bucket", "pipeline-configs/v1.0/dvc.lock", "/app/dvc.lock"
    )

def test_upload_pipeline_configuration(manager):
    """
    Tests the logic for uploading pipeline files to S3.
    """
    # Simulate that the files exist in the workspace
    with patch('pathlib.Path.exists', return_value=True):
        manager.upload_pipeline_configuration()

    manager.s3_client.upload_file.assert_any_call(
        "/app/dvc.yaml", "my-test-bucket", "pipeline-configs/v1.0/dvc.yaml"
    )
    manager.s3_client.upload_file.assert_any_call(
        "/app/dvc.lock", "my-test-bucket", "pipeline-configs/v1.0/dvc.lock"
    )
    manager.s3_client.upload_file.assert_any_call(
        "/app/params.yaml", "my-test-bucket", "pipeline-configs/v1.0/params.yaml"
    )

def test_generate_training_fingerprint(manager):
    """
    Tests the creation of a training fingerprint.
    """
    # Mock file reads
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
    assert fingerprint["params_hash"] == hashlib.md5(mock_params.encode()).hexdigest()
    assert fingerprint["data_fingerprints"] == {"status": "mock_status"}

def test_run_training_with_dvc_success(manager):
    """
    Tests the successful execution of the DVC training pipeline, including streaming.
    """
    # Mock subprocess.Popen to simulate a successful run
    mock_process = MagicMock()
    mock_process.stdout.readline.side_effect = [
        "Running stage 'data_ingestion'\n",
        "Stage 'data_ingestion' finished.\n",
        "" # End of stream
    ]
    mock_process.wait.return_value = None
    mock_process.returncode = 0

    with patch('subprocess.Popen', return_value=mock_process) as mock_popen:
        # Consume the generator to trigger the logic
        events = list(manager.run_training_with_dvc())
    
    # Check that dvc repro and dvc push were called
    mock_popen.assert_any_call(
        ["dvc", "repro", "-f", "-v"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=manager.workspace, bufsize=1
    )
    mock_popen.assert_any_call(
        ["dvc", "push"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=manager.workspace, bufsize=1
    )

    # Check the streamed events
    assert any("Training ID" in e['data'] for e in events if e['type'] == 'log')
    assert any(e['type'] == 'progress' and e['data']['current_stage'] == "Running: data_ingestion" for e in events)
    assert any("DVC pipeline completed successfully" in e['data'] for e in events if e['type'] == 'log')
    assert any("Artifacts pushed to remote storage" in e['data'] for e in events if e['type'] == 'log')

def test_run_training_with_dvc_failure(manager):
    """
    Tests the behavior when the 'dvc repro' command fails.
    """
    mock_process = MagicMock()
    mock_process.stdout.readline.side_effect = ["DVC failed miserably\n", ""]
    mock_process.wait.return_value = None
    mock_process.returncode = 1 # Simulate failure

    with patch('subprocess.Popen', return_value=mock_process):
        # The generator should raise an exception, which pytest can catch
        with pytest.raises(Exception, match="DVC pipeline failed with exit code 1"):
            list(manager.run_training_with_dvc())

def test_reproduce_training(manager):
    """
    Tests the logic for reproducing a previous training run.
    """
    training_id = "train_test_123"
    with patch('subprocess.run') as mock_run:
        # Simulate successful subprocess calls
        mock_run.return_value = MagicMock(returncode=0, stdout="Success")
        result = manager.reproduce_training(training_id)
    
    # Check that S3 downloads were called
    manager.s3_client.download_file.assert_any_call(
        manager.s3_bucket, f"experiments/{training_id}/dvc.lock", manager.workspace / "dvc.lock"
    )
    
    # Check that DVC commands were executed in the correct order
    mock_run.assert_any_call(["dvc", "pull"], check=True, cwd=manager.workspace)
    mock_run.assert_any_call(["dvc", "checkout"], check=True, cwd=manager.workspace)
    mock_run.assert_any_call(["dvc", "repro", "-f"], capture_output=True, text=True, cwd=manager.workspace)

    assert result["status"] == "reproduction_successful"
    assert result["original_training_id"] == training_id