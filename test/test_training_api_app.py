# test/test_training_api_app.py

import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Import the app object
from services.training.api_end_point.training_fastapi_app import app

# --- Helper to set environment variables ---
@pytest.fixture(autouse=True)
def set_env_vars(monkeypatch):
    """Automatically sets required environment variables for every test."""
    monkeypatch.setenv("DVC_S3_BUCKET", "test-bucket")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("PIPELINE_VERSION", "v1.0-test")
    monkeypatch.setenv("COMMIT_SHA", "test_sha")

@pytest.fixture
def client():
    """
    Provides a TestClient and a correctly injected mock for ProductionDVCTrainingManager.
    
    This fixture patches the actual 'training_manager' global variable that the FastAPI
    app uses, ensuring our mock is the one being called by the endpoints.
    """
    # Create a mock instance first
    mock_manager_instance = MagicMock()

    # Patch the global variable in the app's module with our instance
    with patch('services.training.api_end_point.training_fastapi_app.training_manager', mock_manager_instance):
        with TestClient(app) as test_client:
            yield test_client, mock_manager_instance

# --- Tests for /dvc_status ---

def test_get_dvc_status_success(client):
    """Tests the /dvc_status endpoint when 'dvc status' returns a JSON output."""
    test_client, mock_manager = client
    mock_manager.workspace = "/mock/workspace"
    
    mock_output = json.dumps({"data_ingestion": "changed"})
    mock_run = MagicMock(returncode=0, stdout=mock_output, stderr="")
    
    with patch('subprocess.run', return_value=mock_run) as mock_subprocess:
        response = test_client.get("/dvc_status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["dvc_status"] == {"data_ingestion": "changed"}
        mock_subprocess.assert_called_once()

def test_get_dvc_status_up_to_date(client):
    """Tests the /dvc_status endpoint when the pipeline is up-to-date."""
    test_client, mock_manager = client
    mock_manager.workspace = "/mock/workspace"
    mock_run = MagicMock(returncode=0, stdout="", stderr="")
    
    with patch('subprocess.run', return_value=mock_run):
        response = test_client.get("/dvc_status")
        
        assert response.status_code == 200
        assert response.json()["dvc_status"] == {"status": "up_to_date"}

def test_get_dvc_status_command_error(client):
    """Tests the /dvc_status endpoint when the dvc command fails."""
    test_client, mock_manager = client
    mock_manager.workspace = "/mock/workspace"
    mock_run = MagicMock(returncode=1, stdout="", stderr="DVC error")
    
    with patch('subprocess.run', return_value=mock_run):
        response = test_client.get("/dvc_status")

        assert response.status_code == 200
        assert response.json()["dvc_status"] == {"error": "DVC error"}

# --- Tests for /training_history ---

def test_get_training_history(client):
    """Tests the /training_history endpoint."""
    test_client, mock_manager = client
    mock_s3_response = {
        'CommonPrefixes': [
            {'Prefix': 'experiments/train_20250923_103000/'},
            {'Prefix': 'experiments/train_20250922_150000/'}
        ]
    }
    # Configure the mock s3_client on our manager instance
    mock_manager.s3_client.list_objects_v2.return_value = mock_s3_response
    
    response = test_client.get("/training_history")
    
    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] == 2
    assert data["training_runs"] == ["train_20250923_103000", "train_20250922_150000"]
    mock_manager.s3_client.list_objects_v2.assert_called_once()

# --- Tests for /reproduce/{training_id} ---

def test_reproduce_training_success(client):
    """Tests a successful reproduction request."""
    test_client, mock_manager = client
    reproduce_output = {"status": "reproduction_successful", "original_training_id": "train_123"}
    mock_manager.reproduce_training.return_value = reproduce_output
    
    response = test_client.post("/reproduce/train_123")
    
    assert response.status_code == 200
    assert response.json() == reproduce_output
    mock_manager.reproduce_training.assert_called_once_with("train_123")

def test_reproduce_training_failure_in_manager(client):
    """Tests a failed reproduction that is handled gracefully by the manager."""
    test_client, mock_manager = client
    reproduce_output = {"status": "reproduction_failed", "error": "DVC command failed"}
    mock_manager.reproduce_training.return_value = reproduce_output
    
    response = test_client.post("/reproduce/train_456")

    assert response.status_code == 500
    assert "DVC command failed" in response.json()["detail"]
    mock_manager.reproduce_training.assert_called_once_with("train_456")

# --- Tests for /train_stream ---

def test_stream_training_logs_success(client):
    """Tests the streaming endpoint for a successful training run."""
    test_client, mock_manager = client
    
    def mock_stream_generator():
        yield {'type': 'log', 'data': 'Starting DVC repro...'}
        yield {'type': 'progress', 'data': {'current_stage': 'data_ingestion', 'completed': 0, 'total': 6}}
        yield {'type': 'log', 'data': 'Stage data_ingestion finished.'}

    mock_manager.run_training_with_dvc.return_value = mock_stream_generator()
    
    response = test_client.get("/train_stream")
    
    assert response.status_code == 200
    assert "text/event-stream" in response.headers['content-type']
    
    lines = response.text.split('\n\n')
    events = [line for line in lines if line]
    
    assert "data: Starting DVC repro..." in events[0]
    assert "event: progress" in events[1]
    assert "data: Stage data_ingestion finished." in events[2]
    
    mock_manager.run_training_with_dvc.assert_called_once()

# --- Tests for /pipeline_info ---

def test_get_pipeline_info_success(client):
    """Tests the /pipeline_info endpoint."""
    test_client, mock_manager = client
    mock_dag_output = json.dumps({"nodes": ["data_ingestion"]})
    mock_run = MagicMock(returncode=0, stdout=mock_dag_output, stderr="")
    
    mock_manager.pipeline_version = "v1.0-mock"
    mock_manager.workspace = "/mock/workspace"
    mock_manager.s3_bucket = "mock-bucket"
    
    with patch('subprocess.run', return_value=mock_run):
        response = test_client.get("/pipeline_info")
        
        assert response.status_code == 200
        data = response.json()
        assert data["pipeline_version"] == "v1.0-mock"
        assert data["dag"] == {"nodes": ["data_ingestion"]}