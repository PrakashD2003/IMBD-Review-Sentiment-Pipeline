# test/test_training_api_app.py

import json
import pytest
from unittest.mock import patch, MagicMock, create_autospec
from fastapi.testclient import TestClient

# Ensure the app can be imported
from services.training.api_end_point.training_fastapi_app import app, training_manager
from services.training.scripts.dvc_traning_management_script import ProductionDVCTrainingManager

@pytest.fixture(scope="function") 
def client(monkeypatch):
    """Provides a FastAPI TestClient for making requests to the app."""
    # Set required environment variables for the test session
    monkeypatch.setenv("DVC_S3_BUCKET", "test-bucket")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    
    with patch('services.training.api_end_point.training_fastapi_app.ProductionDVCTrainingManager') as mock_manager_cls, \
         TestClient(app) as c:
        # Yield the client and the *instance* of the mocked manager
        yield c, mock_manager_cls.return_value

# --- Tests for /dvc_status ---

def test_get_dvc_status_success(client):
    """
    Tests the /dvc_status endpoint when 'dvc status' returns a JSON output.
    """
    test_client, mock_training_manager = client
    mock_output = json.dumps({"data_ingestion": "changed"})
    mock_run = MagicMock(returncode=0, stdout=mock_output, stderr="")

    with patch('subprocess.run', return_value=mock_run) as mock_subprocess:
        response = test_client.get("/dvc_status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["dvc_status"] == {"data_ingestion": "changed"}
        # Verify that the correct command was called
        mock_subprocess.assert_called_once_with(
            ["dvc", "status", "--json"],
            capture_output=True, text=True, cwd=mock_training_manager.workspace 
        )

def test_get_dvc_status_up_to_date(client):
    """
    Tests the /dvc_status endpoint when the pipeline is up-to-date (empty stdout).
    """
    test_client, _ = client
    mock_run = MagicMock(returncode=0, stdout="", stderr="")
    with patch('subprocess.run', return_value=mock_run):
        response = test_client.get("/dvc_status")
        
        assert response.status_code == 200
        assert response.json()["dvc_status"] == {"status": "up_to_date"}

def test_get_dvc_status_command_error(client):
    """
    Tests the /dvc_status endpoint when the dvc command fails.
    """
    test_client, _ = client
    mock_run = MagicMock(returncode=1, stdout="", stderr="DVC error")
    with patch('subprocess.run', return_value=mock_run):
        response = test_client.get("/dvc_status")

        assert response.status_code == 200
        assert response.json()["dvc_status"] == {"error": "DVC error"}

# --- Tests for /training_history ---

def test_get_training_history(client):
    """
    Tests the /training_history endpoint by mocking the S3 client response.
    """
    test_client, mock_training_manager = client
    
    # This is the correctly structured mock S3 response
    mock_s3_response = {
        'CommonPrefixes': [
            {'Prefix': 'experiments/train_20250923_103000/'},
            {'Prefix': 'experiments/train_20250922_150000/'}
        ]
    }
    
    with patch.object(mock_training_manager, 's3_client', new_callable=MagicMock) as mock_s3:
        # Set the return value of the mocked method
        mock_s3.list_objects_v2.return_value = mock_s3_response
        
        response = test_client.get("/training_history")

        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 2
        assert data["training_runs"] == ["train_20250923_103000", "train_20250922_150000"]
        
        # Verify that the S3 client was called with the correct parameters
        mock_s3.list_objects_v2.assert_called_once_with(
            Bucket=mock_training_manager.s3_bucket,
            Prefix="experiments/",
            Delimiter="/"
        )

# --- Tests for /reproduce/{training_id} ---

def test_reproduce_training_success(client):
    """
    Tests a successful reproduction request.
    """
    test_client, mock_training_manager = client
    reproduce_output = {"status": "reproduction_successful", "original_training_id": "train_123"}
    # Patch the method on the manager instance itself
    with patch.object(mock_training_manager, 'reproduce_training', return_value=reproduce_output) as mock_reproduce:
        response = test_client.post("/reproduce/train_123")
        
        assert response.status_code == 200
        assert response.json() == reproduce_output
        mock_reproduce.assert_called_once_with("train_123")

def test_reproduce_training_failure(client):
    """
    Tests a failed reproduction request.
    """
    test_client, mock_training_manager = client
    with patch.object(mock_training_manager, 'reproduce_training', side_effect=Exception("Repro failed")) as mock_reproduce:
        response = test_client.post("/reproduce/train_456")

        assert response.status_code == 500
        assert response.json() == {"detail": "Repro failed"}
        mock_reproduce.assert_called_once_with("train_456")

# --- Tests for /train_stream ---

def test_stream_training_logs_success(client):
    """
    Tests the streaming endpoint for a successful training run.
    """
    test_client, mock_training_manager = client
    # This generator simulates the output of run_training_with_dvc
    def mock_stream_generator():
        yield {'type': 'log', 'data': 'Starting DVC repro...'}
        yield {'type': 'progress', 'data': {'current_stage': 'data_ingestion', 'completed': 0, 'total': 6}}
        yield {'type': 'log', 'data': 'Stage data_ingestion finished.'}
        yield {'type': 'log', 'data': 'Pushing artifacts to S3...'}
    
    with patch.object(mock_training_manager, 'run_training_with_dvc', return_value=mock_stream_generator()) as mock_stream:
        response = test_client.get("/train_stream")
        
        assert response.status_code == 200
        assert "text/event-stream" in response.headers['content-type']
        
        # Process the streaming response
        lines = response.text.split('\n\n')
        
        # Filter out empty lines
        events = [line for line in lines if line]
        
        assert events[0] == "data: Starting DVC repro..."
        assert events[1] == 'event: progress\ndata: {"current_stage": "data_ingestion", "completed": 0, "total": 6}'
        assert events[2] == "data: Stage data_ingestion finished."
        assert events[3] == "data: Pushing artifacts to S3..."
        assert events[4] == "event: end\ndata: done"
        mock_stream.assert_called_once()

# --- Tests for /pipeline_info ---

def test_get_pipeline_info_success(client):
    """
    Tests the /pipeline_info endpoint when 'dvc dag' returns a JSON output.
    """
    test_client, mock_training_manager = client
    mock_dag_output = json.dumps({"nodes": ["data_ingestion", "model_training"]})
    mock_run = MagicMock(returncode=0, stdout=mock_dag_output, stderr="")

    with patch('subprocess.run', return_value=mock_run) as mock_subprocess:
        response = test_client.get("/pipeline_info")
        
        assert response.status_code == 200
        data = response.json()
        assert data["pipeline_version"] == mock_training_manager.pipeline_version
        assert data["dag"] == {"nodes": ["data_ingestion", "model_training"]}
        # Verify that the correct command was called
        mock_subprocess.assert_called_once_with(
            ["dvc", "dag", "--json"],
            capture_output=True, text=True, cwd=mock_training_manager.workspace 
        )