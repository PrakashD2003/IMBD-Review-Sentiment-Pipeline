# test/test_training_api_app.py

import json
import pytest
from unittest.mock import patch, MagicMock, create_autospec
from fastapi.testclient import TestClient
import subprocess

# Ensure the app can be imported
from services.training.api_end_point.training_fastapi_app import app, training_manager
from services.training.scripts.dvc_traning_management_script import ProductionDVCTrainingManager

# Use a pytest fixture to initialize the TestClient once per module
@pytest.fixture(scope="module")
def client():
    """Provides a FastAPI TestClient for making requests to the app."""
    with TestClient(app) as c:
        yield c

# --- Tests for /dvc_status ---

def test_get_dvc_status_success(client):
    """
    Tests the /dvc_status endpoint when 'dvc status' returns a JSON output.
    """
    mock_output = json.dumps({"data_ingestion": "changed"})
    mock_run = MagicMock(returncode=0, stdout=mock_output, stderr="")

    with patch('subprocess.run', return_value=mock_run) as mock_subprocess:
        response = client.get("/dvc_status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["dvc_status"] == {"data_ingestion": "changed"}
        # Verify that the correct command was called
        mock_subprocess.assert_called_once_with(
            ["dvc", "status", "--json"],
            capture_output=True, text=True, cwd=training_manager.workspace
        )

def test_get_dvc_status_up_to_date(client):
    """
    Tests the /dvc_status endpoint when the pipeline is up-to-date (empty stdout).
    """
    mock_run = MagicMock(returncode=0, stdout="", stderr="")
    with patch('subprocess.run', return_value=mock_run):
        response = client.get("/dvc_status")
        
        assert response.status_code == 200
        assert response.json()["dvc_status"] == {"status": "up_to_date"}

def test_get_dvc_status_command_error(client):
    """
    Tests the /dvc_status endpoint when the dvc command fails.
    """
    mock_run = MagicMock(returncode=1, stdout="", stderr="DVC error")
    with patch('subprocess.run', return_value=mock_run):
        response = client.get("/dvc_status")

        assert response.status_code == 200
        assert response.json()["dvc_status"] == {"error": "DVC error"}

# --- Tests for /training_history ---

def test_get_training_history(client):
    """
    Tests the /training_history endpoint by mocking the S3 client response.
    """
    # This simulates the response from s3_client.list_objects_v2
    mock_s3_response = {
        'CommonPrefixes': [
            {'Prefix': 'experiments/train_20250923_103000/'},
            {'Prefix': 'experiments/train_20250922_150000/'}
        ]
    }
    
    # We need to patch the s3_client attribute within the training_manager instance
    with patch.object(training_manager, 's3_client', new_callable=MagicMock) as mock_s3:
        mock_s3.list_objects_v2.return_value = mock_s3_response
        response = client.get("/training_history")

        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 2
        # It should sort them in reverse order
        assert data["training_runs"] == ["train_20250923_103000", "train_20250922_150000"]
        mock_s3.list_objects_v2.assert_called_once()

# --- Tests for /reproduce/{training_id} ---

def test_reproduce_training_success(client):
    """
    Tests a successful reproduction request.
    """
    reproduce_output = {"status": "reproduction_successful", "original_training_id": "train_123"}
    # Patch the method on the manager instance itself
    with patch.object(training_manager, 'reproduce_training', return_value=reproduce_output) as mock_reproduce:
        response = client.post("/reproduce/train_123")
        
        assert response.status_code == 200
        assert response.json() == reproduce_output
        mock_reproduce.assert_called_once_with("train_123")

def test_reproduce_training_failure(client):
    """
    Tests a failed reproduction request.
    """
    with patch.object(training_manager, 'reproduce_training', side_effect=Exception("Repro failed")) as mock_reproduce:
        response = client.post("/reproduce/train_456")

        assert response.status_code == 500
        assert response.json() == {"detail": "Repro failed"}
        mock_reproduce.assert_called_once_with("train_456")

# --- Tests for /train_stream ---

def test_stream_training_logs_success(client):
    """
    Tests the streaming endpoint for a successful training run.
    """
    # This generator simulates the output of run_training_with_dvc
    def mock_stream_generator():
        yield {'type': 'log', 'data': 'Starting DVC repro...'}
        yield {'type': 'progress', 'data': {'current_stage': 'data_ingestion', 'completed': 0, 'total': 6}}
        yield {'type': 'log', 'data': 'Stage data_ingestion finished.'}
        yield {'type': 'log', 'data': 'Pushing artifacts to S3...'}
    
    with patch.object(training_manager, 'run_training_with_dvc', return_value=mock_stream_generator()) as mock_stream:
        response = client.get("/train_stream")
        
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