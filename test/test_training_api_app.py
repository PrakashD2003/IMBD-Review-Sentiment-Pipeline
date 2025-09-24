# test/test_training_api_app.py

import json
import pytest
from unittest.mock import patch, MagicMock, create_autospec
from fastapi.testclient import TestClient

# Ensure the app can be imported
from services.training.api_end_point.training_fastapi_app import app, training_manager
from services.training.scripts.dvc_traning_management_script import ProductionDVCTrainingManager

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

# --- Tests for /dvc_status ---

def test_get_dvc_status_success():
    """Tests the /dvc_status endpoint when 'dvc status' returns a JSON output."""
    mock_output = json.dumps({"data_ingestion": "changed"})
    mock_run = MagicMock(returncode=0, stdout=mock_output, stderr="")

    # Patch the class used by the app during startup
    with patch('services.training.api_end_point.training_fastapi_app.ProductionDVCTrainingManager') as mock_manager_cls:
        # Get a handle to the instance that WILL be created
        mock_manager_instance = mock_manager_cls.return_value
        # Configure its attributes
        mock_manager_instance.workspace = "/mock/workspace"
        
        # Now, start the app. It will use our pre-configured mock.
        with TestClient(app) as client, patch('subprocess.run', return_value=mock_run) as mock_subprocess:
            response = client.get("/dvc_status")
            
            assert response.status_code == 200
            data = response.json()
            assert data["dvc_status"] == {"data_ingestion": "changed"}
            mock_subprocess.assert_called_once()

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

def test_get_training_history():
    """Tests the /training_history endpoint with a correctly mocked S3 response."""
    mock_s3_response = {
        'CommonPrefixes': [
            {'Prefix': 'experiments/train_20250923_103000/'},
            {'Prefix': 'experiments/train_20250922_150000/'}
        ]
    }
    with patch('services.training.api_end_point.training_fastapi_app.ProductionDVCTrainingManager') as mock_manager_cls:
        mock_manager_instance = mock_manager_cls.return_value
        # Configure the s3_client *on the instance* to return our mock response
        mock_manager_instance.s3_client.list_objects_v2.return_value = mock_s3_response
        
        with TestClient(app) as client:
            response = client.get("/training_history")
            
            assert response.status_code == 200
            data = response.json()
            assert data["total_count"] == 2
            assert data["training_runs"] == ["train_20250923_103000", "train_20250922_150000"]
            mock_manager_instance.s3_client.list_objects_v2.assert_called_once()


# --- Tests for /reproduce/{training_id} ---

def test_reproduce_training_success():
    """Tests a successful reproduction request."""
    reproduce_output = {"status": "reproduction_successful", "original_training_id": "train_123"}
    with patch('services.training.api_end_point.training_fastapi_app.ProductionDVCTrainingManager') as mock_manager_cls:
        mock_manager_instance = mock_manager_cls.return_value
        # Configure the return value for the method
        mock_manager_instance.reproduce_training.return_value = reproduce_output
        
        with TestClient(app) as client:
            response = client.post("/reproduce/train_123")
            
            assert response.status_code == 200
            assert response.json() == reproduce_output
            mock_manager_instance.reproduce_training.assert_called_once_with("train_123")

def test_reproduce_training_failure_in_manager():
    """Tests a failed reproduction that is handled gracefully by the manager."""
    # The manager catches the error and returns a dict with status "reproduction_failed"
    reproduce_output = {"status": "reproduction_failed", "error": "DVC command failed"}
    with patch('services.training.api_end_point.training_fastapi_app.ProductionDVCTrainingManager') as mock_manager_cls:
        mock_manager_instance = mock_manager_cls.return_value
        mock_manager_instance.reproduce_training.return_value = reproduce_output
        
        with TestClient(app) as client:
            response = client.post("/reproduce/train_456")

            # The endpoint should now return 500 because we fixed it to check the status
            assert response.status_code == 500
            assert "DVC command failed" in response.json()["detail"]
            mock_manager_instance.reproduce_training.assert_called_once_with("train_456")


# --- Tests for /train_stream ---

def test_stream_training_logs_success():
    """
    Tests the streaming endpoint for a successful training run.
    """
    # 1. Define the mock generator that simulates the output
    def mock_stream_generator():
        yield {'type': 'log', 'data': 'Starting DVC repro...'}
        yield {'type': 'progress', 'data': {'current_stage': 'data_ingestion', 'completed': 0, 'total': 6}}
        yield {'type': 'log', 'data': 'Stage data_ingestion finished.'}

    # 2. Patch the manager class that the app will instantiate on startup
    with patch('services.training.api_end_point.training_fastapi_app.ProductionDVCTrainingManager') as mock_manager_cls:
        # 3. Pre-configure the instance that will be created
        #    Set its run_training_with_dvc method to return our mock generator
        mock_manager_instance = mock_manager_cls.return_value
        mock_manager_instance.run_training_with_dvc.return_value = mock_stream_generator()
        
        # 4. NOW, start the TestClient. The app will start and use our pre-configured mock.
        with TestClient(app) as client:
            response = client.get("/train_stream")
            
            # 5. Assert the results
            assert response.status_code == 200
            assert "text/event-stream" in response.headers['content-type']
            
            # Process the streaming response
            lines = response.text.split('\n\n')
            events = [line for line in lines if line] # Filter out empty lines
            
            # The test will now pass because the stream yields the correct events
            assert "data: Starting DVC repro..." in events[0]
            assert "event: progress" in events[1]
            assert "data: Stage data_ingestion finished." in events[2]
            
            # Verify the mocked method was called
            mock_manager_instance.run_training_with_dvc.assert_called_once()

# --- Tests for /pipeline_info ---

def test_get_pipeline_info_success():
    """Tests the /pipeline_info endpoint."""
    mock_dag_output = json.dumps({"nodes": ["data_ingestion"]})
    mock_run = MagicMock(returncode=0, stdout=mock_dag_output, stderr="")
    
    with patch('services.training.api_end_point.training_fastapi_app.ProductionDVCTrainingManager') as mock_manager_cls:
        mock_instance = mock_manager_cls.return_value
        # Configure all attributes the endpoint will access
        mock_instance.pipeline_version = "v1.0-mock"
        mock_instance.workspace = "/mock/workspace"
        mock_instance.s3_bucket = "mock-bucket"
        
        with TestClient(app) as client, patch('subprocess.run', return_value=mock_run):
            response = client.get("/pipeline_info")
            
            assert response.status_code == 200
            data = response.json()
            assert data["pipeline_version"] == "v1.0-mock"
            assert data["dag"] == {"nodes": ["data_ingestion"]}
