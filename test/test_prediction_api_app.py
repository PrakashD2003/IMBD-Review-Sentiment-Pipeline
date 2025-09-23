import pytest
from starlette.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd

# Import the FastAPI app object from your service
from services.prediction.api_end_point.prediction_fastapi_app import app

@pytest.fixture(scope="function")
def client(monkeypatch):
    """
    Provides a fully mocked FastAPI TestClient for the prediction app.
    This fixture isolates the app from file system (params.yaml) and network (MLflow)
    dependencies, which is a best practice for unit testing.
    """
    # 1. Set dummy environment variables that the app's config might need
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setenv("VECTORIZER_NAME", "test-vectorizer")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    monkeypatch.setenv("DAGSHUB_REPO_OWNER", "test")
    monkeypatch.setenv("DAGSHUB_REPO_NAME", "test")

    # 2. Mock all external dependencies required by the app on startup
    with patch('services.prediction.pipelines.unified_prediction_pipeline.load_params', return_value={}) as mock_load_params, \
         patch('services.prediction.pipelines.unified_prediction_pipeline.get_latest_model', return_value=MagicMock()) as mock_get_model, \
         TestClient(app) as c:
        yield c

# --- Test Cases ---

def test_health_check_endpoint(client):
    """Tests the /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_endpoint(client):
    """Tests a successful prediction from the /predict endpoint."""
    payload = {"reviews": ["this is a fantastic movie", "this is a terrible movie"]}
    
    # We need to mock the pipeline's predict_single method for the test
    with patch('services.prediction.api_end_point.prediction_fastapi_app.pipeline.predict_single') as mock_predict:
        # Define the mock return value
        mock_predict.return_value = pd.DataFrame({
            "review": payload["reviews"],
            "sentiment": ["positive", "negative"],
            "probability": [0.99, 0.98]
        })
        
        response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["sentiment"] == "positive"

def test_batch_predict_endpoint(client):
    """Tests a successful prediction from the /batch_predict endpoint."""
    reviews = ["good movie", "bad movie"]
    files = {'file': ('reviews.csv', '\n'.join(reviews), 'text/csv')}

    with patch('services.prediction.api_end_point.prediction_fastapi_app.pipeline.predict_batch') as mock_predict:
        mock_predict.return_value = pd.DataFrame({
            "review": reviews,
            "sentiment": ["positive", "negative"],
            "probability": [0.9, 0.8]
        })
        response = client.post("/batch_predict", files=files)

    assert response.status_code == 200
    assert "predictions.csv" in response.headers['content-disposition']

@pytest.mark.parametrize("invalid_payload", [
    {"reviews": []},                   # Empty list
    {"reviews": "not a list"},         # Wrong data type
    {"data": ["a review"]},            # Wrong key
    {},                                # Empty JSON
])
def test_predict_endpoint_invalid_input(client, invalid_payload):
    """Tests that the /predict endpoint handles invalid payloads with a 422 error."""
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422