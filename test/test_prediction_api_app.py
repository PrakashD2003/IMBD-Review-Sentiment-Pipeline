import pytest
import sys
from pathlib import Path
import pandas as pd
from unittest.mock import patch, MagicMock
from prometheus_client import CollectorRegistry
from starlette.testclient import TestClient
from services.prediction.api_end_point.prediction_fastapi_app import app
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


@pytest.fixture(scope="module")
def client():
    """Provides a FastAPI TestClient for the prediction app."""
    with TestClient(app) as c:
        yield c
        
class DummyPipeline:
    def predict_single(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"prediction": [1] * len(df)})

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"prediction": [0] * len(df)})


class DummyMetric:
    def labels(self, *args, **kwargs):
        return self

    def time(self):
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            yield

        return _cm()

    def inc(self, *args, **kwargs):
        pass

class DummyCounter(DummyMetric):
    def __init__(self, *args, **kwargs):
        pass


class DummyHistogram(DummyMetric):
    def __init__(self, *args, **kwargs):
        pass


def test_predict_endpoint():
    dummy = DummyPipeline()
    registry = CollectorRegistry()
    with patch('prometheus_client.REGISTRY', registry), \
         patch('prometheus_client.metrics.REGISTRY', registry), \
         patch('fastapi_app.app.Counter', DummyCounter), \
         patch('fastapi_app.app.Histogram', DummyHistogram), \
         patch('fastapi_app.app.start_client', return_value=None), \
         patch('fastapi_app.app.UnifiedPredictionPipeline',
               return_value=MagicMock()):
        with TestClient(app) as client:
            with patch('fastapi_app.app.pipeline', dummy):
                resp = client.post('/predict',
                                   json={'reviews': ['a', 'b']})
    assert resp.status_code == 200
    assert resp.json() == {'predictions': [1, 1]}

def test_batch_predict_endpoint():
    dummy = DummyPipeline()
    registry = CollectorRegistry()
    with patch('prometheus_client.REGISTRY', registry), \
         patch('prometheus_client.metrics.REGISTRY', registry), \
         patch('fastapi_app.app.Counter', DummyCounter), \
         patch('fastapi_app.app.Histogram', DummyHistogram), \
         patch('fastapi_app.app.start_client', return_value=None), \
         patch('fastapi_app.app.UnifiedPredictionPipeline',
               return_value=MagicMock()):
        with TestClient(app) as client:
            with patch('fastapi_app.app.pipeline', dummy):
                resp = client.post(
                    '/batch_predict',
                    json={'reviews': ['x', 'y', 'z']}
                )
    assert resp.status_code == 200
    assert resp.json() == {'predictions': [0, 0, 0]}

@pytest.mark.parametrize("invalid_payload", [
    {"reviews": []},                   # Empty list
    {"reviews": "not a list"},         # Wrong data type
    {"data": ["a review"]},            # Wrong key
    {},                                # Empty JSON
])
def test_predict_endpoint_invalid_input(client, invalid_payload):
    """
    Tests that the /predict endpoint returns a 422 error for invalid payloads.
    """
    response = client.post('/predict', json=invalid_payload)
    assert response.status_code == 422