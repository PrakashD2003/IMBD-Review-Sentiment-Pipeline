import sys
from pathlib import Path
import pandas as pd
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from prometheus_client import CollectorRegistry

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from fast_api_app.app import app

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
         patch('fast_api_app.app.Counter', DummyCounter), \
         patch('fast_api_app.app.Histogram', DummyHistogram), \
         patch('fast_api_app.app.start_client', return_value=None), \
         patch('fast_api_app.app.UnifiedPredictionPipeline',
               return_value=MagicMock()):
        with TestClient(app) as client:
            with patch('fast_api_app.app.pipeline', dummy):
                resp = client.post('/predict',
                                   json={'reviews': ['a', 'b']})
    assert resp.status_code == 200
    assert resp.json() == {'predictions': [1, 1]}

def test_batch_predict_endpoint():
    dummy = DummyPipeline()
    registry = CollectorRegistry()
    with patch('prometheus_client.REGISTRY', registry), \
         patch('prometheus_client.metrics.REGISTRY', registry), \
         patch('fast_api_app.app.Counter', DummyCounter), \
         patch('fast_api_app.app.Histogram', DummyHistogram), \
         patch('fast_api_app.app.start_client', return_value=None), \
         patch('fast_api_app.app.UnifiedPredictionPipeline',
               return_value=MagicMock()):
        with TestClient(app) as client:
            with patch('fast_api_app.app.pipeline', dummy):
                resp = client.post(
                    '/batch_predict',
                    json={'reviews': ['x', 'y', 'z']}
                )
    assert resp.status_code == 200
    assert resp.json() == {'predictions': [0, 0, 0]}