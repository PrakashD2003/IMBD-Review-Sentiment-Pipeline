import types
from unittest import mock
import pytest

from src.utils.mlflow_utils import get_latest_model
from src.exception import DetailedException

class DummyModel:
    pass


def test_get_latest_model_no_versions():
    client_mock = mock.Mock()
    client_mock.get_latest_versions.return_value = []
    with mock.patch('mlflow.MlflowClient', return_value=client_mock):
       with pytest.raises(RuntimeError):
            get_latest_model('model', ['Production'])


def test_get_latest_model_loads_latest():
    mv = types.SimpleNamespace(version='42')
    client_mock = mock.Mock()
    client_mock.get_latest_versions.return_value = [mv]

    loader = mock.Mock()
    loader.load_model.return_value = DummyModel()

    with mock.patch('mlflow.MlflowClient', return_value=client_mock):
        with mock.patch('mlflow.pyfunc', loader):
            model = get_latest_model('model', ['Production'])

    assert isinstance(model, DummyModel)
    loader.load_model.assert_called_once_with('models:/model/42')
