import types
from unittest.mock import patch, MagicMock

from src.model.register_model import ModelRegistry


def test_register_model_registers_and_transitions():
    mr = ModelRegistry()
    with patch('src.model.register_model.mlflow.register_model') as reg_mock, \
         patch('src.model.register_model.mlflow.tracking.MlflowClient') as client_cls:
        client = MagicMock()
        client_cls.return_value = client
        reg_mock.return_value = types.SimpleNamespace(version='42')

        version = mr.register_model('run1', 'artifact', 'name', 'Stage')

    assert version == '42'
    reg_mock.assert_called_once_with(model_uri='runs:/run1/artifact', name='name')
    client.transition_model_version_stage.assert_called_once_with(
        name='name', version='42', stage='Stage', archive_existing_versions=True
    )


def test_initiate_model_registration_logs(tmp_path):
    mr = ModelRegistry()
    with patch('src.model.register_model.configure_mlflow'), \
         patch('src.model.register_model.mlflow.start_run') as start_run, \
         patch('src.model.register_model.load_json', return_value={'acc': 1.0}), \
         patch('src.model.register_model.load_object', return_value=MagicMock()), \
         patch('src.model.register_model.mlflow.sklearn.log_model') as log_model, \
         patch('src.model.register_model.mlflow.log_params') as log_params, \
         patch('src.model.register_model.mlflow.log_metrics') as log_metrics, \
         patch.object(ModelRegistry, 'register_model') as register_model:
        run = start_run.return_value.__enter__.return_value
        run.info.run_id = '123'
        mr.initiate_model_registration()

    assert log_model.call_count == 2
    assert log_params.call_count >= 1
    log_metrics.assert_called_once_with({'acc': 1.0})
    register_model.assert_any_call(run_id='123', artifact_path=mr.model_registry_config.mlflow_model_artifact_path,
                                   model_name=mr.model_registry_config.mlflow_model_name,
                                   stage=mr.model_registry_config.mlflow_model_stage)