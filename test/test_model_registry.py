import types
from unittest.mock import patch, MagicMock

from services.training.model.register_model import ModelRegistry

# A mock dictionary to simulate the contents of params.yaml
MOCK_PARAMS = {
    "model_registry": {
        "mlflow_model_name": "test-model",
        "mlflow_vectorizer_name": "test-vectorizer",
        "mlflow_model_stage": "Staging",
        "mlflow_model_artifact_path": "model",
        "mlflow_vectorizer_artifact_path": "vectorizer"
    }
}

def test_register_model_registers_and_transitions():
    """
    Tests that the register_model method correctly calls the MLflow API
    to register a model and transition its stage.
    """
    # Use patch to mock the load_params function during instantiation
    with patch('services.training.model.register_model.load_params', return_value=MOCK_PARAMS):
        mr = ModelRegistry()

    with patch('services.training.model.register_model.mlflow.register_model') as reg_mock, \
         patch('services.training.model.register_model.mlflow.tracking.MlflowClient') as client_cls:
        
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
    """
    Tests that the main registration method calls all the necessary MLflow
    logging and registration functions.
    """
    # Patch all external dependencies, including load_params
    with patch('services.training.model.register_model.load_params', return_value=MOCK_PARAMS), \
         patch('services.training.model.register_model.configure_mlflow'), \
         patch('services.training.model.register_model.mlflow.start_run') as start_run, \
         patch('services.training.model.register_model.load_json', return_value={'acc': 1.0}), \
         patch('services.training.model.register_model.load_object', return_value=MagicMock()), \
         patch('services.training.model.register_model.mlflow.sklearn.log_model') as log_model, \
         patch('services.training.model.register_model.mlflow.log_params') as log_params, \
         patch('services.training.model.register_model.mlflow.log_metrics') as log_metrics:
        
        # Instantiate the class inside the mocked context
        mr = ModelRegistry()
        
        # We also need to patch the instance method `register_model`
        with patch.object(mr, 'register_model') as register_model_mock:
            run = start_run.return_value.__enter__.return_value
            run.info.run_id = '123'
            
            mr.initiate_model_registration()

            assert log_model.call_count == 2
            assert log_params.call_count >= 1
            log_metrics.assert_called_once_with({'acc': 1.0})
            register_model_mock.assert_any_call(
                run_id='123', 
                artifact_path=mr.model_registry_config.mlflow_model_artifact_path,
                model_name=mr.model_registry_config.mlflow_model_name,
                stage=mr.model_registry_config.mlflow_model_stage
            )