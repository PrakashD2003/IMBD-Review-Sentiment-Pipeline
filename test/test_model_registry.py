import types  # <-- Add this import
from unittest.mock import patch, MagicMock

from services.training.model.register_model import ModelRegistry

def test_register_model_registers_and_transitions():
    """
    Test that the register_model method correctly calls the MLflow API
    to register a model and transition its stage.
    """
    mr = ModelRegistry()
    with patch('services.training.model.register_model.mlflow.register_model') as reg_mock, \
         patch('services.training.model.register_model.mlflow.tracking.MlflowClient') as client_cls:
        
        # Setup mock return values
        client = MagicMock()
        client_cls.return_value = client
        reg_mock.return_value = types.SimpleNamespace(version='42')

        # Call the method under test
        version = mr.register_model('run1', 'artifact', 'name', 'Stage')

        # Assertions
        assert version == '42'
        reg_mock.assert_called_once_with(model_uri='runs:/run1/artifact', name='name')
        client.transition_model_version_stage.assert_called_once_with(
            name='name', version='42', stage='Stage', archive_existing_versions=True
        )

def test_initiate_model_registration_logs(tmp_path):
    """
    Test that the main registration method calls all the necessary MLflow
    logging and registration functions with the correct parameters.
    """
    mr = ModelRegistry()
    # Corrected the patch paths from 'src.model.register_model' to the actual module path
    with patch('services.training.model.register_model.configure_mlflow'), \
         patch('services.training.model.register_model.mlflow.start_run') as start_run, \
         patch('services.training.model.register_model.load_json', return_value={'acc': 1.0}), \
         patch('services.training.model.register_model.load_object', return_value=MagicMock()), \
         patch('services.training.model.register_model.mlflow.sklearn.log_model') as log_model, \
         patch('services.training.model.register_model.mlflow.log_params') as log_params, \
         patch('services.training.model.register_model.mlflow.log_metrics') as log_metrics, \
         patch.object(ModelRegistry, 'register_model') as register_model:
        
        # Setup mock run object
        run = start_run.return_value.__enter__.return_value
        run.info.run_id = '123'
        
        # Call the method under test
        mr.initiate_model_registration()

        # Assert that the correct MLflow functions were called
        assert log_model.call_count == 2
        assert log_params.call_count >= 1
        log_metrics.assert_called_once_with({'acc': 1.0})
        register_model.assert_any_call(
            run_id='123', 
            artifact_path=mr.model_registry_config.mlflow_model_artifact_path,
            model_name=mr.model_registry_config.mlflow_model_name,
            stage=mr.model_registry_config.mlflow_model_stage
        )