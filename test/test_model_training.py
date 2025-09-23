import pytest
from unittest.mock import patch, MagicMock
import dask.dataframe as dd
import pandas as pd

from services.training.model.train_model import ModelTrainer

def test_model_training_and_save():
    """
    Tests that the ModelTrainer class can initialize a model, train it,
    and save it without errors.
    """
    # 1. Setup: Create dummy Dask DataFrame for training
    df = pd.DataFrame({
        'feature_1': [0.1, 0.5, 0.9],
        'feature_2': [0.2, 0.6, 0.8],
        'sentiment': [0, 1, 1]
    })
    ddf = dd.from_pandas(df, npartitions=1)

    # 2. Mock external dependencies (params.yaml and saving function)
    mock_params = {
        "model_training": {
            "model_to_use": "LightGBM",
            "lightgbm": {
                "n_estimators": 5, # Use a small number for fast testing
                "learning_rate": 0.1
            }
        },
        "global_params": {
            "target_column": "sentiment"
        }
    }

    # 3. Run the method under test within the mocked context
    with patch('services.training.model.train_model.load_params', return_value=mock_params), \
         patch('services.training.model.train_model.save_object') as mock_save_object:
        
        trainer = ModelTrainer()
        trained_model = trainer.train_model(train_ddf=ddf,target_col="sentiment")

        # 4. Assertions
        # Check that the model object was created
        assert trained_model is not None
        # Check that the model has a 'predict' method, indicating it's a scikit-learn compatible model
        assert hasattr(trained_model, 'predict')
        # Check that the save function was called
        mock_save_object.assert_called_once()