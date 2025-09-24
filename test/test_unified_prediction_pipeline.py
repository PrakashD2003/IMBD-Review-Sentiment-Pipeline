import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import dask.dataframe as dd
import numpy as np

from services.prediction.pipelines.unified_prediction_pipeline import UnifiedPredictionPipeline
from common.exception import DetailedException

# Mock parameters to be used in tests, isolating them from the params.yaml file
MOCK_PARAMS = {
    "global_params": {
        "text_column": "review"
    }
}

def test_pipeline_initialization_success():
    """
    Tests that the UnifiedPredictionPipeline can be initialized successfully
    when all its dependencies are mocked.
    """
    with patch('services.prediction.pipelines.unified_prediction_pipeline.load_params', return_value=MOCK_PARAMS), \
         patch('services.prediction.pipelines.unified_prediction_pipeline.get_latest_model', return_value=MagicMock()):
        
        # This should now succeed without trying to access files or network
        pipeline = UnifiedPredictionPipeline()
        assert pipeline is not None
        assert pipeline.params == MOCK_PARAMS

def test_pipeline_initialization_failure():
    """
    Tests that if a dependency (like MLflow) fails during initialization,
    a DetailedException is raised.
    """
    # Simulate that the get_latest_model call fails
    with patch('services.prediction.pipelines.unified_prediction_pipeline.load_params', return_value=MOCK_PARAMS), \
         patch('services.prediction.pipelines.unified_prediction_pipeline.get_latest_model', side_effect=RuntimeError("MLflow is down")):
        
        with pytest.raises(DetailedException) as excinfo:
            UnifiedPredictionPipeline()
        
        # Check that the exception message contains the original error
        assert "MLflow is down" in str(excinfo.value)

def test_predict_single():
    """
    Tests the predict_single method with mocks.
    """
    with patch('services.prediction.pipelines.unified_prediction_pipeline.load_params', return_value=MOCK_PARAMS), \
         patch('services.prediction.pipelines.unified_prediction_pipeline.get_latest_model') as mock_get_model:

        # Setup mocks for model and vectorizer
        mock_model = MagicMock()
        mock_vectorizer = MagicMock()
        mock_get_model.side_effect = [mock_model, mock_vectorizer]

        pipeline = UnifiedPredictionPipeline()
        
        input_df = pd.DataFrame({"review": ["a good movie"]})
        
        # Configure mock return values for the prediction flow
        mock_vectorizer.transform.return_value = [[0.1, 0.9]]
        # Return a numpy array, just like a real model would
        mock_model.predict.return_value = np.array([1]) 
        mock_model.predict_proba.return_value = [[0.1, 0.9]]

        result_df = pipeline.predict_single(input_df)

        assert isinstance(result_df, pd.DataFrame)
        assert "sentiment" in result_df.columns
        assert result_df.iloc[0]["sentiment"] == "positive"

def test_predict_batch():
    """
    Tests the predict_batch method with mocks.
    """
    with patch('services.prediction.pipelines.unified_prediction_pipeline.load_params', return_value=MOCK_PARAMS), \
         patch('services.prediction.pipelines.unified_prediction_pipeline.get_latest_model') as mock_get_model:
        
        # Setup mocks
        mock_model = MagicMock()
        mock_vectorizer = MagicMock()
        mock_get_model.side_effect = [mock_model, mock_vectorizer]

        pipeline = UnifiedPredictionPipeline()

        # Create a dummy Dask DataFrame for input
        input_df = pd.DataFrame({"review": ["a test review"]})
        input_ddf = dd.from_pandas(input_df, npartitions=1)
        
        # Configure the final mock in the chain to return a Pandas DataFrame
        mock_compute = MagicMock(return_value=pd.DataFrame({"sentiment": ["positive"]}))
        mock_map_partitions = MagicMock()
        mock_map_partitions.compute = mock_compute
        
        with patch('dask.dataframe.from_pandas', return_value=MagicMock(map_partitions=MagicMock(return_value=mock_map_partitions))):
             result_df = pipeline.predict_batch(input_ddf)

        assert isinstance(result_df, pd.DataFrame)
        assert "sentiment" in result_df.columns