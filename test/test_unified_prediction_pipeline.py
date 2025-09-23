# test/test_unified_prediction_pipeline.py

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from services.prediction.pipelines.unified_prediction_pipeline import UnifiedPredictionPipeline, batch_predict_func
from common.exception import DetailedException

# --- Mock Objects ---

@pytest.fixture
def mock_vectorizer():
    """Provides a mock scikit-learn vectorizer."""
    vectorizer = MagicMock()
    # The transform method should return a mock sparse matrix or array
    vectorizer.transform.return_value = np.array([[0, 1, 1], [1, 0, 1]])
    vectorizer.vocabulary_ = {"great": 0, "movie": 1, "bad": 2}
    return vectorizer

@pytest.fixture
def mock_model():
    """Provides a mock scikit-learn classifier."""
    model = MagicMock()
    # The predict method returns a numpy array of predictions
    model.predict.return_value = np.array([1, 0])
    model.n_features_in_ = 3
    return model

@pytest.fixture
def mock_pipeline(mock_model, mock_vectorizer):
    """
    Provides an instance of UnifiedPredictionPipeline with all external
    dependencies mocked out. This is the primary fixture for testing the class.
    """
    # Patch all functions that have side effects (file I/O, network calls)
    with patch('services.prediction.pipelines.unified_prediction_pipeline.load_params') as mock_load_params, \
         patch('services.prediction.pipelines.unified_prediction_pipeline.configure_mlflow') as mock_configure_mlflow, \
         patch('services.prediction.pipelines.unified_prediction_pipeline.get_latest_model') as mock_get_latest_model, \
         patch('services.prediction.pipelines.unified_prediction_pipeline.get_client', side_effect=ValueError), \
         patch('services.prediction.pipelines.unified_prediction_pipeline.start_client') as mock_start_client:
        
        # Configure the return values of our mocks
        mock_load_params.return_value = {"global_params": {"text_column": "review"}}
        
        # get_latest_model is called twice: for the model and the vectorizer.
        # The side_effect list provides a different return value for each call.
        mock_get_latest_model.side_effect = [mock_model, mock_vectorizer]

        # Instantiate the pipeline. The __init__ method will use our mocks.
        pipeline = UnifiedPredictionPipeline()
        yield pipeline

# --- Test Cases ---

def test_pipeline_initialization(mock_pipeline, mock_model, mock_vectorizer):
    """
    Tests if the pipeline initializes correctly, loading the mocked model and vectorizer.
    """
    assert mock_pipeline.model is mock_model
    assert mock_pipeline.vectorizer is mock_vectorizer
    assert mock_pipeline.params["global_params"]["text_column"] == "review"

def test_predict_single(mock_pipeline):
    """
    Tests the predict_single method for low-latency, in-memory predictions.
    """
    input_df = pd.DataFrame({"review": ["great movie", "bad movie"]})
    
    # Execute the method under test
    result_df = mock_pipeline.predict_single(input_df)

    # --- Assertions ---
    # Check the output format and values
    assert isinstance(result_df, pd.DataFrame)
    assert "prediction" in result_df.columns
    pd.testing.assert_series_equal(
        result_df["prediction"],
        pd.Series([1, 0], name="prediction", dtype=int),
        check_names=False # The index name might differ slightly
    )
    
    # Check that our mocked dependencies were called correctly
    assert mock_pipeline.vectorizer.transform.call_count == 1
    assert mock_pipeline.model.predict.call_count == 1

def test_predict_batch(mock_pipeline):
    """
    Tests the predict_batch method for Dask-based batch predictions.
    Because the dask client is mocked to run locally, this effectively tests
    the `batch_predict_func` logic.
    """
    input_df = pd.DataFrame({"review": ["great movie", "bad movie"]})
    
    # Execute the method under test
    with patch('dask.dataframe.from_pandas') as mock_from_pandas:
        # We compute locally, so we can just return a Dask DF from the pandas DF
        # to test the map_partitions logic.
        import dask.dataframe as dd
        mock_ddf = dd.from_pandas(input_df, npartitions=1)
        mock_from_pandas.return_value = mock_ddf
        
        result_df = mock_pipeline.predict_batch(input_df)

    # --- Assertions ---
    # Check the output format and values
    assert isinstance(result_df, pd.DataFrame)
    assert "prediction" in result_df.columns
    assert result_df["prediction"].tolist() == [1, 0]
    
    # The underlying functions are called within map_partitions, so we can't
    # easily check call counts on the pipeline's mocks. Instead, this test
    # validates the end-to-end logic of the batch path.

def test_batch_predict_func_logic(mock_model, mock_vectorizer):
    """
    Directly tests the helper function used by `map_partitions` in batch mode.
    """
    partition_df = pd.DataFrame({
        "review": ["great movie", "bad movie"],
        "other_col": ["A", "B"]
    })
    
    result_df = batch_predict_func(
        partition=partition_df,
        text_col="review",
        vectorizer=mock_vectorizer,
        model=mock_model
    )
    
    assert "prediction" in result_df.columns
    assert "other_col" in result_df.columns # Ensures original columns are preserved
    assert result_df["prediction"].tolist() == [1, 0]

def test_pipeline_initialization_failure():
    """
    Tests that an exception during initialization is handled correctly.
    """
    # Mock get_latest_model to raise an error
    with patch('services.prediction.pipelines.unified_prediction_pipeline.get_latest_model', side_effect=RuntimeError("MLflow is down")):
        # Assert that initializing the pipeline raises our custom exception
        with pytest.raises(DetailedException) as excinfo:
            UnifiedPredictionPipeline()
        
        # Check that the original error message is contained in the wrapped exception
        assert "MLflow is down" in str(excinfo.value)