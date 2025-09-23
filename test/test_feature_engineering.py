import pytest
import pandas as pd
import dask.dataframe as dd
from unittest.mock import patch

from services.training.features.feature_engineering import FeatureEngineering

def test_tfidf_vectorization_shape():
    """
    Tests that the TF-IDF vectorization produces the correct output shape and
    that the vectorizer is fitted correctly.
    """
    # 1. Setup: Create dummy Dask DataFrames
    train_df = pd.DataFrame({
        "review": ["good movie", "bad movie", "excellent film"],
        "sentiment": [1, 0, 1],
    })
    test_df = pd.DataFrame({
        "review": ["not a good movie", "really bad film"],
        "sentiment": [0, 0],
    })
    train_ddf = dd.from_pandas(train_df, npartitions=1)
    test_ddf = dd.from_pandas(test_df, npartitions=1)

    # 2. Mock external dependencies (especially params.yaml)
    mock_params = {
        "feature_engineering": {
            "tfidf": {
                "ngram_range": [1, 2],
                "max_features": 100
            }
        },
        "global_params": {
            "text_column": "review"
        }
    }

    # 3. Run the method under test, mocking the file dependency
    with patch('services.training.features.feature_engineering.load_params', return_value=mock_params):
        fe = FeatureEngineering()
        vectorizer, train_features, test_features = fe.vectorize_tfidf(
            train_ddf=train_ddf,
            test_ddf=test_ddf,
            column="sentiment"
        )

    # 4. Assertions
    assert train_features.shape[1] == test_features.shape[1]
    assert "good movie" in vectorizer.get_feature_names_out()
    assert "bad film" not in vectorizer.get_feature_names_out() # Should not be a feature
    # Check that sentiment column is preserved
    assert "sentiment" in train_features.columns
    assert "sentiment" in test_features.columns