import pytest
import pandas as pd
import dask.dataframe as dd
from services.training.data.data_preprocessing import DataPreprocessing
from services.training.data.data_preprocessing import preprocess_text

def test_preprocess_data_removes_nulls_and_cleans():
    df = pd.DataFrame({"review": ["This is GREAT!", None, "BAD movie"], "num": [1,2,3]})
    ddf = dd.from_pandas(df, npartitions=1)
    dp = DataPreprocessing()
    result = dp.preprocess_data(ddf, "review").compute()
    assert result.shape[0] == 2
    assert list(result["review"]) == ["great", "bad movie"]

@pytest.mark.parametrize("raw_text, expected_clean_text", [
    ("", ""),                                # Empty string
    ("  ", ""),                              # Whitespace only
    ("URL http://t.co/xyz removed", "url removed"), # URL removal
    ("it's a movie", "movie"),               # Stopword removal
    ("running ran runs", "running ran run"), # Lemmatization
    ("not a bad movie", "not bad_NOT movie"), # Negation handling
    ("12345 numeric data", "numeric data"),  # Number removal
    ("!?@#$", ""),                           # Punctuation only
])
def test_preprocess_text_edge_cases(raw_text, expected_clean_text):
    """
    Tests the core text cleaning logic with various edge cases.
    """
    assert preprocess_text(raw_text) == expected_clean_text