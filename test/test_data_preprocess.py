import pandas as pd
import dask.dataframe as dd
from src.data.data_preprocessing import DataPreprocessing


def test_preprocess_data_removes_nulls_and_cleans():
    df = pd.DataFrame({"review": ["This is GREAT!", None, "BAD movie"], "num": [1,2,3]})
    ddf = dd.from_pandas(df, npartitions=1)
    dp = DataPreprocessing()
    result = dp.preprocess_data(ddf, "review").compute()
    assert result.shape[0] == 2
    assert list(result["review"]) == ["great", "bad movie"]
