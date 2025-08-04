import pandas as pd
import dask.dataframe as dd
from services.training.features.feature_engineering import FeatureEngineering


def test_tfidf_vectorization_shape():
    train_df = pd.DataFrame({
        "review": ["good movie", "bad movie"],
        "sentiment": [1, 0],
    })
    test_df = pd.DataFrame({
        "review": ["not good", "really bad movie"],
        "sentiment": [0, 0],
    })
    train_ddf = dd.from_pandas(train_df, npartitions=1)
    test_ddf = dd.from_pandas(test_df, npartitions=1)

    fe = FeatureEngineering()
    fe.params["TF-IDF_Params"]["min_df"] = 1
    fe.params["TF-IDF_Params"]["max_df"] = 1.0
    train_vec, test_vec, tfidf = fe.vectorize_tfidf(train_ddf, test_ddf, column="review")
    train_pd = train_vec.compute()
    test_pd = test_vec.compute()

    # Expect three tf-idf features + sentiment column
    assert train_pd.shape[1] == 4
    assert test_pd.shape[1] == 4
