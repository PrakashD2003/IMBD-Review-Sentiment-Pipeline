import pandas as pd
import dask.dataframe as dd
from unittest.mock import patch, MagicMock

from services.training.data.data_ingestion import DataIngestion


def test_basic_preprocessing_filters_and_encodes():
    df = pd.DataFrame({
        "sentiment": ["positive", "negative", "neutral", "positive"],
        "review": ["good", "bad", "meh", "great"]
    })
    ddf = dd.from_pandas(df, npartitions=1)
    di = DataIngestion()
    result = di.basic_preprocessing(ddf).compute()
    assert list(result["sentiment"]) == [1, 0, 1]


def test_initiate_data_ingestion_saves_artifacts(tmp_path):
    df = pd.DataFrame({"sentiment": ["positive", "negative"], "review": ["a", "b"]})
    ddf = dd.from_pandas(df, npartitions=1)

    with patch('services.training.data.data_ingestion.S3Connection') as s3_cls, \
     patch('services.training.data.data_ingestion.save_dask_dataframe_as_parquet') as save_parquet, \
     patch('services.training.data.data_ingestion.dask_train_test_split', return_value=(ddf, ddf)):
        s3_instance = s3_cls.return_value
        s3_instance.load_parquet_from_s3_as_dask_dataframe.return_value = ddf
        di = DataIngestion()
        di.data_ingestion_config.raw_data_file_path = str(tmp_path / 'raw')
        di.data_ingestion_config.training_data_file_path = str(tmp_path / 'train')
        di.data_ingestion_config.test_data_file_path = str(tmp_path / 'test')
        artifact = di.initiate_data_ingestion()

    assert artifact.training_data_file_path == str(tmp_path / 'train')
    assert save_parquet.call_count == 3