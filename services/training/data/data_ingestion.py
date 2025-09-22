"""Data ingestion pipeline.

This module downloads raw review data from S3, filters valid sentiment
labels, creates train/test splits using Dask, and persists the results.
"""

import dask.dataframe as dd
from pathlib import Path
from dask_ml.model_selection import train_test_split as dask_train_test_split
from sklearn.model_selection import train_test_split
from common.logger import configure_logger
from common.exception import DetailedException
from services.training.entity.config_entity import DataIngestionConfig
from services.training.entity.artifact_entity import DataIngestionArtifact
from services.training.connections.s3_connection import S3Connection
from common.utils.main_utils import load_params, save_dask_dataframe_as_parquet
from common.constants import PARAM_FILE_PATH, SINGLE_FILE


module_name = Path(__file__).stem

logger = configure_logger(
    logger_name=module_name,
    level="DEBUG",
    to_console=True,
    to_file=True,
    log_file_name=module_name
)


class DataIngestion:
    """
    Orchestrates the data ingestion pipeline:
      1. Fetch raw CSV from S3 into a Dask DataFrame
      2. Filter to only positive/negative sentiments and encode them as 1/0
      3. Split into train/test Dask DataFrames using Dask-ML
      4. Save each split to disk as CSV (either single file or partitioned)
    """

    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        """
        Initialize the DataIngestion component.

        :param data_ingestion_config: Holds S3 bucket details and file paths for raw/train/test.
        :raises DetailedException: If loading PARAM_FILE_PATH fails.
        """
        try:
            logger.debug("Configuring DataIngestion with supplied configuration.")
            self.data_ingestion_config = data_ingestion_config
            self.params = load_params(params_path=PARAM_FILE_PATH, logger=logger)
            self.s3 = S3Connection(logger=logger)
            logger.info("DataIngestion initialized successfully.")
        except Exception as e:
            raise DetailedException(exc=e, logger=logger) from e

    def basic_preprocessing(
        self, ddf_raw: dd.DataFrame
    ) -> dd.DataFrame:
        """
        Filter to only rows where 'sentiment' is 'positive' or 'negative',
        then encode 'positive'->1 and 'negative'->0 in place (lazily).

        :param ddf_raw: Raw Dask DataFrame containing at least a 'sentiment' column.
        :return: A Dask DataFrame containing only positive/negative rows with sentiment encoded.
        :raises DetailedException: On any error during filtering/encoding.
        """
        try:
            logger.debug("Starting basic_preprocessing on Dask DataFrame.")

            # 1) Keep only rows where sentiment is 'positive' or 'negative'
            ddf_filtered = ddf_raw[ddf_raw["sentiment"].isin(["positive", "negative"])]
            logger.info("Filtered positive/negative rows; now encoding.")

            # 2) Encode: positive->1, negative->0
            #    NOTE: Dask cares if you assign to a new column or use .map.
            ddf_filtered = ddf_filtered.assign(
                sentiment=ddf_filtered["sentiment"]
                .map(
                    lambda x: 1 if x == "positive" else 0,
                    meta=("sentiment", "int64")
                )
            )
            logger.info("Encoding complete (positive->1, negative->0).")

            return ddf_filtered

        except Exception as e:
            raise DetailedException(exc=e, logger=logger) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        1. Load raw CSV from S3 into Dask DataFrame.
        2. Apply basic_preprocessing to filter/encode sentiments.
        3. Use Dask-ML train_test_split to create lazy train/test DataFrames.
        4. Persist each split to disk as CSV (single or partitioned).

        :return: DataIngestionArtifact containing the file paths to train/test CSVs.
        :raises DetailedException: On any failure in load, preprocess, split, or save.
        """
        try:
            logger.info("Entered 'initiate_data_ingestion' function of 'DataIngetion' Class")
            logger.info("\n" + "-" * 80)
            logger.info("Starting Data Ingestion Component...")
            # 1) Fetch raw CSV from S3 into a Dask DataFrame
            logger.debug(
                "Fetching raw CSV from S3: bucket='%s', key='%s'",
                self.data_ingestion_config.s3_bucket,
                self.data_ingestion_config.s3_data_file_key
            )
            ddf_raw = self.s3.load_parquet_from_s3_as_dask_dataframe(
                bucket_name=self.data_ingestion_config.s3_bucket,
                file_key=self.data_ingestion_config.s3_data_file_key,
                n_partitions=self.data_ingestion_config.n_partitions
            )
            logger.info("Successfully loaded raw CSV into Dask DataFrame.")
            
            logger.debug("Saving raw data as csv at: '%s'",self.data_ingestion_config.raw_data_file_path)
            save_dask_dataframe_as_parquet(file_save_path=self.data_ingestion_config.raw_data_file_path, dataframe=ddf_raw, single_file=True, index=False, logger=logger)
            logger.info("Raw data saved successfully to '%s'", self.data_ingestion_config.raw_data_file_path)

            # 2) Basic preprocessing (filter + encode)
            ddf_clean = self.basic_preprocessing(ddf_raw=ddf_raw)

            # 3) Split into train/test using Dask-ML
            test_size = float(self.params.get("test_size", 0.2))
            random_state = int(self.params.get("random_state", 42))
            shuffle = bool(self.params.get("shuffle", True))
            logger.debug(
                "Splitting into train/test with 'test_size=%s', 'random_state=%s', 'shuffle=%s'",
                test_size, random_state, shuffle
            )
            
            train_ddf, test_ddf = dask_train_test_split(ddf_raw,
                                    test_size=self.params["data_ingestion"]["test_size"],
                                    shuffle=self.params["data_ingestion"]["shuffle"],
                                    random_state=self.params["global_params"]["random_state"])
            
            logger.info(
                "Created train/test Dask DataFrames: 'train_partitions=%s', 'test_partitions=%s'",
                train_ddf.npartitions, test_ddf.npartitions
            )

            # 4) Save each split to CSV
            single_out = bool(SINGLE_FILE)  # from constants
            train_path = self.data_ingestion_config.training_data_file_path
            test_path = self.data_ingestion_config.test_data_file_path
        

            logger.debug(
                "Saving train split to '%s' (single_file='%s')", train_path, single_out
            )
            save_dask_dataframe_as_parquet(
                file_save_path=train_path,
                dataframe=train_ddf,
                single_file=single_out,
                index=False,
                logger=logger
            )
            logger.info("Training data saved successfully to '%s'", train_path)

            logger.debug(
                "Saving test split to '%s' (single_file='%s')", test_path, single_out
            )
            save_dask_dataframe_as_parquet(
                file_save_path=test_path,
                dataframe=test_ddf,
                single_file=single_out,
                index=False,
                logger=logger
            )
            logger.info("Test data saved successfully to '%s'", test_path)

            artifact = DataIngestionArtifact(
                training_data_file_path=train_path,
                test_data_file_path=test_path,
                raw_data_file_path=self.data_ingestion_config.raw_data_file_path
            )
            logger.info("DataIngestionArtifact created: '%s'", artifact)

            logger.debug("Exiting initiate_data_ingestion.")
            return artifact

        except Exception as e:
            raise DetailedException(exc=e, logger=logger) from e


if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion()
    