import pandas as pd
from typing import Tuple
from logger import configure_logger
from exception import DetailedException
from entity.config_entity import DataIngestionConfig
from entity.artifact_entity import DataIngestionArtifact
from connections.s3_connection import S3Connection
from utils.main_utils import load_params, save_dataframe_as_csv
from sklearn.model_selection import train_test_split
from constants import PARAM_FILE_PATH

logger = configure_logger(logger_name=__name__, level="DEBUG", to_console=True, to_file=True, log_file_name=__name__)

class DataIngestion:
    """
    Orchestrates the data ingestion pipeline: fetching raw data,
    basic preprocessing, splitting into train/test, and saving to disk.
    """
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        """
        :param config: Configuration parameters for data ingestion,
                       including S3 bucket details and file paths.
        :param logger: Optional logger; defaults to module‐level logger.
        :raises DetailedException: If loading the PARAM_FILE_PATH fails.
        """
        try:
            logger.debug("Configuring DataIngestion class of data module through constructer...")
            self.data_ingestion_config = data_ingestion_config
            self.params = load_params(params_path=PARAM_FILE_PATH,logger=logger)
            self.s3 = S3Connection(logger=logger)
            logger.info("DataIngestion class configured successfully.")
        except Exception as e:
            raise DetailedException(exc=e, logger=logger) from e
        

    def basic_preprocessing(self, df: pd.DataFrame)-> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filter to only positive/negative sentiments, encode them as 1/0,
        then split into training and test sets.

        :param df: Raw DataFrame with a 'sentiment' column.
        :return: (train_df, test_df)
        :raises DetailedException: On any error during processing.
        """
        try:
            logger.debug("Entered basic_preprocessing function of DataIngestion class of data_ingestion module.")
            logger.debug("Removing rows that have any other value than 'positive' or 'negative' in 'sentiment' column...")
            
            final_df = df[df['sentiment'].isin(['positive','negative'])]
            logger.info("Successfully removed rows that have any other value than 'positive' or 'negative' in 'sentiment' column.")
            
            logger.debug("Encoding 'positive':1 and 'negative':0")
            final_df['sentiment'] = final_df['sentiment'].replace({'positive':1,'negative':0})
            logger.info("Encoding completed successfully")

            logger.debug("Performing train-test-split with test_size %s ...", self.params[""])
            train_data, test_data = train_test_split(final_df, test_size=self.params[""], random_state=self.params[""])
            logger.info("Data splited successfully in train-test-data with test_size=%s, random_state=%s ", self.params[""], self.params[""])

            logger.debug("Exiting 'basic_preprocessing' function and returning train-test-data as tuple.")
            return train_data , test_data
        except Exception as e:
            raise DetailedException(exc=e, logger=logger) from e
    
    def initiate_data_ingestion(self) ->DataIngestionArtifact:
        """
        Fetches raw data from S3, applies preprocessing, and writes
        train/test CSVs to the configured file paths.

        :return: DataIngestionArtifact containing the saved file paths.
        :raises DetailedException: On any failure during ingestion.
        """
        try:
            logger.info("Entered initiate_data_ingestion method of DataIngestion class")
            print("\n" + "-"*80)
            print("🚀 Starting Data Ingestion Component...")

            logger.debug("Fetching data from s3 bucket(%s)...",self.data_ingestion_config.s3_bucket)
            df = self.s3.load_csv_from_s3(bucket_name=self.data_ingestion_config.s3_bucket, file_key=self.data_ingestion_config.s3_data_file_key)
            logger.info("Successfully fetch data form s3(%s).", self.data_ingestion_config.s3_bucket)

            logger.debug("Performing basic preprocessing on fetched data...")
            training_data, test_data = self.basic_preprocessing(df=df)

            logger.debug("Saving training_data at :%s", self.data_ingestion_config.training_data_file_path)
            save_dataframe_as_csv(file_save_path=self.data_ingestion_config.training_data_file_path, dataframe=training_data, index=False, logger=logger)
            logger.info("Training Data Saved Successfully at: %s", self.data_ingestion_config.training_data_file_path)
            
            logger.debug("Saving test_data at :%s", self.data_ingestion_config.test_data_file_path)
            save_dataframe_as_csv(file_save_path=self.data_ingestion_config.test_data_file_path, dataframe=test_data, index=False, logger=logger)
            logger.info("Test Data Saved Successfully at: %s", self.data_ingestion_config.test_data_file_path)
            
            data_ingestion_artifact = DataIngestionArtifact(training_data_file_path=self.data_ingestion_config.training_data_file_path,
                                                              test_data_file_path=self.data_ingestion_config.test_data_file_path,
                                                              raw_data_file_path=self.data_ingestion_config.raw_data_file_path)

            logger.info(f"Data Ingestion Artifact Created: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise DetailedException(exc=e, logger=logger)