import os
from dataclasses import dataclass
from logger.global_logging import LOG_SESSION_TIME
from constants import *

@dataclass
class DataIngestionConfig:
    data_ingestion_dir:str = os.path.join(DATA_DIR_NAME,DATA_INGESTION_DIR_NAME)
    raw_data_file_path:str = os.path.join(RAW_DATA_FILE_DIR_NAME, f"{LOG_SESSION_TIME}.csv")
    training_data_file_path:str = os.path.join(data_ingestion_dir,TRAINING_DATA_FILE_NAME)
    test_data_file_path:str = os.path.join(data_ingestion_dir,TEST_DATA_FILE_NAME)
    s3_bucket:str = os.getenv(S3_DATA_BUCKET_ENV) or S3_DATA_BUCKET
    s3_data_file_key:str = os.getenv(S3_DATA_BUCKET_ENV) or S3_DATA_FILE_NAME

@dataclass
class DataPreprocessingConfig:
    preprocessed_data_dir:str = os.path.join(DATA_DIR_NAME, DATA_PREPROCESSING_DIR)
    preprocessed_trained_data_file_path:str = os.path.join(preprocessed_data_dir, PREPROCESSED_TRAINED_DATA_FILE_NAME)
    preprocessed_test_data_file_path:str = os.path.join(preprocessed_data_dir, PREPROCESSED_TEST_DATA_FILE_NAME)