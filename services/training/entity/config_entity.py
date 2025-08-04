"""Configuration dataclasses for each pipeline component."""

import os
from dataclasses import dataclass
from common.constants import *

DATA_DIR_PATH:str = os.path.join(ARTIFACT_DIR_NAME, DATA_DIR_NAME)
METRICS_DIR_PATH:str = os.path.join(ARTIFACT_DIR_NAME, METRICS_DIR_NAME)
OBJECT_SAVE_DIR_PATH:str = os.path.join(ARTIFACT_DIR_NAME, OBJ_SAVE_DIR_NAME)
@dataclass
class DataIngestionConfig:
    data_ingestion_dir:str = os.path.join(DATA_DIR_PATH,DATA_INGESTION_DIR_NAME)
    raw_data_file_path:str = os.path.join(DATA_DIR_PATH, RAW_DATA_FILE_DIR_NAME)
    training_data_file_path:str = os.path.join(data_ingestion_dir,TRAINING_DATA_FILE_NAME)
    test_data_file_path:str = os.path.join(data_ingestion_dir,TEST_DATA_FILE_NAME)
    s3_bucket:str = S3_DATA_BUCKET_ENV or S3_DATA_BUCKET
    s3_data_file_key:str = S3_DATA_FILE_PREFIX_ENV or S3_DATA_FILE_PREFIX
    # dask_partition_block_size = DASK_PARTITION_BLOCK_SIZE
    n_partitions:str = N_PARTITIONS

@dataclass
class DataPreprocessingConfig:
    preprocessed_data_dir:str = os.path.join(DATA_DIR_PATH, DATA_PREPROCESSING_DIR)
    preprocessed_trained_data_file_path:str = os.path.join(preprocessed_data_dir, PREPROCESSED_TRAINED_DATA_FILE_NAME)
    preprocessed_test_data_file_path:str = os.path.join(preprocessed_data_dir, PREPROCESSED_TEST_DATA_FILE_NAME)

@dataclass
class FeatureEngineeringConfig:
    feature_engineered_data_dir:str = os.path.join(DATA_DIR_PATH, FEATURE_ENGINEERING_DATA_DIR)
    vectorizer_obj_dir:str = os.path.join(OBJECT_SAVE_DIR_PATH, VECTORIZER_OBJ_DIR)
    vectorizer_obj_file_path:str = os.path.join(vectorizer_obj_dir, VECTORIZER_OBJ_FILE_NAME)
    feature_engineered_training_data_file_path:str = os.path.join(feature_engineered_data_dir, FEATURE_ENGINEERED_TRAINING_DATA_FILE_NAME)
    feature_engineered_test_data_file_path:str = os.path.join(feature_engineered_data_dir, FEATURE_ENGINEERED_TEST_DATA_FILE_NAME)
@dataclass
class ModelTrainerConfig:
    model_obj_dir:str = os.path.join(OBJECT_SAVE_DIR_PATH, TRAINED_MODEL_OBJ_DIR)
    trained_model_obj_path:str = os.path.join(model_obj_dir, TRAINED_MODEL_OBJ_NAME)

@dataclass
class ModelEvaluationConfig:
    performance_metrics_file_save_path:str = os.path.join(METRICS_DIR_PATH, PERFORMANCE_METRICS_FILE_NAME)
    experiment_info_file_save_paht:str = os.path.join(METRICS_DIR_PATH, EXPERIMENT_INFO_FILE_NAME)
    

@dataclass
class ModelRegistryConfig:
    mlflow_uri:str = MLFLOW_TRACKING_URI
    dagshub_repo_name:str = DAGSHUB_REPO_NAME
    dagshub_repo_owner_name:str = DAGSHUB_REPO_OWNER_NAME
    experiment_name:str = EXPERIMENT_NAME
    mlflow_model_name:str = MODEL_NAME
    mlflow_model_stage:str = MODEL_REGISTRY_MODEL_STAGE
    mlflow_vectorizer_name:str = VECTORIZER_NAME
    mlflow_model_artifact_path:str = MODEL_ARTIFACT_PATH
    mlflow_vectorizer_artifact_path:str = VECTORIZER_ARTIFACT_PATH

@dataclass
class PredictionPipelineConfig:
    mlflow_uri:str = MLFLOW_TRACKING_URI
    dagshub_repo_name:str = DAGSHUB_REPO_NAME
    dagshub_repo_owner_name:str = DAGSHUB_REPO_OWNER_NAME
    mlflow_model_name:str = MODEL_NAME
    mlflow_vectorizer_name:str = VECTORIZER_NAME
    mlflow_model_stages:list[str] = STAGES
    
    
@dataclass
class BatchPredictionConfig:
    mlflow_uri:str = MLFLOW_TRACKING_URI
    dagshub_repo_name:str = DAGSHUB_REPO_NAME
    dagshub_repo_owner_name:str = DAGSHUB_REPO_OWNER_NAME
    mlflow_model_name:str = MODEL_NAME
    mlflow_vectorizer_name:str = VECTORIZER_NAME
    mlflow_model_stages:list[str] = STAGES
    input_path:str = ""
    output_path:str = ""