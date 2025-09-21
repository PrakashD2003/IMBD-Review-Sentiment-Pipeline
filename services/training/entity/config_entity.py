"""Configuration dataclasses for each pipeline component."""

# In services/training/entity/config_entity.py

import os
from dataclasses import dataclass, field
from common.constants import *

# Base directory paths, defined once
DATA_DIR_PATH: str = os.path.join(ARTIFACT_DIR_NAME, DATA_DIR_NAME)
METRICS_DIR_PATH: str = os.path.join(ARTIFACT_DIR_NAME, METRICS_DIR_NAME)
OBJECT_SAVE_DIR_PATH: str = os.path.join(ARTIFACT_DIR_NAME, OBJ_SAVE_DIR_NAME)

@dataclass
class DataIngestionConfig:
    # Define base directories and external configs
    data_ingestion_dir: str = os.path.join(DATA_DIR_PATH, DATA_INGESTION_DIR_NAME)
    raw_data_dir: str = os.path.join(DATA_DIR_PATH, RAW_DATA_FILE_DIR_NAME)
    s3_bucket: str = S3_DATA_BUCKET
    s3_data_file_key: str = S3_DATA_FILE_PREFIX
    n_partitions: int = N_PARTITIONS

    # Initialize dependent paths as empty
    raw_data_file_path: str = field(init=False)
    training_data_file_path: str = field(init=False)
    test_data_file_path: str = field(init=False)

    def __post_init__(self):
        # Construct full paths after the object is created
        os.makedirs(self.data_ingestion_dir, exist_ok=True)
        os.makedirs(self.raw_data_dir, exist_ok=True)
        self.raw_data_file_path = os.path.join(self.raw_data_dir, "raw_data.parquet")
        self.training_data_file_path = os.path.join(self.data_ingestion_dir, TRAINING_DATA_FILE_NAME)
        self.test_data_file_path = os.path.join(self.data_ingestion_dir, TEST_DATA_FILE_NAME)

@dataclass
class DataPreprocessingConfig:
    preprocessed_data_dir: str = os.path.join(DATA_DIR_PATH, DATA_PREPROCESSING_DIR)
    
    # Initialize dependent paths
    preprocessed_trained_data_file_path: str = field(init=False)
    preprocessed_test_data_file_path: str = field(init=False)

    def __post_init__(self):
        os.makedirs(self.preprocessed_data_dir, exist_ok=True)
        self.preprocessed_trained_data_file_path = os.path.join(self.preprocessed_data_dir, PREPROCESSED_TRAINED_DATA_FILE_NAME)
        self.preprocessed_test_data_file_path = os.path.join(self.preprocessed_data_dir, PREPROCESSED_TEST_DATA_FILE_NAME)

@dataclass
class FeatureEngineeringConfig:
    feature_engineered_data_dir: str = os.path.join(DATA_DIR_PATH, FEATURE_ENGINEERING_DATA_DIR)
    vectorizer_obj_dir: str = os.path.join(OBJECT_SAVE_DIR_PATH, VECTORIZER_OBJ_DIR)
    
    # Initialize dependent paths
    vectorizer_obj_file_path: str = field(init=False)
    feature_engineered_training_data_file_path: str = field(init=False)
    feature_engineered_test_data_file_path: str = field(init=False)

    def __post_init__(self):
        os.makedirs(self.feature_engineered_data_dir, exist_ok=True)
        os.makedirs(self.vectorizer_obj_dir, exist_ok=True)
        self.vectorizer_obj_file_path = os.path.join(self.vectorizer_obj_dir, VECTORIZER_OBJ_FILE_NAME)
        self.feature_engineered_training_data_file_path = os.path.join(self.feature_engineered_data_dir, FEATURE_ENGINEERED_TRAINING_DATA_FILE_NAME)
        self.feature_engineered_test_data_file_path = os.path.join(self.feature_engineered_data_dir, FEATURE_ENGINEERED_TEST_DATA_FILE_NAME)

@dataclass
class ModelTrainerConfig:
    model_obj_dir: str = os.path.join(OBJECT_SAVE_DIR_PATH, TRAINED_MODEL_OBJ_DIR)
    
    # Initialize dependent path
    trained_model_obj_path: str = field(init=False)

    def __post_init__(self):
        os.makedirs(self.model_obj_dir, exist_ok=True)
        self.trained_model_obj_path = os.path.join(self.model_obj_dir, TRAINED_MODEL_OBJ_NAME)

@dataclass
class ModelEvaluationConfig:
    metrics_dir: str = METRICS_DIR_PATH
    
    # Initialize dependent paths
    performance_metrics_file_save_path: str = field(init=False)
    experiment_info_file_save_path: str = field(init=False)

    def __post_init__(self):
        os.makedirs(self.metrics_dir, exist_ok=True)
        self.performance_metrics_file_save_path = os.path.join(self.metrics_dir, PERFORMANCE_METRICS_FILE_NAME)
        self.experiment_info_file_save_path = os.path.join(self.metrics_dir, EXPERIMENT_INFO_FILE_NAME)
    
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

