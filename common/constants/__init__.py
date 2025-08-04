"""Central location for project-wide constants and configuration values."""

import os
from dataclasses import field


# Read the SINGLE_FILE variable and interpret common truthy values
_sflag = os.getenv("SINGLE_FILE", "").strip().lower()
SINGLE_FILE: bool = _sflag in ("1", "true", "yes")


### Params(yaml file) Constants ###
PARAM_FILE_PATH = "params.yaml"

### Data Ingestion Constants ###
ARTIFACT_DIR_NAME:str = "artifacts"
DATA_DIR_NAME:str = "data"
DATA_INGESTION_DIR_NAME:str = "interim"
RAW_DATA_FILE_DIR_NAME = "raw"

TRAINING_DATA_FILE_NAME: str = "train"
TEST_DATA_FILE_NAME:     str = "test"
S3_DATA_BUCKET_ENV:str = os.getenv("S3_DATA_BUCKET_ENV")
S3_DATA_BUCKET:str = "imbd-capstone-proj-bucket"
S3_DATA_FILE_PREFIX_ENV:str = os.getenv("S3_DATA_FILE_NAME_ENV")
S3_DATA_FILE_PREFIX:str = "data/IMDB Dataset.parquet"
N_PARTITIONS: int = int(os.getenv("N_PARTITONS", "3"))


### Data Preprocessing Constants ###
DATA_PREPROCESSING_DIR:str = "processed"
PREPROCESSED_TRAINED_DATA_FILE_NAME = "preprocessed_training_data"
PREPROCESSED_TEST_DATA_FILE_NAME = "preprocessed_test_data"

    

### Feature Engineering Constants ###
OBJ_SAVE_DIR_NAME:str = "saved_models"
VECTORIZER_OBJ_DIR:str = "vectorizer"
VECTORIZER_OBJ_FILE_NAME:str = "vectorizer.pkl"
FEATURE_ENGINEERING_DATA_DIR:str = "feature_engineered"
FEATURE_ENGINEERED_TRAINING_DATA_FILE_NAME:str = "feature_engineered_training_data"
FEATURE_ENGINEERED_TEST_DATA_FILE_NAME:str = "feature_engineered_test_data"


### Model Training Constants ###
TRAINED_MODEL_OBJ_DIR:str = "model"
TRAINED_MODEL_OBJ_NAME:str = "model.pkl"

### Model Evaluation Constants ###
METRICS_DIR_NAME:str = "reports"
PERFORMANCE_METRICS_FILE_NAME:str = "performance_metrics.json" 
EXPERIMENT_INFO_FILE_NAME:str = "experiment_info.json"

### Model Registry ###
MODEL_ARTIFACT_PATH      = "model"
VECTORIZER_ARTIFACT_PATH = "vectorizer"


################################################# CSV Variables ####################################################
# ### Params(yaml file) Constants ###
# PARAM_FILE_PATH = "params.yaml"

# ### Data Ingestion Constants ###
# DATA_DIR_NAME:str = "data"
# DATA_INGESTION_DIR_NAME:str = "interim"
# RAW_DATA_FILE_DIR_NAME = "raw"
# if SINGLE_FILE:
#     TRAINING_DATA_FILE_NAME: str = "train.csv"
#     TEST_DATA_FILE_NAME:     str = "test.csv"
# else:
#     TRAINING_DATA_FILE_NAME: str = "train-*.csv"
#     TEST_DATA_FILE_NAME:     str = "test-*.csv"
# S3_DATA_BUCKET_ENV:str = os.getenv("S3_DATA_BUCKET_ENV")
# S3_DATA_BUCKET:str = "imbd-capstone-proj-bucket"
# S3_DATA_FILE_PREFIX_ENV:str = os.getenv("S3_DATA_FILE_NAME_ENV")
# S3_DATA_FILE_PREFIX:str = "data/IMDB Dataset.csv"
# DASK_PARTITION_BLOCK_SIZE:str = os.getenv("DASK_PARTITION_BLOCK_SIZE") or "16 Mib"

# ### Data Preprocessing Constants ###
# DATA_PREPROCESSING_DIR:str = "processed"
# if SINGLE_FILE:
#     PREPROCESSED_TRAINED_DATA_FILE_NAME = "preprocessed_training_data.csv"
#     PREPROCESSED_TEST_DATA_FILE_NAME = "preprocessed_test_data.csv"
# else:
#     PREPROCESSED_TRAINED_DATA_FILE_NAME = "preprocessed_training_data-*.csv"
#     PREPROCESSED_TEST_DATA_FILE_NAME = "preprocessed_test_data-*.csv"

# ### Feature Engineering Constants ###
# OBJ_SAVE_DIR:str = "saved_models"
# VECTORIZER_OBJ_DIR:str = "vectorizer"
# VECTORIZER_OBJ_FILE_NAME:str = "vectorizer.pkl"
# FEATURE_ENGINEERING_DATA_DIR:str = "feature_engineered"
# if SINGLE_FILE:
#     FEATURE_ENGINEERED_TRAINING_DATA_FILE_NAME:str = "feature_engineered_training_data.csv"
#     FEATURE_ENGINEERED_TEST_DATA_FILE_NAME:str = "feature_engineered_test_data.csv"
# else:
#     FEATURE_ENGINEERED_TRAINING_DATA_FILE_NAME:str = "feature_engineered_training_data-*.csv"
#     FEATURE_ENGINEERED_TEST_DATA_FILE_NAME:str = "feature_engineered_test_data-*.csv"

# ### Model Training Constants ###
# TRAINED_MODEL_OBJ_DIR:str = "model"
# TRAINED_MODEL_OBJ_NAME:str = "model.pkl"

# ### Model Evaluation Constants ###
# METRICS_DIR_PATH:str = "reports"
# PERFORMANCE_METRICS_FILE_NAME:str = "performance_metrics.json" 
# EXPERIMENT_INFO_FILE_NAME:str = "experiment_info.json"
# MLFLOW_REGISTRY_MODEL_NAME:str = "model"
# MLFLOW_REGISTRY_VECTORIZER_NAME:str = "vectorizer"

################################################# ENV Variables ####################################################
# SINGLE_FILE

### Data Ingestion Constants ###
# S3_DATA_BUCKET_ENV:str = "S3_DATA_BUCKET_ENV"
# S3_DATA_FILE_PREFIX_ENV:str = "S3_DATA_FILE_NAME_ENV"
# DASK_PARTITION_BLOCK_SIZE:str = "DASK_PARTITION_BLOCK_SIZE"
# N_PARTITIONS: int


### AWS Credentials ###
AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION: str = os.getenv("AWS_REGION")

### Mlflow Variables ###
MLFLOW_TRACKING_URI:str = os.getenv("MLFLOW_TRACKING_URI")
DAGSHUB_REPO_OWNER_NAME:str = os.getenv("DAGSHUB_REPO_OWNER_NAME")
DAGSHUB_REPO_NAME:str = os.getenv("DAGSHUB_REPO_NAME")
EXPERIMENT_NAME:str =  os.getenv("EXPERIMENT_NAME") or "MY-DVC-PIPELINE"
MODEL_NAME:str = os.getenv("MODEL_NAME") or "IMBD-REVIEW-SENTIMENT-ANALYSIS-PROJECT-MODEL"
VECTORIZER_NAME:str = os.getenv("VECTORIZER_NAME") or "IMBD-REVIEW-SENTIMENT-ANALYSIS-PROJECT-VECTORIZER"
MODEL_REGISTRY_MODEL_STAGE:str = "Staging"


### Trainig Pipeline Variables ###
DASK_SCHEDULER_ADDRESS = os.getenv("DASK_SCHEDULER_ADDRESS")
# Environment configuration for local Dask clusters
DASK_WORKERS = int(os.getenv("DASK_WORKERS", "1"))
DASK_THREADS = int(os.getenv("DASK_THREADS", "2"))
DASK_MEMORY_LIMIT = os.getenv("DASK_MEMORY_LIMIT", "12GB")

### Prediction Pipeline Variables ###
STAGES:list[str] = field(default_factory=lambda: ["Staging"])

### Fast Api ###
PORT:int 