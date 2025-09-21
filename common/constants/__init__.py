# In common/constants/__init__.py

import os
from dataclasses import field

# --- General Execution Flags ---
# Controls Dask output behavior; set to true via env var for single-file outputs.
_sflag = os.getenv("SINGLE_FILE", "false").strip().lower()
SINGLE_FILE: bool = _sflag in ("1", "true", "yes")


# --- File Path Constants ---
PARAM_FILE_PATH: str = "params.yaml"  # Name of the DVC parameters file.


# --- Artifact Directory & File Name Structure ---
# These constants define the folder and file names for storing pipeline outputs.
ARTIFACT_DIR_NAME: str = "artifacts"
DATA_DIR_NAME: str = "data"
# Ingestion stage
DATA_INGESTION_DIR_NAME: str = "interim"
RAW_DATA_FILE_DIR_NAME: str = "raw"
TRAINING_DATA_FILE_NAME: str = "train"
TEST_DATA_FILE_NAME: str = "test"
# Preprocessing stage
DATA_PREPROCESSING_DIR: str = "processed"
PREPROCESSED_TRAINED_DATA_FILE_NAME: str = "preprocessed_training_data"
PREPROCESSED_TEST_DATA_FILE_NAME: str = "preprocessed_test_data"
# Feature Engineering stage
FEATURE_ENGINEERING_DATA_DIR: str = "feature_engineered"
FEATURE_ENGINEERED_TRAINING_DATA_FILE_NAME: str = "feature_engineered_training_data"
FEATURE_ENGINEERED_TEST_DATA_FILE_NAME: str = "feature_engineered_test_data"
# Saved Objects (Models, Vectorizers)
OBJ_SAVE_DIR_NAME: str = "saved_models"
VECTORIZER_OBJ_DIR: str = "vectorizer"
VECTORIZER_OBJ_FILE_NAME: str = "vectorizer.pkl"
TRAINED_MODEL_OBJ_DIR: str = "model"
TRAINED_MODEL_OBJ_NAME: str = "model.pkl"
# Reports
METRICS_DIR_NAME: str = "reports"
PERFORMANCE_METRICS_FILE_NAME: str = "performance_metrics.json"
EXPERIMENT_INFO_FILE_NAME: str = "experiment_info.json"


# --- Data Source & Partitioning (from Environment) ---
# S3 location for the raw IMDB dataset.
S3_DATA_BUCKET: str = os.getenv("S3_DATA_BUCKET")
# S3 key (path) to the raw dataset file within the bucket.
S3_DATA_FILE_PREFIX: str = os.getenv("S3_DATA_FILE_PREFIX")
# Default number of partitions for Dask DataFrames.
N_PARTITIONS: int = int(os.getenv("N_PARTITIONS", "3"))


# --- AWS Credentials (from Environment) ---
# Credentials for connecting to AWS services like S3.
AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION: str = os.getenv("AWS_REGION")


# --- MLflow & DagsHub Configuration (from Environment) ---
# Connection details for the MLflow tracking server and DagsHub.
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI")
DAGSHUB_REPO_OWNER_NAME: str = os.getenv("DAGSHUB_REPO_OWNER_NAME")
DAGSHUB_REPO_NAME: str = os.getenv("DAGSHUB_REPO_NAME")
# Names for MLflow experiments and registered models.
EXPERIMENT_NAME: str = os.getenv("EXPERIMENT_NAME")
MODEL_NAME: str = os.getenv("MODEL_NAME")
VECTORIZER_NAME: str = os.getenv("VECTORIZER_NAME")
# Default stage for newly registered models.
MODEL_REGISTRY_MODEL_STAGE: str = "Staging"
# Artifact paths used within an MLflow run.
MODEL_ARTIFACT_PATH: str = "model"
VECTORIZER_ARTIFACT_PATH: str = "vectorizer"


# --- Dask Cluster Configuration (from Environment) ---
# Address of the Dask scheduler for distributed processing.
DASK_SCHEDULER_ADDRESS: str = os.getenv("DASK_SCHEDULER_ADDRESS")
# Fallback configuration for creating a local Dask cluster if no scheduler is specified.
DASK_WORKERS: int = int(os.getenv("DASK_WORKERS") or "1")
DASK_THREADS: int = int(os.getenv("DASK_THREADS") or "2")
DASK_MEMORY_LIMIT: str = os.getenv("DASK_MEMORY_LIMIT") or "12GB"


# --- Prediction Service Configuration ---
# Default model stages for the prediction service to load from the registry.
# The string is split by commas to allow for multiple stages if needed.
STAGES: list[str] = os.getenv("MLFLOW_MODEL_STAGES", "Staging").split(",")
# Port for the FastAPI application.
PORT: int