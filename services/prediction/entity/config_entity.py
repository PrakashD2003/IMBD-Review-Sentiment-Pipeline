"""Configuration dataclasses for each pipeline component."""

from dataclasses import dataclass
from common.constants import *

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