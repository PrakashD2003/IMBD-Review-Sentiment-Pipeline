from dataclasses import dataclass
from src.entity.config_entity import *

data_ingestion_config = DataIngestionConfig()
@dataclass
class DataIngestionArtifact:
    training_data_file_path:str = data_ingestion_config.training_data_file_path 
    test_data_file_path:str = data_ingestion_config.test_data_file_path
    raw_data_file_path:str = data_ingestion_config.raw_data_file_path

data_preprocessing_config = DataPreprocessingConfig()
@dataclass
class DataPreprocessingArtifact:
    preprocessed_training_data_file_path:str = data_preprocessing_config.preprocessed_trained_data_file_path
    preprocessed_test_data_file_path:str = data_preprocessing_config.preprocessed_test_data_file_path

feature_engineering_config = FeatureEngineeringConfig()
@dataclass
class FeatureEngineeringArtifact:
    feature_engineered_training_data_file_path:str = feature_engineering_config.feature_engineered_training_data_file_path
    feature_engineered_test_data_file_path:str = feature_engineering_config.feature_engineered_test_data_file_path
    vectorizer_obj_file_path:str = feature_engineering_config.vectorizer_obj_file_path

model_trainer_config = ModelTrainerConfig()
@dataclass
class ModelTrainerArtifact:
    trained_model_obj_path:str = model_trainer_config.trained_model_obj_path 
    
model_evaluation_config = ModelEvaluationConfig()
@dataclass
class ModelEvaluationArtifact:
    performance_metrics_file_path:str = model_evaluation_config.performance_metrics_file_save_path