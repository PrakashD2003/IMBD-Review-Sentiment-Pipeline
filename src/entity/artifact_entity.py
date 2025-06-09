from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    training_data_file_path:str 
    test_data_file_path:str
    raw_data_file_path:str

@dataclass
class DataPreprocessingArtifact:
    preprocessed_training_data_file_path:str
    preprocessed_test_data_file_path:str

@dataclass
class FeatureEngineeringArtifact:
    feature_engineered_training_data_file_path:str 
    feature_engineered_test_data_file_path:str
    vectorizer_obj_file_path:str 

@dataclass
class ModelTrainerArtifact:
    trained_model_obj_path:str 
    
@dataclass
class ModelEvaluationArtifact:
    performance_metrics_file_path:str