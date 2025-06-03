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
