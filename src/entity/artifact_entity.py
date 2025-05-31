from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    training_data_file_path:str 
    test_data_file_path:str
    raw_data_file_path:str