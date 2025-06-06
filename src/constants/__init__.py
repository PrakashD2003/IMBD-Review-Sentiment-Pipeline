import os
### Dask Credentials ###
# Read the SINGLE_FILE variable and interpret common truthy values
_sflag = os.getenv("SINGLE_FILE", "").strip().lower()
SINGLE_FILE: bool = _sflag in ("1", "true", "yes")

### AWS Credentials ###
AWS_ACCESS_KEY_ID: str = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY: str = "AWS_SECRET_ACCESS_KEY"
AWS_REGION: str = "AWS_REGION"

### Params(yaml file) Constants ###
PARAM_FILE_PATH = ""

### Data Ingestion Constants ###
DATA_DIR_NAME:str = "data"
DATA_INGESTION_DIR_NAME:str = "interim"
RAW_DATA_FILE_DIR_NAME = "raw"
if SINGLE_FILE:
    TRAINING_DATA_FILE_NAME: str = "train.csv"
    TEST_DATA_FILE_NAME:     str = "test.csv"
else:
    TRAINING_DATA_FILE_NAME: str = "train-*.csv"
    TEST_DATA_FILE_NAME:     str = "test-*.csv"
S3_DATA_BUCKET_ENV:str = "S3_DATA_BUCKET_ENV"
S3_DATA_BUCKET:str = "imbd-capstone-proj-bucket"
S3_DATA_FILE_NAME_ENV:str = "S3_DATA_FILE_NAME_ENV"
S3_DATA_FILE_NAME:str = ""

### Data Preprocessing Constants ###
DATA_PREPROCESSING_DIR:str = "processed"
if SINGLE_FILE:
    PREPROCESSED_TRAINED_DATA_FILE_NAME = "preprocessed_training_data.csv"
    PREPROCESSED_TEST_DATA_FILE_NAME = "preprocessed_test_data.csv"
else:
    PREPROCESSED_TRAINED_DATA_FILE_NAME = "preprocessed_training_data-*.csv"
    PREPROCESSED_TEST_DATA_FILE_NAME = "preprocessed_test_data-*.csv"

### Feature Engineering Constants ###
MODEL_OBJ_DIR:str = "saved_models"
FEATURE_ENGINEERING_DATA_DIR:str = "feature_engineered"
VECTORIZER_OBJ_DIR:str = "vectorizer"
if SINGLE_FILE:
    FEATURE_ENGINEERED_TRAINING_DATA_FILE_NAME:str = "feature_engineered_training_data.csv"
    FEATURE_ENGINEERED_TEST_DATA_FILE_NAME:str = "feature_engineered_test_data.csv"
else:
    FEATURE_ENGINEERED_TRAINING_DATA_FILE_NAME:str = "feature_engineered_training_data-*.csv"
    FEATURE_ENGINEERED_TEST_DATA_FILE_NAME:str = "feature_engineered_test_data-*.csv"
VECTORIZER_OBJ_FILE_NAME:str = "vectorizer.pkl"



