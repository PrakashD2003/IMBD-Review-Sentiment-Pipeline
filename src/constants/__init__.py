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
TRAINING_DATA_FILE_NAME:str = "train.csv"
TEST_DATA_FILE_NAME:str = "test.csv"
S3_DATA_BUCKET_ENV:str = "S3_DATA_BUCKET_ENV"
S3_DATA_BUCKET:str = "imbd-capstone-proj-bucket"
S3_DATA_FILE_NAME_ENV:str = "S3_DATA_FILE_NAME_ENV"
S3_DATA_FILE_NAME:str = ""

### Data Preprocessing Constants ###
DATA_PREPROCESSING_DIR:str = "processed"
PREPROCESSED_TRAINED_DATA_FILE_NAME = "preprocessed_training_data.csv"
PREPROCESSED_TEST_DATA_FILE_NAME = "preprocessed_test_data.csv"

