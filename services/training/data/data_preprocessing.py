"""Data preprocessing utilities.

Applies URL removal, tokenization, stop-word filtering and lemmatization
to prepare text data for feature extraction using Dask.
"""

import re
import pandas as pd
from pathlib import Path

import nltk
import dask.dataframe as ddf
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from common.logger import configure_logger
from common.exception import DetailedException
from services.training.entity.config_entity import DataPreprocessingConfig
from services.training.entity.artifact_entity import DataIngestionArtifact, DataPreprocessingArtifact
from common.utils.main_utils import load_params, load_parquet_as_dask_dataframe, save_dask_dataframe_as_parquet
from common.constants import PARAM_FILE_PATH, SINGLE_FILE


# Download required NLTK data
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("stopwords", quiet=True)

# Force WordNet to load once, setting up its private internals
_warmup = WordNetLemmatizer().lemmatize("test")

module_name = Path(__file__).stem

logger = configure_logger(
    logger_name=module_name,
    level="DEBUG",
    to_console=True,
    to_file=True,
    log_file_name=module_name
)

# Precompile regex patterns and resources at module level
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
PUNCT_PATTERN = re.compile(r"[^\w\s]")
# Keep most stopwords, but exclude common negation words
negation_words = {"not", "no", "never", "none", "nor", "don't", "doesn't", "didn't", "couldn't", "won't", "wouldn't", "isn't", "aren't"}
STOPWORDS = set(stopwords.words('english')) - negation_words
LEMMATIZER = WordNetLemmatizer()


def preprocess_text(text: str) -> str:
    """
    Clean a single text string by:
      - Removing URLs
      - Removing punctuation
      - Tokenizing and lowercasing
      - Removing numeric tokens
      - Removing English stopwords
      - Lemmatizing tokens
    """
    # Remove URLs
    cleaned = URL_PATTERN.sub("", text)
    # Remove punctuation
    cleaned = PUNCT_PATTERN.sub("", cleaned)
    # Split into tokens
    tokens = cleaned.split()
    # Lowercase and remove digits
    tokens = [w.lower() for w in tokens if not w.isdigit()]
    # Remove stopwords
    tokens = [w for w in tokens if w not in STOPWORDS]
    # Lemmatize
    tokens = [LEMMATIZER.lemmatize(w) for w in tokens]
    return " ".join(tokens)

class DataPreprocessing:
    """
    Applies text-based cleaning to given column of data using Dask. The steps include:
      1. Removing null rows
      2. Removing URLs
      3. Removing punctuation
      4. Splitting into tokens
      5. Lowercasing
      6. Removing numeric tokens
      7. Removing stopwords
      8. Lemmatization
    """
    def __init__(self, data_preprocessing_config:DataPreprocessingConfig = DataPreprocessingConfig(),
                 data_ingestion_artifact:DataIngestionArtifact = DataIngestionArtifact()):
        try:
            """
            Initialize the DataPreprocessing component.

            :param data_preprocessing_config: Holds file paths for saving preprocessed data.
            :param data_ingestion_artifact: Contains paths to ingested train/test CSVs.
            :param logger: Optional Logger. If None, DEFAULT_LOGGER is used.
            :raises DetailedException: If loading parameters fails or required keys are missing.
            """
            logger.debug("Configuring 'DataPreprocessing' class of data module through constructer...")
            self.data_preprocessing_config = data_preprocessing_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.params = load_params(params_path=PARAM_FILE_PATH,logger=logger)
            logger.info("DataPreprocessing class configured successfully.")
        except Exception as e:
            raise DetailedException(exc=e, logger=logger) from e
        
    def preprocess_data(self,dataframe:ddf.DataFrame, col:str)->ddf.DataFrame:
        """     
        Apply `preprocess_text` to each row in the specified column.

        :param dataframe: Dask DataFrame containing the column to clean.
        :param col: Column name for text preprocessing.
        :return: Dask DataFrame with cleaned text column.
        :raises DetailedException: On any error during mapping.
        """
        try:
            logger.info("Entering 'preprocess_data' function of 'DataPreprocessing' class of 'data' module...")
            logger.debug("Starting Dask preprocessing on partitioned DataFrame.")
            logger.debug("Removing 'Null' values from dataframe... ")
            dataframe = dataframe.dropna()
            logger.info("Removed 'Null' values successfully.")
            
            logger.info(f"Preprocessing column: {col}")
            if isinstance(dataframe, pd.DataFrame):
                # pandas: no meta
                dataframe[col] = dataframe[col].map(preprocess_text)
                logger.info("Column preprocessing complete.")
                return dataframe
            else:
                dataframe[col] = dataframe[col].map(preprocess_text, meta = (col, "object"))
                logger.debug("Exiting 'preprocess_data' function of 'DataPreprocessing' class of 'data' module.")
                return dataframe     
        except Exception as e: 
            raise DetailedException(exc=e, logger=logger) from e
    
    def initiate_data_preprocessing(self) ->DataPreprocessingArtifact:
        try:
            """
            1. Load ingested training CSV into a Dask DataFrame.
            2. Preprocess the specified text column.
            3. Load ingested test CSV into a Dask DataFrame.
            4. Preprocess the specified text column.
            5. Save both preprocessed DataFrames to disk as CSV,
            either as a single file or multiple partitioned files.

            :return: A DataPreprocessingArtifact containing output file paths.
            :raises DetailedException: If any step (load, preprocess, save) fails.
            """
            logger.info("Entered initiate_data_preprocessing method of 'DataPreprocessing' class")
            logger.info("\n" + "-" * 80)
            logger.info("Starting Data Preprocessing Component...")

            logger.debug("Loading Training Data...")
            training_data = load_parquet_as_dask_dataframe(self.data_ingestion_artifact.training_data_file_path, logger=logger)

            logger.debug("Performing Preprocessing on '%s' column of training data...", self.params["data_preprocessing_params"]["text_column_for_preprocessing"])
            training_data = self.preprocess_data(dataframe=training_data, col=self.params["data_preprocessing_params"]["text_column_for_preprocessing"])
            logger.info("Successfully Preprocesses training data.")

            logger.debug("Loading Test Data...")
            test_data = load_parquet_as_dask_dataframe(self.data_ingestion_artifact.test_data_file_path, logger=logger)

            logger.debug("Performing Preprocessing on '%s' column of test data...", self.params["data_preprocessing_params"]["text_column_for_preprocessing"])
            test_data = self.preprocess_data(dataframe=test_data, col=self.params["data_preprocessing_params"]["text_column_for_preprocessing"])
            logger.info("Successfully Preprocesses test data.")

            logger.debug("Saving Preprocessed data...")
            save_dask_dataframe_as_parquet(file_save_path=self.data_preprocessing_config.preprocessed_trained_data_file_path, dataframe=training_data, single_file=SINGLE_FILE,index=False, logger=logger)
            save_dask_dataframe_as_parquet(file_save_path=self.data_preprocessing_config.preprocessed_test_data_file_path, dataframe=test_data, single_file=SINGLE_FILE, index=False, logger=logger)
            logger.info("Preprocessed data saved successfully.")


            logger.debug("Exiting 'initiate_data_preprocessing' function of 'DataPreprocessing' class of 'data' module.")
            return DataPreprocessingArtifact(
                preprocessed_training_data_file_path = self.data_preprocessing_config.preprocessed_trained_data_file_path,
                preprocessed_test_data_file_path = self.data_preprocessing_config.preprocessed_test_data_file_path
            )
        except Exception as e:
            raise DetailedException(exc=e, logger=logger) from e

if __name__ == "__main__":
    data_preprocessing = DataPreprocessing()
    data_preprocessing.initiate_data_preprocessing()     
 