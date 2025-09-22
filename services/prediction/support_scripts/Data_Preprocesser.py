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


def preprocess_data(dataframe:ddf.DataFrame, col:str)->ddf.DataFrame:
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


