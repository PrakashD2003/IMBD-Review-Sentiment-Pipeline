"""Data preprocessing utilities.

Applies URL removal, tokenization, stop-word filtering and lemmatization
to prepare text data for feature extraction using Dask.
"""

import re
import logging
import pandas as pd

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


# This logger will automatically inherit the configuration from the FastAPI app
logger = logging.getLogger("prediction-service")

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
      - Handling negations by adding a _NOT suffix
      - Lemmatizing tokens
    """
    # 1. Initial cleaning and tokenization
    cleaned = URL_PATTERN.sub("", text)
    cleaned = PUNCT_PATTERN.sub("", cleaned)
    tokens = cleaned.split()

    # 2. Lowercase and remove stopwords FIRST
    tokens = [w.lower() for w in tokens if not w.isdigit()]
    # This will correctly remove "a" from ["not", "a", "bad", "movie"]
    tokens = [w for w in tokens if w not in STOPWORDS]

    # 3. NOW apply the more robust negation handling on cleaned tokens
    negation_flags = {"not", "no", "never", "n't", "isnt", "wasnt", "arent", "werent", "couldnt", "wouldnt", "shouldnt"}
    processed_tokens = []
    negate_count = 0
    for token in tokens:
        if negate_count > 0:
            processed_tokens.append(token + "_NOT")
            negate_count -= 1
        else:
            processed_tokens.append(token)

        if token in negation_flags:
            negate_count = 2  # Negate the next 2 words

    # 4. Finally, lemmatize
    final_tokens = [LEMMATIZER.lemmatize(w) for w in processed_tokens]
    
    return " ".join(final_tokens)


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


