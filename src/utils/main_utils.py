import os
import yaml
from logger import configure_logger
from exception import DetailedException
from typing import Optional
import logging
import pandas as pd

# 1) Configuring the module‐level default logger exactly once, on import
DEFAULT_LOGGER = configure_logger(
    logger_name=__name__,
    level="DEBUG",
    to_console=True,
    to_file=True,
    log_file_name=__name__  # or f"{__name__}.log"
)


def load_params(params_path: str, logger: Optional[logging.Logger] = None) -> dict:
    """
    Load a YAML configuration file into a Python dictionary, with logging.

    If `logger` is None, a default one is Used via `configure_logger`.
    Logs at DEBUG level on entry, INFO on success, and ERROR on failure.

    :param params_path: Path to the YAML file containing parameters.
    :param logger: Optional Logger instance. If omitted, one will be configured.
    :return: Dictionary of parameters parsed from the YAML file.
    :raises FileNotFoundError: If the file doesn’t exist.
    :raises yaml.YAMLError: If the file isn’t valid YAML.
    :raises DetailedException: For any other unexpected errors, including traceback info.
    """
    try:
        logger = logger or DEFAULT_LOGGER

        logger.debug("Entered load_params function of utils module.")
        logger.debug("Loading parameters from: %s", params_path)

        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)

        logger.info("Parameters have been successfully loaded from: %s", params_path)
        logger.debug("Exiting 'load_params' function and returning 'params'.")
        return params

    except FileNotFoundError:
        logger.error("File not found: %s", params_path)
        raise

    except yaml.YAMLError as e:
        logger.error("YAML error while loading %s: %s", params_path, e)
        raise

    except Exception as e:
        # Wrap unexpected errors for consistent formatting + traceback info
        raise DetailedException(exc=e, logger=logger) from e

def load_csv_data(file_path: str, logger: Optional[logging.Logger] = None)->pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame, with optional logging.

    This function attempts to read the CSV at `file_path` into a DataFrame.
    It logs DEBUG messages when entering and before reading, an INFO on success,
    and an ERROR if the file is missing or fails to parse. Any other exception
    is wrapped in DetailedException to include traceback location.

    :param file_path: Path to the CSV file to load.
    :param logger: Optional Logger instance. If None, uses DEFAULT_LOGGER.
    :return: pandas.DataFrame containing the CSV data.
    :raises FileNotFoundError: If the CSV file does not exist.
    :raises pd.errors.ParserError: If pandas cannot parse the CSV.
    :raises DetailedException: For any other unexpected errors.
    """
    try:
        logger = logger or DEFAULT_LOGGER

        logger.debug("Entered load_csv_data function of utils module.")
        logger.debug("Loading Csv data from: %s", file_path)
        dataframe  = pd.read_csv(file_path)
        logger.info("Data has been successfully loaded from: %s ",file_path)
        logger.debug("Exiting 'load_csv_data' function and returning 'dataframe'.")
        return dataframe
    except FileNotFoundError as e:
        logger.error("File not found: %s", file_path)
        raise
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the csv file: %s", e)
        raise
    except Exception as e:
        raise DetailedException(exc=e, logger=logger) from e


def save_dataframe_as_csv(file_save_path: str, dataframe: pd.DataFrame, index: bool = False, logger: Optional[logging.Logger] = None) -> None:
    """
    Save a pandas DataFrame to a CSV file, creating parent directories as needed.

    :param file_save_path: Full path (including filename) where the CSV should be written.
    :param dataframe: The pandas DataFrame to save.
    :param index: Whether to write row names (index) into the CSV. Defaults to False.
    :param logger: Optional Logger; if None, uses DEFAULT_LOGGER.
    :raises DetailedException: Wraps any unexpected exception that occurs during directory
                               creation or file writing, adding file/line context.
    """
    logger = logger or DEFAULT_LOGGER
    try:
        logger.debug("Entered save_dataframe_as_csv; target path: %s", file_save_path)

        # Ensure parent directory exists
        parent_dir = os.path.dirname(file_save_path) or "."
        logger.debug("Creating parent directory if missing: %s", parent_dir)
        os.makedirs(parent_dir, exist_ok=True)
        logger.info("Parent directory ready: %s", parent_dir)

        # Write DataFrame to CSV
        logger.debug("Writing DataFrame to CSV at: %s", file_save_path)
        dataframe.to_csv(file_save_path, index=index)
        logger.info("DataFrame successfully saved to CSV: %s", file_save_path)

    except Exception as e:
        # Any failure gets wrapped for consistent, detailed logging
        raise DetailedException(exc=e, logger=logger) from e
