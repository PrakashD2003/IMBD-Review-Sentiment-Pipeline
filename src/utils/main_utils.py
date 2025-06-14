import os
import json
import dask.dataframe as ddf
from dask.diagnostics import ProgressBar
import yaml
import dill
from src.logger import configure_logger
from src.exception import DetailedException
from typing import Optional
import logging
import pandas as pd

# 1) Configuring the module‐level default logger exactly once, on import
DEFAULT_LOGGER = configure_logger(
    logger_name=__name__,
    level="DEBUG",
    to_console=True,
    to_file=True,
    log_file_name=__name__  
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

        if not os.path.exists(params_path):
            logger.error("File Not Found : %s", params_path)
            raise FileNotFoundError(f"{params_path} does not exist.")

        
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

        if not os.path.exists(file_path):
            logger.error("File Not Found : %s", file_path)
            raise FileNotFoundError(f"{file_path} does not exist.")

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
    try:
        logger = logger or DEFAULT_LOGGER
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
        logger.debug("Exiting 'save_dataframe_as_csv' function of 'main_utils' module.")
    except Exception as e:
        # Any failure gets wrapped for consistent, detailed logging
        raise DetailedException(exc=e, logger=logger) from e


    
def load_csv_as_dask_dataframe(file_path: str, logger: Optional[logging.Logger] = None)-> ddf.DataFrame:
    """
    Load a CSV file from disk into a Dask DataFrame, with logging.

    :param file_path: Path to the CSV file to load.
    :param logger: Optional Logger instance. If None, uses DEFAULT_LOGGER.
    :return: A Dask DataFrame representing the contents of the CSV.
    :raises FileNotFoundError: If the specified file does not exist.
    :raises pd.errors.ParserError: If Dask (via pandas under the hood) cannot parse the CSV.
    :raises DetailedException: For any other unexpected errors, wrapped with traceback info.
    """
    try:
        logger = logger or DEFAULT_LOGGER

        logger.debug("Entered 'load_csv_as_dask_dataframe' function of utils module.")

        # if not os.path.exists(file_path):
        #     logger.error("File Not Found : %s", file_path)
        #     raise FileNotFoundError(f"{file_path} does not exist.")
        
        logger.debug("Loading Csv data from: %s", file_path)
        with ProgressBar():
            dataframe  = ddf.read_csv(file_path)
        logger.info("Data has been successfully loaded from: %s ",file_path)
        logger.debug("Exiting 'load_csv_as_dask_dataframe' function and returning 'dask dataframe'.")
        return dataframe
    except FileNotFoundError as e:
        logger.error("File not found: %s", file_path)
        raise
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the csv file: %s", e)
        raise
    except Exception as e:
        raise DetailedException(exc=e, logger=logger) from e

def load_parquet_as_dask_dataframe(file_path: str, logger: Optional[logging.Logger] = None)-> ddf.DataFrame:
    """
    Load a Parquet dataset from disk into a Dask DataFrame, with logging.

    :param file_path: Path or glob pattern to the Parquet file(s) to load.
                      Supports wildcards (e.g. "data/*.parquet") or directories.
    :param logger: Optional Logger instance. If None, uses DEFAULT_LOGGER.
    :return: A Dask DataFrame representing the contents of the Parquet data.
    :raises FileNotFoundError: If no file matches the provided path or pattern.
    :raises IOError: If the Parquet file(s) cannot be read (e.g. corrupted, mismatched schema).
    :raises DetailedException: For any other errors, wrapped with traceback info.
    """
    try:
        logger = logger or DEFAULT_LOGGER

        logger.debug("Entered 'load_parquet_as_dask_dataframe' function of utils module.")

        # if not os.path.exists(file_path):
        #     logger.error("File Not Found : %s", file_path)
        #     raise FileNotFoundError(f"{file_path} does not exist.")
        
        logger.debug("Loading Parquet data from: %s", file_path)
        with ProgressBar():
            dataframe  = ddf.read_parquet(file_path)
        logger.info("Data has been successfully loaded from: %s ",file_path)
        logger.debug("Exiting 'load_parquet_as_dask_dataframe' function and returning 'dask dataframe'.")
        return dataframe
    except FileNotFoundError as e:
        logger.error("File not found: %s", file_path)
        raise
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the csv file: %s", e)
        raise
    except Exception as e:
        raise DetailedException(exc=e, logger=logger) from e

    
def save_dask_dataframe_as_csv(file_save_path: str, dataframe: ddf.DataFrame, single_file: bool = False,index: bool = False, logger: Optional[logging.Logger] = None) -> None:
    """
    Persist a Dask DataFrame to disk as CSV, optionally combining all partitions into one file.

    By default, Dask writes one CSV per partition under the given path. If `single_file=True`,
    Dask collects partitions on the client and writes a single CSV. Note that writing a single
    file requires all data to pass through the driver, so it’s best for moderate-sized datasets.

    :param file_save_path: 
        Destination path or filename pattern. Examples:
        - "out/processed-*.csv": writes multiple partition files like processed-0.csv, processed-1.csv, etc.
        - "out/processed.csv": when single_file=True, creates a single file at that exact path.
    :param dataframe: The Dask DataFrame to save.
    :param single_file: If True, combine all partitions into one CSV. .
                        If False, writes one CSV per partition under a folder. Defaults to False
    :param index: Whether to include the index in the output CSV(s). Defaults to False.
    :param logger: Optional Logger instance. If None, uses DEFAULT_LOGGER.
    :raises DetailedException: If directory creation or the CSV write fails.
    """
    try:
        logger = logger or DEFAULT_LOGGER
        logger.info("Entered save_dask_dataframe_as_csv; target path: %s", file_save_path)

        # Ensure parent directory exists
        parent_dir = os.path.dirname(file_save_path) or "."
        logger.debug("Creating parent directory if missing: %s", parent_dir)
        os.makedirs(parent_dir, exist_ok=True)
        logger.info("Parent directory ready: %s", parent_dir)

        # Write DataFrame to CSV
        logger.debug("Writing DataFrame to CSV at: %s", file_save_path)
        with ProgressBar():
            dataframe.to_csv(file_save_path, index=index, single_file=single_file)
        logger.info("DataFrame successfully saved to CSV: %s", file_save_path)
        logger.debug("Exiting 'save_dask_dataframe_as_csv' function of 'main_utils' module.")
    except Exception as e:
        # Any failure gets wrapped for consistent, detailed logging
        raise DetailedException(exc=e, logger=logger) from e

def save_dask_dataframe_as_parquet(
    file_save_path: str,
    dataframe: ddf.DataFrame,
    single_file: bool = False,
    index: bool = False,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Persist a Dask DataFrame to disk as Parquet, optionally combining all partitions into one file.

    By default, Dask writes one Parquet file per partition under the given path.
    If `single_file=True`, the DataFrame is first collapsed to a single partition
    before writing, resulting in one output Parquet file.

    :param file_save_path:
        Destination path or directory for Parquet output. Examples:
        - "out/data-*.parquet": writes data-0.parquet, data-1.parquet, etc.
        - "out/data.parquet": with single_file=True, writes one file under that path.
    :param dataframe: The Dask DataFrame to save.
    :param single_file: If True, repartition to one partition and write a single file.
                        If False, writes one file per partition. Defaults to False.
    :param index: Whether to include the DataFrame index in the output. Defaults to False.
    :param logger: Optional Logger; if None, uses DEFAULT_LOGGER.
    :raises DetailedException: If directory creation or the Parquet write fails.
    """
    try:
        logger = logger or DEFAULT_LOGGER
        logger.info("Entered save_dask_dataframe_as_parquet; target: %s", file_save_path)

        # Ensure parent directory exists
        parent_dir = os.path.dirname(file_save_path) or "."
        logger.debug("Ensuring output directory exists: %s", parent_dir)
        os.makedirs(parent_dir, exist_ok=True)
        logger.info("Output directory ready: %s", parent_dir)

        # If requested, collapse to a single partition
        if single_file:
            logger.debug("Repartitioning to one partition for single-file output")
            dataframe = dataframe.repartition(npartitions=1)

        logger.debug("Writing DataFrame to Parquet at: %s", file_save_path)
        with ProgressBar():
            dataframe.to_parquet(
                file_save_path,
                write_index=index
            )
        logger.info("DataFrame successfully saved to Parquet: %s", file_save_path)

    except Exception as e:
        # Wrap in your DetailedException for consistency
        raise DetailedException(exc=e, logger=logger) from e
    
def save_object(
    file_path: str,
    obj: object,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Persist a Python object to disk using dill serialization.

    :param file_path:  Full path (including filename) where the object will be written.
    :param obj:        Any picklable Python object to save.
    :param logger:     Optional Logger; if None, DEFAULT_LOGGER is used.
    :raises DetailedException: If the directory cannot be created or the dump fails.
    """
    try:
        log = logger or DEFAULT_LOGGER
        log.info("Entered save_object; target path: %s", file_path)

        # Ensure parent directory exists
        parent_dir = os.path.dirname(file_path) or "."
        log.debug("Creating parent directory if missing: %s", parent_dir)
        os.makedirs(parent_dir, exist_ok=True)
        log.info("Parent directory ready: %s", parent_dir)

        # Write the object
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
            log.info("Object successfully saved at: %s", file_path)

    except Exception as e:
        raise DetailedException(exc=e, logger=log) from e

def load_object(file_path: str, logger: Optional[logging.Logger] = None) -> object:
    """
    Load a pickled Python object from disk using dill.

    :param file_path: Full path to the serialized object file.
    :param logger:    Optional logger; falls back to DEFAULT_LOGGER if None.
    :return:          The deserialized Python object.
    :raises FileNotFoundError: If the file does not exist.
    :raises DetailedException: For any other I/O or deserialization errors.
    """
    try:
        log = logger or DEFAULT_LOGGER
        log.info("Entered load_object; target path: %s", file_path)
        if not os.path.exists(file_path):
            logger.error("File Not Found : %s", file_path)
            raise FileNotFoundError(f"{file_path} does not exist.")

        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)
        return obj
    except Exception as e:
        raise DetailedException(exc=e, logger=logger) from e


def save_json(
    file_path: str,
    data: dict,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Save a Python dictionary to a JSON file.

    :param file_path: Full path where the JSON file will be written.
    :param data:      Dictionary to serialize.
    :param logger:    Optional logger; falls back to DEFAULT_LOGGER if None.
    :raises DetailedException: On I/O errors during directory creation or write.
    """
    try:
        log = logger or DEFAULT_LOGGER
        log.info("Entered save_json; target path: %s", file_path)

        # Ensure parent directory exists
        parent_dir = os.path.dirname(file_path) or "."
        log.debug("Creating parent directory if missing: %s", parent_dir)
        os.makedirs(parent_dir, exist_ok=True)
        log.info("Parent directory ready: %s", parent_dir)

        with open(file_path, "w") as file:
            json.dump(dict, file, indent=4)
            log.info("Json successfully saved at: %s", file_path)

    except Exception as e:
        raise DetailedException(exc=e, logger=log) from e

def load_json(
    file_path: str,
    logger: Optional[logging.Logger] = None
) -> dict:
    """
    Load a JSON file into a Python dictionary.

    :param file_path: Full path to the JSON file.
    :param logger:    Optional logger; falls back to DEFAULT_LOGGER if None.
    :return:          The loaded dictionary.
    :raises FileNotFoundError: If the file does not exist.
    :raises DetailedException: For any I/O or JSON parsing errors.
    """
    try:
        log = logger or DEFAULT_LOGGER
        log.info("Entered load_json; target path: %s", file_path)
        
        if not os.path.exists(file_path):
            logger.error("File Not Found : %s", file_path)
            raise FileNotFoundError(f"{file_path} does not exist.")
        
        with open(file_path, "r") as file:
            loaded_dict = json.load(file)
            log.info("Json Successfully Loaded from: %s", file_path)
            return loaded_dict
    except Exception as e:
        raise DetailedException(exc=e, logger=log) from e









