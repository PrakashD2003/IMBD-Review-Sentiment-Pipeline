import logging
import mlflow
import dagshub
 
from typing import Optional, Iterable
from src.logger import configure_logger
from src.exception import DetailedException
from src.constants import MLFLOW_TRACKING_URI

"""Utility functions for interacting with MLflow."""

DEFAULT_LOGGER = configure_logger(
    logger_name=__name__,
    level="DEBUG",
    to_console=True,
    to_file=True,
    log_file_name=__name__
)


def configure_mlflow(
    mlflow_uri: str = MLFLOW_TRACKING_URI,
    dagshub_repo_owner_name: Optional[str] = None,
    dagshub_repo_name: Optional[str] = None,
    experiment_name: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> None:
    """Configure MLflow and optionally initialize DagsHub."""
    try:
        logger = logger or DEFAULT_LOGGER
        logger.debug("Entered 'configure_mlflow' utility function.")
        mlflow.set_tracking_uri(mlflow_uri)
        if dagshub_repo_owner_name and dagshub_repo_name:
            dagshub.init(
                repo_owner=dagshub_repo_owner_name,
                repo_name=dagshub_repo_name,
                mlflow=True,
            )
        if experiment_name:
            mlflow.set_experiment(experiment_name)
        logger.info("MLflow configured successfully.")
    except Exception as e:
        raise DetailedException(exc=e, logger=logger) from e
    

def get_latest_model(model_name: str, stages: Iterable[str], *, flavor: str = "pyfunc", logger: Optional[logging.Logger] = None) -> object:
    """Load the latest model for ``model_name`` in any of ``stages``.

    Parameters
    ----------
    model_name : str
        Name of the model registered in MLflow.
    stages : Iterable[str]
        Stages to search for (e.g. ["Staging", "Production"]).
    flavor : str, optional
        MLflow model flavor to use when loading the model. Defaults to
        ``"pyfunc"``.
    logger : Optional[logging.Logger]
        Logger to use for debug messages. Defaults to DEFAULT_LOGGER.

    Returns
    -------
    object
        The loaded model instance.

    Raises
    ------
    RuntimeError
        If no version of the model exists in the provided stages.
    """
    try:
        logger = logger or DEFAULT_LOGGER
        client = mlflow.MlflowClient()
        logger.debug("Getting Latest Version of model: '%s' of stages: '%s' from Mlflow Registry...", model_name, stages)
        versions = client.get_latest_versions(model_name, stages=list(stages))
        if not versions:
            raise RuntimeError(f"No versions of '{model_name}' found for stages {list(stages)}")

        model_uri = f"models:/{model_name}/{versions[0].version}"
        loader = getattr(mlflow, flavor)
        logger.info(f"Fetching model from: {model_uri}")
        return loader.load_model(model_uri)
    except Exception as e:
        raise DetailedException(exc=e, logger=logger) from e

def get_latest_version(model_name:str, stages:list[str], logger: Optional[logging.Logger] = None)->object:
    """
    Return the version number of the latest model in the specified stages.

    Parameters
    ----------
    model_name : str
        Name of the registered model in the MLflow Model Registry.
    stages : Iterable[str]
        List of stages to search, e.g. ["Staging", "Production"].
    logger : Optional[logging.Logger]
        Logger to use for debug messages. Defaults to DEFAULT_LOGGER.

    Returns
    -------
    str
        The version string of the latest model.

    Raises
    ------
    DetailedException
        If there is an error contacting the MLflow Registry.
    RuntimeError
        If no versions are found in the given stages.
    """
    try:
        logger = logger or DEFAULT_LOGGER
        client = mlflow.MlflowClient()
        logger.debug("Getting Latest Version of model: '%s' of stages: '%s' from Mlflow Registry...", model_name, stages)
        latest_version = client.get_latest_versions(model_name, stages=stages)
        if not latest_version:
            raise RuntimeError(f"No {stages} version of {model_name}")
        return latest_version 
    except Exception as e:
        raise DetailedException(exc=e, logger=logger) from e