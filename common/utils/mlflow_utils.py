"""Utility functions for interacting with MLflow."""
import os
import logging
import mlflow
import dagshub

from typing import Optional, Iterable
from common.logger import configure_logger
from common.exception import DetailedException
from common.constants import MLFLOW_TRACKING_URI


# This will also inherit the central logger configuration.
DEFAULT_LOGGER = logging.getLogger(__name__)

def _ensure_non_interactive_auth(logger: logging.Logger) -> None:
    """
    Make sure both MLflow and DagsHub clients are authenticated *without* prompting.
    - MLflow: via MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD
    - DagsHub client: via dagshub.auth.add_app_token(...)
    """
    # 1) Backfill MLflow Basic Auth from DagsHub vars if needed
    u = os.getenv("MLFLOW_TRACKING_USERNAME") or os.getenv("DAGSHUB_USERNAME")
    p = os.getenv("MLFLOW_TRACKING_PASSWORD") or os.getenv("DAGSHUB_TOKEN")
    if u and p:
        os.environ["MLFLOW_TRACKING_USERNAME"] = u
        os.environ["MLFLOW_TRACKING_PASSWORD"] = p
        logger.debug("MLflow auth envs set (username present, password token provided).")
    else:
        logger.warning(
            "MLflow auth envs missing. Set MLFLOW_TRACKING_USERNAME + "
            "MLFLOW_TRACKING_PASSWORD (or DAGSHUB_USERNAME + DAGSHUB_TOKEN)."
        )

    # 2) Authenticate DagsHub client *before* dagshub.init to avoid interactive prompt
    du = os.getenv("DAGSHUB_USERNAME")
    dt = os.getenv("DAGSHUB_TOKEN")
    if du and dt:
        try:
            dagshub.auth.add_app_token(username=du, token=dt)
            logger.debug("DagsHub token added non-interactively.")
        except Exception as e:
            logger.warning("Failed to add DagsHub token non-interactively: %s", e)
    else:
        logger.debug("No DAGSHUB_USERNAME/TOKEN found; skipping DagsHub client auth.")

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

        # Always set tracking URI first
        mlflow.set_tracking_uri(mlflow_uri)

        # Ensure non-interactive auth for both MLflow and DagsHub
        _ensure_non_interactive_auth(logger)

        # Optional: link runs to the repo in DagsHub UI (non-interactive if token added)
        if dagshub_repo_owner_name and dagshub_repo_name:
            try:
                dagshub.init(
                    repo_owner=dagshub_repo_owner_name,
                    repo_name=dagshub_repo_name,
                    mlflow=True,
                )
                logger.debug("dagshub.init completed.")
            except Exception as e:
                # Don't block startup if this is cosmetic
                logger.warning("dagshub.init failed (continuing without it): %s", e)

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
        logger.debug(
            "Getting Latest Version of model: '%s' of stages: '%s' from Mlflow Registry...",
            model_name,
            stages,
        )
        versions = client.get_latest_versions(model_name, stages=list(stages))
    except Exception as e:
        raise DetailedException(exc=e, logger=logger) from e
    
    if not versions:
        raise RuntimeError(f"No versions of '{model_name}' found for stages {list(stages)}")


    model_uri = f"models:/{model_name}/{versions[0].version}"
    loader = getattr(mlflow, flavor)
    logger.info(f"Fetching model from: {model_uri}")
    try:
        return loader.load_model(model_uri)
    except Exception as e:
        raise DetailedException(exc=e, logger=logger) from e

def get_latest_version(
    model_name: str,
    stages: list[str],
    logger: Optional[logging.Logger] = None,
) -> mlflow.entities.model_registry.ModelVersion:
    """Return the latest ``ModelVersion`` for ``model_name`` in ``stages``.

    Parameters
    ----------
    model_name : str
        Name of the registered model in the MLflow Model Registry.
    stages : list[str]
        List of stages to search, e.g. ["Staging", "Production"].
    logger : Optional[logging.Logger]
        Logger to use for debug messages. Defaults to ``DEFAULT_LOGGER``..

    Returns
    -------
    mlflow.entities.model_registry.ModelVersion
        The most recent model version in the provided stages

    Raises
    ------
    RuntimeError
        If no versions are found in the given stages.
    DetailedException
        If there is an error contacting the MLflow Registry.
    """
    try:
        logger = logger or DEFAULT_LOGGER
        client = mlflow.MlflowClient()
        logger.debug(
            "Getting Latest Version of model: '%s' of stages: '%s' from Mlflow Registry...",
            model_name,
            stages,
        )
        versions = client.get_latest_versions(model_name, stages=stages)
        if not versions:
            raise RuntimeError(f"No {stages} version of {model_name}")
        return versions[0] 
    except Exception as e:
        raise DetailedException(exc=e, logger=logger) from e