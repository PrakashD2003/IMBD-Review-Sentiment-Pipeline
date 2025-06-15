from pathlib import Path
from mlflow.tracking import MlflowClient

from src.exception import DetailedException
from src.logger import configure_logger
from src.utils.mlflow_utils import configure_mlflow, get_latest_version
from src.constants import MLFLOW_TRACKING_URI, DAGSHUB_REPO_NAME, DAGSHUB_REPO_OWNER_NAME, MODEL_NAME
module_name = Path(__file__).stem

logger = configure_logger(logger_name=module_name,
                          level="DEBUG",
                          to_console=True,
                          to_file=True,
                          log_file_name=module_name)


def main():
    logger.info("Getting Latest Version of Production Model from Mlflow Registry...")
    configure_mlflow(mlflow_uri=MLFLOW_TRACKING_URI)
    prod_version = get_latest_version(model_name=MODEL_NAME,
                                      stages=["Production"],
                                      logger=logger)
    run_prod = MlflowClient().get_run(prod_version.run_id)
    
    logger.info("Fetching Production Model Metrics From Run: %s", prod_version.run_id)
    metrics_prod = run_prod.data.metrics
    accuracy_prod = metrics_prod.get("accuracy")
    logger.info("Production Model Accuracy: %s", accuracy_prod)

    logger.info("Getting Latest Version of Staging Model from Mlflow Registry...")
    configure_mlflow(mlflow_uri=MLFLOW_TRACKING_URI)
    staging_version = get_latest_version(model_name=MODEL_NAME,
                                      stages=["Staging"],
                                      logger=logger)
    run_stage = MlflowClient().get_run(staging_version.run_id)
    
    logger.info("Fetching Latest Staging Model Metrics From Run: %s", staging_version.run_id)
    metrics_staging = run_stage.data.metrics
    accuracy_staging = metrics_staging.get("accuracy")
    logger.info("Latest Staging Model Accuracy: %s", accuracy_staging)

    if accuracy_staging > accuracy_prod:
        logger.info("Latest Staging Model accuracy(%s) is greater than current Production Model accuracy(%s).", accuracy_staging, accuracy_prod)
        logger.info("Promoting Staging Model to Production and archiving current Production Model")
        MlflowClient().transition_model_version_stage(
            name=MODEL_NAME,
            version=prod_version[0].version,
            archive_existing_versions=True
        )
        MlflowClient().transition_model_version_stage(
            name=MODEL_NAME,
            version=staging_version[0].version,
            stage="Production"
        )
    
