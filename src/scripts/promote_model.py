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
    try:
        logger.info("Getting Latest Version of Production Model from Mlflow Registry...")
        configure_mlflow(mlflow_uri=MLFLOW_TRACKING_URI)
        prod_version = get_latest_version(
            model_name=MODEL_NAME,
            stages=["Production"],
            logger=logger,
        )
        run_prod = MlflowClient().get_run(prod_version.run_id)
        
        logger.info("Fetching Production Model Metrics From Run: %s", prod_version.run_id)
        metrics_prod = run_prod.data.metrics
       
        logger.info("Getting Latest Version of Staging Model from Mlflow Registry...")
        configure_mlflow(mlflow_uri=MLFLOW_TRACKING_URI)
        staging_version = get_latest_version(
            model_name=MODEL_NAME,
            stages=["Staging"],
            logger=logger,
        )
        run_stage = MlflowClient().get_run(staging_version.run_id)
        
        logger.info("Fetching Latest Staging Model Metrics From Run: %s", staging_version.run_id)
        metrics_staging = run_stage.data.metrics
        
        common_metrics = metrics_prod.keys() & metrics_staging.keys()
        if all(metrics_staging[key] >= metrics_prod[key] for key in common_metrics):
            logger.info("Latest Staging Model Performance is better than current Production Model Performance.")
            logger.info("Current Production Model Performance: %s", metrics_prod)
            logger.info("Current Staging Model Performance: %s", metrics_staging)
            logger.info("Promoting Staging Model to Production and archiving current Production Model...")
            MlflowClient().transition_model_version_stage(
                name=MODEL_NAME,
                version=prod_version.version,
                stage="Archived",
            )
            MlflowClient().transition_model_version_stage(
                name=MODEL_NAME,
                version=staging_version.version,
                stage="Production",
            )
        else:
            logger.info(
                "Staging model Performance (%s) is not greater than Production Performance (%s). No promotion.",
                metrics_staging,
                metrics_prod,
            )
    except Exception as e:
        raise DetailedException(exc=e, logger=logger)
    
if __name__ == "__main__":
    main()
