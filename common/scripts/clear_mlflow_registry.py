import os
import sys
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

# Add project root to path to allow imports from common
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.utils.mlflow_utils import configure_mlflow
from common.logger import configure_logger

logger = configure_logger(
    logger_name="clear_mlflow_registry",
    level="INFO",
    to_console=True,
    to_file=True,
    log_file_name="clear_mlflow_registry"
)

def clear_model_registry():
    """
    Connects to the MLflow Model Registry and deletes all registered models.
    It first transitions all model versions to 'Archived' before deletion.
    """
    try:
        # Configure MLflow client using existing utility function
        configure_mlflow()
        client = MlflowClient()
        logger.info("Successfully connected to MLflow server.")
        
        # 1. Get all registered models
        registered_models = client.search_registered_models()
        
        if not registered_models:
            logger.info("MLflow Model Registry is already empty. No models to delete.")
            return

        logger.warning(f"Found {len(registered_models)} registered models to delete:")
        for model in registered_models:
            print(f"  - {model.name}")

        # 2. Add a safety confirmation prompt
        confirm = input("\nAre you sure you want to permanently delete ALL of these models? (yes/no): ").strip().lower()
        if confirm != 'yes':
            logger.info("Operation cancelled by user.")
            return

        # 3. Iterate and delete each model
        for model in registered_models:
            model_name = model.name
            try:
                logger.info(f"Processing model: '{model_name}'...")
                
                # Get all versions of the current model
                versions = client.search_model_versions(f"name='{model_name}'")
                
                # Transition all versions to 'Archived' before deleting the model
                for version in versions:
                    if version.current_stage != 'Archived':
                        logger.info(f"  - Archiving version {version.version}...")
                        client.transition_model_version_stage(
                            name=model_name,
                            version=version.version,
                            stage="Archived"
                        )
                
                # Delete the registered model (which also deletes all its versions)
                logger.info(f"Deleting registered model '{model_name}'...")
                client.delete_registered_model(name=model_name)
                logger.info(f"Successfully deleted model '{model_name}'.")

            except MlflowException as e:
                logger.error(f"Failed to delete model '{model_name}': {e}")
                continue # Move to the next model

        logger.info("MLflow Model Registry has been cleared.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    clear_model_registry()