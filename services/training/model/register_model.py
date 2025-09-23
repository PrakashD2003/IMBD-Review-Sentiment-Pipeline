"""Utilities for registering trained models and vectorizers with MLflow."""

import logging
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

import mlflow
import mlflow.sklearn
from pathlib import Path

from common.logger import configure_logger
from common.exception import DetailedException
from common.utils.mlflow_utils import configure_mlflow
from common.utils.main_utils import load_object, load_json, load_params
from services.training.entity.config_entity import ModelRegistryConfig
from services.training.entity.artifact_entity import ModelTrainerArtifact, ModelEvaluationArtifact, FeatureEngineeringArtifact
from common.constants import PARAM_FILE_PATH


# This logger will automatically inherit the configuration from the entrypoint
logger = logging.getLogger("training-service")

class ModelRegistry:
    """
    Handles the registration of a trained model and its associated vectorizer into the MLflow Model Registry.
    Also logs performance metrics, model parameters, and integrates with DagsHub for experiment tracking.

    Attributes:
        model_registry_config (ModelRegistryConfig): Configuration object with MLflow and DagsHub settings.
        model_trainer_artifact (ModelTrainerArtifact): Contains paths to the trained model.
        model_evaluation_artifact (ModelEvaluationArtifact): Contains performance metrics file.
        feature_engineering_artifact (FeatureEngineeringArtifact): Contains path to the vectorizer.
        params (dict): Model and TF-IDF parameters loaded from params.yaml.
    """
    def __init__(self, model_registry_config:ModelRegistryConfig = ModelRegistryConfig(), 
                 model_trainer_artifact:ModelTrainerArtifact = ModelTrainerArtifact(), 
                 model_evaluation_artifact:ModelEvaluationArtifact = ModelEvaluationArtifact(), 
                 feature_engineering_artifact:FeatureEngineeringArtifact = FeatureEngineeringArtifact()):
        """
        Initializes the ModelRegistry class with configuration and artifacts.

        Args:
            model_registry_config (ModelRegistryConfig): Configuration for MLflow and DagsHub.
            model_trainer_artifact (ModelTrainerArtifact): Trained model artifact.
            model_evaluation_artifact (ModelEvaluationArtifact): Evaluation metrics artifact.
            feature_engineering_artifact (FeatureEngineeringArtifact): Feature engineering artifact.
        """
        try:
            logger.debug("Configuring 'ModelRegistry' class of 'model' module through constructer...")
            self.model_registry_config = model_registry_config
            self.model_trainer_artifact = model_trainer_artifact
            self.model_evaluation_artifact = model_evaluation_artifact
            self.feature_engineering_artifact = feature_engineering_artifact
            self.params = load_params(params_path=PARAM_FILE_PATH, logger=logger)
            logger.info("ModelRegistry class configured successfully.")
        except Exception as e:
            raise DetailedException(exc=e, logger=logger) from e
    
    
    def register_model(self, run_id:str, artifact_path: str, model_name:str, stage:str):
        """
        Register a logged MLflow model under a given name and transition it to a stage.

        Args:
            run_id:     MLflow run ID containing the logged model artifact.
            model_name: Registered model name in the MLflow Model Registry.
            stage:      Stage to transition the new version to (e.g. "Staging" or "Production").

        Returns:
            The new model version as a string.

        Raises:
            DetailedException: If registration or transition fails.
        """
    
        try:
            result = mlflow.register_model(model_uri=f"runs:/{run_id}/{artifact_path}", name=model_name)

            mlflow.tracking.MlflowClient().transition_model_version_stage(name=model_name,
                                                                          version=result.version,
                                                                          stage=stage,
                                                                          archive_existing_versions=True
                                                                          )
            return result.version
        except Exception as e:
            raise DetailedException(exc=e, logger=logger) from e
    
    def initiate_model_registration(self):
        """
        Executes the full model registration workflow:
        - Loads and logs performance metrics.
        - Logs the trained model and vectorizer.
        - Logs associated parameters.
        - Registers the model and vectorizer in MLflow Model Registry.
        """
        configure_mlflow(
        mlflow_uri=self.model_registry_config.mlflow_uri,
        dagshub_repo_owner_name=self.model_registry_config.dagshub_repo_owner_name,
        dagshub_repo_name=self.model_registry_config.dagshub_repo_name,
        experiment_name=self.model_registry_config.experiment_name,
        logger=logger,
        )
        with mlflow.start_run() as run:
            try:
                logger.info("Entered 'initiate_model_registration' method of 'ModelRegistry' class")
                logger.info("\n" + "-" * 80)
                logger.info("Starting Model Registry Component...")
                
                # --- Log Metrics ---
                logger.debug("Loading Performance Metrics from: %s", self.model_evaluation_artifact.performance_metrics_file_path)
                performance_metrics = load_json(file_path=self.model_evaluation_artifact.performance_metrics_file_path, logger=logger)
                logger.info("Performance Metrics Loaded Successfully.")
                logger.debug("Logging Performance Metrics in Mlflow experiment: %s", run.info.run_id)
                mlflow.log_metrics(performance_metrics)
                logger.debug("Successfully Logged Performance Metrics.")

                # --- Handle Model ---
                logger.debug("Loading Model from: %s", self.model_trainer_artifact.trained_model_obj_path)
                model = load_object(file_path=self.model_trainer_artifact.trained_model_obj_path, logger=logger)
                logger.info("Model Successfully Loaded.")

                # Conditionally log feature info to prevent AttributeError
                if isinstance(model, LogisticRegression):
                    logger.info("Classifier expects %d features", model.coef_.shape[1])
                elif isinstance(model, lgb.LGBMClassifier):
                    logger.info("Classifier was trained on %d features", model.n_features_in_)

                logger.debug("Logging Model in Mlflow experiment: %s", run.info.run_id)
                mlflow.sklearn.log_model(model, artifact_path=self.model_registry_config.mlflow_model_artifact_path)
                logger.debug("Successfully Logged Model in Mlflow experiment.")

                # Conditionally log correct model parameters
                model_name_trained = self.params.get("model_training", {}).get("model_to_use")
                model_params = {}
                if model_name_trained == "LogisticRegression":
                    model_params = self.params.get("model_training", {}).get("logistic_regression", {})
                elif model_name_trained == "LightGBM":
                    model_params = self.params.get("model_training", {}).get("lightgbm", {})
                logger.debug("Logging Model Parameters for '%s'", model_name_trained)
                mlflow.log_params(model_params)
                mlflow.log_param("model_type", model_name_trained)
                logger.debug("Successfully Logged Model Parameters.")

                logger.debug("Registering Model in Mlflow Registry...")
                self.register_model(
                    run_id=run.info.run_id, 
                    artifact_path=self.model_registry_config.mlflow_model_artifact_path,
                    model_name=self.model_registry_config.mlflow_model_name, 
                    stage=self.model_registry_config.mlflow_model_stage
                )
                logger.debug("Successfully Registered Model in Mlflow Registry.")

                # --- Handle Vectorizer ---
                logger.debug("Loading Vectorizer from: %s", self.feature_engineering_artifact.vectorizer_obj_file_path)
                vectorizer = load_object(file_path=self.feature_engineering_artifact.vectorizer_obj_file_path, logger=logger)
                logger.info("Vectorizer Successfully Loaded.")
                
                logger.debug("Logging Vectorizer in Mlflow experiment: %s", run.info.run_id)
                logger.info("Vectorizer vocab size: %d", len(vectorizer.vocabulary_))
                mlflow.sklearn.log_model(vectorizer, artifact_path=self.model_registry_config.mlflow_vectorizer_artifact_path)
                logger.debug("Successfully Logged Vectorizer in Mlflow experiment.")
                
                # Log vectorizer parameters from correct path
                vectorizer_params = self.params.get("feature_engineering", {}).get("tfidf", {})
                logger.debug("Logging Vectorizer Parameters")
                mlflow.log_params(vectorizer_params)
                logger.debug("Successfully Logged Vectorizer Parameters.")

                logger.debug("Registering Vectorizer in Mlflow Registry...")
                self.register_model(
                    run_id=run.info.run_id, 
                    artifact_path=self.model_registry_config.mlflow_vectorizer_artifact_path,
                    model_name=self.model_registry_config.mlflow_vectorizer_name, 
                    stage=self.model_registry_config.mlflow_model_stage
                )
                logger.debug("Successfully Registered Vectorizer in Mlflow Registry.")

            except Exception as e:
                raise DetailedException(exc=e, logger=logger) from e
                

if __name__ == "__main__":
    model_registry = ModelRegistry()
    model_registry.initiate_model_registration()
    