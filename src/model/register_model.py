import dagshub
import mlflow
import mlflow.sklearn
from pathlib import Path

from src.logger import configure_logger
from src.exception import DetailedException
from src.utils.main_utils import load_object, load_json, load_params
from src.entity.config_entity import ModelRegistryConfig
from src.entity.artifact_entity import ModelTrainerArtifact, ModelEvaluationArtifact, FeatureEngineeringArtifact
from src.constants import PARAM_FILE_PATH

module_name = Path(__file__).stem

logger = configure_logger(logger_name=module_name, 
                          level="DEBUG", to_console=True, 
                          to_file=True, 
                          log_file_name=module_name)

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
    
    def configure_mlflow(self, experiment_name:str)->None:
        """
        Configures MLflow and DagsHub for tracking.

        Args:
            experiment_name (str): The name of the MLflow experiment.
        """
        try:
            logger.info("Entered 'configure_mlflow' function of 'ModelRegistry' class.")
            logger.debug("Setting up MLFlOW and Dagshub...")
            mlflow.set_tracking_uri(uri=self.model_registry_config.mlflow_uri)
            dagshub.init(repo_owner=self.model_registry_config.dagshub_repo_owner_name,
                         repo_name=self.model_registry_config.dagshub_repo_name,
                         mlflow=True)
            mlflow.set_experiment(experiment_name)
            logger.info("MLFlOW and Dagshub Congigured Successfully.")
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
        try:
            self.configure_mlflow(experiment_name=self.model_registry_config.experiment_name)
            with mlflow.start_run() as run:
                logger.info("Entered 'initiate_model_registration' method of 'ModelRegistry' class")
                print("\n" + "-"*80)
                print("🚀 Starting Model Registry Component...")

                logger.debug("Loading Performance Metrics from: %s", self.model_evaluation_artifact.performance_metrics_file_path)
                performance_metrics = load_json(file_path=self.model_evaluation_artifact.performance_metrics_file_path, logger=logger)
                if performance_metrics is None:
                    raise RuntimeError("Loaded performance_metrics is None")
                logger.info("Performance Metrics Loaded Successfully.")

                logger.debug("Logging Performance Metrics: '%s' in Mlflow experiment: %s", performance_metrics, run.info.run_id )
                mlflow.log_metrics(performance_metrics)
                logger.debug("Successfully Logged Performance Metrics in Mlflow experiment.")



                logger.debug("Loading Model from: %s",self.model_trainer_artifact.trained_model_obj_path)
                model = load_object(file_path=self.model_trainer_artifact.trained_model_obj_path,
                                                logger=logger)
                logger.info("Model Successfully Loaded.")

                logger.debug("Logging Model in Mlflow experiment: %s",run.info.run_id )
                logger.info("Classifier expects %d features", model.coef_.shape[1])
                mlflow.sklearn.log_model(model, artifact_path=self.model_registry_config.mlflow_model_artifact_path)
                logger.debug("Successfully Logged Model in Mlflow experiment.")
                model_params = self.params.get("Model_Params", {})
                if model_params is None:
                    raise RuntimeError("Loaded model_params is None")
                logger.debug("Logging Model Parameters: '%s' in Mlflow experiment: %s", model_params, run.info.run_id )
                mlflow.log_params(model_params)
                logger.debug("Successfully Logged Model Parameters in Mlflow experiment.")

                logger.debug("Registering Model in Mlflow Registry...")
                self.register_model(run_id=run.info.run_id, 
                                    artifact_path=self.model_registry_config.mlflow_model_artifact_path,
                                    model_name=self.model_registry_config.mlflow_model_name, 
                                    stage=self.model_registry_config.mlflow_model_stage)
                logger.debug("Successfully Registered Model in Mlflow Registry.")

                logger.debug("Loading Vectorizer from: %s",self.feature_engineering_artifact.vectorizer_obj_file_path)
                vectorizer = load_object(file_path=self.feature_engineering_artifact.vectorizer_obj_file_path,
                                                logger=logger)
                logger.info("Vectorizer Successfully Loaded.")
                
                logger.debug("Logging Vectorizer in Mlflow experiment: %s",run.info.run_id )
                logger.info("Vectorizer vocab size: %d", len(vectorizer.vocabulary_))
                mlflow.sklearn.log_model(vectorizer, artifact_path=self.model_registry_config.mlflow_vectorizer_artifact_path)
                logger.debug("Successfully Logged Vectorizer in Mlflow experiment.")
                
                vectorizer_params = self.params.get("TF-IDF_Params", {})
                if vectorizer_params is None:
                    raise RuntimeError("Loaded vectorizer_params is None")
                logger.debug("Logging Vectorizer Parameters: '%s' in Mlflow experiment: %s", vectorizer_params, run.info.run_id )
                mlflow.log_params(vectorizer_params)
                logger.debug("Successfully Logged Vectorizer Parameters in Mlflow experiment.")

                logger.debug("Registering Vectorizer in Mlflow Registry...")
                self.register_model(run_id=run.info.run_id, 
                                    artifact_path= self.model_registry_config.mlflow_vectorizer_artifact_path,
                                    model_name=self.model_registry_config.mlflow_vectorizer_name, 
                                    stage=self.model_registry_config.mlflow_model_stage)
                logger.debug("Successfully Registered Vectorizer in Mlflow Registry.")
        except Exception as e:
            raise DetailedException(exc=e, logger=logger) from e
                

if __name__ == "__main__":
    model_registry = ModelRegistry()
    model_registry.initiate_model_registration()
    