import dagshub
import mlflow
import mlflow.sklearn
from logger import configure_logger
from exception import DetailedException
from utils.main_utils import load_object, load_json, load_params
from entity.config_entity import ModelRegistryConfig
from entity.artifact_entity import ModelTrainerArtifact, ModelEvaluationArtifact, FeatureEngineeringArtifact
from constants import PARAM_FILE_PATH

logger = configure_logger(logger_name=__name__, level="DEBUG", to_console=True, to_file=True, log_file_name=__name__)


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
    def __init__(self, model_registry_config:ModelRegistryConfig, 
                 model_trainer_artifact:ModelTrainerArtifact, 
                 model_evaluation_artifact:ModelEvaluationArtifact, 
                 feature_engineering_artifact:FeatureEngineeringArtifact):
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
        
    def register_model(self, run_id:str, model_name:str, stage:str):
        """
        Registers the model in the MLflow Model Registry and transitions it to a given stage.

        Args:
            run_id (str): The MLflow run ID where the model was logged.
            model_name (str): The name of the model in the registry.
            stage (str): The stage to transition the model to (default is "Staging").

        Returns:
            str: The version number of the registered model.
        """
        try:
            result = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=model_name)

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
                logger.info("Performance Metrics Loaded Successfully.")

                logger.debug("Logging Performance Metrics in Mlflow experiment: %s",run.info.run_id )
                mlflow.log_params(params = performance_metrics)
                logger.debug("Successfully Logged Performance Metrics in Mlflow experiment.")



                logger.debug("Loading Model from: %s",self.model_trainer_artifact.trained_model_obj_path)
                model = load_object(file_path=self.model_trainer_artifact.trained_model_obj_path,
                                                logger=logger)
                logger.info("Model Successfully Loaded.")

                logger.debug("Logging Model in Mlflow experiment: %s",run.info.run_id )
                mlflow.sklearn.log_model(model, "model")
                logger.debug("Successfully Logged Model in Mlflow experiment.")

                logger.debug("Logging Model Parameters in Mlflow experiment: %s",run.info.run_id )
                mlflow.log_params(params= self.params["Model_Parameters"])
                logger.debug("Successfully Logged Model Parameters in Mlflow experiment.")

                logger.debug("Registering Model in Mlflow Registry...")
                self.register_model(run_id=run.info.run_id, model_name=self.model_registry_config.mlflow_model_name)
                logger.debug("Successfully Registered Model in Mlflow Registry.")

                logger.debug("Loading Vectorizer from: %s",self.feature_engineering_artifact.vectorizer_obj_file_path)
                vectorizer = load_object(file_path=self.feature_engineering_artifact.vectorizer_obj_file_path,
                                                logger=logger)
                logger.info("Vectorizer Successfully Loaded.")
                
                logger.debug("Logging Vectorizer in Mlflow experiment: %s",run.info.run_id )
                mlflow.sklearn.log_model(vectorizer, "vectorizer")
                logger.debug("Successfully Logged Vectorizer in Mlflow experiment.")

                logger.debug("Logging Vectorizer Parameters in Mlflow experiment: %s",run.info.run_id )
                mlflow.log_params(params= self.params["TF-IDF_Params"])
                logger.debug("Successfully Logged Vectorizer Parameters in Mlflow experiment.")

                logger.debug("Registering Vectorizer in Mlflow Registry...")
                self.register_model(run_id=run.info.run_id, model_name=self.model_registry_config.mlflow_vectorizer_name)
                logger.debug("Successfully Registered Vectorizer in Mlflow Registry.")
        except Exception as e:
            raise DetailedException(exc=e, logger=logger) from e
                


                


            

        