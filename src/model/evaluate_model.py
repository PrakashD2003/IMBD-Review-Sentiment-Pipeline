import dagshub
import mlflow
import dask.dataframe as ddf
from dask_ml.metrics import accuracy_score, precision_score, recall_score, f1_score 
from logger import configure_logger
from exception import DetailedException
from constants import MLFLOW_TRACKING_URI
from entity.config_entity import ModelEvaluationConfig
from entity.artifact_entity import ModelTrainerArtifact, ModelEvaluationArtifact

logger = configure_logger(logger_name=__name__, level="DEBUG", to_console=True, to_file=True, log_file_name=__name__)

class ModelEvaluation:
    def __init__(self, model_evaluation_config:ModelEvaluationConfig, model_trainer_artifact:ModelTrainerArtifact):
        try:
            logger.debug("Configuring 'ModelEvaluation' class of 'model' module through constructer...")
            self.model_evaluation_config = model_evaluation_config
            self.model_trainer_artifact = model_trainer_artifact
            logger.info("ModelEvaluation class configured successfully.")
        except Exception as e:
            raise DetailedException(exc=e, logger=logger) from e
        
    def evaluate_model(model:object, test_df)

        