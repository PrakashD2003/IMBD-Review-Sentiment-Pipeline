import dask.dataframe as ddf
from pathlib import Path
from numpy import ndarray
from dask import delayed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 

from src.logger import configure_logger
from src.exception import DetailedException
from src.utils.main_utils import load_parquet_as_dask_dataframe, load_object, save_json, load_params
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, ModelEvaluationArtifact, FeatureEngineeringArtifact
from src.constants import PARAM_FILE_PATH

module_name = Path(__file__).stem

logger = configure_logger(logger_name=module_name, 
                          level="DEBUG", to_console=True, 
                          to_file=True, 
                          log_file_name=module_name)

class ModelEvaluation:
    """
    Component responsible for evaluating a trained model on feature-engineered
    test data, saving performance metrics locally in JSON format, and logging
    them to MLflow.

    Attributes:
        model_evaluation_config (ModelEvaluationConfig): Configuration including
            file paths for metrics output.
        model_trainer_artifact (ModelTrainerArtifact): Artifact containing the
            path to the trained model object.
        feature_engineering_artifact (FeatureEngineeringArtifact): Artifact with
            paths to the feature-engineered test dataset.
    """
    def __init__(self, model_evaluation_config:ModelEvaluationConfig = ModelEvaluationConfig(),
                 model_trainer_artifact:ModelTrainerArtifact = ModelTrainerArtifact(), 
                 feature_engineering_artifact:FeatureEngineeringArtifact = FeatureEngineeringArtifact()):
        """
        Initialize the ModelEvaluation component.

        :param model_evaluation_config: Configuration for saving metrics.
        :param model_trainer_artifact: Contains the path to the trained model.
        :param feature_engineering_artifact: Contains paths to the test data.
        :raises DetailedException: If configuration fails.
        """
        try:
            logger.debug("Configuring 'ModelEvaluation' class of 'model' module through constructer...")
            self.model_evaluation_config = model_evaluation_config
            self.model_trainer_artifact = model_trainer_artifact
            self.feature_engineering_artifact = feature_engineering_artifact
            self.params = load_params(params_path=PARAM_FILE_PATH, logger=logger)
            logger.info("ModelEvaluation class configured successfully.")
        except Exception as e:
            raise DetailedException(exc=e, logger=logger) from e
        
        
    def evaluate_model(self, model:object, test_ddf: ddf.DataFrame, target_col:str)->dict:
        """
        Evaluate a trained model on a Dask‐DataFrame test set.

        Steps:
          1. Persist the Dask DataFrame and split into features (all columns
             except `target_col`) and labels (`target_col`).
          2. Convert both to NumPy arrays and pull into memory.
          3. Wrap sklearn metric calls (accuracy, precision, recall, F1)
             in a Dask‐delayed function for parallel execution.
          4. Compute and return metrics.

        :param model:      A fitted classifier with a .predict() method.
        :param test_ddf:   Dask DataFrame containing feature columns plus
                           the label column named by `target_col`.
        :param target_col: Name of the column in `test_ddf` to use as the label.
        :return:           A dict with keys "accuracy", "precision",
                           "recall", and "f1_score".
        :raises DetailedException: On any failure during splitting,
                                   conversion or metric computation.
        """
        try:
            logger.info("Entered 'evaluate_model' function of 'ModelEvaluation' class")
            logger.debug("Splitting test_data into 'dependent' and 'independet' features...")
            test_ddf = test_ddf.persist()
            X_test = test_ddf.drop(columns=[target_col]).to_dask_array(lengths=True)
            y_test = test_ddf[target_col].astype(int).to_dask_array(lengths=True)

            # Pull into memory as Numpy Array once
            X_test_in_mem = X_test.compute()
            y_test_in_mem = y_test.compute()

            logger.info("Test data splitted successfully.")
            
            @delayed
            def evaluate(model:object, X_test_in_mem:ndarray, y_test_in_mem:ndarray)->dict:
                """
                Compute sklearn metrics on in‐memory NumPy arrays.
                """
                y_pred_raw = model.predict(X_test_in_mem)
                y_pred = y_pred_raw.astype(int)
                return {
                    "accuracy":  accuracy_score(y_test_in_mem, y_pred),
                    "precision": precision_score(y_test_in_mem, y_pred),
                    "recall":    recall_score(y_test_in_mem, y_pred),
                    "f1_score":  f1_score(y_test_in_mem, y_pred),
                }
            
            #  Schedule delayed evaluation
            logger.info("Entering 'evalute' dask delayed function inside 'evaluate_model' function...")
            metrics_delayed = evaluate(model, X_test_in_mem, y_test_in_mem)
            metrics = metrics_delayed.compute()
            logger.info("Exiting 'evalute' dask delayed function inside 'evaluate_model' function...")
            
            return metrics
        except Exception as e:
            raise DetailedException(exc=e, logger=logger) from e
        
    def initiate_model_evaluation(self):
        """
        Orchestrate the full model evaluation workflow:
          1. Load feature-engineered test data.
          2. Load the trained model.
          3. Compute performance metrics via `evaluate_model`.
          4. Save metrics as JSON locally and log to MLflow.

        :return: A ModelEvaluationArtifact containing the metrics file path.
        :raises DetailedException: On any failure in the workflow.
        """
        try:
            logger.info("Entered 'initiate_model_evaluation' method of 'ModelEvaluation' class")
            print("\n" + "-" * 80)
            print("Starting Model Evaluation Component...")

            logger.debug("Loading Test data from: %s",self.feature_engineering_artifact.feature_engineered_test_data_file_path)
            test_ddf = load_parquet_as_dask_dataframe(file_path=self.feature_engineering_artifact.feature_engineered_test_data_file_path,
                                            logger=logger)
            logger.info("Test Data Successfully Loaded.")

            logger.debug("Loading Model from: %s",self.model_trainer_artifact.trained_model_obj_path)
            model = load_object(file_path=self.model_trainer_artifact.trained_model_obj_path,
                                            logger=logger)
            logger.info("Model Successfully Loaded.")

            logger.debug("Initiating Model Evaluation...")
            performance_metrics = self.evaluate_model(model=model, test_ddf=test_ddf, target_col=self.params.get("Target_Col"))
            logger.info("Model Evaluation Completed.")

            logger.debug("Saving Performance Metrics at: %s", self.model_evaluation_config.performance_metrics_file_save_path)
            save_json(file_path=self.model_evaluation_config.performance_metrics_file_save_path, 
                      data=performance_metrics, 
                      logger=logger)
            logger.info("Performance Metircs Successfully Saved.")

            return ModelEvaluationArtifact(performance_metrics_file_path=self.model_evaluation_config.performance_metrics_file_save_path)
        except Exception as e:
            raise DetailedException(exc=e, logger=logger) from e


if __name__ == "__main__":
    model_evalutation = ModelEvaluation()
    model_evalutation.initiate_model_evaluation()
    
            



