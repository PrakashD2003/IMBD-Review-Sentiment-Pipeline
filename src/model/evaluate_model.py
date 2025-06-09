import dask.dataframe as ddf
from pandas import DataFrame
from dask import delayed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
from logger import configure_logger
from exception import DetailedException
from utils.main_utils import load_dask_dataframe, load_object, save_json
from entity.config_entity import ModelEvaluationConfig
from entity.artifact_entity import ModelTrainerArtifact, ModelEvaluationArtifact, FeatureEngineeringArtifact

logger = configure_logger(logger_name=__name__, level="DEBUG", to_console=True, to_file=True, log_file_name=__name__)

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
    def __init__(self, model_evaluation_config:ModelEvaluationConfig, model_trainer_artifact:ModelTrainerArtifact, feature_engineering_artifact:FeatureEngineeringArtifact):
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
            logger.info("ModelEvaluation class configured successfully.")
        except Exception as e:
            raise DetailedException(exc=e, logger=logger) from e
        
        
    def evaluate_model(self, model:object, test_ddf: ddf.DataFrame)->dict:
        """
        Evaluate the trained model on the test set and compute key
        performance metrics (accuracy, precision, recall, F1) using
        Dask-delayed execution for parallelism.

        Steps:
          1. Split the Dask DataFrame into features and label.
          2. Compute the test set into pandas once for in-memory use.
          3. Wrap sklearn metric calls in a Dask-delayed function.
          4. Compute all metrics in parallel.

        :param model:    The trained model with .predict() method.
        :param test_ddf: Dask DataFrame containing the test features and label
                         (label in the last column).
        :return:         A dictionary with keys 'accuracy', 'precision',
                         'recall', and 'f1_score'.
        :raises DetailedException: On any failure during evaluation.
        """
        try:
            logger.info("Enterd 'evaluate_model' function of 'ModelEvaluation' class")
            logger.debug("Splitting test_data into 'dependent' and 'independet' features...")
            X_test = test_ddf.iloc[:, :-1]
            y_test = test_ddf.iloc[:, -1]

            # Pull into memory as pandas once
            X_test_in_mem = X_test.compute()
            y_test_in_mem = y_test.compute()

            logger.info("Test data splitted successfully.")
            
            @delayed
            def evaluate(model:object, X_test_in_mem:DataFrame, y_test_in_mem:DataFrame)->dict:
                """
                Compute sklearn metrics on in‐memory Pandas Dataframe.
                """
                y_pred = model.predict(X_test_in_mem)
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
            logger.info("Entered 'initiate_feature_engineering' method of 'FeatureEngineering' class")
            print("\n" + "-"*80)
            print("🚀 Starting Feature Engineering Component...")

            logger.debug("Loading Test data from: %s",self.feature_engineering_artifact.feature_engineered_test_data_file_path)
            test_ddf = load_dask_dataframe(file_path=self.feature_engineering_artifact.feature_engineered_test_data_file_path,
                                            logger=logger)
            logger.info("Test Data Successfully Loaded.")

            logger.debug("Loading Model from: %s",self.model_trainer_artifact.trained_model_obj_path)
            model = load_object(file_path=self.model_trainer_artifact.trained_model_obj_path,
                                            logger=logger)
            logger.info("Model Successfully Loaded.")

            logger.debug("Initiating Model Evaluation...")
            performance_metrics = self.evaluate_model(model=model, test_ddf=test_ddf)
            logger.info("Model Evaluation Completed.")

            logger.debug("Saving Performance Metrics at: %s", self.model_evaluation_config.performance_metrics_file_save_path)
            save_json(file_path=self.model_evaluation_config.performance_metrics_file_save_path, 
                      dict=performance_metrics, 
                      logger=logger)
            logger.info("Performance Metircs Successfully Saved.")

            return ModelEvaluationArtifact(performance_metrics_file_path=self.model_evaluation_config.performance_metrics_file_save_path)
        except Exception as e:
            raise DetailedException(exc=e, logger=logger) from e



        
            



