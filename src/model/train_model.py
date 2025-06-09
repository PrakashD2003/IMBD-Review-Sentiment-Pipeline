import pandas as pd
import dask.dataframe as ddf
from dask_ml.linear_model import LogisticRegression
from logger import configure_logger
from exception import DetailedException
from utils.main_utils import load_params, save_dask_dataframe_as_csv, load_dask_dataframe, save_object
from entity.config_entity import ModelTrainerConfig
from entity.artifact_entity import FeatureEngineeringArtifact, ModelTrainerArtifact
from constants import PARAM_FILE_PATH

logger = configure_logger(logger_name=__name__, level="DEBUG", to_console=True, to_file=True, log_file_name=__name__)


class ModelTrainer:
    """
    Component to train and persist a distributed Logistic Regression model
    on Dask DataFrames.

    Attributes
    ----------
    client : dask.distributed.Client
        The Dask client for distributed computation.
    """
    def __init__(self, model_trainer_config:ModelTrainerConfig, feature_engineering_artifact:FeatureEngineeringArtifact ):
        """
        Initialize the ModelTrainer.

        :param config:            ModelTrainerConfig with paths and settings.
        :param fe_artifact:       FeatureEngineeringArtifact carrying
                                  paths to feature-engineered CSVs.
        :param dask_scheduler:    Address of an existing Dask scheduler
                                  (e.g. "tcp://…:8786"). If None, uses
                                  a local threaded client.
        :raises DetailedException: If client start or param load fails.
        """
        try:
            logger.debug("Configuring 'ModelTrainer' class of 'model' module through constructer...")
            self.model_trainer_config = model_trainer_config
            self.feature_engineering_artifact = feature_engineering_artifact
            self.params = load_params(params_path=PARAM_FILE_PATH, logger=logger)
            logger.info("FeatureEngineering class configured successfully.")
        except Exception as e:
            raise DetailedException(exc=e, logger=logger) from e

    def train_model(self, train_ddf:ddf.DataFrame)-> LogisticRegression:
        """
        Train a distributed Logistic Regression model on the provided Dask DataFrame.

        :param train_ddf: Dask DataFrame containing feature-engineered training data.
            Assumes the last column is the label and all preceding columns are features.
        :return: A fitted Dask-ML LogisticRegression model.
        :raises DetailedException: If training fails.
        """
        try:
            logger.info("Entered 'train_model' function of 'ModelTrainer' class.")
            logger.debug("Splitting training data into 'dependent' and 'independent' features...")
            y_train = train_ddf.iloc[:,-1]
            x_train = train_ddf.iloc[:,:-1]

            logger.debug("Initializing 'LogisticRegression' object...")
            model_params = self.params("Model_Params", {})
            clf = LogisticRegression(**model_params)
            logger.info("'LogisticRegression' object initialized successfully")

            logger.debug("Fitting 'LogisticRegression' on the entire training data...")
            clf.fit(X=x_train, y=y_train)
            logger.info("Model Trained Successfully.")

            return clf
        except Exception as e:
            raise DetailedException(exc=e, logger=logger)
        
    def initiate_model_training(self) ->ModelTrainerArtifact:
        """
        Orchestrate the full model training workflow:
          1. Load the feature-engineered training data as a Dask DataFrame.
          2. Call train_model() to fit the model.
          3. Persist the trained model to disk.

        :return: A ModelTrainerArtifact containing the path to the saved model.
        :raises DetailedException: If any step of the workflow fails.
        """
        try:
            logger.info("Entered 'initiate_model_training' method of 'ModelTrainer' class")
            print("\n" + "-"*80)
            print("🚀 Starting Model Trainer Component...")

            logger.debug("Loading training data from: %s", self.feature_engineering_artifact.feature_engineered_training_data_file_path)
            train_ddf = load_dask_dataframe(file_path=self.feature_engineering_artifact.feature_engineered_training_data_file_path,
                                            logger=logger)
            
            clf = self.train_model(train_ddf=train_ddf)

            logger.debug("Saving Trained Model Object at: %s", self.model_trainer_config.trained_model_obj_path)
            save_object(file_path=self.model_trainer_config.trained_model_obj_path,
                        obj=clf,
                        logger=logger)
            logger.info("Model Object Saved Successfully.")

            return ModelTrainerArtifact(trained_model_obj_path=self.model_trainer_config.trained_model_obj_path)
        except Exception as e:
            raise DetailedException(exc=e, logger=logger)



            



            
