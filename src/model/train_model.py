import dask.dataframe as ddf
from pathlib import Path
from dask_ml.linear_model import LogisticRegression

from src.logger import configure_logger
from src.exception import DetailedException
from src.utils.main_utils import load_params, load_parquet_as_dask_dataframe, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import FeatureEngineeringArtifact, ModelTrainerArtifact
from src.constants import PARAM_FILE_PATH

module_name = Path(__file__).stem

logger = configure_logger(logger_name=module_name, 
                          level="DEBUG", to_console=True, 
                          to_file=True, 
                          log_file_name=module_name)

class ModelTrainer:
    """
    Component to train and persist a distributed Logistic Regression model
    on Dask DataFrames.

    Attributes
    ----------
    client : dask.distributed.Client
        The Dask client for distributed computation.
    """
    def __init__(self, model_trainer_config:ModelTrainerConfig = ModelTrainerConfig(), 
                 feature_engineering_artifact:FeatureEngineeringArtifact = FeatureEngineeringArtifact()):
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
            logger.info("Model Trainer class configured successfully.")
        except Exception as e:
            raise DetailedException(exc=e, logger=logger) from e

    def train_model(self, train_ddf:ddf.DataFrame, target_col:str)-> LogisticRegression:
        """
        Train a distributed Logistic Regression model on a Dask DataFrame.

        This method will:
        1. Persist the incoming `train_ddf` to materialize partitions and fix divisions,
            so that downstream operations (like computing array lengths) work correctly.
        2. Split off the column named by `target_col` as the label (y) and use all
            other columns as features (X), converting each to a Dask Array.
        3. Initialize a Dask-ML `LogisticRegression` with hyperparameters taken from
            `self.params["Model_Params"]`.
        4. Fit the model on (X, y) in parallel across partitions.

        :param train_ddf: 
            A Dask DataFrame of feature-engineered training data. Must include a
            column matching `target_col`.
        :param target_col: 
            Name of the column in `train_ddf` to use as the target label. All other
            columns become features.
        :return: 
            A fitted `dask_ml.linear_model.LogisticRegression` instance.
        :raises DetailedException: 
            If any step fails (persisting, splitting, model initialization, or fit).
        """
        try:
            logger.info("Entered 'train_model' function of 'ModelTrainer' class.")
            logger.debug("Splitting training data into 'dependent' and 'independent' features...")
            train_ddf = train_ddf.persist()     # materialize all partitions & fix divisions
            y_train = train_ddf[target_col].to_dask_array(lengths=True)
            x_train = train_ddf.drop(columns=[target_col]).to_dask_array(lengths=True)

            model_params = self.params.get("Model_Params", {})
            logger.debug("Initializing 'LogisticRegression' object with Params: %s", model_params)
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
            train_ddf = load_parquet_as_dask_dataframe(file_path=self.feature_engineering_artifact.feature_engineered_training_data_file_path,
                                            logger=logger)
            
            clf = self.train_model(train_ddf=train_ddf, target_col=self.params.get("Target_Col"))

            logger.debug("Saving Trained Model Object at: %s", self.model_trainer_config.trained_model_obj_path)
            save_object(file_path=self.model_trainer_config.trained_model_obj_path,
                        obj=clf,
                        logger=logger)
            logger.info("Model Object Saved Successfully.")

            return ModelTrainerArtifact(trained_model_obj_path=self.model_trainer_config.trained_model_obj_path)
        except Exception as e:
            raise DetailedException(exc=e, logger=logger)

if __name__ == "__main__":
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training()



            
