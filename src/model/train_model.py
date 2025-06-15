import joblib
import dask.dataframe as ddf
from pathlib import Path
from sklearn.linear_model import LogisticRegression

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
    Trains and persists a scikit‐learn LogisticRegression model in parallel,
    using Dask + Joblib for distributing the .fit() call across workers.
    """
    def __init__(self, model_trainer_config:ModelTrainerConfig = ModelTrainerConfig(), 
                 feature_engineering_artifact:FeatureEngineeringArtifact = FeatureEngineeringArtifact()):
        """
        Initialize the ModelTrainer component.

        :param model_trainer_config:
            Configuration holding paths for saving the trained model.
        :param feature_engineering_artifact:
            Artifact containing the path to the feature‐engineered training data.
        :raises DetailedException:
            If loading parameters fails.
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
        Fit a scikit‐learn LogisticRegression on Dask‐backed data in parallel.

        Steps:
          1. Persist the Dask DataFrame to materialize partitions.
          2. Convert to Dask Arrays: X (all features) and y (the target column).
             └─ sklearn only accepts NumPy arrays (or array‐like), so we must convert.
          3. Initialize sklearn.LogisticRegression(**Model_Params).
          4. Call clf.fit(X, y) within joblib.parallel_backend("dask") to distribute work.

        :param train_ddf: Dask DataFrame containing feature columns + `target_col`.
        :param target_col: Name of the column in train_ddf to use as the label.
        :return: A fitted sklearn.linear_model.LogisticRegression instance.
        :raises DetailedException: If any step (persist, split, init, fit) fails.
        """
        try:
            logger.info("Entered 'train_model' function of 'ModelTrainer' class.")
            logger.debug("Splitting training data into 'dependent' and 'independent' features...")
            
            train_ddf = train_ddf.persist()     # materialize all partitions & fix divisions
            y_train = train_ddf[target_col].to_dask_array(lengths=True)
            x_train = train_ddf.drop(columns=[target_col]).to_dask_array(lengths=True)
            
            model_params = self.params.get("Model_Params", {})
            logger.debug("Initializing 'LogisticRegression' with Params: %s", model_params)
            clf = LogisticRegression(**model_params)
            logger.info("'LogisticRegression' object initialized successfully")

            logger.debug("Fitting model in parallel over Dask...")
            with joblib.parallel_backend("dask"):
                clf.fit(X=x_train, y=y_train)
            logger.info("Model trained successfully.")

            return clf
        except Exception as e:
            raise DetailedException(exc=e, logger=logger)
        
    def initiate_model_training(self) ->ModelTrainerArtifact:
        """
        Orchestrate the full training workflow:
          1. Load the feature‐engineered data from Parquet as a Dask DataFrame.
          2. Call train_model() to fit the sklearn model in parallel.
          3. Save the trained model object to disk.

        :return: A ModelTrainerArtifact with the saved model path.
        :raises DetailedException: If loading data, training, or saving fails.
        """
        try:
            logger.info("Entered 'initiate_model_training' method of 'ModelTrainer' class")
            print("\n" + "-" * 80)
            print("Starting Model Trainer Component...")

            logger.debug("Loading training data from: %s", self.feature_engineering_artifact.feature_engineered_training_data_file_path)
            train_ddf = load_parquet_as_dask_dataframe(file_path=self.feature_engineering_artifact.feature_engineered_training_data_file_path,
                                            logger=logger)
            
            logger.info("Training Data Shape (before split): %s", train_ddf.shape)
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



