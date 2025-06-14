import dagshub
import mlflow
import pandas as pd

from dask.distributed import Client, LocalCluster
from pathlib import Path

from src.entity.config_entity import PredictionPipelineConfig
from src.logger import configure_logger
from src.exception import DetailedException
from src.utils.main_utils import load_params
from src.data.data_preprocessing import DataPreprocessing 
from src.constants import DASK_SCHEDULER_ADDRESS, PARAM_FILE_PATH

module_name = Path(__file__).stem
logger = configure_logger(
    logger_name=module_name,
    level="DEBUG",
    to_console=True,
    to_file=True,
    log_file_name=module_name,
)



class PredictionPipeline:
    """
    A per-request inference pipeline that:
      1. Loads model + vectorizer from MLflow Model Registry.
      2. Preprocesses raw text via your existing Dask-based DataPreprocessing.
      3. Vectorizes and predicts entirely in-memory with sklearn.

    This avoids spinning up a cluster on every call, and is optimized
    for low-latency web requests.
    """
    def __init__(self, prediction_pipeline_congfig:PredictionPipelineConfig = PredictionPipelineConfig(),
                 data_preprocessing:DataPreprocessing = DataPreprocessing()):
        """
        Configure Mlflow, load parameters, vectorizer, and model.

        Parameters
        ----------
        prediction_pipeline_congfig : PredictionPipelineConfig
            Contains MLflow & DagsHub URIs, model names, etc.
        data_preprocessing : DataPreprocessing
            Your existing preprocessing component, which returns a Dask dataframe.
        """
        try:
            # >>> if you uncomment one of these, it changes where the work runs <<<

            # 1) Comment this out for local threaded execution:
            # client = start_client()
            logger.debug("Configuring 'PredictionPipeline' class of 'prediction_pipeline' module through constructer...")
            self.prediction_pipeline_congfig = prediction_pipeline_congfig
            self.data_preprocessing = data_preprocessing
            self.params = load_params(params_path=PARAM_FILE_PATH, logger=logger)
            self.configure_mlflow()
            self.vectorizer = self.get_latest_model(model_name=self.prediction_pipeline_congfig.mlflow_vectorizer_name, 
                                                    stages=self.prediction_pipeline_congfig.mlflow_model_stages)
            self.model = self.get_latest_model(model_name=self.prediction_pipeline_congfig.mlflow_model_name,
                                          stages=self.prediction_pipeline_congfig.mlflow_model_stages)
            logger.info("PredictionPipeline class configured successfully.")
        except Exception as e:
            raise DetailedException(exc=e, logger=logger)
    
    def configure_mlflow(self)->None:
        """
        Configures MLflow and DagsHub for tracking.

        Args:
            experiment_name (str): The name of the MLflow experiment.
        """
        try:
            logger.info("Entered 'configure_mlflow' function of 'PredictionPipeline' class.")
            logger.debug("Setting up MLFlOW and Dagshub...")
            mlflow.set_tracking_uri(uri=self.prediction_pipeline_congfig.mlflow_uri)
            dagshub.init(repo_owner=self.prediction_pipeline_congfig.dagshub_repo_owner_name,
                         repo_name=self.prediction_pipeline_congfig.dagshub_repo_name,
                         mlflow=True)
            logger.info("MLFlOW and Dagshub Congigured Successfully.")
        except Exception as e:
            raise DetailedException(exc=e, logger=logger) from e
        
    def get_latest_model(self, model_name:str, stages:list[str])->object:
        """
        Retrieve the latest registered model (or vectorizer) from MLflow Model Registry.

        Parameters
        ----------
        model_name : str
            The name under which the model was registered in MLflow.
        stages : list[str]
            List of stages to consider (e.g. ["Staging", "Production"]).

        Returns
        -------
        object
            A pyfunc-wrapped model or vectorizer.
        """
        try:
            client = mlflow.MlflowClient()
            logger.debug("Getting Latest Version of model: '%s' of stages: '%s' from Mlflow Registry...")
            latest_version = client.get_latest_versions(model_name, stages=stages)
            if not latest_version:
                raise RuntimeError(f"No {stages} version of {model_name}")
            model_uri = f'models:/{model_name}/{latest_version}'
            logger.info(f"Fetching model from: {model_uri}")
            return mlflow.pyfunc.load_model(model_uri)   
        except Exception as e:
            raise DetailedException(exc=e, logger=logger) from e

    
    def predict(self, df:pd.DataFrame) -> pd.Series:
        """
        Run end-to-end prediction on raw input.

        1. Preprocess text (uses Dask under the hood, then .compute()).
        2. Transform with TF-IDF.
        3. Predict with the loaded model.

        Parameters
        ----------
        df : pd.DataFrame
            Incoming request data. Must contain the text column named in
            `prediction_pipeline_params.text_column_for_preprocessing`.

        Returns
        -------
        pd.Series
            Integer predictions (0/1), indexed same as `df`.
        """
        try:
            # 1) preprocess
            logger.debug("Preprocessing Data...")
            text_col = self.params['prediction_pipeline_params']['text_column_for_preprocessing']
            cleaned = self.data_preprocessing.preprocess_data(dataframe=df, col=text_col).compute()
            
            
            logger.debug("Applying TF-IDF Vectorization to Input Data...")
            X  = self.vectorizer.transform(cleaned[text_col])
            y  = self.model.predict(X).astype(int)
            return pd.Series(y, index=df.index, name="prediction")
        except Exception as e:
            raise DetailedException(exc=e, logger=logger) from e


        