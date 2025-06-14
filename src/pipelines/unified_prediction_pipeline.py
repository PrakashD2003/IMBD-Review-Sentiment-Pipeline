import dagshub
import mlflow
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster, get_client
from pathlib import Path
from functools import partial

from src.entity.config_entity import PredictionPipelineConfig
from src.logger import configure_logger
from src.exception import DetailedException
from src.utils.main_utils import load_params
from src.data.data_preprocessing import DataPreprocessing
from src.constants import DASK_SCHEDULER_ADDRESS, PARAM_FILE_PATH, N_PARTITIONS

module_name = Path(__file__).stem
logger = configure_logger(
    logger_name=module_name,
    level="DEBUG",
    to_console=True,
    to_file=True,
    log_file_name=module_name,
)

def start_client() -> Client:
    """Start or connect to a Dask scheduler for batch inference."""
    if DASK_SCHEDULER_ADDRESS:
        logger.info("Connecting to remote Dask scheduler at %s", DASK_SCHEDULER_ADDRESS)
        return Client(DASK_SCHEDULER_ADDRESS)
    else:
        logger.info("Starting local Dask cluster for prediction")
        cluster = LocalCluster(n_workers=8, threads_per_worker=2, memory_limit="4GB")
        return Client(cluster)



def batch_predict_func(partition: pd.DataFrame, text_col: str, preprocessor, vectorizer, model):
    """
    Process one pandas partition: preprocess, vectorize, predict.

    Returns the original columns plus a `prediction` column.
    """
    cleaned = preprocessor.preprocess_data(partition, col=text_col)
    X = vectorizer.transform(cleaned[text_col].tolist())
    y = model.predict(X)
    return pd.DataFrame({
        **{col: partition[col].values for col in partition.columns},
        "prediction": y
    }, index=partition.index)

class UnifiedPredictionPipeline:
    """
    A unified inference pipeline that supports both single-record and
    large-scale batch prediction with Dask.
    """
    def __init__(self):
        """
        Load params, start MLflow, and fetch vectorizer+model from the registry.
        """
        try:
            logger.debug("Configuring 'UnifiedPredictionPipeline' class of 'prediction_pipeline' module through constructer...")
             # Ensure a Dask client is available
            try:
                get_client()
            except ValueError:
                start_client()
            self.config = PredictionPipelineConfig()
            self.params = load_params(params_path=PARAM_FILE_PATH, logger=logger)
            self.preprocessor = DataPreprocessing()

            # MLflow setup
            self.configure_mlflow()

            # Loading Models
            logger.info("Loading classifier model named %r", self.config.mlflow_model_name)
            self.model = self.get_latest_model(model_name=self.config.mlflow_model_name,
                                            stages=self.config.mlflow_model_stages)
            
            logger.info("Loading TF-IDF vectorizer named %r", self.config.mlflow_vectorizer_name)
            self.vectorizer = self.get_latest_model(model_name=self.config.mlflow_vectorizer_name, 
                                                    stages=self.config.mlflow_model_stages)
            logger.info("Vectorizer vocab size: %d", len(self.vectorizer.vocabulary_))
            logger.info("Classifier expects %d features", self.model.coef_.shape[1])
            
            logger.info("'UnifiedPredictionPipeline' class configured successfully.")
        except Exception as e:
            raise DetailedException(exc=e, logger=logger) from e
        
    
    def configure_mlflow(self)->None:
        """
        Configures MLflow and DagsHub for tracking.
        """
        try:
            logger.info("Entered 'configure_mlflow' function of 'PredictionPipeline' class.")
            logger.debug("Setting up MLFlOW and Dagshub...")
            mlflow.set_tracking_uri(uri=self.config.mlflow_uri)
            dagshub.init(repo_owner=self.config.dagshub_repo_owner_name,
                         repo_name=self.config.dagshub_repo_name,
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
            logger.debug("Getting Latest Version of model: '%s' of stages: '%s' from Mlflow Registry...", model_name, stages)
            latest_version = client.get_latest_versions(model_name, stages=stages)
            if not latest_version:
                raise RuntimeError(f"No {stages} version of {model_name}")
            model_uri = f'models:/{model_name}/{latest_version[0].version}'
            logger.info(f"Fetching model from: {model_uri}")
            return mlflow.sklearn.load_model(model_uri=model_uri)   
        except Exception as e:
            raise DetailedException(exc=e, logger=logger) from e

    def predict_single(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run inference on a small pandas DataFrame and return binary sentiment predictions.

        This method is optimized for low-volume / per-request inference: it
        preprocesses in memory, vectorizes via the loaded TF-IDF, and runs
        the model locally.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing at least the configured text column
            (e.g. `"review"`).

        Returns
        -------
        pd.DataFrame
            A new DataFrame, indexed the same as `df`, with a single column
            `"prediction"` of dtype int (0 or 1).

        Raises
        ------
        DetailedException
            If any error occurs during preprocessing or model prediction.
        """
        try:
            logger.info("Running single inference...")
            text_col = self.params["prediction_pipeline_params"]["text_column_for_preprocessing"]
            cleaned = self.preprocessor.preprocess_data(df, col=text_col)
            X = self.vectorizer.transform(cleaned[text_col])
            y = self.model.predict(X).astype(int)
            return pd.DataFrame({"prediction": y}, index=df.index)
        except Exception as e:
            raise DetailedException(exc=e, logger=logger) from e
    
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run distributed batch inference on a pandas DataFrame using Dask.

        This method will spin up (or reuse) a Dask Client, split `df` into
        `N_PARTITIONS` partitions, and map a preprocessing→vectorize→predict
        function across them in parallel. The final result is gathered back
        into a pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing at least the configured text column.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame with all original columns plus a `"prediction"`
            column of dtype int.

        Raises
        ------
        DetailedException
            If any error occurs while scheduling or computing the Dask pipeline.
        """
        try:
            logger.info("Running batch inference...")
            text_col = self.params["prediction_pipeline_params"]["text_column_for_preprocessing"]
            ddf = dd.from_pandas(df, npartitions=N_PARTITIONS)

            batch_func = partial(
                batch_predict_func,
                text_col=text_col,
                preprocessor=self.preprocessor,
                vectorizer=self.vectorizer,
                model=self.model
            )

            result = ddf.map_partitions(
                batch_func,
                meta=ddf._meta.assign(prediction=pd.Series(dtype='int64'))
            )

            return result.compute()
        except Exception as e:
            raise DetailedException(exc=e, logger=logger) from e
        


        