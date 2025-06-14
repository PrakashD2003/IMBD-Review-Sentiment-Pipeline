import dagshub
import mlflow
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
from pathlib import Path

from src.entity.config_entity import BatchPredictionConfig
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
    """
    Start or connect to a Dask scheduler for batch workloads.

    Returns
    -------
    Client
        A Dask distributed Client.
    """
    if DASK_SCHEDULER_ADDRESS:
        logger.info("Connecting to remote Dask scheduler at %s", DASK_SCHEDULER_ADDRESS)
        return Client(DASK_SCHEDULER_ADDRESS)
    else:
        logger.info("Starting local Dask cluster for batch processing")
        cluster = LocalCluster(n_workers=8, threads_per_worker=2, memory_limit="4GB")
        return Client(cluster)

class BatchPredictionPipeline:
    """
    A pipeline for large-scale batch inference using Dask.

    1. Spins up a Dask client
    2. Reads input data as a Dask DataFrame
    3. Applies preprocessing, vectorization, and prediction via map_partitions
    4. Writes out predictions to Parquet
    """
    def __init__(
        self,
        config: BatchPredictionConfig = BatchPredictionConfig(),
        preprocessor: DataPreprocessing = DataPreprocessing()
    ):
        """
        Initialize the batch pipeline.

        Parameters
        ----------
        config : BatchPredictionConfig
            Config object with paths, model names, etc.
        preprocessor : DataPreprocessing
            Preprocessing component.
        """
        try:
            logger.debug("Initializing BatchPredictionPipeline...")
            self.config = config
            self.preprocessor = preprocessor
            self.params = load_params(params_path=PARAM_FILE_PATH, logger=logger)

             # Start Dask client (closed when run methods complete)
            self.client = start_client()

            # Configure MLflow
            mlflow.set_tracking_uri(self.config.mlflow_uri)
            dagshub.init(
                repo_owner=self.config.dagshub_repo_owner_name,
                repo_name=self.config.dagshub_repo_name,
                mlflow=True
            )

            # Load vectorizer and model once
            client = mlflow.MlflowClient()
            vec_versions = client.get_latest_versions(self.config.mlflow_vectorizer_name, stages=self.config.mlflow_model_stages)
            vec_uri = f"models:/{self.config.mlflow_vectorizer_name}/{vec_versions[0].version}"
            self.vectorizer = mlflow.pyfunc.load_model(vec_uri)

            mdl_versions = client.get_latest_versions(self.config.mlflow_model_name, stages=self.config.mlflow_model_stages)
            mdl_uri = f"models:/{self.config.mlflow_model_name}/{mdl_versions[0].version}"
            self.model = mlflow.pyfunc.load_model(mdl_uri)

            logger.info("BatchPredictionPipeline configured.")
        except Exception as e:
            raise DetailedException(exc=e, logger=logger)

    def predict_partition(self, pdf: pd.DataFrame) -> pd.DataFrame:
        """
        Process one pandas partition: preprocess, vectorize, predict.
        """
        # Preprocess
        text_col = self.params['prediction_pipeline_params']['text_column_for_preprocessing']
        cleaned = self.preprocessor.preprocess_data(dataframe=pdf, col=text_col)
        # Vectorize and predict
        X = self.vectorizer.transform(cleaned[text_col].tolist())
        preds = self.model.predict(X)
        return pd.DataFrame({
            **{col: pdf[col].values for col in pdf.columns},
            'prediction': preds
        }, index=pdf.index)

    def run_locally(self) -> None:
        """
        Execute the batch prediction pipeline end-to-end.
        """
        try:
            # 1. Read
            logger.info("Reading input data from %s", self.config.input_path)
            ddf = dd.read_parquet(self.config.input_path)

            # 2. Run predict over partitions
            logger.info("Scheduling predictions over %d partitions", ddf.npartitions)
            result = ddf.map_partitions(
                self.predict_partition,
                meta=ddf._meta.assign(prediction=pd.Series(dtype='int64'))
            )

            # 3. Write out
            logger.info("Writing predictions to %s", self.config.output_path)
            result.to_parquet(self.config.output_path, write_index=False)
            logger.info("Batch predictions saved.")
        except Exception as e:
            raise DetailedException(exc=e, logger=logger)
        finally:
            self.client.close()
    
    def run_on_api(self, df:pd.DataFrame) -> None:
        """
        Execute the batch prediction pipeline end-to-end.
        """
        try:
            # 1. Read
            logger.info("Reading input data from %s", self.config.input_path)
            ddf = dd.from_pandas(data=df, npartitions=N_PARTITIONS)

            # 2. Run predict over partitions
            logger.info("Scheduling predictions over %d partitions", ddf.npartitions)
            result = ddf.map_partitions(
                self.predict_partition,
                meta=ddf._meta.assign(prediction=pd.Series(dtype='int64'))
            )

            # 3. Write out
            logger.info("Returning batch prediction.")
            return result.compute()
        except Exception as e:
            raise DetailedException(exc=e, logger=logger)
        finally:
            self.client.close()

if __name__ == "__main__":
    batch = BatchPredictionPipeline()
    batch.run()
