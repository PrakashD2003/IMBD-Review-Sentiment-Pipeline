"""Batch prediction pipeline using Dask for scalable inference."""

import logging
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster

from common.utils.mlflow_utils import configure_mlflow, get_latest_model
from services.prediction.entity.config_entity import BatchPredictionConfig
from common.logger import configure_logger
from common.exception import DetailedException
from common.utils.main_utils import load_params
from services.prediction.support_scripts.Data_Preprocesser import preprocess_data
from common.constants import DASK_SCHEDULER_ADDRESS, PARAM_FILE_PATH, N_PARTITIONS

# This logger will automatically inherit the configuration from the FastAPI app
logger = logging.getLogger(__name__)

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
        config: BatchPredictionConfig = BatchPredictionConfig()
    ):
        """
        Initialize the batch pipeline.

        Parameters
        ----------
        config : BatchPredictionConfig
            Config object with paths, model names, etc.
        """
        try:
            logger.debug("Initializing BatchPredictionPipeline...")
            self.config = config
            self.params = load_params(params_path=PARAM_FILE_PATH, logger=logger)

             # Start Dask client (closed when run methods complete)
            self.client = start_client()

            # Configure MLflow
            configure_mlflow(
                mlflow_uri=self.config.mlflow_uri,
                dagshub_repo_owner_name=self.config.dagshub_repo_owner_name,
                dagshub_repo_name=self.config.dagshub_repo_name,
                logger=logger,
            )

            # Load vectorizer and model once
            self.vectorizer = get_latest_model(
                model_name=self.prediction_pipeline_congfig.mlflow_vectorizer_name,
                stages=self.prediction_pipeline_congfig.mlflow_model_stages,
                logger=logger
            )
            self.model = get_latest_model(
                model_name=self.prediction_pipeline_congfig.mlflow_model_name,
                stages=self.prediction_pipeline_congfig.mlflow_model_stages,
                logger=logger
            )

            logger.info("BatchPredictionPipeline configured.")
        except Exception as e:
            raise DetailedException(exc=e, logger=logger)

    def predict_partition(self, pdf: pd.DataFrame) -> pd.DataFrame:
        """
        Process one pandas partition: preprocess, vectorize, predict.
        """
        # Preprocess
        text_col = self.params['prediction_pipeline_params']['text_column_for_preprocessing']
        cleaned = preprocess_data(dataframe=pdf, col=text_col)
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
