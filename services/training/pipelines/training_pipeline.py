"""Command-line interface for running individual pipeline steps with Dask."""

import argparse
from dask.distributed import Client, LocalCluster

from common.constants import DASK_SCHEDULER_ADDRESS, DASK_WORKERS, DASK_THREADS, DASK_MEMORY_LIMIT




def start_client():
    """Bring up a Dask client (EKS scheduler or local)."""
    if DASK_SCHEDULER_ADDRESS:
        return Client(DASK_SCHEDULER_ADDRESS)
    else:
         cluster = LocalCluster(
            n_workers=DASK_WORKERS,
            threads_per_worker=DASK_THREADS,
            memory_limit=DASK_MEMORY_LIMIT,
        )
    return Client(cluster)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        choices=["ingestion", "preprocess", "fe", "train", "evaluate", "register"],
        required=True,
        help="Which pipeline step to run",
    )
    args = parser.parse_args()

    client = start_client()
    print("Dask client dashboard: %s", client.dashboard_link)


    try:
        if args.step == "ingestion":
            from services.training.data.data_ingestion import DataIngestion
            DataIngestion().initiate_data_ingestion()

        elif args.step == "preprocess":
            from services.training.data.data_preprocessing import DataPreprocessing
            DataPreprocessing().initiate_data_preprocessing()

        elif args.step == "fe":
            from services.training.features.feature_engineering import FeatureEngineering
            FeatureEngineering().initiate_feature_engineering()

        elif args.step == "train":
            from services.training.model.train_model import ModelTrainer
            ModelTrainer().initiate_model_training()

        elif args.step == "evaluate":
            from services.training.model.evaluate_model import ModelEvaluation
            ModelEvaluation().initiate_model_evaluation()

        elif args.step == "register":
            from services.training.model.register_model import ModelRegistry
            ModelRegistry().initiate_model_registration()
    finally:
        client.close()

if __name__ == "__main__":
    main()
