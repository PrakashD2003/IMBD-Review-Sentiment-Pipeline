import argparse
from dask.distributed import Client, LocalCluster

from src.constants import DASK_SCHEDULER_ADDRESS




def start_client():
    """Bring up a Dask client (EKS scheduler or local)."""
    if DASK_SCHEDULER_ADDRESS:
        return Client(DASK_SCHEDULER_ADDRESS)
    else:
        cluster = LocalCluster(n_workers=1, threads_per_worker=2, memory_limit="12GB")
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

    if args.step == "ingestion":
        from src.data.data_ingestion import DataIngestion
        DataIngestion().initiate_data_ingestion()

    elif args.step == "preprocess":
        from src.data.data_preprocessing import DataPreprocessing
        DataPreprocessing().initiate_data_preprocessing()

    elif args.step == "fe":
        from src.features.feature_engineering import FeatureEngineering
        FeatureEngineering().initiate_feature_engineering()

    elif args.step == "train":
        from src.model.train_model import ModelTrainer
        ModelTrainer().initiate_model_training()

    elif args.step == "evaluate":
        from src.model.evaluate_model import ModelEvaluation
        ModelEvaluation().initiate_model_evaluation()

    elif args.step == "register":
        from src.model.register_model import ModelRegistry
        ModelRegistry().initiate_model_registration()

if __name__ == "__main__":
    main()
