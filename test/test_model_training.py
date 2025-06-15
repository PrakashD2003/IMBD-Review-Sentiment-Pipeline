import pandas as pd
import dask.dataframe as dd
from src.model.train_model import ModelTrainer
from src.utils.main_utils import save_object
from dask.distributed import Client


def test_model_training_and_save(tmp_path):
    df = pd.DataFrame({
        "feat1": [0, 1, 0, 1],
        "feat2": [1, 0, 1, 0],
        "sentiment": [0, 1, 0, 1],
    })
    ddf = dd.from_pandas(df, npartitions=1)

    client = Client(processes=False, threads_per_worker=1, n_workers=1)
    try:
        trainer = ModelTrainer()
        model = trainer.train_model(ddf, target_col="sentiment")
    finally:
        client.close()

    model_path = tmp_path / "model.pkl"
    save_object(str(model_path), model)
    assert model_path.exists()
