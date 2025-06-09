import dask.dataframe as ddf
from pandas import DataFrame
from dask import delayed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
from logger import configure_logger
from exception import DetailedException
from utils.main_utils import load_dask_dataframe, load_object, save_json
from entity.config_entity import ModelEvaluationConfig
from entity.artifact_entity import ModelTrainerArtifact, ModelEvaluationArtifact, FeatureEngineeringArtifact

logger = configure_logger(logger_name=__name__, level="DEBUG", to_console=True, to_file=True, log_file_name=__name__)


class ModelRegistry:
    def __init__(self, model_):
        