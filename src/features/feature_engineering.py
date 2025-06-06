from logger import configure_logger
from exception import DetailedException
from utils.main_utils import load_csv_data, load_params, save_dataframe_as_csv
from entity.config_entity import FeatureEngineeringConfig
from entity.artifact_entity import DataPreprocessingArtifact, FeatureEngineeringArtifact
from constants import PARAM_FILE_PATH

logger = configure_logger(logger_name=__name__, level="DEBUG", to_console=True, to_file=True, log_file_name=__name__)

class FeatureEngineering:
    def __init__(self, feature_engineering_congfig: FeatureEngineeringConfig, data_preprocessing_artifact: DataPreprocessingArtifact):
        try:
            logger.debug("Configuring 'FeatureEngineering' class of 'features' module through constructer...")
            self.feature_engineering_congfig = feature_engineering_congfig
            self.data_preprocessing_artifact = data_preprocessing_artifact
            self.params = load_params(params_path=PARAM_FILE_PATH, logger=logger)
            logger.info("FeatureEngineering class configured successfully.")
        except Exception as e:
            raise DetailedException(exc=e, logger=logger) from e

