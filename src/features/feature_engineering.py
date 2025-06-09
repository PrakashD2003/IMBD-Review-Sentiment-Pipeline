import pandas as pd
import dask.dataframe as ddf
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from logger import configure_logger
from exception import DetailedException
from utils.main_utils import load_params, save_dask_dataframe_as_csv, load_dask_dataframe, save_object
from entity.config_entity import FeatureEngineeringConfig
from entity.artifact_entity import DataPreprocessingArtifact, FeatureEngineeringArtifact
from constants import PARAM_FILE_PATH

logger = configure_logger(logger_name=__name__, level="DEBUG", to_console=True, to_file=True, log_file_name=__name__)

class FeatureEngineering:
    """
    Perform text feature extraction on Dask DataFrames.
    
    - Fits a TF-IDF vectorizer on the entire training set.
    - Transforms training and test partitions in parallel.
    - Saves out both the transformed DataFrames and the fitted vectorizer.
    """

    def __init__(self, feature_engineering_congfig: FeatureEngineeringConfig, data_preprocessing_artifact: DataPreprocessingArtifact):
        """
        Initialize the FeatureEngineering component.

        :param feature_engineering_config:
            Holds output file paths for vectorized train/test data and the vectorizer object.
        :param data_preprocessing_artifact:
            Contains file paths to the preprocessed train/test CSVs.
        :raises DetailedException: If loading parameters fails.
        """
        try:
            logger.debug("Configuring 'FeatureEngineering' class of 'features' module through constructer...")
            self.feature_engineering_congfig = feature_engineering_congfig
            self.data_preprocessing_artifact = data_preprocessing_artifact
            self.params = load_params(params_path=PARAM_FILE_PATH, logger=logger)
            logger.info("FeatureEngineering class configured successfully.")
        except Exception as e:
            raise DetailedException(exc=e, logger=logger) from e

    def vectorize_tfidf(
        self,
        train_ddf: ddf.DataFrame,
        test_ddf:  ddf.DataFrame,
        column:    str,
    ) -> Tuple[ddf.DataFrame, ddf.DataFrame, TfidfVectorizer]:
        """
        Fit a TfidfVectorizer on the entire training column, then transform train & test.

        Steps:
          1. Compute the full `column` from train_ddf into pandas.
          2. Fit TfidfVectorizer(max_features=...) using parameters from `self.params`.
          3. Build a `meta` DataFrame describing the output schema.
          4. Define a partition‐wise transformation that calls `tfidf.transform(...)`
             and wraps the result in a pandas DataFrame.
          5. Apply `map_partitions` on both train_ddf and test_ddf.

        :param train_ddf:  Dask DataFrame for training (must contain `column`).
        :param test_ddf:   Dask DataFrame for testing  (must contain `column`).
        :param column:     Name of the text column to vectorize.
        :return: A tuple of
            - train_tfidf_ddf: Dask DataFrame of TF-IDF features (float64)
            - test_tfidf_ddf:  Dask DataFrame of TF-IDF features (float64)
            - tfidf:           The fitted TfidfVectorizer instance
        :raises DetailedException: If fitting or transformation fails.
        """
        try:
            logger.info("Fitting TfidfVectorizer on the entire training data...")

            # Build vocabulary on the full column in memory
            full_series: pd.Series = train_ddf[column].compute()
            tfidf_params = self.params.get("TF-IDF_Params", {})
            tfidf = TfidfVectorizer(**tfidf_params)
            tfidf.fit(full_series)

            # Prepare meta with float64 TF-IDF columns
            feature_names = tfidf.get_feature_names_out()
            meta = pd.DataFrame({f: pd.Series(dtype="float64") for f in feature_names})

            # Partition-wise transform using the fitted vectorizer
            def _transform_partition(pdf: pd.DataFrame) -> pd.DataFrame:
                X = tfidf.transform(pdf[column])  # sparse (n_rows × n_features)
                return pd.DataFrame.sparse.from_spmatrix(
                    X, index=pdf.index, columns=feature_names
                )

            # Apply to each partition
            logger.debug("Vectorizing training partitions…")
            train_tfidf_ddf = train_ddf.map_partitions(_transform_partition, meta=meta)
            logger.info("Training data vectorized into %d features.", len(feature_names))

            logger.debug("Vectorizing test partitions…")
            test_tfidf_ddf = test_ddf.map_partitions(_transform_partition, meta=meta)
            logger.info("Test data vectorized into %d features.", len(feature_names))

            return train_tfidf_ddf, test_tfidf_ddf, tfidf

        except Exception as e:
            raise DetailedException(exc=e, logger=logger) from e
        
    def initiate_feature_engineering(self)-> FeatureEngineeringArtifact:
        """
        End-to-end feature engineering workflow:

          1. Load preprocessed train/test CSVs as Dask DataFrames.
          2. Vectorize both using TF-IDF.
          3. Save out the vectorized DataFrames (partitioned).
          4. Save out the fitted TfidfVectorizer object.

        :return: A FeatureEngineeringArtifact containing:
            - Paths to the vectorized train/test CSVs
            - Path to the serialized vectorizer object
        :raises DetailedException: On any failure in load, vectorize, or save steps.
        """
        try:
            logger.info("Entered 'initiate_feature_engineering' method of 'FeatureEngineering' class")
            print("\n" + "-"*80)
            print("🚀 Starting Feature Engineering Component...")

            logger.debug("Loading Training data from: %s",self.data_preprocessing_artifact.preprocessed_training_data_file_path)
            train_ddf = load_dask_dataframe(file_path=self.data_preprocessing_artifact.preprocessed_training_data_file_path,
                                            logger=logger)
            logger.info("Training Data Successfully Loaded.")
            logger.debug("Loading Test data from: %s",self.data_preprocessing_artifact.preprocessed_test_data_file_path)
            test_ddf = load_dask_dataframe(file_path=self.data_preprocessing_artifact.preprocessed_test_data_file_path,
                                            logger=logger)
            logger.info("Test Data Successfully Loaded.")

            logger.debug("Vectorizing training and test data using 'Tf-idf' vectorizer with params: %s", **self.params("TF-IDF_Params",{}))
            vectorized_train_ddf, vectorized_test_ddf, vectorizer_obj = self.vectorize_tfidf(train_ddf=train_ddf,
                                                                       test_ddf=test_ddf,
                                                                       column=self.params[""]
                                                                       )
            logger.info("Successfully Vectorized training and test data.")
            
            logger.debug("Saving vectorized training data at: %s ",self.feature_engineering_congfig.feature_engineered_training_data_file_path)
            save_dask_dataframe_as_csv(file_save_path=self.feature_engineering_congfig.feature_engineered_training_data_file_path,
                                  dataframe=vectorized_train_ddf,
                                  single_file=False,
                                  index=False,
                                  logger=logger)
            logger.info("Successfully saved the training data.")
            
            logger.debug("Saving vectorized test data at: %s ",self.feature_engineering_congfig.feature_engineered_test_data_file_path)
            save_dask_dataframe_as_csv(file_save_path=self.feature_engineering_congfig.feature_engineered_test_data_file_path,
                                  dataframe=vectorized_test_ddf,
                                  single_file=False,
                                  index=False,
                                  logger=logger)
            logger.info("Successfully saved the test data.")
            
            logger.debug("Saving Vectorizer Object at: %s ",self.feature_engineering_congfig.vectorizer_obj_file_path)
            save_object(file_path=self.feature_engineering_congfig.vectorizer_obj_file_path,
                        obj=vectorizer_obj,
                        logger=logger)
            logger.info("Successfully saved the Vectorizer Object.")

            return FeatureEngineeringArtifact(feature_engineered_training_data_file_path=self.feature_engineering_congfig.feature_engineered_training_data_file_path,
                                              feature_engineered_test_data_file_path=self.feature_engineering_congfig.feature_engineered_test_data_file_path,
                                              vectorizer_obj_file_path=self.feature_engineering_congfig.vectorizer_obj_file_path)
        except Exception as e:
            raise DetailedException(exc=e, logger=logger)

            


