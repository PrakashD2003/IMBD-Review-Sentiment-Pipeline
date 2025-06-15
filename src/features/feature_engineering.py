import pandas as pd
import dask.dataframe as ddf
from pathlib import Path
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

from src.logger import configure_logger
from src.exception import DetailedException
from src.utils.main_utils import load_params, save_dask_dataframe_as_parquet, load_parquet_as_dask_dataframe, save_object
from src.entity.config_entity import FeatureEngineeringConfig
from src.entity.artifact_entity import DataPreprocessingArtifact, FeatureEngineeringArtifact
from src.constants import PARAM_FILE_PATH

module_name = Path(__file__).stem

logger = configure_logger(logger_name=module_name, 
                          level="DEBUG", to_console=True, 
                          to_file=True, 
                          log_file_name=module_name)


# Top-level partition transform function for TF-IDF
def _transform_partition(
    pdf: pd.DataFrame,
    column: str,
    feature_names: list,
    tfidf: TfidfVectorizer
) -> pd.DataFrame:
    """
    Dask‐partition transform that applies TF-IDF and returns a dense pandas DataFrame.

    :param pdf:    Partition as pandas DataFrame.
    :param column: Text column to vectorize.
    :param feature_names: List of TF-IDF feature names (vocab).
    :param tfidf:  Already‐fitted TfidfVectorizer.
    :return:        pandas.DataFrame with shape (n_rows, n_features) of floats.
    """
    X = tfidf.transform(pdf[column])
    dense = X.toarray()       # ← dense numpy
    df = pd.DataFrame(dense, index=pdf.index, columns=feature_names)
    df["sentiment"] = pdf["sentiment"].values
    return df

class FeatureEngineering:
    """
    Perform text feature extraction on Dask DataFrames.
    
    - Fits a TF-IDF vectorizer on the entire training set.
    - Transforms training and test partitions in parallel.
    - Saves out both the transformed DataFrames and the fitted vectorizer.
    """

    def __init__(self, feature_engineering_congfig: FeatureEngineeringConfig = FeatureEngineeringConfig(),
                 data_preprocessing_artifact: DataPreprocessingArtifact = DataPreprocessingArtifact()):
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
            # Convert list->tuple 
            rng = tfidf_params.get("ngram_range")
            if isinstance(rng, list):
                tfidf_params["ngram_range"] = tuple(rng)
            tfidf = TfidfVectorizer(**tfidf_params)
            tfidf.fit(full_series)

            # Prepare meta with float64 TF-IDF columns
            feature_names = tfidf.get_feature_names_out()
            feature_names = [f"tfidf_{f}" for f in tfidf.get_feature_names_out()]
            meta_cols = {f: "float64" for f in feature_names}
            meta_cols["sentiment"] = "int64"
            meta = ddf.utils.make_meta(meta_cols)

          

            # Apply to each partition
            logger.debug("Vectorizing training partitions...")
            train_vec = train_ddf.map_partitions(_transform_partition,
                                                column=column,
                                                feature_names=feature_names,
                                                tfidf=tfidf,
                                                meta=meta
                                                )
            logger.info("Training data vectorized into %d features.", len(feature_names))

            logger.debug("Vectorizing test partitions...")
            test_vec = test_ddf.map_partitions(_transform_partition,
                                            column=column,
                                            feature_names=feature_names,
                                            tfidf=tfidf,
                                            meta=meta
                                            )
            logger.info("Test data vectorized into %d features.", len(feature_names))

            return train_vec, test_vec, tfidf

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
            logger.info("\n" + "-" * 80)
            logger.info("Starting Feature Engineering Component...")

            logger.debug("Loading Training data from: %s",self.data_preprocessing_artifact.preprocessed_training_data_file_path)
            train_ddf = load_parquet_as_dask_dataframe(file_path=self.data_preprocessing_artifact.preprocessed_training_data_file_path,
                                            logger=logger)
            logger.info("Training Data Successfully Loaded.")
            logger.debug("Loading Test data from: %s",self.data_preprocessing_artifact.preprocessed_test_data_file_path)
            test_ddf = load_parquet_as_dask_dataframe(file_path=self.data_preprocessing_artifact.preprocessed_test_data_file_path,
                                            logger=logger)
            logger.info("Test Data Successfully Loaded.")

            tfidf_params = self.params.get("TF-IDF_Params", {})
            logger.info(
                "Vectorizing training and test data using Tf-IDF vectorizer with params: %s",
                tfidf_params
            )
            vectorized_train_ddf, vectorized_test_ddf, vectorizer_obj = self.vectorize_tfidf(train_ddf=train_ddf,
                                                                       test_ddf=test_ddf,
                                                                       column=self.params["feature_engineering_params"]["text_column_for_engineering"]
                                                                       )
            logger.info("Successfully Vectorized training and test data.")
            
            logger.debug("Saving vectorized training data at: %s ",self.feature_engineering_congfig.feature_engineered_training_data_file_path)
            save_dask_dataframe_as_parquet(file_save_path=self.feature_engineering_congfig.feature_engineered_training_data_file_path,
                                  dataframe=vectorized_train_ddf,
                                  single_file=False,
                                  index=False,
                                  logger=logger)
            logger.info("Successfully saved the training data.")
            
            logger.debug("Saving vectorized test data at: %s ",self.feature_engineering_congfig.feature_engineered_test_data_file_path)
            save_dask_dataframe_as_parquet(file_save_path=self.feature_engineering_congfig.feature_engineered_test_data_file_path,
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

            
if __name__ == "__main__":
    feature_engineering = FeatureEngineering()
    feature_engineering.initiate_feature_engineering()
    