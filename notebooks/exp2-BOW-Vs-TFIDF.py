import logging
import mlflow
import mlflow.sklearn
import dagshub
import os
import setuptools
import string
import re
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,  GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import scipy.sparse
from var import MLFLOW_TRACKING_URI, DAGSHUB_REPO_OWNER, DAGSHUB_REPO_NAME

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# # first time only—downloads the WordNet data to your default nltk_data folder
# nltk.download('wordnet')    
# # also grab the multilingual word database (often needed)
# nltk.download('omw-1.4')

logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(levelname)s - %(message)s")
# ========================== CONFIGURATION ==========================
CONFIG = {
    "data_path": "notebooks/sample-data.csv",
    "test_size": 0.2,
    "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
    "dagshub_repo_owner": DAGSHUB_REPO_OWNER,
    "dagshub_repo_name": DAGSHUB_REPO_NAME,
    "experiment_name": "BOW Vs TFIDF"
}

# ========================== SETUP MLflow & DAGSHUB ==========================
logging.info("Setting up MLFlOW and Dagshub...")
mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
dagshub.init(repo_owner=CONFIG["dagshub_repo_owner"],repo_name=CONFIG["dagshub_repo_name"],mlflow=True)
mlflow.set_experiment(CONFIG["experiment_name"])

# ========================== TEXT PREPROCESSING ==========================

# Define text preprocessing functions
def lemmatization(text:str)->str:
    """
    Lemmatize the text
    Think of lemmatization like finding a word’s “dictionary form.” It’s like this:

    * You start with a word that might be changed by tense, number, or form:

    * running, ran, runs → run
    * better → good
    * geese → goose

    * A lemmatizer looks up or reasons what the base word is (the “lemma”) instead of just chopping off endings.

    So, in plain terms: **lemmatization** turns words into their simplest, real-word form so that all the different versions (“runs,” “running,” “ran”) become just “run.” That way, when you analyze text, you treat them as the same word.
    """
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text:str)->str:
    """Remove stop words from the text.
    Stop words are the most common words in a language that carry very little semantic meaning 
    on their own—words like “a,” “an,” “the,” “in,” “on,” “and,” “but,” etc. 
    In many natural-language processing (NLP) tasks, these words are removed (or “filtered out”) before analysis
    to reduce noise and focus on the more meaningful words.
    """
    stop_word = set(stopwords.words("english")) 
    text = [word for word in text.split() if word not in stop_word]
    return " ".join(text)

def remove_numbers(text:str)->str:
    """Remove numbers from the text."""
    return " ".join([word for word in text.split() if not word.isdigit()])    

def lower_case(text:str)->str:
    """Convert text to lower case."""
    text = [word.lower() for word in text.split()]
    return " ".join(text)

def remove_urls(text:str)->str:
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def normalize_text(df):
    """Normalize the text data."""
    try:
        df['review'] = df['review'].apply(remove_stop_words)
        df['review'] = df['review'].apply(remove_numbers)
        df['review'] = df['review'].apply(remove_urls)
        df['review'] = df['review'].apply(lower_case)
        df['review'] = df['review'].apply(lemmatization)
        return df
    except Exception as e:
        print(f'Error during text normalization: {e}')
        raise 

# ========================== LOAD & PREPROCESS DATA ==========================
def load_data(file_path:str)->pd.DataFrame:
    """Load Csv data from specified location and applies preproccessing to it."""
    try:
        logging.info(f"Loading data from {file_path} ...")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            logging.info("Preprocessing data...")
            df = normalize_text(df)
            df = df[df['sentiment'].isin(['positive', 'negative'])]
            df['sentiment'] = df["sentiment"].map({'positive':1,'negative':0})
            return df
        else:
            raise FileNotFoundError
    except Exception as e:
        print(f"Error loading data: {e}")
        raise 

# ========================== FEATURE ENGINEERING ==========================
VECTORIZERS = {
    'BOW': CountVectorizer(),
    'TF-IDF': TfidfVectorizer()
}

ALGORITHMS = {
    'LogisticRegression': LogisticRegression(),
    'MultinomialNB': MultinomialNB(),
    'XGBoost': XGBClassifier(),
    'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier()
}


# ========================== TRAIN & EVALUATE MODELS ==========================
def log_model_params(algo_name:str, model:BaseEstimator)->None:
    """Logs hyperparameters of the trained model to MLflow."""
    try:
        logging.info(f"Logging model({algo_name}) parameters to mlflow...")
        params_to_log = {}
        if algo_name == 'LogisticRegression':
            params_to_log["C"] = model.C
        elif algo_name == 'MultinomialNB':
            params_to_log["alpha"] = model.alpha
        elif algo_name == 'XGBoost':
            params_to_log["n_estimators"] = model.n_estimators
            params_to_log["learning_rate"] = model.learning_rate
        elif algo_name == 'RandomForest':
            params_to_log["n_estimators"] = model.n_estimators
            params_to_log["max_depth"] = model.max_depth
        elif algo_name == 'GradientBoosting':
            params_to_log["n_estimators"] = model.n_estimators
            params_to_log["learning_rate"] = model.learning_rate
            params_to_log["max_depth"] = model.max_depth

        mlflow.log_params(params_to_log)
    except Exception as e:
        logging.info(f"Error occured while loffing model parameters:{e}")
        raise
        

def train_and_evaluate(df:pd.DataFrame)->None:
    try:
        logging.info("Setting up mlflow experiment..." )
        with mlflow.start_run(run_name="ALL Experiments") as parent_run:
            mlflow.log_artifact(local_path=r"notebooks\exp2-BOW-Vs-TFIDF.py",artifact_path="notebooks")
            for algo_name, algorithm in ALGORITHMS.items():
                for vec_name, vectorizer in VECTORIZERS.items():
                    with mlflow.start_run(run_name=f"{algo_name} with {vec_name}", nested=True) as child_run:
                        # Feature extraction
                        logging.info("Perfoming vectorization on the data...")
                        X = vectorizer.fit_transform(df["review"])
                        Y = df["sentiment"]
                        logging.info("Performing train-test-split...")
                        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=CONFIG["test_size"],random_state=42)

                        # Log preprocessing parameters
                        logging.info("Logging preproccessing parameters to mlflow...")
                        mlflow.log_params(params={"Vectorizer":vec_name,
                                                  "Algorithm":algo_name,
                                                  "test_size":CONFIG["test_size"]
                                                  })
                        
                        
                        # Train model
                        logging.info(f"Starting Model({algo_name}) Training...")
                        model = algorithm
                        model.fit(X_train, Y_train)

                        # Log model parameters
                        log_model_params(algo_name, model)
                        
                        # Evaluate model
                        logging.info(f"Evaluating model({algo_name}) performance...")
                        Y_pred = model.predict(X_test)
                        metrics = {
                            "accuracy": accuracy_score(Y_test, Y_pred),
                            "precision": precision_score(Y_test, Y_pred),
                            "recall": recall_score(Y_test, Y_pred),
                            "f1_score": f1_score(Y_test, Y_pred)
                        }    
                        logging.info(f"Logging model({algo_name}) metrics to mlflow...")
                        mlflow.log_metrics(metrics)

                        # Log model
                        input_example = X_test[:5] if not scipy.sparse.issparse(X_test) else X_test[:5].toarray()
                        mlflow.sklearn.log_model(model, "model", input_example=input_example)

                        # Print results for verification
                        print(f"\nAlgorithm: {algo_name}, Vectorizer: {vec_name}")
                        print(f"Metrics: {metrics}")

    except Exception as e:
         logging.error(f"Error in training {algo_name} with {vec_name}: {e}")
         mlflow.log_param("error", str(e))

# ========================== EXECUTION ==========================
if __name__ == "__main__":
    df = load_data(CONFIG["data_path"])
    train_and_evaluate(df)