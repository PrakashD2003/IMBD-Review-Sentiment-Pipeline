import logging
import mlflow
import mlflow.sklearn
import dagshub
import os
import setuptools
import string
import re
import pandas as pd


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import TfidfVectorizer
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
    "experiment_name": "LOGISTIC REGRESSION WITH BOW VECTORIZER"
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

# ==========================
# Train & Log Model
# ==========================
def train_and_log_model(X_train, Y_train, X_test, Y_test):
    """Trains a Logistic Regression model with GridSearch and logs results to MLflow."""
    param_grid = [
    {
        "penalty": ["elasticnet"],
        "solver": ["saga"],           # only solvers that support all penalties
        "l1_ratio": [0.0, 0.5, 1.0],  # only used when penalty="elasticnet"
        "C": [0.01, 0.1, 1, 10, 100], # regularization strength
        "max_iter": [100, 500, 1000, 2000, 5000],
    },
    {
        "penalty": ["l1", "l2"],
        "solver": ["liblinear", "saga"],
        "C": [0.01, 0.1, 1, 10, 100],
        "max_iter": [100, 500, 1000, 2000, 5000],
    },
]

    with mlflow.start_run() as parent:
        model = LogisticRegression(random_state=42)
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid,scoring="accuracy", cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X_train,Y_train)
         
        for params, mean_score, std_score in zip(  grid_search.cv_results_["params"],
                                                  grid_search.cv_results_["mean_test_score"],
                                                  grid_search.cv_results_["std_test_score"]
                                                ):
            with mlflow.start_run(run_name=f"LR with params: {params}", nested=True) as child:
                model = LogisticRegression(**params)
                model.fit(X_train,Y_train)

                Y_pred = model.predict(X_test)

                metrics = {
                    "accuracy": accuracy_score(Y_test, Y_pred),
                    "precision": precision_score(Y_test, Y_pred),
                    "recall": recall_score(Y_test, Y_pred),
                    "f1_score": f1_score(Y_test, Y_pred),
                    "mean_cv_score": mean_score,
                    "std_cv_score": std_score
                }

                # Log parameters & metrics
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)

                print(f"Params: {params} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}")
        
                # Log the best model
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        best_f1 = grid_search.best_score_

        mlflow.log_params(best_params)
        mlflow.log_metric("best_f1_score", best_f1)
        mlflow.sklearn.log_model(best_model, "model")
        
        print(f"\nBest Params: {best_params} | Best F1 Score: {best_f1:.4f}")

if __name__ == "__main__":
    df = load_data(r"notebooks\sample-data.csv")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['review'])
    Y = df["sentiment"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=CONFIG["test_size"], random_state=42)
    
    train_and_log_model(X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)