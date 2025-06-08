import os
import logging
import mlflow
import mlflow.sklearn
import dagshub
import string
import re
import pandas as pd
import scipy.sparse

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from dask_ml.feature_extraction.text import HashingVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Constants / Config
from var import MLFLOW_TRACKING_URI, DAGSHUB_REPO_OWNER, DAGSHUB_REPO_NAME

CONFIG = {
    "DATA_PATH":      "notebooks/sample-data.csv",
    "TEST_SIZE":      0.2,
    "RANDOM_STATE":   42,
    "MLFLOW_URI":     MLFLOW_TRACKING_URI,
    "DAGSHUB_OWNER":  DAGSHUB_REPO_OWNER,
    "DAGSHUB_NAME":   DAGSHUB_REPO_NAME,
    "EXPERIMENT":     "TF-IDF vs Hashing"
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Vectorizer and algorithm definitions
VECTORIZERS = {
    "TF-IDF": TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        stop_words='english',
        sublinear_tf=True,
        norm='l2'
    ),
    "Hashing": HashingVectorizer(
        n_features=50_000,
        alternate_sign=False,
        norm=None,
        binary=False
    )
}

ALGORITHMS = {
    "LogisticRegression":    LogisticRegression(random_state=CONFIG["RANDOM_STATE"], max_iter=1000),
    "MultinomialNB":          MultinomialNB(),
    "XGBoost":                XGBClassifier(random_state=CONFIG["RANDOM_STATE"],
                                           use_label_encoder=False,
                                           eval_metric="logloss"),
    "RandomForest":           RandomForestClassifier(random_state=CONFIG["RANDOM_STATE"]),
    "GradientBoosting":       GradientBoostingClassifier(random_state=CONFIG["RANDOM_STATE"])
}

# Text preprocessing helpers
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
url_pattern = re.compile(r'https?://\S+|www\.\S+')

def normalize_text(text: str) -> str:
    text = url_pattern.sub("", text)
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.lower().split()
    tokens = [tok for tok in tokens if not tok.isdigit()]
    tokens = [tok for tok in tokens if tok not in stop_words]
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens]
    return " ".join(tokens)

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    df = pd.read_csv(path)
    df["review"] = df["review"].map(normalize_text)
    df = df[df["sentiment"].isin(["positive", "negative"])]
    df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})
    return df

def log_model_params(algo_name: str, model):
    params = {}
    if algo_name == "LogisticRegression":
        params["C"] = model.C
    elif algo_name == "MultinomialNB":
        params["alpha"] = model.alpha
    elif algo_name == "XGBoost":
        params["n_estimators"] = model.n_estimators
        params["learning_rate"] = model.learning_rate
    elif algo_name == "RandomForest":
        params["n_estimators"] = model.n_estimators
        params["max_depth"] = model.max_depth
    elif algo_name == "GradientBoosting":
        params["n_estimators"] = model.n_estimators
        params["learning_rate"] = model.learning_rate
        params["max_depth"] = model.max_depth
    mlflow.log_params(params)

def train_and_evaluate(df: pd.DataFrame):
    # MLflow & DagsHub setup
    mlflow.set_tracking_uri(CONFIG["MLFLOW_URI"])
    dagshub.init(repo_owner=CONFIG["DAGSHUB_OWNER"],
                 repo_name=CONFIG["DAGSHUB_NAME"],
                 mlflow=True)
    mlflow.set_experiment(CONFIG["EXPERIMENT"])
    
    with mlflow.start_run(run_name="All Experiments") as parent:
        # Log this script as artifact
        try:
            mlflow.log_artifact(local_path=__file__, artifact_path="experiments")
        except:
            pass  # __file__ not available in notebook mode
        
        for algo_name, algorithm in ALGORITHMS.items():
            for vec_name, vectorizer in VECTORIZERS.items():
                with mlflow.start_run(run_name=f"{algo_name} + {vec_name}", nested=True):
                    logging.info(f"Running {algo_name} with {vec_name}")

                    # Vectorize
                    X = vectorizer.fit_transform(df["review"])
                    y = df["sentiment"]

                    # Split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y,
                        test_size=CONFIG["TEST_SIZE"],
                        random_state=CONFIG["RANDOM_STATE"]
                    )

                    # Log preprocessing params
                    mlflow.log_params({
                        "vectorizer": vec_name,
                        "algorithm": algo_name,
                        "test_size": CONFIG["TEST_SIZE"],
                        **({"n_features": vectorizer.n_features} if vec_name=="Hashing" else {})
                    })

                    # Train
                    model = algorithm
                    model.fit(X_train, y_train)
                    log_model_params(algo_name, model)

                    # Evaluate
                    y_pred = model.predict(X_test)
                    metrics = {
                        "accuracy":  accuracy_score(y_test, y_pred),
                        "precision": precision_score(y_test, y_pred),
                        "recall":    recall_score(y_test, y_pred),
                        "f1_score":  f1_score(y_test, y_pred)
                    }
                    mlflow.log_metrics(metrics)

                    # Log the model artifact
                    example = X_test[:5]
                    if scipy.sparse.issparse(example):
                        example = example.toarray()
                    mlflow.sklearn.log_model(model, "model", input_example=example)

                    # Console output
                    print(f"{algo_name} + {vec_name}: {metrics}")

if __name__ == "__main__":
    df = load_data(CONFIG["DATA_PATH"])
    train_and_evaluate(df)
