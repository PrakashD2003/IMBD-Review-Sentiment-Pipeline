import os
import logging
import mlflow
import mlflow.sklearn
import dagshub
import pandas as pd
import re
import scipy.sparse
import itertools

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Constants
from var import MLFLOW_TRACKING_URI, DAGSHUB_REPO_OWNER, DAGSHUB_REPO_NAME

CONFIG = {
    "DATA_PATH":      "notebooks/sample-data.csv",
    "TEST_SIZE":      0.2,
    "RANDOM_STATE":   42,
    "MLFLOW_URI":     MLFLOW_TRACKING_URI,
    "DAGSHUB_OWNER":  DAGSHUB_REPO_OWNER,
    "DAGSHUB_NAME":   DAGSHUB_REPO_NAME,
    "EXPERIMENT":     "LogisticRegression Hyperparameter Tuning Extended"
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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

def train_logistic_regression(X_train, X_test, y_train, y_test, params):
    mlflow.log_params(params)

    model = LogisticRegression(
        C=params["C"],
        penalty=params["penalty"],
        solver=params["solver"],
        max_iter=params["max_iter"],
        class_weight=params["class_weight"],
        tol=params["tol"],
        random_state=CONFIG["RANDOM_STATE"]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1_score":  f1_score(y_test, y_pred)
    }

    mlflow.log_metrics(metrics)

    if scipy.sparse.issparse(X_test[:5]):
        example = X_test[:5].toarray()
    else:
        example = X_test[:5]

    mlflow.sklearn.log_model(model, "model", input_example=example)
    print(f"Params: {params} | Metrics: {metrics}")

def is_valid_combination(params):
    penalty = params["penalty"]
    solver = params["solver"]

    if penalty == "l1" and solver not in ["liblinear", "saga"]:
        return False
    if penalty == "l2" and solver not in ["liblinear", "saga", "lbfgs", "newton-cg", "sag"]:
        return False
    if penalty == "elasticnet" and solver != "saga":
        return False
    if penalty == "none" and solver not in ["lbfgs", "newton-cg", "sag", "saga"]:
        return False
    return True

def hyperparameter_search(df: pd.DataFrame):
    mlflow.set_tracking_uri(CONFIG["MLFLOW_URI"])
    dagshub.init(repo_owner=CONFIG["DAGSHUB_OWNER"],
                 repo_name=CONFIG["DAGSHUB_NAME"],
                 mlflow=True)
    mlflow.set_experiment(CONFIG["EXPERIMENT"])

    X = df["review"]
    y = df["sentiment"]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG["TEST_SIZE"], random_state=CONFIG["RANDOM_STATE"]
    )

    vectorizer = TfidfVectorizer(
        max_df=0.95,
        min_df=2,
        ngram_range=(1, 2),
        max_features=50000,
        stop_words='english',
        sublinear_tf=True,
        norm='l2'
    )

    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    # Extended Grid
    param_grid = {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l1", "l2", "elasticnet", "none"],
        "solver": ["liblinear", "saga", "lbfgs"],
        "max_iter": [100, 300],
        "class_weight": [None, "balanced"],
        "tol": [1e-4, 1e-3]
    }

    keys, values = zip(*param_grid.items())
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        if not is_valid_combination(params):
            continue

        with mlflow.start_run(nested=False):
            try:
                train_logistic_regression(X_train, X_test, y_train, y_test, params)
            except Exception as e:
                print(f"Skipping combo {params} due to error: {e}")
                continue

if __name__ == "__main__":
    df = load_data(CONFIG["DATA_PATH"])
    hyperparameter_search(df)
