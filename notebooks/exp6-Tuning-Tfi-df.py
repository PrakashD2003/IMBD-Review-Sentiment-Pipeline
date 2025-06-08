import os
import logging
import mlflow
import mlflow.sklearn
import dagshub
import re
import pandas as pd
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
    "EXPERIMENT":     "TF-IDF Hyperparameter Tuning"
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

def train_with_params(X_train, X_test, y_train, y_test, vectorizer, params):
    mlflow.log_params(params)

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(random_state=CONFIG["RANDOM_STATE"], max_iter=1000)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1_score":  f1_score(y_test, y_pred)
    }
    mlflow.log_metrics(metrics)

    if scipy.sparse.issparse(X_test_vec[:5]):
        example = X_test_vec[:5].toarray()
    else:
        example = X_test_vec[:5]

    mlflow.sklearn.log_model(model, "model", input_example=example)
    print(f"Params: {params} | Metrics: {metrics}")

def hyperparameter_search(df: pd.DataFrame):
    mlflow.set_tracking_uri(CONFIG["MLFLOW_URI"])
    dagshub.init(repo_owner=CONFIG["DAGSHUB_OWNER"],
                 repo_name=CONFIG["DAGSHUB_NAME"],
                 mlflow=True)
    mlflow.set_experiment(CONFIG["EXPERIMENT"])

    X = df["review"]
    y = df["sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG["TEST_SIZE"], random_state=CONFIG["RANDOM_STATE"]
    )

    # Grid values
    param_grid = {
        "max_df":       [0.9, 0.95],
        "min_df":       [1, 2],
        "ngram_range":  [(1, 1), (1, 2)],
        "max_features": [10_000, 50_000]
    }

    keys, values = zip(*param_grid.items())
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        with mlflow.start_run(nested=False):
            vectorizer = TfidfVectorizer(
                max_df=params["max_df"],
                min_df=params["min_df"],
                ngram_range=params["ngram_range"],
                max_features=params["max_features"],
                stop_words='english',
                sublinear_tf=True,
                norm='l2'
            )
            train_with_params(X_train, X_test, y_train, y_test, vectorizer, params)

if __name__ == "__main__":
    df = load_data(CONFIG["DATA_PATH"])
    hyperparameter_search(df)
