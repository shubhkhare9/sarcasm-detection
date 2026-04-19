import logging
import joblib
import os
import mlflow
import mlflow.sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

MODELS = {
    "naive_bayes": MultinomialNB(),
    "logistic_regression": LogisticRegression(max_iter=1000, C=1.0),
    "svm": LinearSVC(max_iter=2000, C=1.0),
}


def train_all(X_train, y_train, X_val, y_val, tfidf) -> dict:
    mlflow.set_experiment("sarcasm-detection")
    X_tr = tfidf.transform(X_train)
    X_v = tfidf.transform(X_val)
    trained = {}

    for name, clf in MODELS.items():
        with mlflow.start_run(run_name=name):
            logger.info(f"Training {name}")
            clf.fit(X_tr, y_train)

            val_acc = accuracy_score(y_val, clf.predict(X_v))
            val_f1 = f1_score(y_val, clf.predict(X_v), average="weighted")

            mlflow.log_param("model", name)
            mlflow.log_metric("val_accuracy", val_acc)
            mlflow.log_metric("val_f1", val_f1)
            mlflow.sklearn.log_model(clf, name)

            logger.info(f"{name} — val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

            path = os.path.join(DATA_DIR, f"{name[:2]}.pkl")
            joblib.dump(clf, path)
            trained[name] = clf

    return trained
