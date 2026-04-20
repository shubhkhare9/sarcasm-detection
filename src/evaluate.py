import logging
import json
import os
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
METRICS_PATH = os.path.join(DATA_DIR, "metrics.json")


def evaluate(model, X_test_tfidf, y_test, model_name: str = "model") -> dict:
    preds = model.predict(X_test_tfidf)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, average="weighted"),
        "recall": recall_score(y_test, preds, average="weighted"),
        "f1": f1_score(y_test, preds, average="weighted"),
    }
    logger.info(f"{model_name} test results: {metrics}")
    logger.info(f"\n{classification_report(y_test, preds)}")

    with mlflow.start_run(run_name=f"{model_name}_eval"):
        for k, v in metrics.items():
            mlflow.log_metric(f"test_{k}", v)

    # Write/update metrics.json for DVC tracking
    all_metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            all_metrics = json.load(f)
    all_metrics[model_name] = metrics
    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)

    return metrics
