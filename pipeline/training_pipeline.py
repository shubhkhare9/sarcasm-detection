import logging
import sys
import os
import pickle
import mlflow

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_ingestion import load_raw_data
from src.preprocessing import preprocess, get_splits, fit_tfidf
from src.train import train_all
from src.evaluate import evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def log_pretrained_models():
    """Log BiLSTM and BERT results from results.pkl into MLflow."""
    results_path = os.path.join(DATA_DIR, "results.pkl")
    if not os.path.exists(results_path):
        logger.warning("results.pkl not found, skipping BiLSTM/BERT logging")
        return

    with open(results_path, "rb") as f:
        results = pickle.load(f)

    mlflow.set_experiment("sarcasm-detection")
    for model_name, model_type in [("BiLSTM", "Deep Learning"), ("BERT", "Transformer")]:
        if model_name not in results:
            continue
        metrics = results[model_name]
        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model", model_name)
            mlflow.log_param("type", model_type)
            mlflow.log_metric("test_accuracy", metrics["Accuracy"])
            mlflow.log_metric("test_precision", metrics["Precision"])
            mlflow.log_metric("test_recall", metrics["Recall"])
            mlflow.log_metric("test_f1", metrics["F1"])
            logger.info(f"{model_name} — acc={metrics['Accuracy']:.4f} f1={metrics['F1']:.4f}")


def run_pipeline():
    logger.info("=== Starting training pipeline ===")

    df = load_raw_data()
    df = preprocess(df)
    X_train, X_val, X_test, y_train, y_val, y_test = get_splits(df)
    tfidf = fit_tfidf(X_train)
    trained_models = train_all(X_train, y_train, X_val, y_val, tfidf)

    X_test_tfidf = tfidf.transform(X_test)
    for name, model in trained_models.items():
        evaluate(model, X_test_tfidf, y_test, model_name=name)

    log_pretrained_models()

    logger.info("=== Pipeline complete ===")


if __name__ == "__main__":
    run_pipeline()
