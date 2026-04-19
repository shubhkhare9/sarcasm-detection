import logging
import sys
import os

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

    logger.info("=== Pipeline complete ===")


if __name__ == "__main__":
    run_pipeline()
