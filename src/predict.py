import logging
import joblib
import os
import numpy as np

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Training-time mean word count (approximate baseline for drift detection)
TRAIN_MEAN_WORD_COUNT = 7.2


def load_model(model_name: str = "svm"):
    path = os.path.join(DATA_DIR, f"{model_name[:2]}.pkl")
    logger.info(f"Loading model from {path}")
    return joblib.load(path)


def load_tfidf():
    path = os.path.join(DATA_DIR, "tfidf.joblib")
    return joblib.load(path)


def check_drift(texts: list):
    mean_wc = np.mean([len(t.split()) for t in texts])
    if abs(mean_wc - TRAIN_MEAN_WORD_COUNT) > 3:
        logger.warning(f"Data drift detected: mean word count={mean_wc:.2f} vs training={TRAIN_MEAN_WORD_COUNT}")


def predict(texts: list, model_name: str = "svm") -> list:
    if isinstance(texts, str):
        texts = [texts]
    check_drift(texts)
    tfidf = load_tfidf()
    model = load_model(model_name)
    X = tfidf.transform(texts)
    preds = model.predict(X).tolist()
    logger.info(f"Predicted {len(preds)} samples using {model_name}")
    return preds
