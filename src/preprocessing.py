import re
import logging
import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning text")
    df = df.copy()
    df["clean"] = df["headline"].apply(clean_text)
    return df


def get_splits(df: pd.DataFrame, test_size: float = 0.15, val_size: float = 0.15):
    X, y = df["clean"].values, df["is_sarcastic"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size / (1 - test_size), random_state=42, stratify=y_train
    )
    logger.info(f"Split sizes — train:{len(X_train)} val:{len(X_val)} test:{len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_tfidf(X_train, max_features: int = 20000) -> TfidfVectorizer:
    logger.info("Fitting TF-IDF vectorizer")
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    tfidf.fit(X_train)
    path = os.path.join(DATA_DIR, "tfidf.joblib")
    joblib.dump(tfidf, path)
    logger.info(f"TF-IDF saved to {path}")
    return tfidf
