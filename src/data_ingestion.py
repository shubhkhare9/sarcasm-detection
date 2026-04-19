import json
import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_raw_data(path: str = None) -> pd.DataFrame:
    if path is None:
        path = os.path.join(DATA_DIR, "Sarcasm_Headlines_Dataset_v2.json")
    logger.info(f"Loading raw data from {path}")
    records = []
    with open(path, "r") as f:
        for line in f:
            records.append(json.loads(line))
    df = pd.DataFrame(records)[["headline", "is_sarcastic"]]
    logger.info(f"Loaded {len(df)} records")
    return df


def load_clean_data(path: str = None) -> pd.DataFrame:
    if path is None:
        path = os.path.join(DATA_DIR, "sarcasm_clean.csv")
    logger.info(f"Loading clean data from {path}")
    return pd.read_csv(path)
