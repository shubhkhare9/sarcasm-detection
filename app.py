# import logging
# import os
# from contextlib import asynccontextmanager

# import torch
# from fastapi import FastAPI, HTTPException
# from fastapi.responses import HTMLResponse
# from pydantic import BaseModel
# from transformers import BertForSequenceClassification, BertTokenizer

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
# )
# logger = logging.getLogger(__name__)

# ROOT = os.path.dirname(os.path.abspath(__file__))
# DATA_DIR = os.path.join(ROOT, "data")
# MODEL_PATH = os.path.join(DATA_DIR, "best_bert.pt")
# HTML_PATH = os.path.join(ROOT, "demo.html")

# MODEL_NAME = os.getenv("MODEL_NAME", "bert-base-uncased")
# MAX_LEN = 64
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TOKENIZER = None
# MODEL = None
# MODEL_LOAD_ERROR = None


# def load_assets():
#     if not os.path.exists(MODEL_PATH):
#         raise FileNotFoundError("Missing data/best_bert.pt — run notebook 05 first.")
#     os.environ["TRANSFORMERS_VERBOSITY"] = "error"
#     os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
#     tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
#     model = BertForSequenceClassification.from_pretrained(
#         MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True
#     )
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
#     model.to(device)
#     model.eval()
#     logger.info("BERT model loaded successfully")
#     return tokenizer, model


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     global TOKENIZER, MODEL, MODEL_LOAD_ERROR
#     try:
#         TOKENIZER, MODEL = load_assets()
#         MODEL_LOAD_ERROR = None
#     except Exception as exc:
#         TOKENIZER = None
#         MODEL = None
#         MODEL_LOAD_ERROR = str(exc)
#         logger.exception("Model failed to load during startup")
#     yield


# app = FastAPI(title="Sarcasm Detection API", version="1.0", lifespan=lifespan)


# class PredictRequest(BaseModel):
#     headline: str


# class PredictResponse(BaseModel):
#     label: int
#     probability: float
#     explain: str


# @app.get("/", response_class=HTMLResponse)
# def serve_ui():
#     with open(HTML_PATH, "r") as f:
#         return f.read()


# @app.get("/health")
# def health():
#     return {"ok": True, "model_loaded": MODEL is not None}


# @app.get("/ready")
# def ready():
#     if MODEL is None or TOKENIZER is None:
#         raise HTTPException(
#             status_code=503,
#             detail=MODEL_LOAD_ERROR or "Model is still unavailable",
#         )
#     return {"ready": True}


# @app.post("/predict", response_model=PredictResponse)
# def predict(req: PredictRequest):
#     if MODEL is None or TOKENIZER is None:
#         raise HTTPException(
#             status_code=503,
#             detail=MODEL_LOAD_ERROR or "Model is unavailable",
#         )

#     headline = req.headline.strip()
#     if not headline:
#         raise HTTPException(status_code=400, detail="headline is required")

#     logger.info(f"Prediction request: '{headline}'")

#     text = headline.lower()
#     enc = TOKENIZER(text, max_length=MAX_LEN, truncation=True, padding="max_length", return_tensors="pt")
#     input_ids = enc["input_ids"].to(device)
#     attention_mask = enc["attention_mask"].to(device)

#     with torch.no_grad():
#         logits = MODEL(input_ids=input_ids, attention_mask=attention_mask).logits

#     proba = torch.softmax(logits, dim=1)[0][1].item()
#     label = 1 if proba >= 0.5 else 0
#     explain = (
#         "Model sees strong sarcasm cues in phrasing and tone."
#         if label == 1
#         else "Model reads this as a straightforward headline."
#     )

#     logger.info(f"Result: label={label}, probability={proba:.4f}")
#     return PredictResponse(label=label, probability=float(proba), explain=explain)


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)


import logging
import os
from contextlib import asynccontextmanager

import joblib
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

ROOT       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(ROOT, "data")
SPLITS_PATH = os.path.join(DATA_DIR, "splits.joblib")   # contains the TF-IDF vectorizer
SVM_PATH    = os.path.join(DATA_DIR, "svm.pkl")          # LinearSVC model
HTML_PATH   = os.path.join(ROOT, "demo.html")

VECTORIZER = None
MODEL      = None
MODEL_LOAD_ERROR = None

TFIDF_PATH = os.path.join(DATA_DIR, "tfidf.joblib")   # ← already exists ✅
SVM_PATH   = os.path.join(DATA_DIR, "sv.pkl")          # ← use sv.pkl (newer) or svm.pkl

def load_assets():
    if not os.path.exists(TFIDF_PATH):
        raise FileNotFoundError("Missing data/tfidf.joblib")
    if not os.path.exists(SVM_PATH):
        raise FileNotFoundError("Missing data/sv.pkl")

    vectorizer = joblib.load(TFIDF_PATH)

    with open(SVM_PATH, "rb") as f:
        model = pickle.load(f)

    logger.info("SVM + TF-IDF loaded successfully")
    return vectorizer, model


@asynccontextmanager
async def lifespan(app: FastAPI):
    global VECTORIZER, MODEL, MODEL_LOAD_ERROR
    try:
        VECTORIZER, MODEL = load_assets()
        MODEL_LOAD_ERROR = None
    except Exception as exc:
        VECTORIZER = None
        MODEL      = None
        MODEL_LOAD_ERROR = str(exc)
        logger.exception("Model failed to load during startup")
    yield


app = FastAPI(title="Sarcasm Detection API", version="1.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    headline: str

class PredictResponse(BaseModel):
    label: int
    probability: float
    explain: str


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    with open(HTML_PATH, "r") as f:
        return f.read()

@app.get("/health")
def health():
    return {"ok": True, "model_loaded": MODEL is not None}

@app.get("/ready")
def ready():
    if MODEL is None or VECTORIZER is None:
        raise HTTPException(status_code=503, detail=MODEL_LOAD_ERROR or "Model unavailable")
    return {"ready": True}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if MODEL is None or VECTORIZER is None:
        raise HTTPException(status_code=503, detail=MODEL_LOAD_ERROR or "Model unavailable")

    headline = req.headline.strip()
    if not headline:
        raise HTTPException(status_code=400, detail="headline is required")

    logger.info(f"Prediction request: '{headline}'")

    features = VECTORIZER.transform([headline.lower()])
    label    = int(MODEL.predict(features)[0])

    # LinearSVC has no predict_proba — use decision_function score instead
    # and convert to a 0–1 range with sigmoid
    raw_score = float(MODEL.decision_function(features)[0])
    probability = float(1 / (1 + np.exp(-raw_score)))   # sigmoid

    explain = (
        "Model sees strong sarcasm cues in phrasing and tone."
        if label == 1
        else "Model reads this as a straightforward headline."
    )

    logger.info(f"Result: label={label}, probability={probability:.4f}")
    return PredictResponse(label=label, probability=probability, explain=explain)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)