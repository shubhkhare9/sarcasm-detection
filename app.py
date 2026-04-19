import logging
import os
import sys

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import BertForSequenceClassification, BertTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_PATH = os.path.join(DATA_DIR, "best_bert.pt")
HTML_PATH = os.path.join(ROOT, "demo.html")

MODEL_NAME = "bert-base-uncased"
MAX_LEN = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="Sarcasm Detection API", version="1.0")


def load_assets():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Missing data/best_bert.pt — run notebook 05 first.")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    logger.info("BERT model loaded successfully")
    return tokenizer, model


TOKENIZER, MODEL = load_assets()


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
    return {"ok": True}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    headline = req.headline.strip()
    if not headline:
        raise HTTPException(status_code=400, detail="headline is required")

    logger.info(f"Prediction request: '{headline}'")

    text = headline.lower()
    enc = TOKENIZER(text, max_length=MAX_LEN, truncation=True, padding="max_length", return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        logits = MODEL(input_ids=input_ids, attention_mask=attention_mask).logits

    proba = torch.softmax(logits, dim=1)[0][1].item()
    label = 1 if proba >= 0.5 else 0
    explain = (
        "Model sees strong sarcasm cues in phrasing and tone."
        if label == 1
        else "Model reads this as a straightforward headline."
    )

    logger.info(f"Result: label={label}, probability={proba:.4f}")
    return PredictResponse(label=label, probability=float(proba), explain=explain)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
