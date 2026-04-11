#!/usr/bin/env python3
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

import torch
from transformers import BertTokenizer, BertForSequenceClassification

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")

MODEL_PATH = os.path.join(DATA_DIR, "best_bert.pt")
HTML_PATH = os.path.join(ROOT, "demo.html")

MODEL_NAME = "bert-base-uncased"
MAX_LEN = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_assets():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Missing data/best_bert.pt. Run the BERT notebook first.")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return tokenizer, model


TOKENIZER, MODEL = load_assets()


def predict_headline(text: str):
    text = text.lower()  # training data was lowercased
    enc = TOKENIZER(
        text,
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
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
    return {"label": label, "probability": float(proba), "explain": explain}


class Handler(BaseHTTPRequestHandler):
    def _set_headers(self, status=200, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200, "text/plain")

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/" or path == "/demo.html":
            if not os.path.exists(HTML_PATH):
                self._set_headers(404, "text/plain")
                self.wfile.write(b"demo.html not found")
                return
            self._set_headers(200, "text/html")
            with open(HTML_PATH, "rb") as f:
                self.wfile.write(f.read())
            return
        if path == "/health":
            self._set_headers(200, "application/json")
            self.wfile.write(json.dumps({"ok": True}).encode("utf-8"))
            return

        self._set_headers(404, "text/plain")
        self.wfile.write(b"Not found")

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/predict":
            self._set_headers(404, "text/plain")
            self.wfile.write(b"Not found")
            return

        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
            headline = (payload.get("headline") or "").strip()
            if not headline:
                raise ValueError("headline is required")
        except Exception as exc:
            self._set_headers(400, "application/json")
            self.wfile.write(json.dumps({"error": str(exc)}).encode("utf-8"))
            return

        result = predict_headline(headline)
        self._set_headers(200, "application/json")
        self.wfile.write(json.dumps(result).encode("utf-8"))


if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8000
    print(f"Serving demo on http://{host}:{port}")
    print("Press Ctrl+C to stop")
    HTTPServer((host, port), Handler).serve_forever()
