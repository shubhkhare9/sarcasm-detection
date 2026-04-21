# Sarcasm Detection in News Headlines — NLP + MLOps Project

## Problem Statement
Sarcasm detection is a critical NLP challenge because sarcastic text systematically misleads downstream sentiment analysis models. A headline like *"Nation's Economists Agree Economy Doing Great"* carries the opposite sentiment to its literal meaning, causing failures in automated content understanding.

## Business Use Case
- Improve sentiment analysis accuracy in social media monitoring
- Customer review analytics and feedback classification
- Social media moderation and toxicity filtering
- News aggregation and content tagging pipelines

---

## Dataset
- **Name:** News Headlines Dataset for Sarcasm Detection (Rishabh Misra)
- **Source:** https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection
- **Task:** Sarcastic (1) vs Not Sarcastic (0)
- **Size:** ~28,000 headlines

### Dataset Setup
1. Download `Sarcasm_Headlines_Dataset_v2.json` from Kaggle
2. Place it in the `data/` folder

### GloVe Setup (for Notebook 4)
1. Download from: https://nlp.stanford.edu/data/glove.6B.zip
2. Unzip and place `glove.6B.100d.txt` in `data/`

---

## Models & Performance
| # | Notebook | Model | Test F1 | Test Accuracy |
|---|----------|-------|---------|---------------|
| 1 | 03 | Naive Bayes (TF-IDF) | 0.7878 | 0.7878 |
| 2 | 03 | Logistic Regression (TF-IDF) | 0.7854 | 0.7862 |
| 3 | 03 | SVM (TF-IDF) | 0.7954 | 0.7957 |
| 4 | 04 | BiLSTM + GloVe | 0.8689 | 0.8691 |
| 5 | 05 | BERT (fine-tuned) | 0.9294 | 0.9294 |

---

## Project Structure
```
├── src/
│   ├── data_ingestion.py      ← load raw/clean data
│   ├── preprocessing.py       ← clean text, TF-IDF, splits
│   ├── train.py               ← train NB/LR/SVM with MLflow tracking
│   ├── evaluate.py            ← evaluate and log test metrics
│   └── predict.py             ← inference + drift detection
├── pipeline/
│   └── training_pipeline.py   ← end-to-end orchestrator
├── tests/
│   └── test_preprocessing.py  ← pytest unit tests
├── notebook/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_traditional_ml.ipynb
│   ├── 04_bilstm.ipynb
│   ├── 05_bert.ipynb
│   └── 06_comparative_analysis.ipynb
├── data/
│   ├── Sarcasm_Headlines_Dataset_v2.json  ← original dataset (Kaggle)
│   ├── sarcasm_clean.csv                  ← cleaned dataset
│   ├── tfidf.joblib                       ← fitted TF-IDF vectorizer
│   ├── best_bert.pt                       ← fine-tuned BERT weights
│   ├── best_bilstm.keras                  ← trained BiLSTM model
│   └── results.pkl                        ← all model evaluation results
├── outputs/                               ← saved plots and charts
├── .github/workflows/ci.yml               ← GitHub Actions CI
├── app.py                                 ← FastAPI prediction server
├── demo.html                              ← frontend UI
├── Dockerfile                             ← container definition
├── dvc.yaml                               ← DVC pipeline stages
├── requirements.txt
└── .gitignore
```

---

## Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Data Versioning with DVC
```bash
dvc init
dvc add data/Sarcasm_Headlines_Dataset_v2.json
git add data/.gitignore data/Sarcasm_Headlines_Dataset_v2.json.dvc
git commit -m "Track dataset with DVC"
```

Run the full DVC pipeline:
```bash
dvc repro
```

---

## Experiment Tracking with MLflow
Training automatically logs parameters and metrics to MLflow.

Run the pipeline:
```bash
python pipeline/training_pipeline.py
```

View the MLflow UI:
```bash
mlflow ui
```
Then open `http://127.0.0.1:5000`

---

## Run Notebooks (in order)
```
01_data_exploration.ipynb     → EDA
02_preprocessing.ipynb        → Cleaning, TF-IDF, sequences
03_traditional_ml.ipynb       → NB, LR, SVM
04_bilstm.ipynb               → BiLSTM + GloVe
05_bert.ipynb                 → Fine-tuned BERT
06_comparative_analysis.ipynb → Final comparison
```

---

## API Deployment (FastAPI)

### Run locally
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
Then open `http://127.0.0.1:8000`

### Run with Docker
```bash
docker build -t sarcasm-detection .
docker run -p 8000:8000 sarcasm-detection
```

### Production readiness notes
- `GET /health` checks API liveness
- `GET /ready` checks whether the BERT model loaded successfully
- The API expects `data/best_bert.pt` to exist inside the deployment artifact
- On first startup, Hugging Face may download `bert-base-uncased` tokenizer/config files unless they are already cached in the image or host environment
- For repeatable cloud deploys, store large training artifacts in DVC / object storage and fetch them during build or release if you do not commit them

### API Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web UI |
| GET | `/health` | Health check |
| POST | `/predict` | Predict sarcasm |

Example request:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"headline": "Area Man Solves All Problems"}'
```

---

## CI/CD
GitHub Actions runs on every push:
- Installs dependencies
- Runs `pytest tests/`

See `.github/workflows/ci.yml`

Important:
- `protobuf<5` is pinned because newer protobuf releases break `mlflow==2.13.2`
- `tensorflow==2.16.1` is pinned because newer TensorFlow releases pull protobuf versions that conflict with MLflow 2.13.2
- Do not commit `.dvc/config.local`; keep DVC credentials in repository secrets or deployment environment variables

---

## Monitoring & Logging
- All modules use Python `logging` (INFO level)
- Prediction inputs and outputs are logged
- Basic data drift detection: flags if incoming headline word count deviates significantly from training distribution

---

## Improving Generalization (Babylon Bee & Other Sources)

### Step 1 — Add Babylon Bee samples
```bash
python add_babylonbee_samples.py
```

### Step 2 — Scrape other satire sources
```bash
python scrape_headlines.py
```

### Step 3 — Fine-tune BERT
```bash
python finetune_bert.py
```

### Step 4 — Restart the API
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## Reproducibility Checklist
- [x] `requirements.txt` — all dependencies pinned
- [x] `dvc.yaml` — reproducible data + training pipeline
- [x] `data/*.dvc` — dataset tracked with DVC
- [x] MLflow — all experiments logged with parameters and metrics
- [x] Random seeds fixed (`random_state=42`) throughout
- [x] `Dockerfile` — fully containerized deployment
