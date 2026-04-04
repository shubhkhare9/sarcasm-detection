# Sarcasm Detection in News Headlines — NLP Project

## Overview
Binary text classification to detect sarcasm in news headlines.
- **Dataset:** News Headlines Dataset for Sarcasm Detection (Rishabh Misra)
- **Source:** https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection
- **Task:** Sarcastic (1) vs Not Sarcastic (0)
- **Size:** ~28,000 headlines

## Dataset Setup
1. Download `Sarcasm_Headlines_Dataset_v2.json` from Kaggle
2. Place it in the `data/` folder

## GloVe Setup (for Notebook 4)
1. Download from: https://nlp.stanford.edu/data/glove.6B.zip
2. Unzip and place `glove.6B.100d.txt` in `data/`

## Models
| # | Notebook | Model | Expected F1 |
|---|----------|-------|-------------|
| 1 | 03 | Naive Bayes (TF-IDF) | 0.78–0.82 |
| 2 | 03 | Logistic Regression (TF-IDF) | 0.81–0.85 |
| 3 | 03 | SVM (TF-IDF) | 0.83–0.87 |
| 4 | 04 | BiLSTM + GloVe | 0.86–0.90 |
| 5 | 05 | BERT (fine-tuned) | 0.91–0.94 |

## Run Order
```
01_data_exploration.ipynb   → EDA
02_preprocessing.ipynb      → Cleaning, TF-IDF, sequences
03_traditional_ml.ipynb     → NB, LR, SVM
04_bilstm.ipynb             → BiLSTM + GloVe
05_bert.ipynb               → Fine-tuned BERT
06_comparative_analysis.ipynb → Final comparison
```

## Install Dependencies
```bash
pip install -r requirements.txt
```
# sarcasm-detection
