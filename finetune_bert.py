#!/usr/bin/env python3
"""
Fine-tune best_bert.pt on extra_headlines.csv (scraped data) + babylonbee_samples.csv.
Saves updated weights back to data/best_bert.pt.

Usage:
    python finetune_bert.py
"""

import os
import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

ROOT       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(ROOT, "data")
EXTRA_CSV  = os.path.join(DATA_DIR, "extra_headlines.csv")
BABYLONBEE_CSV = os.path.join(DATA_DIR, "babylonbee_samples.csv")
MODEL_PATH = os.path.join(DATA_DIR, "best_bert.pt")

MODEL_NAME = "bert-base-uncased"
MAX_LEN    = 64
BATCH_SIZE = 16
EPOCHS     = 3
LR         = 5e-6  # Lower LR to prevent overwriting original BERT knowledge (catastrophic forgetting)
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Dataset ───────────────────────────────────────────────────────────────────

class HeadlineDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]).lower(),
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids"     : enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label"         : torch.tensor(self.labels[idx], dtype=torch.long),
        }

# ── Train / eval loop ─────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer=None, scheduler=None, train=True):
    model.train() if train else model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc="Train" if train else "Val", leave=False):
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labs = batch["label"].to(device)
            out  = model(input_ids=ids, attention_mask=mask, labels=labs)
            loss, logits = out.loss, out.logits
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labs.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return total_loss / len(loader), f1

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(EXTRA_CSV):
        print(f"❌ {EXTRA_CSV} not found. Run scrape_headlines.py first.")
        return

    # Load scraped data
    df = pd.read_csv(EXTRA_CSV).dropna(subset=["headline"])
    
    # Merge with Babylon Bee samples if available
    if os.path.exists(BABYLONBEE_CSV):
        bb_df = pd.read_csv(BABYLONBEE_CSV).dropna(subset=["headline"])
        df = pd.concat([df, bb_df], ignore_index=True)
        print(f"✓ Merged {len(bb_df)} Babylon Bee samples")
    
    df["headline"] = df["headline"].astype(str).str.lower().str.strip()
    df = df[df["headline"].str.len() > 5].reset_index(drop=True)
    print(f"Loaded {len(df)} headlines  (sarcastic: {df['is_sarcastic'].sum()}, real: {(df['is_sarcastic']==0).sum()})")

    X_train, X_val, y_train, y_val = train_test_split(
        df["headline"].tolist(),
        df["is_sarcastic"].tolist(),
        test_size=0.15,
        random_state=42,
        stratify=df["is_sarcastic"],
    )

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    train_loader = DataLoader(
        HeadlineDataset(X_train, y_train, tokenizer),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        HeadlineDataset(X_val, y_val, tokenizer),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    # Load existing fine-tuned model
    print(f"\nLoading model from {MODEL_PATH} ...")
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)

    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(0.1 * total_steps)
    optimizer    = AdamW(model.parameters(), lr=LR, eps=1e-8, weight_decay=0.01)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_f1 = 0.0
    print(f"\nFine-tuning for {EPOCHS} epochs on {len(X_train)} samples...\n")

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}  {'-'*40}")
        tr_loss, tr_f1 = run_epoch(model, train_loader, optimizer, scheduler, train=True)
        vl_loss, vl_f1 = run_epoch(model, val_loader, train=False)
        print(f"  Train → Loss: {tr_loss:.4f}  F1: {tr_f1:.4f}")
        print(f"  Val   → Loss: {vl_loss:.4f}  F1: {vl_f1:.4f}")
        if vl_f1 > best_f1:
            best_f1 = vl_f1
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✓ Best saved (val_f1={best_f1:.4f})")

    print(f"\n✅ Done! Updated weights saved to {MODEL_PATH}")
    print(f"   Best val F1: {best_f1:.4f}")
    print("\nRestart demo.py to use the updated model.")


if __name__ == "__main__":
    main()
