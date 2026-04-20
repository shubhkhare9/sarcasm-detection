#!/usr/bin/env python3
"""
Data Drift Monitoring with Evidently.
Compares training data (reference) vs incoming headlines (current).

Usage:
    python monitor.py
    
Output:
    outputs/drift_report.html  — full visual report
    outputs/drift_summary.json — drift metrics summary
"""

import os
import json
import pandas as pd
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

ROOT     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR  = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load reference data (training set) ───────────────────────────────────────

def load_reference():
    path = os.path.join(DATA_DIR, "sarcasm_clean.csv")
    df = pd.read_csv(path)
    df["text_length"]  = df["clean"].str.split().str.len()
    df["char_length"]  = df["clean"].str.len()
    df["label"]        = df["is_sarcastic"].astype(int)
    return df[["text_length", "char_length", "label"]]


# ── Simulate current/production data ─────────────────────────────────────────
# In production this would be real incoming headlines logged from demo.py
# Here we simulate drift by using Babylon Bee + scraped headlines

def load_current():
    frames = []

    bb_path = os.path.join(DATA_DIR, "babylonbee_samples.csv")
    if os.path.exists(bb_path):
        bb = pd.read_csv(bb_path)
        bb["clean"] = bb["headline"].str.lower().str.strip()
        frames.append(bb[["clean", "is_sarcastic"]])

    extra_path = os.path.join(DATA_DIR, "extra_headlines.csv")
    if os.path.exists(extra_path):
        ex = pd.read_csv(extra_path).dropna(subset=["headline"])
        ex["clean"] = ex["headline"].str.lower().str.strip()
        frames.append(ex[["clean", "is_sarcastic"]])

    if not frames:
        raise FileNotFoundError("No current data found. Run scrape_headlines.py first.")

    df = pd.concat(frames, ignore_index=True).drop_duplicates()
    df["text_length"] = df["clean"].str.split().str.len()
    df["char_length"]  = df["clean"].str.len()
    df["label"]        = df["is_sarcastic"].astype(int)
    return df[["text_length", "char_length", "label"]]


# ── Run Evidently Report ──────────────────────────────────────────────────────

def run_monitoring():
    print("Loading reference data (training set)...")
    reference = load_reference()
    print(f"  Reference: {len(reference)} samples")

    print("Loading current data (production/new headlines)...")
    current = load_current()
    print(f"  Current:   {len(current)} samples")

    definition = DataDefinition(
        numerical_columns=["text_length", "char_length"],
        categorical_columns=["label"],
    )

    ref_dataset = Dataset.from_pandas(reference, data_definition=definition)
    cur_dataset = Dataset.from_pandas(current,   data_definition=definition)

    print("\nRunning Evidently drift report...")
    report = Report(metrics=[DataSummaryPreset(), DataDriftPreset()])
    result = report.run(reference_data=ref_dataset, current_data=cur_dataset)

    # Save HTML report
    html_path = os.path.join(OUT_DIR, "drift_report.html")
    result.save_html(html_path)
    print(f"✅ HTML report saved → {html_path}")

    # Save JSON summary
    result_dict = result.dict()
    summary = {
        "reference_size": len(reference),
        "current_size":   len(current),
        "reference_avg_word_count": round(reference["text_length"].mean(), 2),
        "current_avg_word_count":   round(current["text_length"].mean(), 2),
        "reference_label_balance":  round(reference["label"].mean(), 4),
        "current_label_balance":    round(current["label"].mean(), 4),
    }

    # Check for drift in metrics
    ref_wc = reference["text_length"].mean()
    cur_wc = current["text_length"].mean()
    drift_pct = abs(cur_wc - ref_wc) / ref_wc * 100
    summary["word_count_drift_pct"] = round(drift_pct, 2)
    summary["drift_detected"] = bool(drift_pct > 20)

    json_path = os.path.join(OUT_DIR, "drift_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ JSON summary saved → {json_path}")

    # Print summary to console
    print("\n" + "="*50)
    print("DRIFT MONITORING SUMMARY")
    print("="*50)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if summary["drift_detected"]:
        print("\n⚠️  WARNING: Significant word count drift detected!")
        print("   Consider retraining the model on new data.")
    else:
        print("\n✅ No significant drift detected. Model should be stable.")


if __name__ == "__main__":
    run_monitoring()
