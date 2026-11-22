#!/usr/bin/env python3
"""
Phase K.6: Demo CSV Evaluation Script for E-Commerce Review Intelligence System.

Reads a CSV file containing reviews and ground-truth sentiments, queries the live
prediction API (/predict/bulk or /predict), evaluates empirical metrics (Accuracy,
Macro F1, Per-Class Precision/Recall/F1), and outputs a detailed misclassification table.

Usage:
    python3 scripts/eval_demo_csv.py --csv data/demo_eval_sample.csv --api-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate sentiment prediction API on an annotated demo CSV dataset."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data/demo_eval_sample.csv",
        help="Path to evaluation CSV containing review text and expected sentiment columns.",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="Base URL of the ML microservice (default: http://127.0.0.1:8000).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP request timeout in seconds.",
    )
    return parser.parse_args()


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {col.lower().strip(): col for col in df.columns}
    for cand in candidates:
        if cand.lower().strip() in lower_map:
            return lower_map[cand.lower().strip()]
    return None


def run_evaluation(csv_path: Path, api_url: str, timeout: float = 30.0):
    if not csv_path.exists():
        print(f"[ERROR] CSV file not found at: {csv_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 76)
    print(" E-COMMERCE REVIEW INTELLIGENCE: DEMO CSV EVALUATION SUITE")
    print(f" Target Dataset: {csv_path} (exists, size: {csv_path.stat().st_size} bytes)")
    print(f" ML Service URL: {api_url}")
    print("=" * 76)

    # 1. Load CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"[ERROR] Failed to parse CSV: {exc}", file=sys.stderr)
        sys.exit(1)

    if df.empty:
        print("[ERROR] Evaluation CSV is empty.", file=sys.stderr)
        sys.exit(1)

    # Detect text column
    text_col = find_column(df, ["review", "text", "summary", "review_text", "comment", "full_text"])
    if not text_col:
        print(
            f"[ERROR] Could not find review text column in CSV. Found columns: {list(df.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Detect expected sentiment column
    label_col = find_column(df, ["expected_sentiment", "sentiment", "label", "true_sentiment", "target"])
    if not label_col:
        print(
            f"[ERROR] Could not find expected sentiment column in CSV. Found columns: {list(df.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f" [INFO] Loaded {len(df)} rows. Text column: '{text_col}', Label column: '{label_col}'.")

    # Clean expected labels
    df["expected_clean"] = df[label_col].astype(str).str.strip().str.lower()
    valid_classes = {"positive", "negative", "neutral"}
    invalid_mask = ~df["expected_clean"].isin(valid_classes)
    if invalid_mask.any():
        print(
            f" [WARN] Dropping {invalid_mask.sum()} rows with unrecognized sentiment labels: "
            f"{df.loc[invalid_mask, 'expected_clean'].unique()}"
        )
        df = df[~invalid_mask].copy()

    # 2. Query Prediction Endpoint
    # We first try /predict/bulk, and if unavailable we try fallback to single /predict or local model
    api_url = api_url.rstrip("/")
    bulk_endpoint = f"{api_url}/predict/bulk"
    predictions = []
    confidences = []

    t0 = time.time()
    try:
        print(f" [INFO] Sending {len(df)} reviews to {bulk_endpoint}...")
        payload = {"reviews": [{"review": str(t)} for t in df[text_col]]}
        resp = requests.post(bulk_endpoint, json=payload, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            for item in data.get("predictions", []):
                predictions.append(item.get("predicted_sentiment"))
                confidences.append(float(item.get("confidence", 1.0)))
        else:
            raise RuntimeError(f"Bulk endpoint returned status {resp.status_code}: {resp.text}")
    except Exception as exc:
        print(f" [WARN] Bulk API query failed ({exc}). Retrying via individual /predict calls...")
        predictions = []
        confidences = []
        for text in df[text_col]:
            try:
                single_resp = requests.post(
                    f"{api_url}/predict",
                    json={"text": str(text)},
                    timeout=5.0,
                )
                if single_resp.status_code == 200:
                    s_data = single_resp.json()
                    predictions.append(s_data.get("sentiment"))
                    confidences.append(float(s_data.get("confidence", 1.0)))
                else:
                    predictions.append("error")
                    confidences.append(0.0)
            except Exception as single_exc:
                print(f"[ERROR] Inference call failed for text: {single_exc}", file=sys.stderr)
                predictions.append("error")
                confidences.append(0.0)

    elapsed = time.time() - t0
    df["predicted_sentiment"] = predictions
    df["confidence"] = confidences

    # 3. Calculate Performance Metrics
    classes = ["negative", "neutral", "positive"]
    y_true = df["expected_clean"].values
    y_pred = df["predicted_sentiment"].values

    total = len(df)
    correct = int((y_true == y_pred).sum())
    accuracy = correct / total if total > 0 else 0.0

    print("\n" + "-" * 76)
    print(" EVALUATION SUMMARY")
    print("-" * 76)
    print(f" Total Evaluated Reviews: {total}")
    print(f" Correct Predictions:    {correct} / {total}")
    print(f" Overall Accuracy:       {accuracy * 100:.2f}%")
    print(f" Total Elapsed Time:     {elapsed:.2f}s ({elapsed / total * 1000:.1f}ms per review)")

    # Per-class breakdown
    print("\n" + "=" * 76)
    print(f"{'Class':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 76)

    macro_precisions = []
    macro_recalls = []
    macro_f1s = []

    for c in classes:
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        support = int((y_true == c).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        macro_precisions.append(prec)
        macro_recalls.append(rec)
        macro_f1s.append(f1)

        print(f"{c.capitalize():<12} | {prec * 100:>8.2f}% | {rec * 100:>8.2f}% | {f1:>10.4f} | {support:>8}")

    print("-" * 76)
    macro_p = float(np.mean(macro_precisions))
    macro_r = float(np.mean(macro_recalls))
    macro_f = float(np.mean(macro_f1s))
    print(f"{'Macro Average':<12} | {macro_p * 100:>8.2f}% | {macro_r * 100:>8.2f}% | {macro_f:>10.4f} | {total:>8}")
    print("=" * 76)

    # 4. Confusion Matrix
    print("\nCONFUSION MATRIX (Rows: True, Columns: Predicted):")
    header = f"{'':<12} | " + " | ".join(f"{c.capitalize():^10}" for c in classes)
    print(header)
    print("-" * len(header))
    for true_c in classes:
        row_counts = [int(((y_true == true_c) & (y_pred == pred_c)).sum()) for pred_c in classes]
        print(f"{true_c.capitalize():<12} | " + " | ".join(f"{count:^10}" for count in row_counts))

    # 5. Misclassified Reviews Table
    misclassified = df[y_true != y_pred]
    print("\n" + "=" * 76)
    print(f" MISCLASSIFIED REVIEWS AUDIT ({len(misclassified)} errors found)")
    print("=" * 76)

    if misclassified.empty:
        print(" [PERFECT MATCH] Zero classification errors found on this evaluation set!")
    else:
        for idx, row in misclassified.iterrows():
            text_preview = str(row[text_col])
            if len(text_preview) > 65:
                text_preview = text_preview[:62] + "..."
            print(f" Row #{idx + 1:<3} | True: {row['expected_clean'].upper():<8} | Pred: {row['predicted_sentiment'].upper():<8} (Conf: {row['confidence']*100:.1f}%)")
            print(f"         Text: \"{text_preview}\"")
            print("-" * 76)


if __name__ == "__main__":
    args = parse_args()
    csv_file = Path(args.csv)
    if not csv_file.is_absolute():
        csv_file = Path(__file__).resolve().parent.parent / csv_file
    run_evaluation(csv_path=csv_file, api_url=args.api_url, timeout=args.timeout)
