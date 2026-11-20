"""
Unit and integration tests for Phase D: Exploratory Data Analysis (EDA).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import main as main_module
from eda import clean_encoding_text, compute_eda


@pytest.fixture
def sample_eda_csv(tmp_path: Path) -> Path:
    csv_file = tmp_path / "eda_test_reviews.csv"
    pd.DataFrame({
        "product_name": [
            "Noise Cancelling Wireless Headphones??????(Black)",
            "Smart Fitness Watch",
            "Nova Dry Iron?ÃƒÂ¿?ÃƒÂ¿(Grey)",
            "Ultra HD 4K Monitor",
            "Corrupted Row With Product In Rate Column",
        ],
        "product_price": ["1999", "2499", "799", "14999", "999"],
        "Rate": ["5", "1", "4", "3", "Pigeon Electric Kettle"],  # 5th row corrupted
        "Review": [
            "Great audio quality",
            "Stopped working",
            None,  # Null title
            "Average monitor",
            "Bad",
        ],
        "Summary": [
            "The sound clarity is wonderful and battery lasts long.",
            "Stopped working on day 3, customer service ignored me.",
            "Heats quickly, decent build for the price.",
            "The screen is okay for basic tasks, nothing exceptional.",
            "Useless item",
        ],
        "Sentiment": ["positive", "negative", "positive", "neutral", "negative"],
    }).to_csv(csv_file, index=False)
    return csv_file


def test_clean_encoding_text():
    assert clean_encoding_text("Headphones??????(Black)") == "Headphones (Black)"
    assert clean_encoding_text("Iron?ÃƒÂ¿?ÃƒÂ¿(Grey)") == "Iron (Grey)"
    assert clean_encoding_text("   Regular clean string   ") == "Regular clean string"
    assert clean_encoding_text(None) == ""


def test_compute_eda_on_sample(sample_eda_csv: Path, tmp_path: Path):
    summary = compute_eda(csv_path=sample_eda_csv, save_plots=False)

    assert summary["total_raw_rows"] == 5
    assert summary["cleaning_report"]["corrupted_rate_rows_dropped"] == 1
    assert summary["total_clean_rows"] == 4

    # Verify class distribution
    assert "clean_counts" in summary["class_distribution"]
    assert summary["class_distribution"]["clean_counts"]["positive"] == 2
    assert summary["class_distribution"]["clean_counts"]["negative"] == 1
    assert summary["class_distribution"]["clean_counts"]["neutral"] == 1

    # Verify rating distribution
    assert summary["rating_distribution"]["counts"]["5"] == 1
    assert summary["rating_distribution"]["counts"]["1"] == 1

    # Verify review length distribution
    assert "all" in summary["review_length_distribution"]
    assert "words" in summary["review_length_distribution"]["all"]
    assert summary["review_length_distribution"]["all"]["words"]["mean"] > 0

    # Verify key findings are generated
    assert len(summary["key_findings"]) >= 4


def test_eda_summary_api_endpoint(sample_eda_csv: Path):
    client = TestClient(main_module.app)
    resp = client.get("/eda/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_raw_rows" in data
    assert "class_distribution" in data
    assert "rating_vs_sentiment" in data
    assert "key_findings" in data
    assert len(data["key_findings"]) > 0


def test_eda_report_image_endpoint():
    client = TestClient(main_module.app)
    resp = client.get("/eda/reports/class_distribution.png")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"

    resp_404 = client.get("/eda/reports/non_existent.png")
    assert resp_404.status_code == 404
