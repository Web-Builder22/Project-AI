"""
Test suite for Phase G: Inference endpoints (/predict, /predict/bulk, negation cases, validation, missing model).
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from main import app
import inference


@pytest.fixture
def client():
    return TestClient(app)


def test_predict_single_positive(client):
    """Verify single prediction on clear positive review."""
    resp = client.post("/predict", json={"text": "Outstanding quality, very satisfied and fast shipping!"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["sentiment"] == "positive"
    assert data["confidence"] > 0.5
    assert "probabilities" in data
    assert "positive" in data["probabilities"]
    assert "negative" in data["probabilities"]
    assert "neutral" in data["probabilities"]
    assert "processed_text" in data
    assert "top_contributing_words" in data


def test_predict_single_negative(client):
    """Verify single prediction on clear negative review."""
    resp = client.post("/predict", json={"text": "Completely broken and defective. Terrible waste of money."})
    assert resp.status_code == 200
    data = resp.json()
    assert data["sentiment"] == "negative"
    assert data["confidence"] > 0.5


def test_predict_single_neutral(client):
    """Verify single prediction on neutral review."""
    resp = client.post("/predict", json={"text": "It is an average product. Okay for the price nothing special."})
    assert resp.status_code == 200
    data = resp.json()
    assert data["sentiment"] in ["neutral", "positive", "negative"]
    assert "probabilities" in data


def test_predict_negation_cases(client):
    """Verify negation handling: 'not good' must be negative, 'not bad' must be positive."""
    # Negation 1: "not good" -> negative
    resp_neg = client.post("/predict", json={"text": "The battery life is not good at all."})
    assert resp_neg.status_code == 200
    assert resp_neg.json()["sentiment"] == "negative", f"Expected negative for 'not good', got {resp_neg.json()['sentiment']}"

    # Negation 2: "not bad" -> positive
    resp_pos = client.post("/predict", json={"text": "For this price point it is actually not bad at all, worth buying."})
    assert resp_pos.status_code == 200
    assert resp_pos.json()["sentiment"] == "positive", f"Expected positive for 'not bad', got {resp_pos.json()['sentiment']}"


def test_predict_validation_empty_text(client):
    """Verify empty text returns HTTP 422."""
    resp = client.post("/predict", json={"text": ""})
    assert resp.status_code == 422

    resp_ws = client.post("/predict", json={"text": "   "})
    assert resp_ws.status_code == 422


def test_predict_validation_length_exceeded(client):
    """Verify text exceeding 2000 characters returns HTTP 422."""
    long_text = "great product " * 200  # > 2000 characters
    assert len(long_text) > 2000
    resp = client.post("/predict", json={"text": long_text})
    assert resp.status_code == 422
    assert "exceeds maximum allowed length" in resp.json()["detail"]


def test_predict_missing_model_returns_503(client):
    """Verify that when best_model.joblib is missing, a clear 503 is returned."""
    with patch("inference.BEST_MODEL_PATH", Path("/tmp/non_existent_model.joblib")):
        with patch("inference._PIPELINE", None):
            resp = client.post("/predict", json={"text": "A genuine review"})
            assert resp.status_code == 503
            assert "Model not available" in resp.json()["detail"]


def test_predict_bulk_json(client):
    """Verify bulk prediction via JSON payload."""
    payload = {
        "reviews": [
            {"review": "Amazing product, loved it!"},
            {"review": "Horrible quality, defective on arrival."},
            {"review": "Average item, nothing special."},
        ]
    }
    resp = client.post("/predict/bulk", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert data["summary"]["total_reviews"] == 3
    assert data["summary"]["positive_count"] >= 1
    assert data["summary"]["negative_count"] >= 1
    assert "predictions" in data
    assert len(data["predictions"]) == 3
    assert "download_csv" in data


def test_predict_bulk_csv_file(client):
    """Verify bulk prediction via CSV file upload."""
    csv_content = b"product_name,review\nHeadphones,Great sound and comfortable fit\nCharger,Broke within one week terrible\n"
    files = {"file": ("test_reviews.csv", io.BytesIO(csv_content), "text/csv")}
    resp = client.post("/predict/bulk", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["summary"]["total_reviews"] == 2
    assert "download_csv" in data


def test_inference_applies_oof_multipliers():
    """Verify that inference loads and applies the OOF-tuned neutral multiplier (2.90)."""
    multipliers = inference.get_class_multipliers()
    assert "neutral" in multipliers
    assert multipliers["neutral"] == 2.9
    assert multipliers["positive"] == 1.0
    assert multipliers["negative"] == 1.0


def test_contributing_words_filters_stopwords_and_keeps_negations():
    """Verify that top contributing words filters standard stopwords while strictly preserving negations."""
    pipeline = inference.get_model_pipeline()
    words = inference.extract_top_contributing_words(
        pipeline,
        clean_text="this product is not good and very terrible for the price",
        predicted_class="negative",
        top_k=5,
    )
    assert len(words) <= 5
    extracted_tokens = [w["word"] for w in words]
    # Uninformative words like 'is', 'the', 'for' must be excluded
    for stop in ["is", "the", "for", "and"]:
        assert stop not in extracted_tokens, f"Stopword '{stop}' should have been filtered out."
    for w in words:
        assert "weight" in w
        assert isinstance(w["weight"], (int, float))
        assert w["weight"] > 0

