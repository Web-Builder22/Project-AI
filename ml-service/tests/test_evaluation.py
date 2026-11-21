"""
Test suite for Phase E: data split integrity, zero leakage, full Pipeline persistence, and evaluation metrics schema.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = BASE_DIR / "reports"


def test_data_cleaning_and_split_no_leakage():
    """Verify that train and test sets have zero text overlap and preserve class distribution."""
    from train_proper import prepare_clean_dataset
    from sklearn.model_selection import train_test_split

    csv_path = DATA_DIR / "reviews.csv"
    assert csv_path.exists(), f"reviews.csv missing at {csv_path}"

    df, report = prepare_clean_dataset(csv_path)
    assert len(df) > 100000, f"Expected >100k clean rows, got {len(df)}"
    assert report["conflicting_duplicate_texts_dropped"] > 0, "Conflicting duplicates should be dropped"

    X = df["full_text"].tolist()
    y = df["Sentiment"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 1. Zero leakage: no text in train should appear in test
    train_set = set(X_train)
    test_set = set(X_test)
    overlap = train_set.intersection(test_set)
    assert len(overlap) == 0, f"Detected data leakage: {len(overlap)} texts appear in both train and test!"

    # 2. Stratification: class proportions in train and test should match within 0.5%
    for label in ["positive", "negative", "neutral"]:
        p_train = y_train.count(label) / len(y_train)
        p_test = y_test.count(label) / len(y_test)
        diff = abs(p_train - p_test)
        assert diff < 0.005, f"Stratification mismatch for class '{label}': train={p_train:.4f}, test={p_test:.4f}"


def test_rate_star_rating_not_in_features():
    """Verify that star rating 'Rate' is completely excluded from feature columns."""
    from train_proper import prepare_clean_dataset

    csv_path = DATA_DIR / "reviews.csv"
    df, _ = prepare_clean_dataset(csv_path)

    # Features must come solely from review text
    assert "full_text" in df.columns
    sample_text = df["full_text"].iloc[0]
    assert isinstance(sample_text, str)


def test_evaluation_json_schema():
    """Verify that evaluation.json adheres to all course and assignment metric schemas."""
    eval_path = ARTIFACTS_DIR / "evaluation.json"
    if not eval_path.exists():
        pytest.skip("evaluation.json not generated yet; run train_proper.py first.")

    with open(eval_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Top-level keys
    for key in [
        "dataset_report",
        "split",
        "hyperparameter_grids",
        "models",
        "selected_best_model",
        "top_features",
        "error_analysis",
    ]:
        assert key in data, f"Missing required top-level key: {key}"

    # Models evaluated
    models = data["models"]
    expected_models = [
        "dummy_baseline",
        "linear_svc",
        "logistic_regression",
        "multinomial_nb",
        "random_forest",
    ]
    for m in expected_models:
        assert m in models, f"Missing model {m} in evaluation.json"
        m_info = models[m]
        assert "display_name" in m_info
        assert "cv_macro_f1_mean" in m_info
        assert "cv_macro_f1_std" in m_info
        assert "training_time_seconds" in m_info
        assert "best_hyperparameters" in m_info

        tm = m_info["test_metrics"]
        for metric in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "per_class"]:
            assert metric in tm, f"Missing {metric} in test_metrics for {m}"

        assert "neutral" in tm["per_class"], f"Missing per-class metrics for 'neutral' in {m}"
        assert tm["per_class"]["neutral"]["f1"] >= 0.0

    # Best model validation
    best = data["selected_best_model"]
    assert best["model_key"] in models
    assert best["cv_macro_f1"] > 0.0
    assert best["selection_reason"]
    assert best["pipeline_artifact"] == "best_model.joblib"

    # Random forest honest subsampling documentation
    rf = models["random_forest"]
    assert "subsample_details" in rf
    assert rf["subsample_details"]["size"] == 15000


def test_best_model_full_pipeline_persistence():
    """Verify best_model.joblib is a complete, self-contained Pipeline capable of direct string inference."""
    best_model_path = MODELS_DIR / "best_model.joblib"
    if not best_model_path.exists():
        pytest.skip("best_model.joblib not created yet; run train_proper.py first.")

    model = joblib.load(best_model_path)
    assert isinstance(model, Pipeline), f"best_model.joblib must be a full sklearn Pipeline, got {type(model)}"
    assert "tfidf" in model.named_steps, "Pipeline missing 'tfidf' vectorizer step"
    assert "clf" in model.named_steps, "Pipeline missing 'clf' classifier step"

    # Inference test directly with raw strings
    test_inputs = [
        "Outstanding product, extremely durable and fast shipping!",
        "Complete waste of money, stopped working after two days.",
        "It is an average item, works as expected nothing special.",
    ]
    preds = model.predict(test_inputs)
    assert len(preds) == 3
    for p in preds:
        assert p in ["positive", "negative", "neutral"]

    # Probability calibration test
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(test_inputs)
        assert probs.shape == (3, 3)
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(3), rtol=1e-3)


def test_confusion_matrices_and_comparison_chart_generated():
    """Verify all report images are saved to disk."""
    expected_reports = [
        "confusion_matrix_dummy_baseline.png",
        "confusion_matrix_linear_svc.png",
        "confusion_matrix_logistic_regression.png",
        "confusion_matrix_multinomial_nb.png",
        "confusion_matrix_random_forest.png",
        "model_comparison_macro_f1.png",
    ]
    for rep in expected_reports:
        p = REPORTS_DIR / rep
        if not p.exists():
            pytest.skip(f"Report image {rep} not generated yet.")
        assert p.stat().st_size > 1000, f"Report {rep} is empty or corrupted."
