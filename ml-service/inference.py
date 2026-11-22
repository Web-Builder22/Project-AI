"""
Phase G: Production inference engine for single and bulk sentiment prediction.
Loads best_model.joblib (full scikit-learn Pipeline) once at startup.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger("ml-service.inference")

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"

CLASSES = ["negative", "neutral", "positive"]

# Global cached model pipeline and OOF-tuned probability multipliers
_PIPELINE: Pipeline | None = None
_MULTIPLIERS: dict[str, float] | None = None


class ModelUnavailableError(Exception):
    """Raised when best_model.joblib is missing or corrupted."""


def get_class_multipliers(force_reload: bool = False) -> dict[str, float]:
    """Load and cache OOF-tuned class probability multipliers (default: neutral=2.90 for Linear SVM)."""
    global _MULTIPLIERS
    if _MULTIPLIERS is not None and not force_reload:
        return _MULTIPLIERS

    multipliers = {"negative": 1.0, "neutral": 2.9, "positive": 1.0}
    # Check evaluation.json first, then training_metadata.json
    eval_file = BASE_DIR / "artifacts" / "evaluation.json"
    meta_file = BASE_DIR / "artifacts" / "training_metadata.json"

    if eval_file.exists():
        try:
            with open(eval_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                best_m = data.get("selected_best_model", {})
                opt_neu = best_m.get("optimal_neutral_multiplier")
                if opt_neu is not None:
                    multipliers["neutral"] = float(opt_neu)
        except Exception as exc:
            logger.warning("Could not load multipliers from evaluation.json: %s", exc)
    elif meta_file.exists():
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                cms = data.get("class_multipliers")
                if isinstance(cms, dict):
                    multipliers.update({k: float(v) for k, v in cms.items()})
        except Exception as exc:
            logger.warning("Could not load multipliers from training_metadata.json: %s", exc)

    _MULTIPLIERS = multipliers
    return _MULTIPLIERS


def get_model_pipeline(force_reload: bool = False) -> Pipeline:
    """Load and cache best_model.joblib along with OOF-tuned multipliers. Raises ModelUnavailableError if missing."""
    global _PIPELINE
    if _PIPELINE is not None and not force_reload:
        return _PIPELINE

    if not BEST_MODEL_PATH.exists():
        logger.error("Best model pipeline missing at %s", BEST_MODEL_PATH)
        raise ModelUnavailableError("Model not available: best_model.joblib missing. Please train models first.")

    try:
        loaded = joblib.load(BEST_MODEL_PATH)
        if not isinstance(loaded, Pipeline):
            raise TypeError(f"Expected Pipeline object in {BEST_MODEL_PATH}, got {type(loaded)}")
        _PIPELINE = loaded
        # Warm multiplier cache simultaneously
        get_class_multipliers(force_reload=force_reload)
        logger.info("Successfully loaded best_model.joblib and OOF multipliers into inference cache.")
        return _PIPELINE
    except Exception as exc:
        logger.exception("Failed to load best_model.joblib")
        raise ModelUnavailableError(f"Failed to load machine learning model: {exc}") from exc


def clean_encoding_text(text: Any) -> str:
    """Same text cleaning function as used during model training."""
    if text is None or pd.isna(text):
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(\?[ÃƒÂ¿]+|[ÃƒÂ¿]+\?*|\?{2,})", " ", text)
    return re.sub(r"\s+", " ", text).strip()


NEGATION_TOKENS = {
    "not", "no", "never", "neither", "nor", "none", "without",
    "cannot", "cant", "can't", "dont", "don't", "doesnt", "doesn't",
    "didnt", "didn't", "wasnt", "wasn't", "werent", "weren't",
    "isnt", "isn't", "arent", "aren't", "wont", "won't", "wouldnt", "wouldn't",
    "couldnt", "couldn't", "shouldnt", "shouldn't", "hardly", "barely",
}

COMMON_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "for", "with", "at", "by", "from",
    "up", "about", "into", "over", "after", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "this", "that",
    "these", "those", "it", "its", "to", "in", "on", "of", "as", "my", "your",
    "we", "they", "them", "he", "she", "you", "i", "me", "so", "very", "just",
    "will", "can", "than", "too", "also", "then", "there", "here", "all", "any",
    "both", "each", "few", "more", "most", "some", "such", "only", "own", "same",
}

FILTER_STOPWORDS = COMMON_STOPWORDS - NEGATION_TOKENS


def _is_informative_token(token: str) -> bool:
    """Return True if token contains meaningful semantic words or negations."""
    words = [w.strip() for w in token.lower().split() if w.strip()]
    if not words:
        return False
    # If phrase contains any negation word (e.g. 'not good', 'no complaints'), preserve it!
    if any(w in NEGATION_TOKENS for w in words):
        return True
    # If all words in the token are in the stopword list, reject it
    if all(w in FILTER_STOPWORDS for w in words):
        return False
    # Must have at least one alphabetic character
    return any(c.isalpha() for c in token)


def extract_top_contributing_words(
    pipeline: Pipeline, clean_text: str, predicted_class: str, top_k: int = 5
) -> list[dict[str, Any]]:
    """
    Compute local linear attribution (TF-IDF weight * classifier coefficient) for the review text.
    Filters out uninformative stopwords (and, for, the, is, a...) while STRICTLY preserving negation words.
    """
    try:
        vec = pipeline.named_steps.get("tfidf")
        clf = pipeline.named_steps.get("clf")
        if vec is None or clf is None:
            return []

        classes = list(getattr(clf, "classes_", getattr(pipeline, "classes_", CLASSES)))
        if predicted_class not in classes:
            return []
        class_idx = classes.index(predicted_class)

        # Get coefficients
        if hasattr(clf, "calibrated_classifiers_"):
            coefs = np.mean([sub.estimator.coef_ for sub in clf.calibrated_classifiers_], axis=0)
        elif hasattr(clf, "coef_"):
            coefs = clf.coef_
        else:
            return []

        # Transform single document
        tfidf_vec = vec.transform([clean_text])
        feature_names = vec.get_feature_names_out()
        nonzeros = tfidf_vec.nonzero()[1]

        scored_terms = []
        for idx in nonzeros:
            word = str(feature_names[idx])
            if not _is_informative_token(word):
                continue
            val = float(tfidf_vec[0, idx])
            c_val = float(coefs[class_idx, idx])
            attribution = val * c_val
            if attribution > 0:
                scored_terms.append((word, attribution))

        scored_terms.sort(key=lambda item: item[1], reverse=True)
        return [{"word": word, "weight": round(score, 4)} for word, score in scored_terms[:top_k]]
    except Exception as exc:
        logger.warning("Could not extract contributing words: %s", exc)
        return []


def predict_single(text: str, product_name: str = "") -> dict[str, Any]:
    """
    Predict sentiment for single review text applying OOF-tuned class probability multipliers.
    Validates input:
    - Empty or whitespace-only -> ValueError(422)
    - Length > 2000 chars -> ValueError(422)
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Field 'text' is required and cannot be empty.")
    if len(text) > 2000:
        raise ValueError("Text exceeds maximum allowed length of 2000 characters.")

    pipeline = get_model_pipeline()
    multipliers = get_class_multipliers()
    cleaned = clean_encoding_text(text)
    if not cleaned:
        raise ValueError("Text contains no valid alphanumeric content after cleaning.")

    classes = [str(c) for c in getattr(pipeline, "classes_", CLASSES)]
    if hasattr(pipeline, "predict_proba"):
        raw_probs = pipeline.predict_proba([cleaned])[0]
        # Apply OOF-tuned probability multiplier (e.g. neutral * 2.90)
        scaled_probs = np.array([float(raw_probs[i]) * multipliers.get(classes[i], 1.0) for i in range(len(classes))])
        sum_scaled = np.sum(scaled_probs)
        normalized_probs = scaled_probs / sum_scaled if sum_scaled > 0 else scaled_probs
        best_idx = int(np.argmax(scaled_probs))
        predicted_sentiment = classes[best_idx]
        confidence = round(float(normalized_probs[best_idx]), 4)
        probabilities = {cls_name: round(float(prob), 4) for cls_name, prob in zip(classes, normalized_probs)}
    else:
        predicted_sentiment = str(pipeline.predict([cleaned])[0])
        probabilities = {str(cls_name): 1.0 if cls_name == predicted_sentiment else 0.0 for cls_name in classes}
        confidence = 1.0

    contributing_words = extract_top_contributing_words(pipeline, cleaned, predicted_sentiment, top_k=5)

    return {
        "sentiment": predicted_sentiment,
        "confidence": confidence,
        "probabilities": probabilities,
        "processed_text": cleaned,
        "product_name": product_name,
        "top_contributing_words": contributing_words,
    }


def predict_bulk_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """
    Batch inference for uploaded reviews dataframe applying OOF-tuned multipliers. Max 5,000 rows.
    """
    if df.empty:
        raise ValueError("Uploaded CSV dataset is empty.")
    if len(df) > 5000:
        raise ValueError(f"Uploaded CSV exceeds maximum allowed limit of 5,000 rows (got {len(df)} rows).")

    # Locate review text column
    lower_cols = {col.lower().strip(): col for col in df.columns}
    target_col = None
    for candidate in ["review", "summary", "text", "review_text", "reviews", "comment"]:
        if candidate in lower_cols:
            target_col = lower_cols[candidate]
            break

    if target_col is None:
        raise ValueError(
            f"CSV must contain a 'review' or 'summary' text column. Found columns: {list(df.columns)}"
        )

    pipeline = get_model_pipeline()
    multipliers = get_class_multipliers()

    # Clean text
    cleaned_texts = df[target_col].map(clean_encoding_text).tolist()
    # Replace empty with placeholder to prevent vectorizer collapse
    safe_texts = [t if t else "average" for t in cleaned_texts]

    classes = list(getattr(pipeline, "classes_", CLASSES))

    if hasattr(pipeline, "predict_proba"):
        raw_probs = pipeline.predict_proba(safe_texts)
        mult_vec = np.array([multipliers.get(c, 1.0) for c in classes])
        scaled_probs = raw_probs * mult_vec
        norm_probs = scaled_probs / np.sum(scaled_probs, axis=1, keepdims=True)
        best_indices = np.argmax(scaled_probs, axis=1)
        preds = np.array([classes[i] for i in best_indices])
        confs = np.max(norm_probs, axis=1)
    else:
        preds = pipeline.predict(safe_texts)
        confs = np.ones(len(preds))

    # Add prediction columns to dataframe
    result_df = df.copy()
    result_df["predicted_sentiment"] = preds
    result_df["confidence"] = np.round(confs, 4)

    # Summary metrics
    total = len(preds)
    pos_count = int(np.sum(preds == "positive"))
    neg_count = int(np.sum(preds == "negative"))
    neu_count = int(np.sum(preds == "neutral"))
    avg_conf = round(float(np.mean(confs)), 4) if total > 0 else 0.0

    summary = {
        "total_reviews": total,
        "positive_count": pos_count,
        "negative_count": neg_count,
        "neutral_count": neu_count,
        "positive_percentage": round(pos_count / total * 100, 2) if total else 0.0,
        "negative_percentage": round(neg_count / total * 100, 2) if total else 0.0,
        "neutral_percentage": round(neu_count / total * 100, 2) if total else 0.0,
        "average_confidence": avg_conf,
    }

    # Generate CSV string
    output_stream = io.StringIO()
    result_df.to_csv(output_stream, index=False)
    csv_string = output_stream.getvalue()

    # Format list of items (cap preview at first 500 for lightweight response)
    preview_rows = result_df.head(500).to_dict(orient="records")

    return {
        "summary": summary,
        "predictions": preview_rows,
        "download_csv": csv_string,
        "total_processed": total,
    }
