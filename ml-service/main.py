"""
FastAPI service exposing dataset inspection, preprocessing, and Phase C training pipeline.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import io
import json
import os
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from config import API_HOST, API_PORT, MAX_TEXT_LENGTH, REPORTS_DIR, get_dataset_config
from dataset import DatasetError, inspect_dataset
from inference import (
    ModelUnavailableError,
    get_model_pipeline,
    predict_bulk_dataframe,
    predict_single,
)
from model_manager import get_training_status
from preprocess import check_nltk_resources, preprocess_text
from train import TrainingError, train_all_models
from training_config import get_training_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ml-service")

app = FastAPI(
    title="E-Commerce Review Intelligence — ML Service",
    version="0.3.0-phase-c",
    description="Dataset inspection, text preprocessing, and machine learning model training pipeline.",
)

cors_origins_env = os.getenv("CORS_ALLOWED_ORIGINS", "*")
allowed_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()] or ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lock and flag to prevent concurrent simultaneous training requests
_TRAINING_LOCK = threading.Lock()
_TRAINING_IN_PROGRESS = False


@app.on_event("startup")
def startup_event():
    """Attempt to load the best model pipeline on startup for zero cold-start latency."""
    try:
        get_model_pipeline()
        logger.info("Best model pipeline loaded on startup.")
    except Exception as exc:
        logger.warning("Model pipeline not loaded on startup (may not be trained yet): %s", exc)


@app.exception_handler(DatasetError)
def dataset_error_handler(_req, exc: DatasetError):
    return JSONResponse(status_code=503, content={"error": str(exc), "detail": "dataset_unavailable"})


@app.get("/health")
def health():
    status_info = get_training_status()
    return {
        "status": "ok",
        "service": "ml-service",
        "phase": "J",
        "ml_trained": status_info["has_trained"],
        "nltk_resources": check_nltk_resources(),
    }


@app.get("/models/status")
def models_status():
    """Report trained model artifacts status."""
    status_info = get_training_status()
    metadata = status_info.get("metadata")
    bm = metadata.get("best_model") if metadata else None
    selected_model = bm.get("display_name") if isinstance(bm, dict) else (bm or "Linear SVM (Calibrated)")
    trained_models = [k for k, v in status_info.get("models_exist", {}).items() if v]
    if status_info.get("best_model_exists") and "best_model.joblib" not in trained_models:
        trained_models.append("best_model.joblib (linear_svc)")
    return {
        "models_trained": status_info["has_trained"],
        "trained_models": trained_models,
        "selected_model": selected_model,
        "details": status_info,
    }


@app.get("/models/comparison")
def models_comparison():
    """Return comprehensive model comparison, cross-validation metrics, test-set performance, and best model justification."""
    from model_manager import load_evaluation
    eval_data = load_evaluation()
    if not eval_data:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models have not been evaluated yet. Run train_proper.py first.",
        )
    return eval_data


@app.get("/models/reports/{filename}")
def model_report_image(filename: str):
    """Serve generated model evaluation visualizations (confusion matrices, comparison charts)."""
    safe_name = os.path.basename(filename)
    path = REPORTS_DIR / safe_name
    if not path.exists() or not safe_name.endswith(".png"):
        raise HTTPException(status_code=404, detail=f"Report image '{safe_name}' not found.")
    return FileResponse(path, media_type="image/png")


@app.get("/dataset/inspect")
def dataset_inspect():
    """Factual inspection of the configured CSV. Returns 503 when unavailable or invalid."""
    config = get_dataset_config()
    report = inspect_dataset(config)
    status_code = 200 if report.exists else 503
    return JSONResponse(status_code=status_code, content=jsonable_encoder({"data": report.to_dict()}))


@app.get("/dataset/config")
def dataset_config():
    """Show active dataset configuration and validation status."""
    config = get_dataset_config()
    validation_issues = config.validate()
    return {
        "csv_path": str(config.resolve_csv_path()),
        "review_text_column": config.review_text_column,
        "label_column": config.label_column,
        "product_column": config.product_column,
        "rating_column": config.rating_column,
        "ratings_are_labels": config.ratings_are_labels,
        "expected_classes": list(config.expected_classes),
        "configured_correctly": config.has_required_columns(),
        "validation_issues": validation_issues,
    }


@app.post("/preprocess")
def preprocess_endpoint(payload: dict):
    """Utility endpoint applying the shared preprocessing pipeline to text."""
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=400, detail="Field 'text' (non-empty string) is required.")
    if len(text) > MAX_TEXT_LENGTH:
        raise HTTPException(status_code=400, detail=f"Text exceeds maximum length of {MAX_TEXT_LENGTH} characters.")
    return {"data": {"original_length": len(text), "processed": preprocess_text(text)}}


# --- Phase G Inference Endpoints ---


@app.post("/predict")
async def predict_endpoint(request: Request):
    """
    Predict sentiment for a single customer review.
    Validates:
    - Empty or non-string text -> 422
    - Text length > 2000 characters -> 422
    - Model missing -> 503
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=422, detail="Request body must be a valid JSON object.")

    if not isinstance(body, dict):
        raise HTTPException(status_code=422, detail="Request body must be a JSON object.")

    text = body.get("text")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=422, detail="Field 'text' is required and cannot be empty.")
    if len(text) > 2000:
        raise HTTPException(status_code=422, detail="Text exceeds maximum allowed length of 2000 characters.")

    product_name = body.get("product_name", "")

    try:
        prediction = predict_single(text=text, product_name=product_name)
        return prediction
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Inference error in /predict")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")


@app.post("/predict/bulk")
async def predict_bulk_endpoint(
    request: Request,
    file: UploadFile = File(None),
):
    """
    Bulk sentiment classification for uploaded CSV files or JSON payloads (up to 5,000 rows).
    Returns per-row predictions, aggregate sentiment summary, and downloadable CSV string.
    """
    df = None
    if file is not None and file.filename:
        content = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Could not parse uploaded CSV file: {exc}")
    else:
        # Check JSON payload
        try:
            body = await request.json()
            if isinstance(body, dict) and "reviews" in body:
                reviews_list = body["reviews"]
                if not isinstance(reviews_list, list) or len(reviews_list) == 0:
                    raise HTTPException(status_code=422, detail="Field 'reviews' must be a non-empty list.")
                if isinstance(reviews_list[0], dict):
                    df = pd.DataFrame(reviews_list)
                else:
                    df = pd.DataFrame({"review": reviews_list})
            else:
                raise HTTPException(status_code=422, detail="Provide either a CSV file or JSON body with 'reviews' array.")
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=422, detail="Must provide either a CSV file or JSON body with 'reviews' array.")

    if df is None or df.empty:
        raise HTTPException(status_code=422, detail="Uploaded review dataset is empty.")

    try:
        bulk_results = predict_bulk_dataframe(df)
        return bulk_results
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Bulk inference error in /predict/bulk")
        raise HTTPException(status_code=500, detail=f"Bulk inference error: {exc}")


# --- Phase D Exploratory Data Analysis (EDA) Endpoints ---


@app.get("/eda/summary")
def eda_summary():
    """Return factual Exploratory Data Analysis summary, distributions, crosstabs, and key findings."""
    json_path = REPORTS_DIR / "eda_summary.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Could not load cached EDA summary: %s", exc)

    config = get_dataset_config()
    csv_path = config.resolve_csv_path()
    if not csv_path.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Dataset CSV not found at '{csv_path}' to compute EDA summary.",
        )
    from eda import compute_eda
    summary = compute_eda(csv_path=csv_path, save_plots=True)
    return summary


@app.get("/eda/reports/{filename}")
def eda_report_image(filename: str):
    """Serve generated EDA visualization plots."""
    safe_name = os.path.basename(filename)
    path = REPORTS_DIR / safe_name
    if not path.exists() or not safe_name.endswith(".png"):
        raise HTTPException(status_code=404, detail=f"Report image '{safe_name}' not found.")
    return FileResponse(path, media_type="image/png")


# --- Phase C Training Endpoints ---


@app.post("/training/validate")
def training_validate():
    """Validate dataset readiness, sample counts, class distributions, and split feasibility without training."""
    config = get_dataset_config()
    report = inspect_dataset(config)

    if not report.exists:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "dataset_not_found",
                "error": f"Dataset CSV file not found at '{report.file_path}'.",
                "ready_for_training": False,
            },
        )

    if not report.configured_correctly or report.issues:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "invalid_dataset",
                "error": "Dataset validation failed.",
                "issues": report.issues,
                "ready_for_training": False,
            },
        )

    classes_valid = len(report.label_classes) >= 2
    min_count = min(report.label_classes.values()) if report.label_classes else 0
    stratifiable = classes_valid and min_count >= 2

    return {
        "status": "ready" if (classes_valid and stratifiable) else "warning",
        "ready_for_training": classes_valid,
        "total_rows": report.total_rows,
        "classes": report.label_classes,
        "class_percentages": report.label_percentages,
        "stratifiable": stratifiable,
        "issues": report.issues,
        "notes": report.notes,
    }


@app.post("/train")
def train_models():
    """Run the complete Phase C machine learning training pipeline across all four classifiers."""
    global _TRAINING_IN_PROGRESS

    if not _TRAINING_LOCK.acquire(blocking=False):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Training is already in progress. Please wait for completion.",
        )

    try:
        _TRAINING_IN_PROGRESS = True
        config = get_dataset_config()
        report = inspect_dataset(config)

        if not report.exists:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "dataset_not_found",
                    "error": f"Dataset CSV file not found at '{report.file_path}'. Training cannot proceed.",
                },
            )

        summary = train_all_models()
        return JSONResponse(status_code=status.HTTP_201_CREATED, content=jsonable_encoder(summary))

    except TrainingError as exc:
        logger.error("Training failed: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"status": "failed", "error": str(exc)},
        )
    except Exception as exc:
        logger.exception("Unexpected training error")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "error": str(exc)},
        )
    finally:
        _TRAINING_IN_PROGRESS = False
        _TRAINING_LOCK.release()


@app.get("/training/status")
def training_status():
    """Report whether training has occurred, artifact status, and compatibility."""
    status_info = get_training_status()
    config = get_dataset_config()
    dataset_exists = config.resolve_csv_path().exists()

    return {
        "training_in_progress": _TRAINING_IN_PROGRESS,
        "dataset_available": dataset_exists,
        "dataset_path": str(config.resolve_csv_path()),
        **status_info,
    }


@app.get("/training/config")
def training_config():
    """Return active training configuration and hyperparameter settings."""
    train_cfg = get_training_config()
    dataset_cfg = get_dataset_config()
    return {
        "training_parameters": train_cfg.to_dict(),
        "dataset_configuration": {
            "csv_path": str(dataset_cfg.resolve_csv_path()),
            "review_text_column": dataset_cfg.review_text_column,
            "label_column": dataset_cfg.label_column,
            "expected_classes": list(dataset_cfg.expected_classes),
            "ratings_are_labels": dataset_cfg.ratings_are_labels,
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=API_HOST, port=API_PORT)
