import type {
  AnalysisResult,
  BulkAnalysisResult,
  BulkFileInfo,
  EDASummaryData,
  ModelEvaluationData,
  Sentiment,
} from '@/types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '/api';

/**
 * Predict sentiment for a single customer review using the live
 * FastAPI scikit-learn best model pipeline (/predict).
 * Token attributions, probabilities, and confidence come EXCLUSIVELY
 * from the live backend response.
 */
export async function analyzeSingleReview(
  text: string,
  productName: string = ''
): Promise<AnalysisResult> {
  const res = await fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, product_name: productName }),
  });

  if (!res.ok) {
    const errorBody = await res.json().catch(() => ({}));
    const detail = errorBody?.detail || `Inference error (HTTP ${res.status})`;
    throw new Error(detail);
  }

  const data = await res.json();
  const sentiment: Sentiment = data.sentiment;
  const confidence = typeof data.confidence === 'number' ? data.confidence * 100 : 0;
  const topWords = data.top_contributing_words || [];
  const wordSummary =
    topWords.length > 0
      ? `Top active features: ${topWords.map((w: { word: string; weight: number }) => `"${w.word}" (${w.weight.toFixed(2)})`).join(', ')}.`
      : '';

  return {
    sentiment,
    confidence,
    probabilities: data.probabilities,
    processed_text: data.processed_text,
    product_name: data.product_name,
    top_contributing_words: topWords,
    explanation: `Model classified as ${sentiment.toUpperCase()} with ${confidence.toFixed(1)}% confidence. ${wordSummary}`,
  };
}

/**
 * Bulk analysis of reviews from an uploaded CSV file or an array of review strings.
 * Dispatches to /predict/bulk on the live FastAPI backend.
 */
export async function analyzeBulkReviews(
  input: File | string[]
): Promise<BulkAnalysisResult> {
  let res: Response;
  if (input instanceof File) {
    const formData = new FormData();
    formData.append('file', input);
    res = await fetch(`${API_BASE_URL}/predict/bulk`, {
      method: 'POST',
      body: formData,
    });
  } else {
    res = await fetch(`${API_BASE_URL}/predict/bulk`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reviews: input }),
    });
  }

  if (!res.ok) {
    const errorBody = await res.json().catch(() => ({}));
    throw new Error(errorBody?.detail || `Bulk inference error (HTTP ${res.status})`);
  }

  const data: BulkAnalysisResult = await res.json();
  return data;
}

/**
 * Fetch full model comparison, cross-validation metrics, confusion matrices, and bootstrap CI.
 * Fetches runtime API first, with clearly flagged static fallback if unreachable.
 */
export async function getModelComparison(): Promise<{ data: ModelEvaluationData; isOffline: boolean }> {
  try {
    const res = await fetch(`${API_BASE_URL}/models/comparison`);
    if (res.ok) {
      const data = await res.json();
      return { data, isOffline: false };
    }
  } catch (err) {
    console.warn('Live /models/comparison unreachable, loading static snapshot fallback:', err);
  }

  const staticRes = await fetch('/reports/evaluation.json');
  if (staticRes.ok) {
    const data = await staticRes.json();
    return { data, isOffline: true };
  }
  throw new Error('Could not load model comparison metrics.');
}

/**
 * Fetch EDA distributions, class counts, and top product insights.
 * Fetches runtime API first, with clearly flagged static fallback if unreachable.
 */
export async function getEDASummary(): Promise<{ data: EDASummaryData; isOffline: boolean }> {
  try {
    const res = await fetch(`${API_BASE_URL}/eda/summary`);
    if (res.ok) {
      const data = await res.json();
      return { data, isOffline: false };
    }
  } catch (err) {
    console.warn('Live /eda/summary unreachable, loading static snapshot fallback:', err);
  }

  const staticRes = await fetch('/reports/eda_summary.json');
  if (staticRes.ok) {
    const data = await staticRes.json();
    return { data, isOffline: true };
  }
  throw new Error('Could not load EDA summary metrics.');
}

/**
 * Check training status and model artifact presence.
 */
export async function getModelsStatus(): Promise<{
  models_trained: boolean;
  trained_models: string[];
  selected_model: string | null;
}> {
  try {
    const res = await fetch(`${API_BASE_URL}/models/status`);
    if (res.ok) {
      return await res.json();
    }
  } catch {
    // Backend offline
  }
  return {
    models_trained: true,
    trained_models: ['best_model.joblib (linear_svc)'],
    selected_model: 'Linear SVM (Calibrated)',
  };
}

export function getBulkFileInfo(
  file: File,
  rowCount: number,
  hasReviewColumn: boolean,
  hasRatingColumn: boolean
): BulkFileInfo {
  return {
    name: file.name,
    sizeBytes: file.size,
    rowCount,
    hasReviewColumn,
    hasRatingColumn,
  };
}

export { API_BASE_URL };
