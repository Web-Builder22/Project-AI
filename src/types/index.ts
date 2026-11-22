export type Sentiment = 'positive' | 'negative' | 'neutral';

export interface DashboardStats {
  totalReviews: number;
  positive: number;
  negative: number;
  neutral: number;
  recentActivity: { id: number; product: string; snippet: string; sentiment: Sentiment; date: string; confidence?: number }[];
}

export interface ContributingWord {
  word: string;
  weight: number;
}

export interface AnalysisResult {
  sentiment: Sentiment;
  confidence: number;
  probabilities?: Record<Sentiment, number>;
  processed_text?: string;
  product_name?: string;
  top_contributing_words?: ContributingWord[];
  explanation?: string;
}

export interface BulkFileInfo {
  name: string;
  sizeBytes: number;
  rowCount: number;
  hasReviewColumn: boolean;
  hasRatingColumn: boolean;
}

export interface BulkPredictionRow {
  row_index: number;
  review: string;
  predicted_sentiment: Sentiment;
  confidence: number;
  probabilities: Record<Sentiment, number>;
  error?: string;
}

export interface BulkAnalysisResult {
  total_reviews: number;
  counts: Record<Sentiment, number>;
  percentages: Record<Sentiment, number>;
  average_confidence: number;
  predictions: BulkPredictionRow[];
  downloadable_csv: string;
}

export interface ModelMetricsSummary {
  name: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  cvMean?: number;
  cvStd?: number;
  neutralF1?: number;
  trainingTime?: number;
  isBest?: boolean;
}

export interface PerClassMetric {
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface ModelDetail {
  display_name: string;
  model_type: string;
  best_hyperparameters: Record<string, any>;
  cv_macro_f1_mean: number;
  cv_macro_f1_std: number;
  test_metrics: {
    accuracy: number;
    macro_precision: number;
    macro_recall: number;
    macro_f1: number;
    weighted_f1: number;
    per_class: Record<Sentiment, PerClassMetric>;
    confusion_matrix: number[][];
  };
  training_time_seconds: number;
  subsample_details?: { size: number; sampling_method: string };
}

export interface ModelEvaluationData {
  dataset_report: {
    raw_rows: number;
    corrupted_rate_rows_dropped: number;
    conflicting_duplicate_texts_dropped: number;
    conflicting_rows_dropped: number;
    final_clean_rows: number;
    class_counts: Record<Sentiment, number>;
    class_percentages: Record<Sentiment, number>;
  };
  split: {
    test_size: number;
    random_state: number;
    train_samples: number;
    test_samples: number;
  };
  hyperparameter_grids: Record<string, any>;
  models: Record<string, ModelDetail>;
  generalization_gaps: Record<string, { cv_macro_f1: number; test_macro_f1: number; gap: number; gap_pct: number }>;
  threshold_tuning_oof: Record<string, any>;
  bootstrap_significance_test: Record<string, any>;
  selected_best_model: {
    model_key: string;
    display_name: string;
    selection_metric: string;
    cv_macro_f1: number;
    test_macro_f1: number;
    selection_reason: string;
    pipeline_artifact: string;
  };
  top_features: {
    source: string;
    best_model_features: Record<Sentiment, { word: string; coefficient: number }[]>;
  };
  error_analysis: {
    patterns: { pattern: string; count: number; description: string; proposed_fix: string }[];
    sample_errors: Record<Sentiment, { text: string; true_label: Sentiment; predicted_label: Sentiment }[]>;
  };
  timestamp_utc?: string;
}

export interface ProductInsight {
  product_name: string;
  total_reviews: number;
  average_rating: number;
  positive: number;
  negative: number;
  neutral: number;
  positive_pct: number;
  negative_pct: number;
}

export interface EDASummaryData {
  dataset_name: string;
  total_raw_rows: number;
  total_raw_columns: number;
  total_clean_rows: number;
  cleaning_report: {
    null_counts_raw: Record<string, number>;
    corrupted_rate_rows_dropped: number;
    exact_duplicate_rows_dropped: number;
    text_duplicates_dropped: number;
    conflicting_duplicate_texts: number;
  };
  class_distribution: {
    raw_counts: Record<Sentiment, number>;
    raw_percentages: Record<Sentiment, number>;
    clean_counts: Record<Sentiment, number>;
    clean_percentages: Record<Sentiment, number>;
  };
  rating_distribution: {
    counts: Record<string, number>;
    percentages: Record<string, number>;
  };
  rating_vs_sentiment: Record<string, Record<Sentiment, number>>;
  review_length_distribution: Record<string, any>;
  top_words_per_sentiment: Record<Sentiment, { word: string; count: number }[]>;
  top_products: ProductInsight[];
  key_findings: string[];
}

export interface InsightItem {
  topic: string;
  count: number;
  share: number;
}
