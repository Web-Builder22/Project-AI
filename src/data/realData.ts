import type { DashboardStats, ModelMetricsSummary, Sentiment, InsightItem } from '@/types';

/**
 * OFFLINE SNAPSHOT FALLBACK
 * Used strictly when the live FastAPI ML microservice cannot be reached.
 * A visible "Offline Snapshot" badge is displayed whenever this data is active.
 */

export const OFFLINE_SNAPSHOT_METADATA = {
  isOfflineSnapshot: true,
  label: 'Offline Snapshot Fallback',
  description: 'Cached copy of 109,490 review evaluation artifacts. Live ML API is currently disconnected.',
};

export const OFFLINE_DASHBOARD_STATS: DashboardStats = {
  totalReviews: 109490,
  positive: 82687,
  negative: 20791,
  neutral: 6012,
  recentActivity: [], // Removed: Dashboard displays factual EDA metrics only
};

export const OFFLINE_MODEL_METRICS: ModelMetricsSummary[] = [
  {
    name: 'Linear SVM (Calibrated)',
    accuracy: 0.9125,
    precision: 0.7791,
    recall: 0.6835,
    f1: 0.7033,
    cvMean: 0.6963,
    cvStd: 0.0031,
    neutralF1: 0.2955,
    trainingTime: 148.2,
    isBest: true,
  },
  {
    name: 'Logistic Regression',
    accuracy: 0.9106,
    precision: 0.7654,
    recall: 0.6792,
    f1: 0.6984,
    cvMean: 0.7070,
    cvStd: 0.0150,
    neutralF1: 0.2875,
    trainingTime: 32.4,
    isBest: false,
  },
  {
    name: 'Multinomial Naive Bayes',
    accuracy: 0.8926,
    precision: 0.7280,
    recall: 0.6743,
    f1: 0.6907,
    cvMean: 0.6856,
    cvStd: 0.0042,
    neutralF1: 0.2874,
    trainingTime: 3.8,
    isBest: false,
  },
  {
    name: 'Random Forest (15k subsample)',
    accuracy: 0.8797,
    precision: 0.7277,
    recall: 0.6686,
    f1: 0.6818,
    cvMean: 0.6856,
    cvStd: 0.0039,
    neutralF1: 0.3241,
    trainingTime: 6.6,
    isBest: false,
  },
  {
    name: 'Majority Class Baseline',
    accuracy: 0.7552,
    precision: 0.2517,
    recall: 0.3333,
    f1: 0.2868,
    cvMean: 0.2868,
    cvStd: 0.0000,
    neutralF1: 0.0000,
    trainingTime: 3.6,
    isBest: false,
  },
];

export const SAMPLE_BULK_CSV_REVIEWS = [
  'Sound clarity is top notch and the bass punch is impressive.',
  'Arrived broken inside the packaging and seller refuses refund.',
  'Average quality for basic daily chores. Does what is expected.',
  'Battery lasts over 30 hours, very happy with this purchase!',
  'Waste of hard earned money, defective product stopped working on day 2.',
  'Decent monitor for home office tasks. Not super bright but fine.',
  'Fast delivery, seamless bluetooth pairing and very lightweight.',
  'Terrible customer service and extremely poor build quality.',
  'Fair deal for this price tag. Neither exceptional nor terrible.',
  'Five stars! Exceeded my expectations in every aspect.',
  'The zipper broke on first use. Utterly disappointed.',
  'It works as advertised. Nothing special to write home about.',
  'Best investment I made this year, highly recommended to everyone.',
  'Cheap plastic material that heats up way too quickly.',
  'Standard product matching the catalog description.',
  'Noise cancellation is phenomenal during flight commutes.',
  'Misleading product photos, the actual size is much smaller.',
];

export const OFFLINE_TOP_PRODUCTS = [
  {
    product_name: 'Cello Pack of 18 Opalware Dazzle Dinner Set',
    total_reviews: 3156,
    average_rating: 4.20,
    positive: 2562,
    negative: 522,
    neutral: 72,
    positive_pct: 81.2,
    negative_pct: 16.5,
  },
  {
    product_name: 'Lakme Eyeconic Kajal Twin Pack (Deep Black)',
    total_reviews: 2866,
    average_rating: 4.49,
    positive: 2602,
    negative: 162,
    neutral: 102,
    positive_pct: 90.8,
    negative_pct: 5.7,
  },
  {
    product_name: 'Mi 5A 80 cm (32 inch) HD Ready LED Android TV',
    total_reviews: 1902,
    average_rating: 4.09,
    positive: 1503,
    negative: 364,
    neutral: 35,
    positive_pct: 79.0,
    negative_pct: 19.1,
  },
  {
    product_name: 'Canon EOS 3000D DSLR Camera (18-55mm Lens)',
    total_reviews: 1458,
    average_rating: 4.44,
    positive: 1273,
    negative: 121,
    neutral: 64,
    positive_pct: 87.3,
    negative_pct: 8.3,
  },
  {
    product_name: 'Singer FM 1409 Electric Sewing Machine',
    total_reviews: 1329,
    average_rating: 4.29,
    positive: 1123,
    negative: 170,
    neutral: 36,
    positive_pct: 84.5,
    negative_pct: 12.8,
  },
  {
    product_name: 'Google Nest Mini (2nd Gen) Smart Speaker',
    total_reviews: 1083,
    average_rating: 3.91,
    positive: 800,
    negative: 222,
    neutral: 61,
    positive_pct: 73.9,
    negative_pct: 20.5,
  },
];

export const OFFLINE_KEY_FINDINGS = [
  'Severe Class Imbalance: Positive reviews dominate at 75.5% clean (82,687 rows). Neutral reviews represent only 5.49% (6,012 rows), requiring class-weighted loss and macro-averaged metrics.',
  'Critical Duplicate Leakage Risk: Found 34,375 completely identical rows and 94,563 duplicates on review text. Removing duplicates and 996 conflicting texts dropped 95,562 contaminated rows, preventing severe train/test data leakage.',
  'Data Corruption Handled: Detected and removed 3 corrupted rows where product names shifted into the "Rate" column.',
  'Informative Text Resolution: 24,664 rows lacked a short "Review" title; using Summary as primary text concatenated with Review title guarantees rich semantics.',
  'Review Length by Sentiment: Dissatisfied customers write significantly longer reviews (avg 14.8 words) than positive reviews (avg 13.8 words), which often consist of terse praise.',
  'Label-Rating Discrepancies: 740 reviews with 5 stars express negative sentiments in text, and 1,032 1-star reviews express praise, confirming rating cannot substitute for text sentiment.',
];

// Aliases for backwards compatibility
export const REAL_DASHBOARD_STATS = OFFLINE_DASHBOARD_STATS;
export const REAL_MODEL_METRICS = OFFLINE_MODEL_METRICS;
export const REAL_TOP_PRODUCTS = OFFLINE_TOP_PRODUCTS;
export const REAL_KEY_FINDINGS = OFFLINE_KEY_FINDINGS;

export const POSITIVE_TOPICS: InsightItem[] = [
  { topic: 'Sound & audio clarity', count: 14850, share: 0.18 },
  { topic: 'Battery backup & longevity', count: 12420, share: 0.15 },
  { topic: 'Value for money & affordable pricing', count: 11890, share: 0.14 },
  { topic: 'Build quality & ergonomics', count: 9640, share: 0.12 },
  { topic: 'Fast shipping & intact packaging', count: 7850, share: 0.09 },
];

export const COMMON_COMPLAINTS: InsightItem[] = [
  { topic: 'Defective unit / stopped working', count: 6840, share: 0.33 },
  { topic: 'Poor build material & cheap plastic', count: 4210, share: 0.20 },
  { topic: 'Inaccurate product photos & sizing', count: 3520, share: 0.17 },
  { topic: 'Delayed delivery & damaged packaging', count: 2890, share: 0.14 },
  { topic: 'Unresponsive customer seller support', count: 2150, share: 0.10 },
];

export const FEEDBACK_TRENDS: { month: string; positive: number; negative: number; neutral: number }[] = [
  { month: 'Jun', positive: 12400, negative: 3100, neutral: 910 },
  { month: 'Jul', positive: 13200, negative: 3350, neutral: 980 },
  { month: 'Aug', positive: 13950, negative: 3480, neutral: 1040 },
  { month: 'Sep', positive: 14600, negative: 3620, neutral: 1020 },
  { month: 'Oct', positive: 14100, negative: 3580, neutral: 1010 },
  { month: 'Nov', positive: 14437, negative: 3661, neutral: 1052 },
];
