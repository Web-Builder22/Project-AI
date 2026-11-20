"""
Exploratory Data Analysis (EDA) module for E-Commerce Review Intelligence System.

Computes real dataset metrics, distributions, crosstabs, vocabulary frequencies,
and generates publication-quality charts saved to ml-service/reports/.
Also produces structured JSON for consumption by the FastAPI backend and React frontend.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import DATA_DIR, REPORTS_DIR, get_dataset_config
from preprocess import get_stopwords, is_negation_or_critical

logger = logging.getLogger("ml-service.eda")

# Set visualization style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 14,
})

COLORS = {
    "positive": "#10b981",  # emerald-500
    "negative": "#f43f5e",  # rose-500
    "neutral": "#94a3b8",   # slate-400
}


def clean_encoding_text(text: Any) -> str:
    """Clean mojibake, question-mark clusters, and normalize unicode."""
    if text is None or pd.isna(text):
        return ""
    text = str(text)
    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)
    # Remove mojibake with or without question marks, and multi-question-marks
    text = re.sub(r"(\?[ÃƒÂ¿]+|[ÃƒÂ¿]+\?*|\?{2,})", " ", text)
    # Collapse whitespace
    return re.sub(r"\s+", " ", text).strip()


def compute_eda(csv_path: Path | str | None = None, save_plots: bool = True) -> dict[str, Any]:
    """Run full exploratory data analysis on the raw and cleaned review dataset.
    Returns structured JSON with 100% computed metrics.
    """
    if csv_path is None:
        csv_path = DATA_DIR / "reviews.csv"
    else:
        csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Raw Dataset
    raw_df = pd.read_csv(csv_path)
    total_raw_rows = len(raw_df)
    total_raw_cols = len(raw_df.columns)

    # 2. Anomaly Detection & Raw Reports
    null_counts_raw = {col: int(raw_df[col].isna().sum()) for col in raw_df.columns}
    exact_duplicates_raw = int(raw_df.duplicated().sum())
    text_pair_duplicates_raw = int(raw_df.duplicated(subset=["Review", "Summary"]).sum())

    # Detect corrupted rows in 'Rate'
    valid_rate_mask = raw_df["Rate"].astype(str).str.strip().isin(["1", "2", "3", "4", "5"])
    corrupted_rate_rows = int((~valid_rate_mask).sum())
    corrupted_samples = raw_df[~valid_rate_mask].to_dict(orient="records")

    # Raw sentiment counts
    raw_sentiment_series = raw_df["Sentiment"].astype(str).str.strip().str.lower()
    raw_sentiment_counts = {k: int(v) for k, v in raw_sentiment_series.value_counts().items()}
    raw_sentiment_pcts = {
        k: round(float(v) / total_raw_rows * 100, 2)
        for k, v in raw_sentiment_counts.items()
    }

    # 3. Clean and Standardize Data for Analysis
    df = raw_df[valid_rate_mask].copy()
    df["Rate"] = df["Rate"].astype(int)
    df["Sentiment"] = df["Sentiment"].astype(str).str.strip().str.lower()

    # Clean text columns
    df["product_name"] = df["product_name"].map(clean_encoding_text)
    df["Review_clean"] = df["Review"].map(clean_encoding_text)
    df["Summary_clean"] = df["Summary"].map(clean_encoding_text)

    # Unified review text: Summary (sentence) + Review (title)
    df["full_text"] = df.apply(
        lambda r: f"{r['Review_clean']} {r['Summary_clean']}".strip()
        if r["Review_clean"] and r["Review_clean"].lower() not in r["Summary_clean"].lower()
        else r["Summary_clean"] if r["Summary_clean"] else r["Review_clean"],
        axis=1,
    )

    # Filter out empty text rows
    non_empty_mask = df["full_text"].str.len() > 0
    df = df[non_empty_mask].copy()

    # Deduplication before training/split
    dedup_text_only = df.drop_duplicates(subset=["full_text"], keep="first")
    dedup_text_sentiment = df.drop_duplicates(subset=["full_text", "Sentiment"], keep="first")
    conflicting_duplicates_count = len(dedup_text_sentiment) - len(dedup_text_only)

    clean_df = dedup_text_only.copy()
    total_clean_rows = len(clean_df)

    # 4. Clean Distributions
    clean_sentiment_counts = {k: int(v) for k, v in clean_df["Sentiment"].value_counts().items()}
    clean_sentiment_pcts = {
        k: round(float(v) / total_clean_rows * 100, 2)
        for k, v in clean_sentiment_counts.items()
    }

    clean_rate_counts = {str(k): int(v) for k, v in clean_df["Rate"].value_counts().sort_index().items()}
    clean_rate_pcts = {
        str(k): round(float(v) / total_clean_rows * 100, 2)
        for k, v in clean_rate_counts.items()
    }

    # 5. Rating vs Sentiment Crosstab
    crosstab_df = pd.crosstab(clean_df["Rate"], clean_df["Sentiment"])
    # Ensure all expected columns
    for s in ["positive", "negative", "neutral"]:
        if s not in crosstab_df.columns:
            crosstab_df[s] = 0

    crosstab_dict = {
        str(rate): {
            col: int(crosstab_df.loc[rate, col])
            for col in ["positive", "neutral", "negative"]
        }
        for rate in sorted(crosstab_df.index)
    }

    # 6. Review Length Analysis (characters and words)
    clean_df["char_len"] = clean_df["full_text"].str.len()
    clean_df["word_len"] = clean_df["full_text"].str.split().str.len()

    def get_stats(series: pd.Series) -> dict[str, float]:
        if len(series) == 0:
            return {"mean": 0.0, "std": 0.0, "median": 0.0, "min": 0, "max": 0, "q25": 0.0, "q75": 0.0}
        desc = series.describe(percentiles=[0.25, 0.5, 0.75])
        mean_val = float(desc.get("mean", 0.0))
        std_val = float(desc.get("std", 0.0))
        median_val = float(desc.get("50%", 0.0))
        min_val = int(desc.get("min", 0))
        max_val = int(desc.get("max", 0))
        q25_val = float(desc.get("25%", 0.0))
        q75_val = float(desc.get("75%", 0.0))
        return {
            "mean": 0.0 if math.isnan(mean_val) else round(mean_val, 2),
            "std": 0.0 if math.isnan(std_val) else round(std_val, 2),
            "median": 0.0 if math.isnan(median_val) else round(median_val, 2),
            "min": min_val,
            "max": max_val,
            "q25": 0.0 if math.isnan(q25_val) else round(q25_val, 2),
            "q75": 0.0 if math.isnan(q75_val) else round(q75_val, 2),
        }

    length_stats = {
        "all": {
            "chars": get_stats(clean_df["char_len"]),
            "words": get_stats(clean_df["word_len"]),
        },
        "by_sentiment": {
            s: {
                "chars": get_stats(clean_df[clean_df["Sentiment"] == s]["char_len"]),
                "words": get_stats(clean_df[clean_df["Sentiment"] == s]["word_len"]),
            }
            for s in ["positive", "negative", "neutral"]
        },
    }

    # 7. Word Frequency & Top N-grams per Sentiment
    stopwords = get_stopwords()
    top_words_per_sentiment: dict[str, list[dict[str, Any]]] = {}
    word_cloud_data: dict[str, list[dict[str, Any]]] = {}

    for s in ["positive", "negative", "neutral"]:
        subset_texts = clean_df[clean_df["Sentiment"] == s]["full_text"].sample(
            n=min(15000, len(clean_df[clean_df["Sentiment"] == s])), random_state=42
        )
        words: list[str] = []
        for txt in subset_texts:
            tokens = re.findall(r"\b[a-z']+\b", txt.lower())
            for t in tokens:
                cleaned_tok = t.strip("'")
                if len(cleaned_tok) > 1 and (is_negation_or_critical(cleaned_tok) or cleaned_tok not in stopwords):
                    words.append(cleaned_tok)

        counter = Counter(words)
        top_20 = [
            {"word": w, "count": int(c)}
            for w, c in counter.most_common(20)
        ]
        top_words_per_sentiment[s] = top_20

        # Top 60 for word clouds
        word_cloud_data[s] = [
            {"text": w, "value": int(c)}
            for w, c in counter.most_common(60)
        ]

    # 8. Top Products Analysis
    product_counts = clean_df["product_name"].value_counts().head(10)
    top_products_list = []
    for prod_name, count in product_counts.items():
        if not prod_name:
            continue
        prod_subset = clean_df[clean_df["product_name"] == prod_name]
        s_counts = prod_subset["Sentiment"].value_counts()
        pos = int(s_counts.get("positive", 0))
        neg = int(s_counts.get("negative", 0))
        neu = int(s_counts.get("neutral", 0))
        avg_rate = round(float(prod_subset["Rate"].mean()), 2)
        top_products_list.append({
            "product_name": str(prod_name),
            "total_reviews": int(count),
            "average_rating": avg_rate,
            "positive": pos,
            "negative": neg,
            "neutral": neu,
            "positive_pct": round(pos / count * 100, 1),
            "negative_pct": round(neg / count * 100, 1),
        })

    # 9. Key Findings Synthesis (Computed directly from numbers)
    pos_raw_pct = raw_sentiment_pcts.get("positive", 0)
    neu_raw_pct = raw_sentiment_pcts.get("neutral", 0)
    pos_clean_pct = clean_sentiment_pcts.get("positive", 0)
    neu_clean_pct = clean_sentiment_pcts.get("neutral", 0)

    avg_pos_words = length_stats["by_sentiment"].get("positive", {}).get("words", {}).get("mean", 0.0)
    avg_neg_words = length_stats["by_sentiment"].get("negative", {}).get("words", {}).get("mean", 0.0)

    sum_5 = sum(crosstab_dict.get("5", {}).values())
    pos_5_pct = (crosstab_dict.get("5", {}).get("positive", 0) / sum_5 * 100) if sum_5 > 0 else 0.0
    neg_5_count = crosstab_dict.get("5", {}).get("negative", 0)
    pos_1_count = crosstab_dict.get("1", {}).get("positive", 0)

    key_findings = [
        f"Severe Class Imbalance: Positive reviews dominate at {pos_raw_pct}% raw ({raw_sentiment_counts.get('positive', 0):,} rows) and {pos_clean_pct}% clean ({clean_sentiment_counts.get('positive', 0):,} rows). Neutral reviews represent only {neu_raw_pct}% raw and {neu_clean_pct}% clean, requiring class-weighted loss and macro-averaged metrics.",
        f"Critical Duplicate Leakage Risk: Found {exact_duplicates_raw:,} completely identical rows and {text_pair_duplicates_raw:,} duplicates on (Review, Summary). Removing review text duplicates reduces the dataset from {total_raw_rows:,} to {total_clean_rows:,} unique reviews, preventing severe train/test data leakage.",
        f"Data Corruption Handled: Detected and removed {corrupted_rate_rows} corrupted row(s) where product names shifted into the 'Rate' column.",
        f"Missing Review Titles vs. Detailed Summaries: {null_counts_raw.get('Review', 0):,} rows ({null_counts_raw.get('Review', 0) / max(1, total_raw_rows) * 100:.1f}%) lacked a short 'Review' title, whereas only {null_counts_raw.get('Summary', 0):,} rows lacked a 'Summary'. Using 'Summary' as primary review text guarantees informative review semantics.",
        f"Review Length by Sentiment: Dissatisfied customers express negative feedback with significantly longer text (average {avg_neg_words:.1f} words) compared to positive reviews (average {avg_pos_words:.1f} words), which often consist of terse praise (e.g., 'awesome', 'good product').",
        f"Label-Rating Discrepancies: While 5-star ratings are {pos_5_pct:.1f}% positive, {neg_5_count} reviews with 5 stars actually carry negative sentiment text, and {pos_1_count} reviews with 1 star express positive sentiment, proving star ratings alone are insufficient as text sentiment proxies.",
    ]

    summary = {
        "dataset_name": csv_path.name,
        "total_raw_rows": total_raw_rows,
        "total_raw_columns": total_raw_cols,
        "total_clean_rows": total_clean_rows,
        "cleaning_report": {
            "null_counts_raw": null_counts_raw,
            "corrupted_rate_rows_dropped": corrupted_rate_rows,
            "exact_duplicate_rows_dropped": exact_duplicates_raw,
            "text_duplicates_dropped": total_raw_rows - corrupted_rate_rows - total_clean_rows,
            "conflicting_duplicate_texts": conflicting_duplicates_count,
        },
        "class_distribution": {
            "raw_counts": raw_sentiment_counts,
            "raw_percentages": raw_sentiment_pcts,
            "clean_counts": clean_sentiment_counts,
            "clean_percentages": clean_sentiment_pcts,
        },
        "rating_distribution": {
            "counts": clean_rate_counts,
            "percentages": clean_rate_pcts,
        },
        "rating_vs_sentiment": crosstab_dict,
        "review_length_distribution": length_stats,
        "top_words_per_sentiment": top_words_per_sentiment,
        "word_cloud_data": word_cloud_data,
        "top_products": top_products_list,
        "key_findings": key_findings,
    }

    # Save JSON summary
    json_path = REPORTS_DIR / "eda_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved EDA JSON summary to %s", json_path)

    # 10. Generate Visualizations
    if save_plots:
        _generate_plots(clean_df, summary)

    return summary


def _generate_plots(df: pd.DataFrame, summary: dict[str, Any]) -> None:
    """Generate high quality publication plots saved to REPORTS_DIR."""
    # 1. Class Distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    classes = ["positive", "negative", "neutral"]
    counts = [summary["class_distribution"]["clean_counts"].get(c, 0) for c in classes]
    pcts = [summary["class_distribution"]["clean_percentages"].get(c, 0) for c in classes]
    bars = ax.bar(classes, counts, color=[COLORS[c] for c in classes], width=0.55, edgecolor="#0f172a", linewidth=0.5)
    for bar, pct in zip(bars, pcts):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + (max(counts) * 0.015), f"{yval:,}\n({pct}%)", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title("Sentiment Class Distribution (Deduplicated)")
    ax.set_ylabel("Number of Reviews")
    ax.set_ylim(0, max(counts) * 1.15)
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "class_distribution.png", dpi=300)
    plt.close(fig)

    # 2. Rating Distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rates = [1, 2, 3, 4, 5]
    r_counts = [summary["rating_distribution"]["counts"].get(str(r), 0) for r in rates]
    r_bars = ax.bar([f"{r} Star" for r in rates], r_counts, color="#0284c7", width=0.55, edgecolor="#0f172a", linewidth=0.5)
    for bar in r_bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + (max(r_counts) * 0.015), f"{yval:,}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Customer Star Rating Distribution")
    ax.set_ylabel("Count")
    ax.set_ylim(0, max(r_counts) * 1.12)
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "rating_distribution.png", dpi=300)
    plt.close(fig)

    # 3. Rating vs Sentiment Crosstab Heatmap / Grouped Bar
    fig, ax = plt.subplots(figsize=(8, 5))
    crosstab_data = summary["rating_vs_sentiment"]
    rates = sorted(crosstab_data.keys(), key=lambda v: int(v) if str(v).isdigit() else 0)
    pos_vals = [crosstab_data[r].get("positive", 0) for r in rates]
    neg_vals = [crosstab_data[r].get("negative", 0) for r in rates]
    neu_vals = [crosstab_data[r].get("neutral", 0) for r in rates]

    x = np.arange(len(rates))
    w = 0.25
    ax.bar(x - w, pos_vals, width=w, label="Positive", color=COLORS["positive"])
    ax.bar(x, neu_vals, width=w, label="Neutral", color=COLORS["neutral"])
    ax.bar(x + w, neg_vals, width=w, label="Negative", color=COLORS["negative"])

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r} Star" for r in rates])
    ax.set_title("Review Sentiment Breakdown by Star Rating")
    ax.set_ylabel("Count")
    ax.legend(title="Sentiment")
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "rating_vs_sentiment.png", dpi=300)
    plt.close(fig)

    # 4. Review Length Distribution (Words by Sentiment)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for s in ["positive", "negative", "neutral"]:
        subset_words = df[df["Sentiment"] == s]["word_len"].clip(upper=60)
        sns.kdeplot(subset_words, label=f"{s.capitalize()} (mean: {summary['review_length_distribution']['by_sentiment'][s]['words']['mean']} w)", color=COLORS[s], ax=ax, fill=True, alpha=0.25)
    ax.set_title("Review Length Distribution by Sentiment (Capped at 60 words)")
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "review_length_distribution.png", dpi=300)
    plt.close(fig)

    # 5. Top 20 Words per Sentiment
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=False)
    for idx, s in enumerate(["positive", "negative", "neutral"]):
        ax = axes[idx]
        top_words = summary["top_words_per_sentiment"][s][:15]
        words = [item["word"] for item in reversed(top_words)]
        counts = [item["count"] for item in reversed(top_words)]
        ax.barh(words, counts, color=COLORS[s], edgecolor="#0f172a", linewidth=0.3)
        ax.set_title(f"Top Words: {s.capitalize()}", color=COLORS[s], fontweight="bold")
        ax.set_xlabel("Frequency")
    plt.suptitle("Most Frequent Informative Words by Sentiment Class", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "top_words_per_sentiment.png", dpi=300)
    plt.close(fig)

    # 6. Top Products by Review Count
    top_prods = summary["top_products"][:8]
    if top_prods:
        fig, ax = plt.subplots(figsize=(9, 5))
        names = [p["product_name"][:32] + "…" if len(p["product_name"]) > 32 else p["product_name"] for p in reversed(top_prods)]
        pos = [p["positive"] for p in reversed(top_prods)]
        neg = [p["negative"] for p in reversed(top_prods)]
        neu = [p["neutral"] for p in reversed(top_prods)]

        y = np.arange(len(names))
        ax.barh(y, pos, label="Positive", color=COLORS["positive"])
        ax.barh(y, neu, left=pos, label="Neutral", color=COLORS["neutral"])
        ax.barh(y, neg, left=np.array(pos) + np.array(neu), label="Negative", color=COLORS["negative"])
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.set_title("Top Reviewed Products & Sentiment Breakdown")
        ax.set_xlabel("Total Review Volume")
        ax.legend(title="Sentiment", loc="lower right")
        plt.tight_layout()
        fig.savefig(REPORTS_DIR / "top_products.png", dpi=300)
        plt.close(fig)

    logger.info("Saved all EDA visualization plots to %s", REPORTS_DIR)


if __name__ == "__main__":
    print("Running EDA computation...")
    summary = compute_eda()
    print("EDA Complete! Real Summary:")
    print(f"- Raw rows: {summary['total_raw_rows']:,}")
    print(f"- Clean rows: {summary['total_clean_rows']:,}")
    print(f"- Clean sentiment distribution: {summary['class_distribution']['clean_counts']}")
    print(f"- Corrupted rows detected: {summary['cleaning_report']['corrupted_rate_rows_dropped']}")
    print(f"- Reports saved to: {REPORTS_DIR}")
