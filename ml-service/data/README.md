# E-Commerce Review Dataset

This directory contains instructions for obtaining the e-commerce customer review dataset used for model training and evaluation.

## Dataset Source
- **Origin**: E-Commerce Product Customer Reviews Dataset (Flipkart product catalog)
- **Source Link**: [DATASET_SOURCE_LINK](https://www.kaggle.com/datasets) <!-- Replace with specific dataset URL if hosted externally -->
- **Filename Expected**: `reviews.csv` placed at `ml-service/data/reviews.csv`

## Dataset Schema
The raw dataset contains **205,052 rows** and 6 columns:
- `product_name`: Title of the catalog product (string)
- `product_price`: Price string (string)
- `Rate`: Star rating from 1 to 5 (float/int, contains 3 corrupted string rows)
- `Review`: Short title of the review (string, 24,664 null values)
- `Summary`: Full sentence customer review (string, 11 null values)
- `Sentiment`: Ground truth customer sentiment label (`positive`, `negative`, `neutral`)

## Data Hygiene & Leakage Prevention Rules
Before model training, the raw dataset undergoes rigorous preprocessing (`preprocess.py`):
1. **Corrupted Rate Rows**: 3 records where text was shifted into numeric columns are discarded.
2. **Missing Text Handling**: `Summary` is used as the primary review text; when available, `Review` title is prepended.
3. **Exact Deduplication**: 34,375 completely identical rows are removed.
4. **Text Deduplication**: 94,563 duplicated `(Review, Summary)` text pairs are pruned.
5. **Conflicting Duplicate Removal**: 996 text groups where identical review text was labeled with conflicting sentiments (e.g., both positive and negative) are dropped entirely (eliminating 27,284 corrupted rows).
6. **Feature Leakage Prevention**: The `Rate` column is strictly excluded from feature representation. Text features are extracted solely via TF-IDF fitted inside stratified cross-validation folds.

**Clean Corpus Size**: **109,490 verified reviews** (82,687 positive, 20,791 negative, 6,012 neutral).

## Setup & Preprocessing Command
```bash
# Place downloaded CSV at ml-service/data/reviews.csv
cd ml-service
python preprocess.py
python train_proper.py
```
