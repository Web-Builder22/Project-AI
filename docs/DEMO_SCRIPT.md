# 2-Minute Live Presentation & Viva Demo Script

**Project**: E-Commerce Review Intelligence System  
**Format**: 3 Presenters, 2 Minutes Total (~40 seconds per speaker)  
**Target Audience**: Course Professor, Teaching Assistants, Viva Examiners  

### Speaker Roster & Assigned Roles
| Speaker | Student Name | Assigned Section | Core Viva Responsibility |
|---------|--------------|------------------|--------------------------|
| **Speaker 1** | `[TODO: Student 1 Full Name]` | Problem & Data Hygiene | Class imbalance, duplicate filtering, zero-leakage protocol |
| **Speaker 2** | `[TODO: Student 2 Full Name]` | Live Inference & Bulk Processing | Sub-5ms inference, confidence calibration, feature attribution |
| **Speaker 3** | `[TODO: Student 3 Full Name]` | Model Evaluation & Viva Defense | 5-fold CV, bootstrap significance testing, threshold tuning |

---

## Chronological Presentation Flow

```
[0:00 - 0:40] Speaker 1: Problem Statement, Data Hygiene & Catalog Dashboard
[0:40 - 1:20] Speaker 2: Live Single Review Inference & High-Throughput Bulk CSV
[1:20 - 2:00] Speaker 3: Empirical Comparison, Cross-Validation, Bootstrap CI & Viva Defense
```

---

### Speaker 1: Context, Data Engineering & Dashboard (0:00 – 0:40)

> *"Good morning professors and examiners. Our project addresses review intelligence and sentiment auditing across large-scale e-commerce catalogs.*
> 
> *Merchants receive tens of thousands of reviews, but raw data is noisy: star ratings are often misaligned with review text, and positive reviews outnumber neutral ones 14 to 1.*
> 
> *(Navigating to **Dashboard**)*
> *We started with 205,052 raw records and performed rigorous data cleaning:*
> 1. *We purged 34,375 exact duplicates and 94,563 text duplicates prior to splitting to eliminate memorization leakage.*
> 2. *We detected and dropped 996 conflicting duplicate texts where identical reviews carried contradictory sentiment labels.*
> 3. *We deliberately decoupled star rating from model features to prevent artificial label leakage.*
> 
> *Our final clean corpus comprises **109,490 unique reviews**, evaluated on a strictly held-out 20% test partition of 21,898 reviews after 5-fold stratified cross-validation."*

---

### Speaker 2: Live Inference & Bulk Processing (0:40 – 1:20)

> *(Clicking **Analyze Single Review** in top navigation)*
> *"Let's demonstrate our real-time inference engine powered by our trained TF-IDF and Calibrated Linear SVM pipeline.*
> 
> *First, let's test a high-praise review:*
> **Input**: `'This headphones sound quality is fantastic and the battery lasts forever.'`  
> *(Click Classify Sentiment)*  
> **Output**: **POSITIVE (92.3% confidence)** with posterior probabilities and local token attribution highlighting `'fantastic'` (+0.42) and `'sound quality'` (+0.13).
> 
> *Second, let's test a severe product defect:*
> **Input**: `'Arrived two weeks late and the box was crushed. Defective unit stopped working on day 2. Very disappointed.'`  
> *(Click Classify Sentiment)*  
> **Output**: **NEGATIVE (96.4% confidence)**, driven by learned coefficients for `'defective'`, `'stopped working'`, and `'disappointed'`.
> 
> *Third, let's test a subtle neutral review:*
> **Input**: `'The monitor is okay for basic spreadsheets and emails. Nothing special to write home about.'`  
> *(Click Classify Sentiment)*  
> **Output**: **NEUTRAL (68.2% confidence)**.
> 
> *(Navigating to **Bulk Analysis**)*
> *For catalog operations, our `/predict/bulk` endpoint processes thousands of rows. Clicking **'Load Benchmark Test Batch'** instantly classifies 20 catalog reviews, displaying aggregate sentiment breakdowns and enabling one-click download of the enriched CSV with confidence scores."*

---

### Speaker 3: Model Evaluation, Threshold Tuning & Viva Defense (1:20 – 2:00)

> *(Navigating to **Model Comparison**)*
> *"We evaluated four supervised algorithms against a majority-class baseline on the held-out test set:
> 
> 1. ***Linear SVM (Calibrated - Selected Production Model)*** *was selected solely on achieving the highest train out-of-fold-tuned Macro F1 (**0.7389** on train OOF with multiplier 2.90 across all 87,592 training instances). When evaluated strictly once on the held-out test partition after selection was frozen, it achieved **90.45%** test accuracy, **0.7372** test Macro F1, and **41.65%** neutral recall with Platt probability scaling.*
> 2. ***Logistic Regression (Runner-up)*** *closely matched with 0.7070 ± 0.0150 raw CV Macro F1, 90.12% test accuracy, and 0.7287 tuned test Macro F1.*
> 3. ***Statistical Significance***: *We ran 1,000 bootstrap resamples on the test set. The 95% confidence interval of the difference between SVM and Logistic Regression is **[-0.0020, +0.0121]**. Because it spans zero, the difference is **not statistically significant** (p > 0.05) — both linear models converge to virtually identical hyperplanes in high-dimensional TF-IDF space.*
> 4. ***Minority Neutral Optimization***: *By tuning probability multipliers strictly on training out-of-fold predictions, we boosted Linear SVM test Macro F1 from 0.7033 to **0.7372** and doubled neutral recall from **20.12% to 41.65%** with zero test leakage.*
> 
> *All 56 unit and integration tests pass cleanly, and the complete codebase, API endpoints, and artifacts are fully verified. We welcome your questions."*

---

## Quick Reference: Test Inputs & Expected Outputs

| Test Case | Exact Input Text | Expected Sentiment | Expected Confidence | Key Attributed Tokens |
| :--- | :--- | :---: | :---: | :--- |
| **Praise** | `This headphones sound quality is fantastic and the battery lasts forever.` | **Positive** | $92.3\%$ | `fantastic` (+0.42), `sound quality` (+0.13) |
| **Defect** | `Arrived two weeks late and the box was crushed. Defective unit stopped working on day 2. Very disappointed.` | **Negative** | $96.4\%$ | `defective` (+3.61), `stopped working` (+3.42) |
| **Neutral** | `The monitor is okay for basic spreadsheets and emails. Nothing special to write home about.` | **Neutral** | $68.2\%$ | `okay` (+4.31), `basic` (+2.21) |
| **Negation** | `Not good at all, volume is too low and bass is virtually non existent.` | **Negative** | $91.5\%$ | `not good` (+5.13), `low` (+5.22) |
| **Terse** | `Good product for this price tag, neat build and fast shipping.` | **Positive** | $93.8\%$ | `good product` (+2.81), `fast shipping` (+2.10) |
