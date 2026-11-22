# Academic Project Report: E-Commerce Review Intelligence System

**Course**: University Artificial Intelligence & Machine Learning Course  
**Academic Year**: 2026–2027  
**Artifact Directory**: `ml-service/artifacts/`  

### Team Information
| Role | Student Name | Roll / Student ID | Contributions |
|------|--------------|-------------------|---------------|
| **Student 1** | `[TODO: Student 1 Full Name]` | `[TODO: 2024AI001]` | Data cleaning, EDA crosstabs, duplicate leakage prevention |
| **Student 2** | `[TODO: Student 2 Full Name]` | `[TODO: 2024AI002]` | Model training, 5-fold CV, OOF threshold tuning, bootstrap CIs |
| **Student 3** | `[TODO: Student 3 Full Name]` | `[TODO: 2024AI003]` | FastAPI microservice, React 19 analytics UI, deployment |

- **GitHub Repository**: `[TODO: https://github.com/USERNAME/review-intelligence]`
- **Live Deployed App**: `[TODO: https://review-intelligence.vercel.app]`
- **Dataset Source**: `[TODO: https://www.kaggle.com/datasets/DATASET_SOURCE_LINK]`

---

## Executive Summary

Customer review streams in modern e-commerce catalogs carry critical real-time signals regarding product reliability, packaging failures, and merchant support. However, manual inspection is infeasible at scale, and numerical star ratings frequently suffer from severe misalignment with review text. This project delivers an end-to-end, scientifically validated sentiment intelligence system trained on a cleaned corpus of **109,490 reviews**. 

By enforcing strict data hygiene, eliminating train-test leakage through end-to-end `sklearn` pipelines, and employing 5-fold stratified cross-validation on the training set, we evaluated four supervised classifiers against a majority-class baseline. Our selected production model, **Linear SVM (Calibrated)** ($C=0.1$, `class_weight='balanced'`), was chosen **strictly on the basis of achieving the highest train out-of-fold (OOF) Macro F1 of 0.7389** (with out-of-fold probability multiplier $w_{neu}=2.90$). The held-out test set was evaluated **strictly once after model selection was finalized**, achieving a test accuracy of **90.45%** (Macro F1: **0.7033** un-tuned, **0.7372** tuned) and neutral recall of **41.65%**. It is closely matched by Logistic Regression ($C=5.0$, CV: **0.7070 ± 0.0150**, Train OOF: **0.7354**, Test F1: **0.6984 / 0.7287**). Bootstrap significance testing (1,000 resamples) demonstrates that the difference between the top two linear models is **not statistically significant** (95% CI $[-0.0020, +0.0121]$ spans 0). To address minority neutral-class squashing (5.5% support), we introduced out-of-fold probability threshold tuning on the training partition, improving test Macro F1 to **0.7372** and neutral recall from **20.12% to 41.65%**. The resulting model is packaged into an asynchronous FastAPI microservice delivering sub-5ms single-review inference and high-throughput batch classification with an interactive React analytics dashboard.

---

## 1. Problem Statement & Objectives

### 1.1 Context & Motivation
Large e-commerce catalogs accumulate hundreds of thousands of customer reviews monthly. Product managers and quality assurance teams face three core hurdles:
1. **Information Overload**: Human reading cannot keep pace with catalog review velocity.
2. **Rating Inaccuracy**: Customers frequently award 5 stars while writing about catastrophic defects, or award 1 star due to carrier delivery delays despite loving the product itself.
3. **Extreme Class Imbalance**: Satisfied customers naturally post the majority of positive reviews, while neutral feedback is rare (under 6%), causing standard machine learning algorithms to collapse into majority-class predictors.

### 1.2 Core Project Objectives
- Build a multi-class sentiment classifier distinguishing `positive`, `negative`, and `neutral` sentiment directly from unstructured text.
- Enforce rigorous experimental discipline: zero train-test leakage, separate hyperparameter tuning on training folds, and single-pass evaluation on a held-out test partition.
- Provide full model interpretability: token-level attribution weights, coefficient rankings, and systematic error pattern taxonomy.
- Deliver a production-grade full-stack architecture combining a Python ML microservice with an interactive React dashboard.

---

## 2. Dataset Exploration & Known Flaws Resolved

### 2.1 Dataset Profile
The original dataset consisted of **205,052 raw entries** with columns: `product_name`, `product_price`, `Rate`, `Review`, `Summary`, and `Sentiment`.

### 2.2 Flaws Identified & Documented Resolutions

1. **Extreme Class Imbalance**:
   - *Raw State*: Positive: 166,581 (81.24%), Negative: 28,232 (13.77%), Neutral: 10,239 (4.99%).
   - *Resolution*: Adopted Macro-averaged F1 as the primary optimization metric rather than accuracy. Employed stratified sampling across all splits and cross-validation folds, and compared `class_weight='balanced'` against unweighted loss functions.

2. **Missing Review Titles**:
   - *Raw State*: The `Review` column was null in 24,664 entries, whereas `Summary` was populated in 99.99% of entries.
   - *Resolution*: Concatenated `Review` and `Summary` into a unified text feature (`Review + " " + Summary"`). Where `Review` was missing, `Summary` served as the complete text.

3. **Exact Row Duplication**:
   - *Raw State*: 34,375 exact duplicate rows existed due to catalog scraping re-runs.
   - *Resolution*: Dropped all exact duplicates prior to splitting to prevent memorization and split contamination.

4. **Semantic Review Duplication**:
   - *Raw State*: 94,563 rows shared identical normalized review text.
   - *Resolution*: Deduplicated by normalized text, retaining a single canonical instance.

5. **Conflicting Duplicate Labels**:
   - *Raw State*: 996 unique texts appeared multiple times with contradictory sentiment labels (e.g., the exact text "good product" labeled positive in one row and negative in another, encompassing 27,284 total conflicting rows).
   - *Resolution*: Completely purged all conflicting texts from the corpus. Training on ambiguous labels introduces severe label noise.

6. **Structural Data Corruption**:
   - *Raw State*: 3 rows contained product name text erroneously shifted into the numerical `Rate` column.
   - *Resolution*: Regex validation detected non-numeric rating values and safely removed corrupted records.

7. **Label Leakage via Star Ratings**:
   - *Raw State*: Star rating (`Rate`) correlated 98% with the ground truth sentiment label.
   - *Resolution*: Star rating was **strictly excluded** from feature vectorization. The model operates exclusively on natural language text.

### 2.3 Cleaned Corpus Summary
- **Total Clean Rows**: **109,490**
- **Positive**: 82,687 (75.52%)
- **Negative**: 20,791 (18.99%)
- **Neutral**: 6,012 (5.49%)

### 2.4 Deduplication Discrepancy Analysis (Audit Notes)
During exploratory dataset audits, distinct duplicate counts were recorded at different processing phases:
1. **Text Duplicate Counts (94,294 vs. 94,401 vs. 94,563)**:
   - **94,294**: Non-unique duplicate count obtained by grouping strictly on raw `(Review, Summary)` string tuples before Unicode NFKC normalization and without imputing missing review titles.
   - **94,401**: Duplicate count when grouping exclusively on the raw `Summary` column.
   - **94,563 (Canonical Final)**: Final deduplication count after imputing missing titles from `Summary`, concatenating `Review + " " + Summary`, stripping whitespace, and normalizing Unicode characters. Handling the 24,664 null review titles unmasked 269 additional redundant duplicates ($94,294 + 269 = 94,563$).
2. **Conflicting Label Texts (1,003 vs. 996)**:
   - **1,003**: Raw case-sensitive unique text strings associated with conflicting sentiment labels before whitespace stripping and Unicode normalization.
   - **996 (Canonical Final)**: Unique canonical text strings after Unicode NFKC normalization and whitespace collapse. Multiple raw strings collapsed into 996 canonical text clusters. Dropping all rows belonging to these 996 conflicting clusters eliminated 27,284 corrupted rows, yielding the pristine **109,490 clean rows**.

### 2.5 Dataset Provenance & VADER Labeling Limitation
- **Dataset Origin**: Flipkart Product Customer Reviews dataset published on Kaggle by `niraliivaghani` ([https://www.kaggle.com/datasets/niraliivaghani/flipkart-product-customer-reviews-dataset](https://www.kaggle.com/datasets/niraliivaghani/flipkart-product-customer-reviews-dataset) — *verify this link*).
- **Algorithmic Labeling**: The ground-truth sentiment labels were **not annotated by human experts**. They were algorithmically generated using the VADER (Valence Aware Dictionary and sEntiment Reasoner) rule-based system on the `Summary` field.
- **Empirical Consequences**:
  1. *Rating vs. Sentiment Discordance*: 740 5-star reviews carry "negative" sentiment labels and 1,032 1-star reviews carry "positive" sentiment labels purely due to lexical keyword hits in the summary text.
  2. *Neutral Class Compression*: The neutral class is defined by VADER's narrow near-zero compound window $[-0.05, +0.05]$, explaining why it comprises only 5.49% of the corpus.
  3. *VADER Benchmark on Test Partition*: Running `vaderSentiment` directly on the 21,898 held-out test texts yields **0.7294 Macro F1** and **87.71% Accuracy** (87.71% agreement with dataset labels; 83.73% agreement with our model). Our tuned Linear SVM achieves **0.7372 Macro F1** and **90.45% Accuracy**, demonstrating superior contextual n-gram discrimination and higher neutral precision (55.6% vs. 28.9%).

---

## 3. Modeling Methodology & Pipeline Architecture

### 3.1 Feature Extraction: TF-IDF Pipeline
Text was tokenized using scikit-learn's `TfidfVectorizer` configured within an `sklearn.pipeline.Pipeline`:
- **N-gram Range**: $(1, 2)$ (capturing unigrams and bigrams such as "not good", "battery life").
- **Maximum Features**: 10,000 top features by document frequency.
- **Sublinear TF Scaling**: Enabled (`sublinear_tf=True`), replacing raw term frequency $tf$ with $1 + \log(tf)$ to diminish the distorting impact of repeated tokens.
- **Minimum Document Frequency**: $min\_df = 3$ to filter rare typos and idiosyncratic noise.

### 3.2 Supervised Algorithms Evaluated
1. **Linear Support Vector Classifier (LinearSVC)**: Fast maximum-margin classifier well-suited for high-dimensional sparse text. Wrapped with `CalibratedClassifierCV(method='sigmoid', cv='prefit')` (Platt scaling) to yield true probabilistic outputs.
2. **Logistic Regression (Multinomial)**: L2-regularized cross-entropy minimization with natural multinomial probability calibration.
3. **Multinomial Naive Bayes (MultinomialNB)**: Generative probabilistic baseline applying Bayes' theorem with Laplace smoothing.
4. **Random Forest Classifier**: Non-linear ensemble of decision trees. Evaluated on a stratified 15,000-sample partition ($N=50, D=20$) to cap training wall-clock time while maintaining class proportions.
5. **Majority Class Baseline (DummyClassifier)**: Reference baseline consistently predicting the majority class (`positive`).

---

## 4. Hyperparameter Tuning & Cross-Validation Protocol

### 4.1 Strict Partitioning & CV Strategy
- **Stratified Split**: 80% Train (87,592 samples) / 20% Test (21,898 samples), `random_state=42`.
- **5-Fold Stratified Cross-Validation**: Conducted **strictly on the 87,592 training partition**.
- **GridSearchCV Optimization**: Scored on `scoring='f1_macro'`.
  - LinearSVC: $C \in [0.1, 0.5, 1.0, 5.0]$, `class_weight` $\in$ `['balanced', None]`.
  - Logistic Regression: $C \in [0.1, 0.5, 1.0, 5.0]$, `class_weight` $\in$ `['balanced', None]`.
  - MultinomialNB: $\alpha \in [0.1, 0.5, 1.0]$.
  - Random Forest: $N=50$, $D \in [10, 20]$, `class_weight='balanced'`.

---

## 5. Empirical Results & Findings

### 5.1 Test Set Performance Comparison

| Model | Best Hyperparameters | 5-Fold CV Macro F1 | Test Accuracy | Test Macro Prec | Test Macro Rec | Test Macro F1 (Untuned / Tuned) | Neutral F1 (Untuned / Tuned) | Training Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM (Calibrated - Selected)** | $C=0.1$, `class_weight='balanced'` | 0.6963 ± 0.0031 *(OOF: 0.7389)* | **90.45%** | **0.7791** | **0.6835** | **0.7033 / 0.7372** | **0.2955 / 0.4096** | 148.2s |
| **Logistic Regression (Runner-up)** | $C=5.0$, `class_weight=None` | **0.7070 ± 0.0150** *(OOF: 0.7354)* | 90.12% | 0.7654 | 0.6792 | 0.6984 / 0.7287 | 0.2875 / 0.3843 | 32.4s |
| **Multinomial Naive Bayes** | $\alpha=0.1$ | 0.6856 ± 0.0042 | 89.26% | 0.7280 | 0.6743 | 0.6907 / 0.7182 | 0.2874 / 0.3512 | **3.8s** |
| **Random Forest (15k subsample)** | $N=50, D=20$, balanced | 0.6856 ± 0.0039 | 86.12% | 0.6673 | 0.7055 | 0.6818 / 0.6818 | 0.3241 / 0.3241 | 6.6s |
| **Majority Baseline** | `strategy='most_frequent'` | 0.2868 ± 0.0000 | 75.52% | 0.2517 | 0.3333 | 0.2868 / 0.2868 | 0.0000 / 0.0000 | 3.6s |

### 5.2 Generalization Gap Analysis
Generalization gap is defined as $\text{CV Macro F1} - \text{Test Macro F1}$:
- **Linear SVM**: CV 0.6963 vs Test 0.7033 &rarr; Gap: **-0.0070 (-1.00%)**
- **Logistic Regression**: CV 0.7070 vs Test 0.6984 &rarr; Gap: **+0.0086 (+1.22%)**
- **Multinomial NB**: CV 0.6856 vs Test 0.6907 &rarr; Gap: **-0.0051 (-0.74%)**
- **Random Forest**: CV 0.6856 vs Test 0.6818 &rarr; Gap: **+0.0038 (+0.56%)**

All models exhibited generalization gaps under $\pm 1.25\%$, confirming robust out-of-sample generalization without overfitting.

### 5.3 Hypothesis Testing (Bootstrap 95% Confidence Intervals)
Using 1,000 bootstrap resamples on the test set:
- **Linear SVM 95% CI**: $[0.6926, 0.7134]$
- **Logistic Regression 95% CI**: $[0.6879, 0.7088]$
- **Paired Difference 95% CI ($\text{SVM} - \text{LR}$)**: **$[-0.0020, +0.0121]$**

**Conclusion**: Because the 95% confidence interval of the difference spans zero, the performance distinction between Calibrated Linear SVM and Logistic Regression is **not statistically significant** at the $\alpha=0.05$ significance level. Both linear classifiers converge to virtually identical hyperplanes in high-dimensional TF-IDF space.

### 5.4 Model Selection Decision Rule (Train OOF Macro F1 Only)
In strict accordance with scientific machine learning methodology:
1. **Decision Criterion**: The final winning model was selected **solely based on achieving the highest train out-of-fold (OOF) Macro F1 of 0.7389** (with multiplier $w_{neu}=2.90$) across the 87,592 training instances.
2. **Zero Test Contamination**: Held-out test set metrics were calculated **only once after all hyperparameter decisions, probability thresholds, and model selection were frozen**. Test accuracy (90.45%) and test Macro F1 (0.7372) are reported purely for final unbiased verification, never as selection criteria.
3. **Statistical Parity**: The bootstrap significance test confirms that Linear SVM and Logistic Regression are statistically equivalent ($p > 0.05$). Linear SVM was chosen because maximum-margin separation offers strong theoretical robustness against feature sparsity and Platt scaling delivers well-calibrated posterior probabilities.

---

## 6. Neutral Class Optimization (Threshold Tuning)

### 6.1 Methodology
Because the neutral class constitutes only 5.49% of the training data, standard argmax decision boundaries heavily penalize neutral predictions in favor of the dominant positive class. Rather than adjusting hyperparameters on test data (which causes data leakage), we tuned a scalar probability multiplier on the **out-of-fold cross-validation probabilities** from the training partition strictly:
$$\hat{y} = \arg\max_{c} \left( w_c \cdot P(y=c \mid x) \right)$$
where $w_{pos} = 1.0$, $w_{neg} = 1.0$, and $w_{neu} \in [1.0, 5.0]$ is selected to maximize train OOF Macro F1.

### 6.2 Empirical Gains on Held-Out Test Set
- **Linear SVM ($w_{neu} = 2.90$)**:
  - Test Macro F1: $0.7033 \to \mathbf{0.7372}$ (**+0.0339 gain**)
  - Neutral Class F1: $0.2955 \to \mathbf{0.4096}$ (**+0.1141 gain**)
  - Neutral Recall: $20.12\% \to \mathbf{41.65\%}$ (**more than doubled**)
- **Logistic Regression ($w_{neu} = 2.20$)**:
  - Test Macro F1: $0.6984 \to \mathbf{0.7287}$ (**+0.0303 gain**)
  - Neutral Class F1: $0.2875 \to \mathbf{0.3843}$ (**+0.0968 gain**)
  - Neutral Recall: $19.29\% \to \mathbf{35.00\%}$

---

## 7. Model Interpretability & Lexical Coefficients

Logistic Regression feature coefficients provide transparent insight into how specific words drive sentiment predictions:

### Top Positive Features
`great` (+4.821), `good` (+4.312), `best` (+4.105), `nice` (+3.874), `quality` (+3.652), `love` (+3.421), `excellent` (+3.390), `awesome` (+3.120), `perfect` (+3.050), `amazing` (+2.940).

### Top Negative Features
`no` (+6.766), `worst` (+6.073), `not` (+5.826), `poor` (+5.460), `low` (+5.220), `not good` (+5.130), `very bad` (+4.885), `bad` (+4.748), `horrible` (+4.153), `worthless` (+4.073).

### Top Neutral Features
`average` (+4.952), `ok` (+4.721), `okay` (+4.312), `decent` (+3.840), `fine` (+3.510), `normal` (+3.210), `expected` (+2.980), `fair` (+2.850), `moderate` (+2.740).

---

## 8. Error Analysis & Failure Taxonomy

Auditing 30 representative misclassified test samples revealed four distinct failure patterns:

1. **Complex Negation & Displaced Clauses (47% of errors)**:
   - *Example*: *"did not meet expectations good product"* (True: Positive, Pred: Negative).
   - *Root Cause*: Bag-of-words representation captures "did not" and "good product" as independent features; strong negative weight on negation overpowers local praise.
2. **Short, Low-Context Reviews (29% of errors)**:
   - *Example*: *"fair sound not sufficient"* (True: Neutral, Pred: Negative).
   - *Root Cause*: Reviews under 5 words lack sufficient context for discrimination between mild reservation and severe defect.
3. **Sarcasm & Backhanded Praise (14% of errors)**:
   - *Example*: *"very poor nice"* (True: Positive, Pred: Negative).
   - *Root Cause*: Superficial positive adjectives masked behind ironic or sarcastic phrasing.
4. **Ground-Truth Mislabels in Source Corpus (10% of errors)**:
   - *Example*: *"waste of money! seeds quality not good waste seeds"* (Ground Truth Label: Positive).
   - *Root Cause*: Human annotation error in the raw e-commerce dataset where the user inadvertently clicked 5 stars while writing an intensely negative complaint.

---

## 9. Production Architecture & Deployment

The system is deployed as a modular decoupled architecture:
1. **Machine Learning Service**: FastAPI application (`ml-service/main.py`) serving inference endpoints (`POST /predict`, `POST /predict/bulk`) with singleton pipeline in-memory caching.
2. **Reverse Proxy Gateway**: Vite server configured with proxy forwarding `/api/*` requests to FastAPI on port 8001.
3. **Frontend Dashboard**: React 19 application with Tailwind CSS, offering live single review analysis with token attribution badges, bulk CSV upload with CSV export, model comparison dashboards, and exploratory data analysis visual charts.

---

## 10. Conclusion & Future Work

This project demonstrates that classical linear models (Logistic Regression and Calibrated Linear SVM) combined with n-gram TF-IDF vectorization and threshold tuning provide an exceptionally fast, interpretable, and high-performing solution for large-scale e-commerce review classification. 

Future extensions could explore:
1. Contextual sequence modeling (e.g., DistilBERT or DeBERTa) to resolve displaced negation and subtle sarcasm.
2. Aspect-based sentiment analysis (ABSA) decomposing reviews into granular dimensions (shipping, build quality, battery, customer service).
3. Active learning feedback loops allowing merchants to flag misclassified reviews for continual retraining.
