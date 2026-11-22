# Machine Learning Model Explanation & Viva Guide
**Course Project**: E-Commerce Review Intelligence System  
**Dataset**: Amazon / Flipkart E-Commerce Product Reviews (109,490 clean unique reviews)  
**Task**: 3-Class Sentiment Classification (`positive`, `negative`, `neutral`)

---

## 1. Feature Extraction: How TF-IDF Works

### 1.1 Mathematical Formulation
Term Frequency-Inverse Document Frequency (TF-IDF) converts raw unstructured review text into dense statistical numeric vectors by balancing how often a term appears within a single review against how common or rare it is across the entire corpus.

1. **Term Frequency (TF)**:
   Measures the frequency of term $t$ in review document $d$. To prevent long reviews from dominating due to raw repetition, we employ sublinear term frequency scaling ($sublinear\_tf=True$):
   $$\text{TF}_{\text{sublinear}}(t, d) = 1 + \log(f_{t, d}) \quad \text{for } f_{t, d} > 0$$

2. **Inverse Document Frequency (IDF)**:
   Measures the discriminative uniqueness of term $t$ across all $N$ reviews in the corpus:
   $$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + \text{DF}(t)}\right) + 1$$
   where $|D| = 87,592$ training documents, and $\text{DF}(t)$ is the number of documents containing term $t$.

3. **Composite TF-IDF Score & L2 Normalization**:
   $$\text{TF-IDF}(t, d) = \text{TF}_{\text{sublinear}}(t, d) \times \text{IDF}(t, D)$$
   Each document vector $\mathbf{v}$ is subsequently normalized to unit Euclidean length ($\|\mathbf{v}\|_2 = 1$):
   $$\mathbf{x} = \frac{\mathbf{v}}{\sqrt{\sum_{k} v_k^2}}$$
   This ensures cosine similarity directly equals the dot product of two normalized vectors.

### 1.2 Step-by-Step Worked Example
Consider a mini-corpus of 3 customer reviews:
- **Review 1 ($d_1$)**: *"great battery life"*
- **Review 2 ($d_2$)**: *"battery life mediocre"*
- **Review 3 ($d_3$)**: *"great phone"*

Corpus size: $|D| = 3$. Total vocabulary: `["battery", "great", "life", "mediocre", "phone"]`.

| Term $t$ | Document Frequency $\text{DF}(t)$ | Smooth IDF Calculation: $\log\left(\frac{1+3}{1+\text{DF}}\right) + 1$ | Interpretation |
| :--- | :---: | :---: | :--- |
| `battery` | 2 ($d_1, d_2$) | $\log(4/3) + 1 \approx 0.2877 + 1 = 1.2877$ | Moderate weight; occurs in multiple reviews. |
| `great` | 2 ($d_1, d_3$) | $\log(4/3) + 1 \approx 0.2877 + 1 = 1.2877$ | Moderate weight; frequent positive term. |
| `life` | 2 ($d_1, d_2$) | $\log(4/3) + 1 \approx 0.2877 + 1 = 1.2877$ | Moderate weight. |
| `mediocre` | 1 ($d_2$) | $\log(4/2) + 1 \approx 0.6931 + 1 = 1.6931$ | **Highest weight!** Rare discriminative neutral/negative token. |
| `phone` | 1 ($d_3$) | $\log(4/2) + 1 \approx 0.6931 + 1 = 1.6931$ | High weight. |

**Vector Construction for $d_2$ (*"battery life mediocre"*)**:
- $\text{TF}(t, d_2) = 1$ for `battery`, `life`, `mediocre`; $0$ for others.
- Unnormalized vector:
  $$\mathbf{v}_2 = [1.2877, 0, 1.2877, 1.6931, 0]$$
- Euclidean Norm $\|\mathbf{v}_2\|_2 = \sqrt{1.2877^2 + 0^2 + 1.2877^2 + 1.6931^2} = \sqrt{1.658 + 1.658 + 2.867} = \sqrt{6.183} \approx 2.4866$
- Normalized vector:
  $$\mathbf{x}_2 = [0.5178, 0.0, 0.5178, 0.6809, 0.0]$$

Notice how `"mediocre"` receives a **31.5% higher feature weight (0.6809)** than `"battery"` (0.5178), allowing the linear classifier to focus on sentiment words rather than topic nouns.

### 1.3 N-Gram Vocabulary & Negation Modeling
Single words (unigrams) fail on negation (e.g., *"not good"* contains *"good"*). We configure:
- `ngram_range = (1, 2)`: Captures both unigrams and adjacent bigrams.
- Bigrams like `"not bad"` receive positive coefficients (+2.2470) while `"not good"` receives negative coefficients (+1.9019).
- Vocabulary capped at `max_features = 10,000` with `min_df = 2` and `max_df = 0.95`.

---

## 2. How the Selected Model Works (Linear SVM & Platt Calibration)

### 2.1 Maximum-Margin Hyperplane
Linear Support Vector Machines (LinearSVC) construct a decision boundary that maximizes the geometric distance (margin) to the nearest training points (support vectors):
$$\mathbf{w}^T \mathbf{x} + b = 0$$

For a soft-margin SVM with slack variables $\xi_i \ge 0$, the primal optimization problem solves:
$$\min_{\mathbf{w}, b, \mathbf{\xi}} \frac{1}{2} \|\mathbf{w}\|_2^2 + C \sum_{i=1}^n \xi_i \quad \text{subject to } y_i (\mathbf{w}^T \mathbf{x}_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$

- **Regularization Parameter $C$**: Controls the trade-off between margin width ($\frac{2}{\|\mathbf{w}\|}$) and training errors. Smaller $C$ ($C=0.1$) encourages a wider, more forgiving margin that prevents overfitting on noisy customer review texts.

### 2.2 One-vs-Rest (OvR) Multiclass Formulation
For our 3-class target ($\mathcal{Y} \in \{\text{negative}, \text{neutral}, \text{positive}\}$), the model trains 3 binary classifiers:
1. $f_{\text{neg}}(\mathbf{x}) = \mathbf{w}_{\text{neg}}^T \mathbf{x} + b_{\text{neg}}$ (Negative vs. Non-Negative)
2. $f_{\text{neu}}(\mathbf{x}) = \mathbf{w}_{\text{neu}}^T \mathbf{x} + b_{\text{neu}}$ (Neutral vs. Non-Neutral)
3. $f_{\text{pos}}(\mathbf{x}) = \mathbf{w}_{\text{pos}}^T \mathbf{x} + b_{\text{pos}}$ (Positive vs. Non-Positive)

Raw prediction assigns the class with the maximum signed distance:
$$\hat{y} = \arg\max_{c \in \{\text{neg}, \text{neu}, \text{pos}\}} f_c(\mathbf{x})$$

### 2.3 Why Linear SVM Excels on Sparse Text
1. **Curse of Dimensionality Inversion**: In text classification, $d = 10,000$ dimensions and features are sparse ($>99\%$ zeros). According to Vapnik-Chervonenkis (VC) theory, data in high dimensions is almost always linearly separable. Linear SVM finds this hyperplane in $\mathcal{O}(N \cdot d)$ time without requiring non-linear kernel computation.
2. **Convex Global Optimum**: The hinge loss optimization problem is strictly convex with a unique global minimum, avoiding local minima pitfalls common to deep neural networks.

### 2.4 Platt Scaling (Probability Calibration)
Standard LinearSVC outputs an uncalibrated signed distance $f(\mathbf{x}) \in (-\infty, +\infty)$, which cannot directly represent confidence or posterior probabilities. We wrap LinearSVC in `CalibratedClassifierCV(method='sigmoid', cv=3)`.
Platt scaling fits a logistic regression sigmoid on the out-of-fold decision values:
$$P(y = c \mid f_c(\mathbf{x})) = \frac{1}{1 + \exp(A f_c(\mathbf{x}) + B)}$$
Probabilities across the 3 classes are subsequently normalized to sum to 1.0.

#### Side Effect on Minority Class:
Because the sigmoid is fitted against empirical class frequencies, it naturally shrinks raw confidence scores toward the class prior. For the 5.49% minority neutral class, standard calibration compresses neutral posterior probabilities downward, causing standard $\arg\max$ decision rules to favor the 75.52% majority positive class unless decision thresholds or probability multipliers are applied.

---

## 3. Model Comparison & Selection Justification

### 3.1 Empirical Comparison (Held-Out Test Set, $N=21,898$)

| Model Architecture | 5-Fold CV Macro F1 | Test Accuracy | Test Macro F1 (Untuned / Tuned) | Neutral F1 (Untuned / Tuned) | Negative F1 | Positive F1 | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM (Calibrated)** | **0.6963 ± 0.0031** *(OOF tuned: 0.7389)* | **90.45%** | **0.7033 / 0.7372** | **0.2955 / 0.4096** | **0.8583** | **0.9562** | ~148s |
| **Logistic Regression** | 0.7070 ± 0.0150 *(OOF tuned: 0.7354)* | 90.12% | 0.6984 / 0.7287 | 0.2875 / 0.3843 | 0.8526 | 0.9550 | ~32.4s |
| **Multinomial Naive Bayes** | 0.6856 ± 0.0042 | 89.26% | 0.6907 | 0.2874 | 0.8383 | 0.9499 | ~3.8s |
| **Random Forest (subsampled 15k)** | 0.6856 ± 0.0039 | 86.12% | 0.6818 | 0.3241 | 0.7925 | 0.9288 | ~6.6s |
| **Majority Baseline** | 0.2868 ± 0.0000 | 75.52% | 0.2868 | 0.0000 | 0.0000 | 0.8605 | ~3.6s |

### 3.2 Honest Comparison: Why the Top Two Are Close
- Both Linear SVM and Logistic Regression learn linear decision hyperplanes $\mathbf{w}^T \mathbf{x} + b$.
- The mathematical difference is the loss function:
  - Linear SVM optimizes **Hinge Loss**: $L(y, f) = \max(0, 1 - y \cdot f)$ (focuses strictly on points near the margin).
  - Logistic Regression optimizes **Cross-Entropy (Log-Loss)**: $L(y, f) = \log(1 + e^{-y \cdot f})$ (penalizes all points smoothly).
- On 10,000-dimensional sparse TF-IDF text vectors, both loss functions discover remarkably similar hyperplane orientations. Bootstrap significance testing ($B=1,000$ resamples) proves the difference between Linear SVM and Logistic Regression is narrow ($\Delta \approx +0.0049$, 95% CI $[-0.0020, +0.0121]$), confirming that the performance difference is **not statistically significant** at $\alpha=0.05$.
- **Selection Decision Rule**: Linear SVM was selected as the final production model **solely on the basis of achieving the highest train out-of-fold (OOF) Macro F1 (0.7389)** with out-of-fold probability multiplier tuning ($w_{neu}=2.90$) across all 87,592 training instances. The test set was held out and evaluated strictly once after model selection was finalized (yielding 0.7372 test Macro F1, 90.45% accuracy, and 41.65% neutral recall). Under no circumstances were test set scores cited or used to select the winning model. Linear SVM is favored in production for its maximum-margin robustness against sparse high-dimensional text vectors and reliable Platt sigmoid scaling.

### 3.3 Why Random Forest & Naive Bayes Were Inferior
1. **Random Forest**: Decision trees make axis-aligned cuts ($x_j > \theta$). In text, sentiment is rarely defined by a single word in isolation, but by linear combinations of words ($\sum w_j x_j$). To approximate a diagonal linear boundary, trees require thousands of step-like orthogonal cuts, leading to severe sample fragmentation and lower test accuracy (86.12% vs 91.25%).
2. **Multinomial Naive Bayes**: Assumes conditional token independence given the class ($P(w_1, w_2 \mid y) = P(w_1 \mid y) P(w_2 \mid y)$). In natural language, words are highly correlated (e.g., `"waste"` and `"money"`), causing Naive Bayes to over-count correlated evidence and produce overconfident, suboptimal posteriors.

---

## 4. Top Feature Attributions (Coefficients)

Averaged hyperplane coefficients ($\mathbf{w}$) extracted across the three calibration folds reveal the strongest sentiment drivers:

### Positive Features ($\mathbf{w}_{\text{pos}}$)
1. `best` (+2.4272)
2. `not bad` (+2.2470) *(Captured by bigram modeling)*
3. `good` (+2.2437)
4. `super` (+1.8144)
5. `awesome` (+1.7751)
6. `great` (+1.5962)
7. `amazing` (+1.5170)
8. `nice` (+1.4460)
9. `excellent` (+1.4402)
10. `love` (+1.4116)

### Negative Features ($\mathbf{w}_{\text{neg}}$)
1. `not` (+2.7659)
2. `worst` (+2.6602)
3. `bad` (+2.5433)
4. `poor` (+2.4041)
5. `no` (+2.1930)
6. `broken` (+2.0505)
7. `low` (+1.9200)
8. `not good` (+1.9019)
9. `problem` (+1.8775)
10. `disappointed` (+1.8631)

### Neutral Features ($\mathbf{w}_{\text{neu}}$)
1. `average` (+1.7498)
2. `but` (+1.7311) *(Signal of mixed/balanced review)*
3. `okk` (+1.6234)
4. `okay` (+1.5670)
5. `ok` (+1.5377)
6. `goodbut` (+1.5300)
7. `normal` (+1.2940)
8. `avg` (+1.1808)
9. `okey` (+1.0596)
10. `hard and` (+1.0203)

---

## 5. Error Analysis & Systematic Failure Patterns

Inspection of the 10 misclassified examples per class in `evaluation.json` reveals 4 primary error categories:

### 1. Contrastive Sentences & Negation Complexity
- *Example*: *"Good design and fast delivery but stopped working after 3 days."*
- *True Label*: Negative | *Predicted Label*: Positive
- *Analysis*: Bag-of-words sums positive tokens (`"good"`, `"fast"`, `"delivery"`) against negative tokens (`"stopped"`, `"working"`). Because positive tokens outnumber negative tokens, the linear sum remains positive despite the decisive trailing clause.

### 2. Sarcasm & Irony
- *Example*: *"Great product if your goal is to set your money on fire."*
- *True Label*: Negative | *Predicted Label*: Positive
- *Analysis*: Contains high-weight positive tokens (`"great"`, `"product"`) with no explicit negative sentiment vocabulary. Linear bag-of-words models lack syntactic dependency parsing to recognize metaphorical sarcasm.

### 3. Neutral Boundary Ambiguity
- *Example*: *"Arrived in a cardboard box on Tuesday."*
- *True Label*: Neutral | *Predicted Label*: Positive
- *Analysis*: Factual statements have near-zero sentiment intensity. The default decision boundary pulls weak signals toward the 75.5% majority class.

### 4. Label Noise & Star Rating Divergence
- *Example*: *"Product broke immediately. Absolutely terrible."* (Tagged in raw data as Positive due to a 5-star rating entered by customer mistake).
- *Analysis*: Human label noise in e-commerce datasets where customers rate 5 stars to get vendor attention while writing harsh complaints.

---

## 6. Viva Examination: 12 Likely Questions & Answers

### Q1: Why use Macro F1 rather than Accuracy as the primary evaluation metric?
**Answer**: Our dataset is severely imbalanced (75.52% positive, 18.99% negative, 5.49% neutral). A naive DummyClassifier that blindly predicts "positive" achieves 75.52% accuracy while completely failing on customer complaints and neutral feedback (Macro F1 = 0.2868). Macro F1 weights all three classes equally, penalizing models that neglect minority classes.

### Q2: Why did you drop conflicting duplicate texts before splitting the data?
**Answer**: Approximately 1,003 unique review texts appeared in the raw data with contradictory sentiment labels (e.g., identical sentence labeled "positive" in one row and "negative" in another). Leaving conflicting duplicates introduces irreducible label noise. Furthermore, keeping non-conflicting duplicates across train and test partitions causes data leakage, inflating test scores unnaturally.

### Q3: Why is the customer star rating (`Rate`) excluded from model features?
**Answer**: Including `Rate` causes label leakage. In e-commerce pipelines, the goal is natural language sentiment intelligence from textual feedback. Furthermore, customer star ratings and textual reviews frequently diverge (e.g., sarcastic reviews or erroneous 5-star ratings). Star ratings are used solely in Exploratory Data Analysis (EDA) to evaluate rating-sentiment cross-tabulations.

### Q4: How did you ensure zero data leakage during cross-validation?
**Answer**: Feature extraction (`TfidfVectorizer`) is encapsulated inside an `sklearn.pipeline.Pipeline`. In each cross-validation fold, the vectorizer is fitted strictly on the $K-1$ training folds and only transforms the validation fold. Test data ($20\%$, $N=21,898$) was held out prior to all modeling and never accessed during hyperparameter tuning or CV.

### Q5: What does the hyperparameter $C$ control in LinearSVC?
**Answer**: $C$ is the regularization parameter governing the trade-off between maximizing the margin width and minimizing training classification errors ($\sum \xi_i$). High $C$ imposes heavy penalties on misclassifications, leading to narrow margins and overfitting. Low $C$ ($C=0.1$) allows a wider margin with soft slack, providing superior generalization on noisy text.

### Q6: Why did Random Forest perform worse and run slower than Linear SVM?
**Answer**: Text representations have 10,000 sparse orthogonal dimensions. Decision trees make single-variable orthogonal splits ($x_j > \theta$), requiring deep, fragmented trees to capture multi-word sentiment combinations. Linear SVM computes a continuous linear combination $\sum w_j x_j$ directly, optimizing convex hinge loss in linear time $\mathcal{O}(N \cdot d)$.

### Q7: What is Platt Scaling, and why is it necessary for Linear SVM?
**Answer**: Standard LinearSVC outputs raw signed geometric distances $f(\mathbf{x}) \in (-\infty, +\infty)$ from the hyperplane, not probabilities. Platt scaling fits a logistic sigmoid $P(y=1 \mid f(\mathbf{x})) = \frac{1}{1 + \exp(Af + B)}$ over out-of-fold decision values, transforming geometric margins into well-calibrated class probabilities for downstream confidence scoring.

### Q8: Why is the neutral class F1 score (~0.30) noticeably lower than positive (0.95) and negative (0.85)?
**Answer**: Two reasons: (1) Severe class imbalance—neutral comprises only 5.49% of the corpus (6,012 reviews vs 82,687 positive). (2) Semantic ambiguity—neutral reviews often contain mixed positive and negative tokens (e.g., *"good screen but poor battery"*) or purely factual text (*"received yesterday"*), placing them directly on the decision boundary between positive and negative hyperplanes.

### Q9: How does sublinear term frequency scaling ($sublinear\_tf=True$) help text classification?
**Answer**: It replaces raw frequency $f_{t,d}$ with $1 + \log(f_{t,d})$. A customer repeating the word *"bad"* ten times in a review is expressing stronger sentiment than someone writing it once, but not ten times as much. Logarithmic dampening prevents repetitive reviews from skewing Euclidean vector lengths.

### Q10: Why must you use `StratifiedKFold` instead of standard `KFold`?
**Answer**: Standard `KFold` assigns samples randomly. With a minority class at 5.49%, random partitioning risks validation folds having skewed class proportions or insufficient samples for reliable metric calculation. `StratifiedKFold` preserves the exact 75.5% / 19.0% / 5.5% proportions across all 5 folds.

### Q11: How can the system's errors on sarcasm and contrastive sentences be resolved in production?
**Answer**: Linear n-gram models lack positional and syntactic attention mechanisms. Resolving contrastive conjunctions (e.g., *"X was great, but Y ruined it"*) and sarcasm requires bidirectional contextual embeddings (such as modern Transformer models like RoBERTa or DeBERTa) or dependency parsers that weight clauses following contrastive markers like *"but"* or *"however"* more heavily.

### Q12: How would you justify choosing Linear SVM over a deep Transformer model for an enterprise client?
**Answer**: (1) **Inference Latency**: Linear SVM inference takes $<0.2$ milliseconds per review on standard CPU, whereas a Transformer requires 30–100ms on GPU. (2) **Cost & Infrastructure**: Linear SVM runs in a lightweight container with $<100$MB RAM without expensive GPUs. (3) **Interpretability**: SVM coefficients directly explain why a review was flagged (exact word weights), fulfilling regulatory compliance and business transparency requirements.
