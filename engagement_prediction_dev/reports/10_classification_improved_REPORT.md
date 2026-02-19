# Notebook 10: Leakage-Free 3-Class Engagement Classification — Report
## LinkedIn Engagement Prediction — TrendPilot

**Date:** February 2026
**Notebook:** `10_model_training_v2_classification.ipynb`
**Follows:** NB09 (`09_model_training_classification.ipynb`)
**Dataset:** 772 LinkedIn posts × 74 features (72 content + 2 author-context)
**Target:** 3-class relative engagement for reactions (Below / Average / Above author baseline)

---

## 1. What This Notebook Fixes

Three data pipeline issues were identified in NB08 and NB09 and corrected here.

### Fix 1 — LOO Computed After Split (Leakage)

**Bug in NB08/NB09:**
Both notebooks called `compute_loo_medians(df)` on the full 772-post dataset, then performed the train/test split. This means a test post's reaction count could influence the LOO median computation for training posts by the same author.

**Example of the leakage:**
If Author X has 4 posts (3 in train, 1 in test), the LOO median for each training post used the test post's reactions in the calculation. The test post's outcome "leaked" into the training target.

**Fix in NB10:**
```
1. train_test_split(df)                    # split FIRST
2. compute_loo_train(df_train)             # LOO using training data only
3. compute_loo_test(df_test, df_train)     # test baseline from training data only
```

### Fix 2 — Imbalanced Classes from Fixed Thresholds

**Bug in NB09:**
Fixed thresholds (< 0.75 = Below, 0.75–1.5 = Average, ≥ 1.5 = Above) produced:
- Class 0 (Below):   283 posts — 36.7%
- Class 1 (Average): 184 posts — **23.8% ← too small**
- Class 2 (Above):   305 posts — 39.5%

The squeezed Class 1 explains Class 1 F1 = 0.27–0.31 across all NB09 models.

**Fix in NB10:**
Thresholds derived from **training set quantiles** (33rd and 67th percentile):
```python
p33 = df_train['relative_reactions'].quantile(1/3)
p67 = df_train['relative_reactions'].quantile(2/3)
```
This guarantees approximately equal class sizes in the training set, giving each class
roughly equal representation for the model to learn from.

### Fix 3 — Comments Dropped

Comments show macro F1 = 0.35–0.37 across all NB09 models — barely above the random
baseline of 0.33. Comment behaviour is too noisy and zero-inflated (19.3% zeros) to
model reliably with content features alone. NB10 focuses on **reactions only**.

---

## 2. Understanding LOO (Leave-One-Out)

### What is LOO?

Leave-One-Out (LOO) is a method for computing a reference statistic for each data point
while **excluding that specific data point itself** from the calculation.

In this project, for each LinkedIn post by author A:
```
LOO_median(post_i) = median( reactions of all other posts by author A, excluding post_i )
```

### Why use LOO instead of a simple author average?

**1. Prevents circular self-reference (self-leakage):**
If we computed `author_avg` including post_i, then `reactions_i / author_avg` would be
partly self-referential — the numerator and denominator share information. LOO breaks
this circularity by excluding the current post from its own baseline.

**2. Simulates production correctly:**
At inference time (scoring a new draft post), you use the author's **historical** posts
as context. LOO exactly mirrors this: we compare each post against all the author's
other posts, just as we would in a deployed system.

### Why median instead of mean?

Engagement distributions are right-skewed. A single viral post (e.g., 5,000 reactions
out of a typical 100) would inflate the mean dramatically, making all of the author's
other posts appear to underperform. The median is robust to these outliers and provides
a stable "typical performance" reference point.

### How we compute LOO — training vs test

The correct approach requires different handling for training and test posts to
prevent test data from influencing the training targets:

```
Training posts:
  LOO_median(post_i) = median of OTHER training posts by same author
  Fallback            = global training median (if author has < 2 other training posts)

Test posts:
  LOO_median(post_j) = median of ALL training posts by same author
  Fallback            = global training median (if author has 0 training posts)

KEY: test post reactions are NEVER used in any LOO computation.
```

For test posts, using *all* training posts (not strictly LOO) is more appropriate
because at inference time, the model has access to the author's full history.

### The relative engagement ratio

Once we have the LOO median, we compute the relative ratio:
```
relative_reactions = reactions / LOO_median
```

| Ratio | Interpretation |
|-------|---------------|
| 1.0   | Post at exactly the author's typical performance |
| 2.0   | Post performed 2× the author's typical |
| 0.5   | Post performed at half the author's typical |

This ratio is then used to assign class labels (see Section 3).

---

## 3. Class Definition

### Quantile-Based Thresholds

Classes are defined by the 33rd and 67th percentile of the training set relative ratio:

| Class | Label | Threshold | Training % |
|-------|-------|-----------|------------|
| 0 | Below Average | ratio < p33 | ~33% |
| 1 | Average | p33 ≤ ratio < p67 | ~33% |
| 2 | Above Average | ratio ≥ p67 | ~33% |

The actual threshold values (p33, p67) are computed at runtime and shown in the notebook.
The test set class distribution may differ slightly from 33/33/33 — this is expected
and correct; the model should not know the test distribution in advance.

---

## 4. Critical Data Limitation: Cold-Start

**495 unique authors, 772 posts = average 1.56 posts per author**

The vast majority of authors in this dataset appear only once. For these single-post
authors, there is no LOO baseline available — the fallback to the global training median
is used. This means approximately 60% of posts use the same baseline (the global median),
regardless of the author's actual typical performance.

**Impact:** The LOO personalisation is effectively absent for ~60% of the data.
The relative ratio for these posts reduces to `reactions / global_median`, which is
essentially just scaled absolute engagement.

**Consequence:** The classification is partly driven by absolute engagement level
(high absolute = Above, low absolute = Below) for cold-start posts, rather than
purely relative content performance.

**The `is_cold_start` feature** flags these posts so the model can account for
the reduced reliability of the LOO baseline.

**To fix:** Collect more posts per author. Target 5+ posts per author for
meaningful LOO baselines. With the current data, this is a hard ceiling on
the LOO approach's effectiveness.

---

## 5. New Input Features (NB10 Addition)

Two author-context features are added to the 72 content features:

| Feature | Description | Why included |
|---------|-------------|-------------|
| `author_train_n` | Number of training posts available for this author as LOO context | Tells the model how reliable the LOO baseline is (0 = no context, 5+ = good context) |
| `is_cold_start` | 1 if author_train_n < 1 (using global median fallback) | Explicit flag for the most unreliable baseline cases |

**Zero leakage:** Both features are derived entirely from training data. For test posts,
`author_train_n` = count of training posts by the same author, which is information
that would be available in production.

---

## 6. Model and Evaluation

**Best model from NB09:** XGBoost (macro F1 = 0.4782)
**Approach in NB10:** RandomizedSearchCV (40 iterations, 5-fold stratified CV) on XGBoost

**Evaluation metrics:**
- **Primary:** Macro F1 — averages F1 across all 3 classes equally; not biased toward majority
- **Secondary:** Per-class F1 (especially Class 1, previously the weakest)
- **Baseline:** Random 3-class guess = macro F1 ≈ 0.33

---

## 7. Results

*(Results populated after running the notebook)*

### Comparison Table

| Notebook | Model | Macro F1 | Notes |
|----------|-------|----------|-------|
| NB09 | XGBoost | 0.4782 | Fixed thresholds, pre-split LOO |
| NB10 | XGBoost_base | — | Fixed pipeline, same params |
| NB10 | XGBoost_tuned | — | Fixed pipeline + RandomizedSearchCV |
| Baseline | Random | 0.33 | Random 3-class guess |

---

## 8. Production Inference

For a deployed system using this model:

```python
def predict_engagement_class(post_text, author_history):
    # 1. Extract content features (same 72 as training)
    content_feats = extract_features(post_text)

    # 2. Compute author baseline
    if len(author_history) >= 1:
        author_median = author_history['reactions'].median()
        author_train_n = len(author_history)
        is_cold_start = 0
    else:
        author_median = GLOBAL_TRAINING_MEDIAN   # stored at training time
        author_train_n = 0
        is_cold_start = 1

    # 3. Build full feature vector
    all_feats = content_feats + [author_train_n, is_cold_start]

    # 4. Predict class
    pred_class = model.predict([all_feats])[0]
    # 0 = likely Below average for this author
    # 1 = likely Average for this author
    # 2 = likely Above average for this author

    return pred_class, author_median
```

**Note:** The class boundaries (p33, p67) must be stored at training time and applied
consistently at inference. Save them alongside the model.

---

## 9. Limitations and Next Steps

### Current limitations
1. **Cold-start dominates:** ~60% of posts use the global median as baseline — the LOO personalisation is largely absent from this dataset
2. **Keyword-based features:** The 72 content features are rule-based (keyword matching, style counts). They capture surface-level patterns but not semantic meaning
3. **Small dataset:** 772 posts. With more data (especially multiple posts per author), both the LOO baseline quality and model stability would improve

### Next steps to improve further
1. **Collect more posts per author:** Target 5+ posts/author. This is the single highest-impact improvement available
2. **Sentence embeddings:** Replace or supplement keyword features with SBERT/TF-IDF embeddings of the post text. This captures what content features cannot — tone, specificity, emotional weight
3. **Posting time features:** Day of week, hour of day, recency — estimated to explain 10–20% additional engagement variance
