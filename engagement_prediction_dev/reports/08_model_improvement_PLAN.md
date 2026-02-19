# Model Improvement Plan
## LinkedIn Engagement Prediction — TrendPilot

**Date:** February 2026
**Author:** Capstone Team
**Preceding Notebook:** `06_model_training_v4_FIXED.ipynb`
**Status:** Approved for Implementation

---

## 1. Executive Summary

All models in Notebook 06 (v4 FIXED) produce **negative R² on the test set**, meaning they perform worse than simply predicting the mean engagement value. This plan diagnoses the root causes and defines two distinct improvement strategies, each implemented as a separate notebook:

| Notebook | Strategy | Primary Metric |
|----------|----------|----------------|
| `08_model_training_loo_relative.ipynb` | LOO Relative Regression | Spearman ρ, Direction Accuracy |
| `09_model_training_classification.ipynb` | 3-Class Classification | Macro F1, Direction Accuracy |

---

## 2. Current State — What Went Wrong

### 2.1 Results from Notebook 06 (v4 FIXED)

| Model | Reactions R² | Comments R² |
|-------|-------------|-------------|
| Linear Regression | -0.2999 | -0.5455 |
| Ridge Regression | -0.2397 | -0.4117 |
| **Random Forest** | **-0.1321** | **-0.0575** |
| XGBoost | -0.6909 | -1.3011 |
| LightGBM | -0.4743 | -0.4079 |

All R² values are **negative**, meaning every model performs worse than a trivial baseline of always predicting the dataset mean.

Cross-validation R² was positive (0.07–0.12) but test R² was consistently negative — indicating the problem is not just overfitting but a fundamental mismatch between the problem formulation and the data.

### 2.2 Root Cause Analysis

#### Problem 1 — Severely Skewed Target Variables (Critical)

The raw engagement counts are extreme right-tailed distributions:

- **Reactions:** range 1–5,942, mean=353, **median=64** (mean is 5.5× the median)
- **Comments:** range 0–529, mean=32, **median=4** (mean is 8× the median, 19% are zero)

Standard regression models (RF, XGBoost, LightGBM) minimize MSE/RMSE. With these distributions, a handful of viral posts (2,000–6,000 reactions) dominate the loss function. The model is penalized more for missing one viral post than for correctly predicting 50 average posts. It can never win against these outliers.

**Visual evidence:** Mean >> Median by a large factor is the signature of extreme right skew. Models cannot recover from this without target transformation.

#### Problem 2 — Content Features Cannot Predict Absolute Engagement

On LinkedIn, raw engagement is determined by:
1. **Author audience size** (~70–80% of variance) — follower count, network size, posting frequency
2. **Posting time** (~10–20% of variance) — day of week, hour, recency
3. **Content quality** (~5–15% of variance) — what this model is trying to predict

By removing all influencer/author features (correctly, to avoid the features dominating the model), we removed signals that explain 70–80% of raw engagement variance. The model is attempting to predict an outcome using only 5–15% of the relevant information.

**Key insight:** Removing influencer features was the right decision — they dominated because they explain most of the raw variance. But the solution is not to force the model to predict absolute counts without that context. The solution is to **reframe the problem so content features predict what content features actually explain**.

#### Problem 3 — Wrong Problem Formulation

A post with 200 reactions means nothing in isolation:
- **200 reactions from an author who averages 50** → excellent post (4× typical performance)
- **200 reactions from an author who averages 1,000** → poor post (0.2× typical performance)

Predicting absolute engagement is asking content features to answer a question they cannot answer. The right question is: **"Given that this author posts content, will this specific post over- or underperform relative to their typical output?"**

#### Problem 4 — Evaluation Metric Mismatch

R² is sensitive to outliers in a right-skewed distribution. A single viral post in the test set can collapse R² from positive to deeply negative. With only 155 test samples, this is sampling noise — not a meaningful signal about model quality.

The CV R² being positive (0.07–0.12) while test R² is negative confirms this: the model has some genuine predictive signal, but R² on raw targets is an unstable metric for this problem.

---

## 3. Solution Strategy

### 3.1 Core Insight: Normalize Out the Author Baseline

Instead of predicting **how many reactions** a post gets, predict **how many times better or worse than the author's typical performance** the post performs.

This is achieved by:
1. Computing each author's typical performance using **Leave-One-Out (LOO) median** — the median reactions from all their *other* posts, excluding the current one
2. Dividing the post's actual reactions by this LOO median to get a **relative ratio**
3. Log-transforming the ratio to create a near-normal distribution

```
relative_reactions  = reactions / loo_median_reactions
log_rel_reactions   = log(relative_reactions)
```

**Why LOO (Leave-One-Out)?**
Using all posts including the current one would be data leakage — the target (reactions of post A) would influence the LOO median used to compute the target for post A. LOO excludes each post from its own author baseline computation, preventing this leakage.

**Why median (not mean)?**
Engagement is right-skewed. One viral post from an author would inflate their mean, making every other post look like underperformance. The median is a more stable and representative "typical performance" measure.

**Why log-transform the ratio?**
Even after dividing by the LOO median, ratios are still right-skewed (a post can be 10–50× the median, but can't be less than 0× the median). Log-transforming makes the distribution symmetric around 0:
- `0.0` = post at author's median (1× typical)
- `+0.69` = 2× author's typical (above average)
- `-0.69` = 0.5× author's typical (below average)

### 3.2 Why This Lets Content Features "Shine"

With the raw target:
- Most variance is explained by author identity (who posted)
- Content features compete against this overwhelming author signal

With the log-relative target:
- Author identity is "divided out" — it's already factored into the denominator
- The remaining variance is exactly what content features can explain: why some posts outperform and others underperform for the same author
- Content quality, hook strength, readability, and storytelling elements now explain meaningful variance

### 3.3 Cold-Start Handling

Authors with fewer than 2 posts don't have a meaningful LOO baseline. For these cases, we substitute the **dataset-wide global median**. At inference time, the same rule applies for new users with no posting history.

---

## 4. Implementation Plan

### 4.1 Notebook 08 — LOO Relative Regression

**File:** `08_model_training_loo_relative.ipynb`

**Approach:**
1. Compute LOO author medians for reactions and comments
2. Create log-relative targets: `log(reactions / loo_median)`
3. Keep the same 72 content features from Notebook 06 (no influencer features added as inputs)
4. Train Random Forest, XGBoost, LightGBM on the new targets
5. Evaluate with Spearman rank correlation and direction accuracy (primary) + R²/MAE on log scale (secondary)
6. Back-transform predictions to original scale for interpretability
7. Show feature importance — content features should now rank at the top

**New Evaluation Metrics:**

| Metric | Description | Beat This |
|--------|-------------|-----------|
| Spearman ρ | Rank correlation between predicted and actual | ρ > 0 means signal |
| Direction Accuracy | % posts correctly predicted above vs below baseline | > 50% beats random |
| R² (log scale) | Variance explained on log-relative target | > 0 beats mean baseline |
| MAE (log scale) | Mean absolute error in log units | Lower is better |

**Why Spearman over Pearson R²?**
- Spearman is rank-based: robust to the remaining skew in log-relative targets
- Spearman answers "do we rank posts correctly?" — the most actionable question for content optimization
- Pearson R² is still sensitive to outliers even after log-transformation

**Expected range:** Spearman ρ = 0.2–0.5, Direction Accuracy = 55–65%

### 4.2 Notebook 09 — 3-Class Engagement Classification

**File:** `09_model_training_classification.ipynb`

**Approach:**
1. Use the same LOO relative ratios from Notebook 08
2. Convert to 3-class labels based on relative ratio thresholds:
   - **Class 0 (Below Average):** `relative_reactions < 0.75` (post underperformed by >25%)
   - **Class 1 (Average):** `0.75 ≤ relative_reactions < 1.5` (within ±50% of baseline)
   - **Class 2 (Above Average):** `relative_reactions ≥ 1.5` (post outperformed by >50%)
3. Train Random Forest, XGBoost, LightGBM classifiers with class balancing
4. Evaluate with macro F1, confusion matrix, per-class precision/recall
5. Also train binary variant: **Below** vs **At/Above** baseline

**Why 3-class classification?**
- More robust than regression: the model only needs to predict the *direction and magnitude* of deviation, not the exact ratio
- More actionable: "Will this post go viral?" is more useful than "this post will get 127 reactions"
- Handles class imbalance explicitly (since viral posts are rare)
- Easier to communicate to non-technical stakeholders

**Expected range:** Macro F1 = 0.35–0.55 (random baseline = 0.33), Binary accuracy = 55–65%

---

## 5. Metrics Comparison Framework

When comparing results across notebooks, use this decision framework:

| Outcome | Interpretation |
|---------|---------------|
| Spearman ρ > 0.3, Dir Acc > 60% | Strong signal — model useful for content ranking |
| Spearman ρ 0.1–0.3, Dir Acc 55–60% | Moderate signal — model provides guidance but noisy |
| Spearman ρ < 0.1, Dir Acc ≤ 52% | Weak/no signal — content features insufficient alone |
| Macro F1 > 0.45 | Classification working well |
| Macro F1 = 0.33 | No better than random 3-class guess |

---

## 6. Realistic Performance Expectations

Based on academic literature on social media engagement prediction with content-only features:

- **Typical Spearman ρ range:** 0.2–0.5 for content-only models on normalized targets
- **With author features:** 0.5–0.8
- **Absolute ceiling for content-only:** ~R² 0.15–0.25 on normalized targets

This is not a failure — it reflects the reality that content quality is one of several factors driving engagement. The model's value is in **ranking content by expected relative performance**, not in predicting exact counts.

---

## 7. Production Inference Flow

After training, the model would be used as follows:

```
Given: a new LinkedIn post draft + known author

1. Extract content features (same 72 features)
2. Retrieve author's historical median reactions (from their past posts)
3. Predict: log_relative = model.predict(content_features)
4. Interpret: relative_ratio = exp(log_relative)
   → ratio > 1.0: post expected to outperform author's typical
   → ratio < 1.0: post expected to underperform author's typical
5. Optionally scale to estimate absolute count:
   predicted_reactions = relative_ratio × author_historical_median
```

For new authors (cold-start), substitute global dataset median in step 2.

---

## 8. File Deliverables

| File | Location | Description |
|------|----------|-------------|
| `08_model_improvement_PLAN.md` | `reports/` | This document |
| `08_model_training_loo_relative.ipynb` | `notebooks/` | LOO relative regression implementation |
| `09_model_training_classification.ipynb` | `notebooks/` | 3-class classification implementation |

---

## 9. Decision Criteria for Proceeding

After running both notebooks, the following thresholds determine whether to proceed to production:

- **Minimum bar:** Spearman ρ > 0.15 OR Direction Accuracy > 53% on test set
- **Good result:** Spearman ρ > 0.25 AND Direction Accuracy > 57%
- **Excellent result:** Spearman ρ > 0.35 OR Macro F1 > 0.45

If neither notebook clears the minimum bar, the next step is collecting more data or exploring NLP-based features (e.g., sentence embeddings, BERTopic).
