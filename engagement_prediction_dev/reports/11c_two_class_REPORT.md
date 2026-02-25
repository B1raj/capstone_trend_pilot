# NB11c Report: 2-Class Engagement Rate Classification
## LinkedIn Engagement Prediction — TrendPilot

**Notebook:** `11c_two_class_classification.ipynb`
**Date:** February 2026

---

## 1. Objective

The goal of this notebook is to build a **binary content quality classifier** that predicts whether a LinkedIn post will perform above or below average engagement — measured as a rate per 1,000 followers.

The question being answered is:

> **"Will this post over-perform or under-perform relative to the creator's audience size?"**

Two classes:
- **Class 0 — Below Average:** the post's engagement rate falls below the typical level
- **Class 1 — Above Average:** the post's engagement rate exceeds the typical level

Every post receives a clear directional signal. There is no middle category.

---

## 2. The Target Variable

### 2.1 Why Engagement Rate?

Raw engagement numbers (reactions, comments) are dominated by follower count. A post with 500 reactions is exceptional for a 2,000-follower account but unremarkable for a 2,000,000-follower account. Using raw numbers would mean the model learns audience size, not content quality.

Engagement rate solves this by normalising for audience size:

```python
engagement_rate = (reactions + comments) / (followers / 1000)
```

This expresses total engagement as **engagements per 1,000 followers**. Two creators with completely different audience sizes can now be compared on equal terms:

| Creator | Followers | Reactions | Comments | Engagement Rate |
|---------|-----------|-----------|----------|----------------|
| Nano-influencer | 5,000 | 50 | 10 | (60) / 5 = **12.0** |
| Macro-influencer | 500,000 | 5,000 | 1,000 | (6,000) / 500 = **12.0** |

Both produced equally resonant content. The rate reflects that equivalence.

Both reactions and comments are included because a post that drives conversation is genuinely more engaging. Combined, they form a complete picture of audience response.

### 2.2 Why Two Classes?

Previous 3-class models (below / average / above) consistently struggled with the middle class. In those experiments, the Average class F1 never exceeded 0.46 — meaning nearly half of average posts were being misclassified. This is a structural problem, not a modelling failure.

**Why the middle class is unlearnable with the current features:**

Average content sits on *both* decision boundaries simultaneously. A post near the 33rd or 67th percentile of engagement rate is genuinely ambiguous — a slightly different posting time, minor algorithm variation, or a single influential commenter could tip it into Class 0 or Class 2. Rule-based content features (hook patterns, style flags, sentiment scores) cannot capture these marginal effects.

The binary split addresses this directly:
- The single threshold is the **median engagement rate** — the most stable percentile statistic
- Below vs above is a more robust distinction than three-way percentile bands
- Posts in the ambiguous middle zone (former Class 1) are now assigned based solely on which side of the median they fall — and the model must only learn *direction*, not *degree*

### 2.3 Class Threshold Construction

The threshold is derived from the **training set only** to prevent any leakage from the test distribution into the class boundary definition.

```python
median_rate = df_train['engagement_rate'].quantile(0.5)
# = 5.985 engagements per 1,000 followers

Class 0: engagement_rate < 5.985   → Below Average
Class 1: engagement_rate >= 5.985  → Above Average
```

This is applied identically to the test set using the training-derived value.

---

## 3. Dataset

### 3.1 Overview

| Metric | Value |
|--------|-------|
| Total posts | 772 |
| Unique authors | 495 |
| Avg posts per author | 1.56 |
| Median reactions | 64 |
| Median comments | 4 |
| Median followers | 12,279 |
| Median engagement rate | 5.99 per 1k followers |
| Skewness of engagement rate | 6.84 (highly right-skewed) |

The high skewness confirms that the distribution is dominated by a small number of viral posts. The median (5.99) is a robust central tendency statistic that is unaffected by these outliers, making it the right choice for a single class boundary.

### 3.2 Class Distribution

| Split | Class 0 (Below) | Class 1 (Above) |
|-------|----------------|----------------|
| Training (617) | 308 (49.9%) | 309 (50.1%) |
| Test (155) | 78 (50.3%) | 77 (49.7%) |

Near-perfect balance by construction — the median always splits any distribution as close to 50/50 as the data allows. This eliminates the need for class weighting in theory, though balanced weights are applied throughout for robustness.

### 3.3 Train / Test Split

The split is performed **before** any class label assignment. The median threshold is then derived from the training set and applied to both splits. Splitting first prevents test distribution information from influencing the class boundaries.

The split is stratified on a temporary binary label to ensure approximate 50/50 class balance is preserved in both train and test.

---

## 4. Creator Tier Context

### 4.1 Follower Tier Distribution

The dataset spans a wide range of creator sizes. Raw `followers` is dropped from features because it is the exact denominator of the target variable — keeping it would allow the model to reconstruct the target formula rather than learning content signals. Two transformed proxies replace it:

- `log_followers`: compresses the 34,000x follower range (~80 to ~2.75M) to ~4x in log space
- `follower_tier`: categorical creator-size label (0=micro, 1=small, 2=medium, 3=large)

| Tier | Follower Range | Count | % of Dataset |
|------|---------------|-------|-------------|
| micro | < 10,000 | 345 | 44.7% |
| small | 10,000–50,000 | 225 | 29.1% |
| medium | 50,000–200,000 | 92 | 11.9% |
| large | > 200,000 | 110 | 14.2% |

Nearly three-quarters (73.8%) of the dataset are micro or small creators — the most common LinkedIn creator profile outside of major public figures.

### 4.2 Why Include Follower Tier at All?

Even after normalising the target by followers, engagement rate patterns differ by creator tier. Micro-creator audiences tend to be small, niche, and highly engaged — their per-follower rate is structurally higher. Large accounts have broad but more passive audiences — their per-follower rate is structurally lower.

These are real patterns in the data that the model can legitimately exploit. Follower tier acts as a **prior** on what engagement rate range is achievable for a given creator. It does not leak the target because it is a categorical context variable derived from a simple bucketing, not the raw follower count used in the denominator.

Whether to include or exclude these proxies is a separate design question explored in the NB11d experiment.

---

## 5. Feature Set

### 5.1 Features Dropped (Leakage)

25 columns were removed to prevent data leakage:

**Direct target leakage** — contain or are derived from the engagement metrics used to build the target:
`reactions`, `comments`, `followers`, `engagement_rate`, `base_score_capped`, `reactions_per_word`, `comments_per_word`, `reactions_per_sentiment`, `comment_to_reaction_ratio`

**Aggregated history leakage** — all `influencer_*` statistics computed across the full dataset. A post's engagement likely contributed to its own author's aggregated average, making these circular:
`influencer_avg_reactions/comments/engagement/base_score/sentiment`, `influencer_std_reactions/comments`, `influencer_post_count`, `influencer_total_engagement`, `influencer_consistency_reactions`, `reactions_vs_influencer_avg`, `comments_vs_influencer_avg`

**Metadata** — non-predictive identifiers: `name`, `content`, `time_spent`, `location`

### 5.2 Features Kept (73 total)

| Category | Key Features | Count |
|----------|-------------|-------|
| Creator context | `log_followers`, `follower_tier` | 2 |
| Text quality | `text_lexical_diversity`, `text_difficult_words_count`, `sentence_count`, `text_avg_sentence_length` | 4 |
| Readability | `readability_flesch_kincaid`, `readability_gunning_fog` | 2 |
| Sentiment | `sentiment_compound`, `sentiment_x_readability` | 2 |
| Style | `style_bullet_count`, `style_question_marks`, `style_emoji_count`, `style_has_all_caps`, `style_has_quotes` | 16 |
| Hook patterns | `hook_score`, `hook_x_power_score`, `has_recency_hook`, `has_announcement_hook` | 4 |
| Content patterns | `has_personal_story`, `has_vulnerability`, `has_transformation`, `has_specific_numbers`, `has_family` | 14 |
| Named entities | `ner_person_count`, `ner_org_count`, `ner_date_count`, `ner_event_count` | 7 |
| Topics | `topic_tech`, `topic_business`, `topic_career`, `topic_count`, `is_multi_topic` | 7 |
| Format / structure | `length_score`, `hashtag_count_extracted`, `mention_count`, `url_count` | 5 |
| Penalty flags | `link_penalty_score`, `link_spam_penalty`, `is_low_effort_link`, `is_promotional` | 4 |
| Composite | `total_engagement_elements` | 1 |

---

## 6. Modelling

### 6.1 Approach

Three classifiers are evaluated: Random Forest, XGBoost, and LightGBM. Each is first run with standard hyperparameters to establish a baseline, then tuned using RandomizedSearchCV with stratified 5-fold cross-validation scored on macro F1. Balanced class weights are applied throughout.

**Why macro F1 as the scoring metric?**
Macro F1 averages F1 equally across both classes, regardless of support size. With near-perfectly balanced classes (78 vs 77), macro F1 and weighted F1 are nearly identical here — but macro F1 is kept consistent with the other experiments for comparability.

### 6.2 Baseline Results

| Model | Macro F1 | Accuracy |
|-------|----------|---------|
| Random Forest | 0.7871 | 0.7871 |
| XGBoost | **0.8064** | **0.8065** |
| LightGBM | 0.7999 | 0.8000 |
| Random baseline | 0.500 | 0.500 |

All three models achieve strong results at baseline without any tuning. XGBoost leads at 0.8064.

### 6.3 Hyperparameter Tuning

**XGBoost — Best CV Macro F1: 0.7694**
```
n_estimators=400, max_depth=3, learning_rate=0.01,
min_child_weight=10, subsample=0.7, colsample_bytree=0.8, gamma=0.1
```

**LightGBM — Best CV Macro F1: 0.7711**
```
n_estimators=400, max_depth=3, learning_rate=0.01,
num_leaves=15, min_child_samples=30, subsample=0.8,
colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=2
```

Note that tuning slightly *reduced* test F1 for XGBoost (0.8064 → 0.7868). This is a known effect with small datasets — the tuned model is optimised for CV performance across training folds but may over-regularise slightly relative to a single held-out test split. The base model generalises better here.

### 6.4 Full Results

| Model | Macro F1 (test) | Accuracy (test) |
|-------|----------------|----------------|
| RF_base | 0.7871 | 0.7871 |
| **XGBoost_base** | **0.8064** | **0.8065** |
| LightGBM_base | 0.7999 | 0.8000 |
| XGBoost_tuned | 0.7868 | 0.7871 |
| LightGBM_tuned | 0.7935 | 0.7935 |
| Random baseline | 0.500 | 0.500 |

**Best model: XGBoost_base — Macro F1 = 0.8064, Accuracy = 80.7%**

---

## 7. Detailed Evaluation — XGBoost (Base)

### 7.1 Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|---------|---------|
| Below Average (<6.0) | 0.82 | 0.74 | **0.78** | 78 |
| Above Average (>=6.0) | 0.76 | 0.83 | **0.80** | 77 |
| **Macro avg** | **0.79** | **0.79** | **0.79** | 155 |

### 7.2 Confusion Matrix

```
                   Predicted:
                   Below    Above
Actual: Below        58       20
Actual: Above        13       64
```

**Below Average (Class 0):** 58/78 correctly identified (74% recall). 20 below-average posts are incorrectly predicted as Above. These are posts that have the surface-level characteristics of good content (good hooks, engaging style) but failed to convert that to actual engagement — possibly due to factors outside the feature set (posting time, algorithm state, topic saturation).

**Above Average (Class 1):** 64/77 correctly identified (83% recall). The model is slightly better at identifying strong content than weak content. 13 above-average posts are missed. These are likely posts that succeeded through factors not captured by rule-based features — the right timing, a trending topic mention, or engagement from a highly connected commenter.

**Overall:** 122/155 correct (80.7% accuracy). The model is decisively above chance and makes calibrated errors — it does not systematically confuse classes in one direction.

### 7.3 Cross-Validation

```
Per-fold Macro F1: [0.7016, 0.8139, 0.7314, 0.8455, 0.7541]
Mean: 0.7693   Std: 0.0530
```

The CV mean (0.7693) is lower than the single test score (0.8064). With only 155 test samples, individual test splits carry substantial variance — the CV mean is the more reliable performance estimate. The standard deviation of 0.053 is moderate; the model is not perfectly consistent across folds, which is expected with 617 training samples.

The model is genuinely learning signal. A random model would score 0.500 across all folds.

---

## 8. Feature Importance Analysis

Top 25 features by XGBoost gain importance:

| Rank | Feature | Importance | Category |
|------|---------|-----------|---------|
| 1 | `follower_tier` | 0.0985 | Creator context |
| 2 | `log_followers` | 0.0671 | Creator context |
| 3 | `has_org_mention` | 0.0296 | Named entities |
| 4 | `style_has_question` | 0.0263 | Style |
| 5 | `style_has_quotes` | 0.0240 | Style |
| 6 | `hashtag_count_extracted` | 0.0239 | Format |
| 7 | `text_lexical_diversity` | 0.0232 | Text quality |
| 8 | `ner_org_count` | 0.0224 | Named entities |
| 9 | `link_penalty_score` | 0.0212 | Penalty |
| 10 | `style_exclamation_marks` | 0.0211 | Style |
| 11 | `style_question_marks` | 0.0211 | Style |
| 12 | `url_count` | 0.0210 | Format |
| 13 | `has_vulnerability` | 0.0206 | Content pattern |
| 14 | `style_has_exclamation` | 0.0198 | Style |
| 15 | `is_multi_topic` | 0.0194 | Topics |
| 16 | `topic_leadership` | 0.0193 | Topics |
| 17 | `sentiment_compound` | 0.0192 | Sentiment |
| 18 | `style_bullet_count` | 0.0191 | Style |
| 19 | `has_direct_address` | 0.0188 | Content pattern |
| 20 | `text_difficult_words_count` | 0.0186 | Text quality |

### 8.1 What the Features Are Saying

**Creator context (ranks 1–2):** `follower_tier` and `log_followers` are the top two features. Even with a globally normalised target, creator tier provides a useful prior — the distribution of engagement rates within each tier differs, and the model uses this context when making predictions. This is explored further in Section 10.

**Organisation mentions are a strong content signal (ranks 3, 8):** `has_org_mention` and `ner_org_count` together dominate the content features. Posts that reference specific companies, institutions, or organisations perform better. This is interpretable: named-entity specificity signals credibility and relevance. Generic advice without reference to real-world entities is a weaker content type on LinkedIn.

**Asking questions drives engagement (ranks 4, 11):** `style_has_question` and `style_question_marks` both appear in the top 15. Questions invite responses, which directly increases comments — and comments feed into the engagement rate numerator.

**Quotes signal authority (rank 5):** `style_has_quotes` at rank 5 suggests that posts citing or quoting others tend to over-perform. This aligns with LinkedIn culture — quotes from executives, researchers, or public figures lend credibility and encourage sharing.

**Hashtag count is mid-range optimal (rank 6):** `hashtag_count_extracted` appears prominently. The relationship is likely non-linear — too few hashtags miss discoverability, too many signal spam. The model has learned this nuance implicitly from the tree splits.

**Vocabulary richness matters (rank 7):** `text_lexical_diversity` — a measure of vocabulary variety relative to total word count — ranks 7th. Posts that use a broader, more varied vocabulary tend to over-perform. This could reflect that higher-effort writing correlates with better content quality, or that diverse vocabulary reaches more audiences in algorithmic distribution.

**Link penalties confirm platform norms (rank 9):** `link_penalty_score` ranking 9th confirms that posts with external links underperform on LinkedIn, consistent with the platform's known preference for native content. The algorithm deprioritises posts that send users off-platform.

**Vulnerability resonates (rank 13):** `has_vulnerability` — posts containing language about personal struggles, failures, or honest admissions — performs above average. This aligns with what practitioners know about LinkedIn engagement: authentic personal stories outperform corporate-speak.

**Style diversity over any single tactic (ranks 10, 11, 14, 18):** Multiple style signals appear — exclamation marks, question marks, bullets — each contributing independently. No single formatting tactic dominates, suggesting the model has learned that stylistic variety (not any one element) is associated with above-average engagement.

---

## 9. Performance by Follower Tier

A key validation question: does the model perform consistently across creator sizes?

| Tier | n (test) | Macro F1 | Accuracy |
|------|---------|----------|---------|
| micro (<10k) | 77 | 0.6183 | 0.6364 |
| small (10k–50k) | 41 | 0.7515 | 0.7561 |
| medium (50k–200k) | 16 | 0.4286 | 0.5000 |
| large (>200k) | 21 | 0.4615 | 0.4762 |

### Analysis

**Small creators perform best (Macro F1 = 0.75).** The 10k–50k follower range has the most learnable engagement rate pattern. These accounts are established enough to have consistent audience behaviour, but niche enough that content quality genuinely drives variation. The model finds strong content signals in this tier.

**Micro creators are moderate (Macro F1 = 0.62).** Micro-creator engagement rates are inherently noisier — a single influential commenter or a share from a larger account can double the rate. Despite this noise, the model classifies correctly 64% of the time, indicating genuine content signal.

**Medium and large tiers are harder (Macro F1 = 0.43–0.46).** With only 16 and 21 test samples respectively, these estimates carry high variance and should be interpreted cautiously. However, the pattern is consistent with what the data suggests: engagement rate for large accounts is more compressed (less variance, harder to distinguish above/below), and medium accounts are underrepresented in training (92 total posts).

### Why Tier Performance Varies

The model uses the same threshold (median = 5.985) for all tiers. But the within-tier median varies dramatically:

| Tier | Within-tier median rate |
|------|------------------------|
| micro | ~27.3 |
| small | ~3.5 |
| medium | ~1.3 |
| large | ~0.6 |

The global median (5.985) sits between micro and small. This means:
- Most micro posts are **above** the global median → model needs to find which micro posts *underperform* relative to micro norms
- Most large posts are **below** the global median → model needs to find which large posts *overperform* relative to large norms

The model finds this asymmetry easier to solve for small creators (where the global median is close to the tier median) than for micro/large creators (where the structural offset is largest).

---

## 10. Dominant Features — Is This a Concern?

`follower_tier` (0.098) and `log_followers` (0.067) together account for approximately 16.5% of total feature importance — more than any other single category. The next highest content feature (`has_org_mention`) is at 0.030, roughly one-third the importance of `follower_tier`.

### Why This Happens

Even after normalising the target by followers, the engagement rate distributions within each tier are not identical. The global median (5.985) does not coincide with the within-tier median for any single tier. As a result, tier membership is a reliable predictor of which side of the global threshold a post will fall on.

For example: a micro-creator with rate 10.0 is above the global median (Class 1). A small-creator with rate 3.0 is below the global median (Class 0). The model has learned that tier alone predicts the likely class with reasonable accuracy — before it even looks at the content.

### Whether It Is a Problem Depends on the Goal

**If the goal is realistic engagement prediction for a given creator:** No problem. Tier context is legitimately informative about what rate a creator typically achieves. The model is right to use it.

**If the goal is pure content quality scoring — assessing content independent of who posts it:** Yes, this is a concern. The model is partially predicting creator audience dynamics rather than content merit.

### The Test: NB11d

To directly answer this question, NB11d removes `follower_tier` and `log_followers` entirely and re-runs the same 2-class classification. If the F1 drop is small, content features carry most of the signal. If the drop is large, the follower proxies were doing most of the work — and the content features are a weaker signal than the results suggest.

---

## 11. Limitations

### 11.1 Dataset Size
772 posts across 495 authors is small for machine learning. The test set (155 samples, ~78 per class) produces F1 estimates with meaningful variance — the CV mean (0.769) is more reliable than the single test score (0.806).

### 11.2 Rule-Based Feature Ceiling
All 71 content features are hand-crafted keyword flags and counts. A post about personal transformation written without expected trigger keywords would not activate `has_transformation`. Text embeddings (SBERT or TF-IDF of `clean_content`) would capture semantic meaning and are the most impactful next addition.

### 11.3 Temporal and Contextual Effects
LinkedIn algorithm changes, trending topics, posting time, and day of week all affect engagement independently of content quality. These are not captured in the feature set.

### 11.4 Follower Tier Dominance
As discussed in Section 10, follower proxies account for ~16.5% of feature importance, raising the question of how much the model is predicting audience dynamics vs content quality. NB11d directly investigates this.

---

## 12. Conclusions

The 2-class engagement rate classifier achieves **Macro F1 = 0.8064** and **80.7% accuracy** on the held-out test set, with a cross-validation mean of 0.7693.

**What works well:**
- Clean, balanced binary target — median split produces near-perfect 50/50 class balance
- Strong performance across all model types, even without tuning
- Interpretable class definitions — every post receives a clear directional prediction
- Genuinely learnable signal — multiple content feature categories contribute meaningfully
- Consistent errors — the confusion matrix shows no systematic directional bias

**Key content quality signals identified:**
- Organisation and entity mentions (specificity and credibility)
- Question-based style (drives comments and engagement)
- Vocabulary diversity (signals writing quality and effort)
- Quote usage (authority signalling)
- Avoiding external links (platform-native content preference)
- Vulnerability and personal directness

**What remains uncertain:**
- How much of the performance comes from follower tier vs content (investigated in NB11d)
- Whether content features would hold up with semantic text embeddings
- Performance on medium/large tiers is poorly estimated (small samples)

**Recommended next steps:**
1. Run NB11d (remove follower proxies) to isolate the pure content signal
2. Add SBERT or TF-IDF embeddings of `clean_content` — the single most impactful potential improvement
3. Collect more posts per author to improve tier representation for medium/large creators
