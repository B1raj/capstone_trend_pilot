# Engagement Rate Classification Report
## LinkedIn Engagement Prediction — TrendPilot

**Notebook:** `11_engagement_rate_classification.ipynb`
**Date:** February 2026

---

## 1. Objective

The goal of this notebook is to build a **3-class content quality classifier** that predicts whether a LinkedIn post will perform below average, at average, or above average engagement — measured fairly across creators of any audience size.

The core question being answered is:

> **"Is this content genuinely good, regardless of who posted it?"**

This is distinct from predicting raw engagement numbers, which would simply reflect audience size. The intended use case is a content scoring tool: a creator drafts a post, and the model assesses whether the content itself carries the characteristics of a high-performing post, before publishing.

---

## 2. The Challenge: Follower Count Dominates Raw Engagement

On LinkedIn, a post with 500 reactions is exceptional for a 2,000-follower account but unremarkable for a 2,000,000-follower account. Any model trained directly on raw engagement numbers ends up learning audience size rather than content quality.

The solution is to normalise engagement by follower count, putting all creators on the same scale and allowing the model to focus on what the content achieves *relative to the audience it reaches*.

---

## 3. Approach: Engagement Rate Normalisation

### 3.1 Target Variable

```python
engagement_rate = (reactions + comments) / (followers / 1000)
```

This expresses total engagement as a **rate per 1,000 followers**. Two examples:

- Nano-influencer: 5,000 followers, 50 reactions + 10 comments → rate = 60 / 5 = **12.0**
- Macro-influencer: 500,000 followers, 5,000 reactions + 1,000 comments → rate = 6,000 / 500 = **12.0**

Both produced equally resonant content. The rate reflects that equivalence.

Both reactions and comments are included in the numerator. A post that drives conversation is genuinely more engaging — combined they form a more complete picture of audience response than reactions alone.

### 3.2 Dataset Statistics

| Metric | Value |
|--------|-------|
| Total posts | 772 |
| Unique authors | 495 |
| Avg posts per author | 1.56 |
| Median reactions | 64 |
| Median comments | 4 |
| Median followers | 12,279 |
| Median engagement rate | 5.88 per 1k followers |
| Mean engagement rate | 31.31 per 1k followers |
| Max engagement rate | 1,132.07 per 1k followers |
| Skewness of rate | 6.84 (highly right-skewed) |

The large gap between median (5.88) and mean (31.31) confirms the distribution is dominated by a small number of viral outliers. This validates the choice of percentile-based class thresholds over mean-based ones.

### 3.3 Train / Test Split

The dataset was split **before** class labels were assigned (80% train, 20% test, stratified). Class thresholds are then derived exclusively from the training set and applied to the test set. This prevents any leakage from the test distribution into the class boundary definitions.

| Split | Posts |
|-------|-------|
| Training | 617 (80%) |
| Test | 155 (20%) |

### 3.4 Class Definitions

Thresholds are the 33rd and 67th percentiles of the training set engagement rate distribution, producing near-perfectly balanced classes.

| Class | Threshold | Meaning |
|-------|-----------|---------|
| 0 — Below Average | rate < 2.20 per 1k followers | Under-performing relative to audience size |
| 1 — Average | 2.20 ≤ rate ≤ 15.16 per 1k | In line with typical creator performance |
| 2 — Above Average | rate > 15.16 per 1k followers | Significantly over-performing — strong content quality signal |

### 3.5 Class Distribution

| Split | Class 0 (Below) | Class 1 (Average) | Class 2 (Above) |
|-------|----------------|------------------|----------------|
| Training (617) | 206 (33.4%) | 205 (33.2%) | 206 (33.4%) |
| Test (155) | 54 (34.8%) | 48 (31.0%) | 53 (34.2%) |

Training classes are near-perfectly balanced. The test set drifts slightly, as expected when fixed thresholds are applied to a different sample.

---

## 4. Creator Tier Distribution

Raw `followers` is dropped as a feature — it is the exact denominator used to construct the target variable, so keeping it would allow the model to partially reconstruct the target formula rather than learning content signals. Two transformed proxies replace it:

- `log_followers` — compresses the 34,000x follower range (~80 to ~2.75M) to ~4x in log space
- `follower_tier` — categorical creator-size context (0=micro, 1=small, 2=medium, 3=large)

| Tier | Follower Range | Count | % of Dataset |
|------|---------------|-------|-------------|
| micro | < 10k | 345 | 44.7% |
| small | 10k–50k | 225 | 29.1% |
| medium | 50k–200k | 92 | 11.9% |
| large | > 200k | 110 | 14.2% |

Nearly three-quarters of the dataset (73.8%) are micro or small creators, which is typical of a LinkedIn dataset collected without bias toward high-profile accounts.

---

## 5. Feature Set

### 5.1 Features Dropped

25 columns were removed across three categories to prevent data leakage:

**Direct target leakage** — columns that contain or are derived from the engagement metrics used to build the target:
`reactions`, `comments`, `followers`, `engagement_rate`, `base_score_capped`, `reactions_per_word`, `comments_per_word`, `reactions_per_sentiment`, `comment_to_reaction_ratio`

**Aggregated history leakage** — all `influencer_*` statistics, `reactions_vs_influencer_avg`, `comments_vs_influencer_avg`, `influencer_consistency_reactions`. These aggregates were computed across the whole dataset, meaning a post may have contributed to its own author's average — a form of circular reasoning.

**Metadata** — non-learnable identifiers and session data: `name`, `content`, `time_spent`, `location`

### 5.2 Features Kept (73 total)

All retained features are pure content and creator-context signals:

| Category | Examples | Count |
|----------|---------|-------|
| Text quality | `text_lexical_diversity`, `text_difficult_words_count`, `sentence_count`, `text_avg_sentence_length` | 4 |
| Readability | `readability_flesch_kincaid`, `readability_gunning_fog` | 2 |
| Sentiment | `sentiment_compound`, `sentiment_x_readability` | 2 |
| Style | `style_bullet_count`, `style_question_marks`, `style_emoji_count`, `style_has_all_caps`, `style_has_quotes` | 16 |
| Hook patterns | `hook_score`, `hook_x_power_score`, `has_recency_hook`, `has_announcement_hook` | 4 |
| Content patterns | `has_personal_story`, `has_vulnerability`, `has_transformation`, `has_specific_numbers`, `has_family` | 14 |
| Named entities | `ner_person_count`, `ner_org_count`, `ner_date_count`, `ner_event_count`, `ner_money_count` | 7 |
| Topics | `topic_tech`, `topic_business`, `topic_career`, `topic_count`, `is_multi_topic` | 7 |
| Format / structure | `length_score`, `hashtag_count_extracted`, `mention_count`, `url_count` | 5 |
| Penalty flags | `link_penalty_score`, `link_spam_penalty`, `is_low_effort_link`, `is_promotional` | 4 |
| Follower proxies | `log_followers`, `follower_tier` | 2 |
| Composite | `total_engagement_elements` | 1 |

---

## 6. Modelling

Three classifiers were evaluated — Random Forest, XGBoost, and LightGBM — first with standard hyperparameters to establish baselines, then tuned via RandomizedSearchCV with stratified 5-fold cross-validation scored on macro F1. Balanced class weights were applied throughout to prevent any class from dominating training.

### 6.1 Baseline Results

| Model | Macro F1 | Accuracy |
|-------|----------|---------|
| Random Forest | 0.5124 | 0.5290 |
| XGBoost | 0.5660 | 0.5677 |
| LightGBM | 0.5459 | 0.5484 |
| Random baseline | ~0.333 | ~0.333 |

All three models substantially outperform random guessing at baseline. XGBoost leads.

### 6.2 Hyperparameter Tuning

**XGBoost — Best CV Macro F1: 0.6091**
```
n_estimators=600, max_depth=6, learning_rate=0.05,
min_child_weight=3, subsample=0.9, colsample_bytree=0.6, gamma=0.3
```

**LightGBM — Best CV Macro F1: 0.5982**
```
n_estimators=600, max_depth=5, learning_rate=0.1,
num_leaves=20, min_child_samples=20, subsample=0.9,
colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1
```

### 6.3 Full Results Summary

| Model | Macro F1 (test) | Accuracy (test) |
|-------|----------------|----------------|
| RandomForest_base | 0.5124 | 0.5290 |
| XGBoost_base | 0.5660 | 0.5677 |
| LightGBM_base | 0.5459 | 0.5484 |
| **XGBoost_tuned** | **0.5734** | **0.5742** |
| LightGBM_tuned | 0.5655 | 0.5677 |
| Random baseline | ~0.333 | ~0.333 |

**Best model: XGBoost (tuned) — Macro F1 = 0.5734, Accuracy = 57.4%**

---

## 7. Detailed Evaluation — XGBoost (Tuned)

### 7.1 Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|---------|---------|
| Below Average (<2.2) | 0.64 | 0.67 | **0.65** | 54 |
| Average (2.2–15.2) | 0.43 | 0.50 | **0.46** | 48 |
| Above Average (>15.2) | 0.67 | 0.55 | **0.60** | 53 |
| **Macro avg** | **0.58** | **0.57** | **0.57** | 155 |

### 7.2 Confusion Matrix

```
                   Predicted:
                   Below   Average   Above
Actual: Below        36       14       4
Actual: Average      14       24      10
Actual: Above         6       18      29
```

**Below Average (Class 0):** 36 of 54 correctly identified (67% recall). The majority of errors fall into Average (14), with very few falsely predicted as Above Average (4). The model rarely praises genuinely poor content — a useful property for a content scoring tool.

**Average (Class 1):** The hardest class with only 24 of 48 correct (50% recall). This is structurally expected — average content sits on both decision boundaries and can legitimately tip either direction. Errors split roughly evenly toward Below (14) and Above (10), showing no systematic directional bias.

**Above Average (Class 2):** 29 of 53 correctly identified (55% recall). The 18 misclassified as Average represent content that resonated despite lacking strong detectable rule-based signals — likely driven by factors outside the feature set such as timing, trending topics, or audience-specific resonance that keyword-based features cannot capture.

### 7.3 Cross-Validation

```
Per-fold Macro F1: [0.6494, 0.5965, 0.5971, 0.6255, 0.5433]
Mean: 0.6023   Std: 0.0355
```

The CV mean (0.6023) is somewhat higher than the single held-out test score (0.5734). With only 155 test samples (~50 per class), individual test splits carry high variance. The CV mean across the full training set is the more reliable estimate of true generalisation performance. A standard deviation of 0.035 indicates the model is moderately stable across different data slices.

---

## 8. Feature Importance Analysis

Top 25 features ranked by XGBoost gain importance:

| Rank | Feature | Importance | Category |
|------|---------|-----------|---------|
| 1 | `follower_tier` | 0.0774 | Creator context |
| 2 | `log_followers` | 0.0359 | Creator context |
| 3 | `has_recency_hook` | 0.0287 | Hook pattern |
| 4 | `has_entities` | 0.0244 | Named entities |
| 5 | `is_promotional` | 0.0215 | Content type |
| 6 | `style_question_marks` | 0.0175 | Style |
| 7 | `ner_event_count` | 0.0171 | Named entities |
| 8 | `url_count` | 0.0167 | Format |
| 9 | `has_org_mention` | 0.0159 | Named entities |
| 10 | `text_lexical_diversity` | 0.0159 | Text quality |
| 11 | `unique_emoji_count` | 0.0153 | Style |
| 12 | `ner_date_count` | 0.0150 | Named entities |
| 13 | `link_penalty_score` | 0.0150 | Penalty |
| 14 | `length_score` | 0.0149 | Format |
| 15 | `topic_count` | 0.0149 | Topics |
| 16 | `text_difficult_words_count` | 0.0147 | Text quality |
| 17 | `has_family` | 0.0147 | Content pattern |
| 18 | `style_emoji_count` | 0.0147 | Style |
| 19 | `style_has_question` | 0.0146 | Style |
| 20 | `style_bullet_count` | 0.0143 | Style |

### Key Findings

**Creator context leads (ranks 1–2).** `follower_tier` and `log_followers` are the most important features. Even after normalisation, engagement rate patterns differ meaningfully by creator tier. Micro-creators tend to have highly variable rates, while larger creators are more consistent. The model uses creator tier as a prior on what rate range is achievable — this is a legitimate and expected signal.

**Hook quality is the strongest pure content signal (rank 3).** `has_recency_hook` — posts framed with a current, timely angle — is the top-ranked content feature. LinkedIn's algorithm rewards posts that tap into trending topics, and the data confirms this produces higher engagement rates.

**Grounded, specific content performs better (ranks 4, 7, 9, 12).** The cluster of named entity features — `has_entities`, `ner_event_count`, `has_org_mention`, `ner_date_count` — suggests that posts referencing real organisations, events, and dates outperform generic advice. Specificity signals credibility and relevance.

**Promotional content is consistently penalised (rank 5).** `is_promotional` is the fifth most important feature. Promotional posts reliably underperform on engagement rate, reflecting LinkedIn audience behaviour — users engage less with content perceived as selling rather than sharing.

**Style signals are distributed, not concentrated (ranks 6, 11, 18–20).** No single style feature dominates; instead, a cluster of signals — question marks, emojis, bullets — each contribute modest but consistent importance. This indicates that style quality is multidimensional and no single tactic is decisive on its own.

**Vocabulary quality matters (ranks 10, 16).** `text_lexical_diversity` and `text_difficult_words_count` appear mid-table, suggesting that varied, substantive language is a mild positive signal. Content using a richer vocabulary tends to perform marginally better.

---

## 9. Performance by Follower Tier

A key validation question is whether the model is fair across creator sizes — i.e., can it assess content quality for both a micro-influencer and a large account without being systematically biased toward either.

| Tier | n (test) | Macro F1 | Accuracy | Class distribution in test |
|------|---------|----------|---------|--------------------------|
| micro (<10k) | 63 | 0.3942 | 0.6032 | {Below: 6, Avg: 20, Above: 37} |
| small (10k–50k) | 51 | 0.3016 | 0.3922 | {Below: 20, Avg: 15, Above: 16} |
| medium (50k–200k) | 17 | 0.3630 | 0.5294 | {Below: 10, Avg: 7} |
| large (>200k) | 24 | **0.8889** | **0.9167** | {Below: 18, Avg: 6} |

### Analysis

**Large creators score 0.89 Macro F1** — an unusually high result. Examining the class distribution reveals why: 18 of 24 large-creator test posts are Class 0 (below average rate). Large accounts have inherently low per-follower engagement rates because their massive audiences are more passive. The model has correctly learned this pattern, but it is learning an audience-dynamics effect rather than content quality per se.

**Micro creators score 0.39 Macro F1** with high accuracy (0.60). The inverse pattern applies — 37 of 63 micro-creator posts are Class 2. Small, engaged niche audiences generate high per-follower rates almost regardless of content. Again the model learns the tier dynamic accurately, but it is not purely assessing the content.

**Small creators are the hardest tier (Macro F1 = 0.30).** This tier has the most balanced class distribution and the most genuine ambiguity. Content quality has its most unpredictable relationship to engagement rate here — this is where the model would need to rely most heavily on pure content signals, and where those signals are currently weakest.

**Medium creators (Macro F1 = 0.36)** have only 17 test samples — too small for a reliable estimate.

### What This Reveals

The tier evaluation exposes a **structural limitation of engagement rate normalisation**. While dividing by followers removes the gross follower-count bias, it does not fully decouple engagement from audience-type effects. Micro-creator audiences on LinkedIn are inherently more engaged per follower; large-creator audiences are inherently more passive per follower. The model partially learns these structural patterns rather than purely assessing content quality — particularly at the tier extremes.

This is not necessarily wrong: tier context is a real and relevant signal for what engagement rate a given post can achieve. But it does mean the model is not a pure content quality judge.

---

## 10. Limitations

### 10.1 Dataset Size
772 posts across 495 authors is small for 3-class classification. With 155 test samples (~50 per class), evaluation metrics carry high variance. The CV mean (0.60) is the more reliable performance estimate.

### 10.2 Engagement Rate Noise at Small Scale
For accounts with very few followers, a single extra reaction can shift the engagement rate dramatically. The metric becomes more stable and meaningful at larger follower counts. Some of the high variance in the micro tier likely reflects this noise.

### 10.3 Rule-Based Content Features
All 73 content features are hand-crafted keyword flags and counts. They capture surface-level signals well but miss semantic meaning. A post about personal transformation written without the expected trigger words would not activate `has_transformation`. Text embeddings (SBERT, TF-IDF) would capture semantic content quality that the current feature set cannot reach.

### 10.4 Tier Confounding
As the tier evaluation shows, the model partially learns audience-type dynamics rather than pure content quality. Follower count is simultaneously a confound to control and a legitimate context signal — fully separating the two is not straightforward with this dataset size and feature set.

### 10.5 Temporal and Contextual Effects Not Modelled
LinkedIn algorithm changes, trending topics, posting time of day, and day of week all affect engagement independently of content quality. None of these signals are captured.

---

## 11. Conclusions

This notebook builds a follower-normalised engagement rate classifier that fairly compares content across creators of any audience size. The approach produces a well-calibrated 3-class target, clean leakage controls, and a practical content quality signal.

**What works well:**
- Balanced, interpretable class definitions (below / average / above average engagements per 1k followers)
- Strong overall performance — Macro F1 = 0.57 on test, CV mean = 0.60, well above the 0.33 random baseline
- Works for any creator without requiring post history — no cold-start problem
- Clean separation of leaky signals from content features
- Identified interpretable drivers: recency hooks, named entities, promotional penalty, style diversity

**What remains difficult:**
- Class 1 (Average) is structurally hard — only 50% recall, as these posts sit on both decision boundaries
- Tier-level performance is uneven — strong at the extremes (large/micro) partly for the wrong reasons
- Feature set is capped by rule-based signals — semantic meaning is not captured

**Recommended next step:** Add TF-IDF or SBERT sentence embeddings of `clean_content` as additional features. The current rule-based features represent a performance ceiling — semantic embeddings would unlock the content quality signal that keyword counting cannot access, and are the most likely avenue for meaningful further improvement.

---

## Appendix: Best Hyperparameters

### XGBoost (Best Model — Macro F1 = 0.5734)
```
n_estimators:     600
max_depth:        6
learning_rate:    0.05
min_child_weight: 3
subsample:        0.9
colsample_bytree: 0.6
gamma:            0.3
eval_metric:      mlogloss
```

### LightGBM
```
n_estimators:      600
max_depth:         5
learning_rate:     0.1
num_leaves:        20
min_child_samples: 20
subsample:         0.9
colsample_bytree:  0.8
reg_alpha:         0.1
reg_lambda:        1
```
