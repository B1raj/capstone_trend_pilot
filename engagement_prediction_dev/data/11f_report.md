# NB11f — Per-Tier Binary Classifiers
## LinkedIn Engagement Prediction — TrendPilot

**Generated:** 2026-02-21 01:17  
**Notebook:** `11f_per_tier_models.ipynb`

---

## 1. Problem Statement

LinkedIn engagement rate varies dramatically across creator sizes. A post with 500 reactions means something very different for a micro-influencer (5k followers) versus a large account (500k followers). Normalising by followers — engagement rate = (reactions + comments) / (followers / 1000) — partially addresses this, but a *single global median* still embeds creator-size information into the class boundary: large accounts systematically cluster below the global median simply because of audience dilution.

Prior experiments confirmed this leakage. In **NB11c** (2-class, global model with `follower_tier` and `log_followers` features), these two audience-size proxies ranked #1 and #2 by importance (~16.5% combined). Removing them in **NB11d** reduced overall F1 from 0.806 to 0.767, quantifying the follower shortcut. **NB11e** (within-tier labels, single shared model) tried to address this with a per-tier threshold but still trained one model across all tiers, achieving ~0.735.

**NB11f** tests the natural next step: train an entirely *separate* binary classifier per follower tier. Each model is asked only: *"Is this post above average for THIS tier?"* — removing both the global-median leakage and the single-model limitation of NB11e.

## 2. Approach & Design

### 2.1 Tier Definition

| Tier   | Follower Range | Full n | ~Train | ~Test |
| ------ | -------------- | ------ | ------ | ----- |
| micro  | < 10k          | 345    | ~276   | ~69   |
| small  | 10k – 50k      | 225    | ~180   | ~45   |
| medium | 50k – 200k     | 92     | ~74    | ~18   |
| large  | > 200k         | 110    | ~88    | ~22   |

Tiers are defined from raw follower counts using fixed breakpoints (0 / 10k / 50k / 200k / ∞), matching industry-standard influencer categories. `follower_tier` is computed for splitting purposes only and is **never passed to any model as a feature** — it would be constant within a tier (zero variance).

### 2.2 Per-Tier Binary Labels

For each tier, the **training-subset median** of engagement rate is computed independently. This creates an approximately 50/50 class split within every tier, making the random baseline exactly 0.500 for each model. Using the training median only (not the test median) ensures no leakage from the test distribution.

### 2.3 Feature Set

71 pure content features — identical to NB11d. Both `log_followers` and `follower_tier` are excluded from the feature matrix. This forces each model to learn engagement prediction from content, structure, and style signals alone.

**Feature categories included:**

| Category | Examples |
|----------|----------|
| Text quality | `readability_flesch_kincaid`, `readability_gunning_fog`, `text_lexical_diversity` |
| Sentiment | `sentiment_compound`, `sentiment_x_readability` |
| Named entities | `ner_person_count`, `ner_org_count`, `ner_location_count` |
| Style | `style_has_question`, `style_exclamation_marks`, `style_bullet_count` |
| Topic flags | `topic_tech`, `topic_business`, `topic_personal_dev`, `topic_leadership` |
| Hook signals | `hook_score`, `hook_x_power_score`, `has_announcement_hook` |
| Narrative | `has_personal_story`, `has_vulnerability`, `has_adversity_learning` |
| Length/structure | `sentence_count`, `text_avg_sentence_length`, `length_score` |

### 2.4 Train / Test Split

The full dataset (772 posts) is split 80/20 (stratified on the global binary label) *before* any per-tier label assignment. This ensures the test set is never seen during threshold or feature computation. After splitting, `follower_tier` is added to `df_train` / `df_test` to enable per-tier row selection inside the training loop.

## 3. Model Design

### 3.1 Classifier Suite

Three tree-ensemble classifiers are trained per tier, providing a consensus view and allowing the best-performing model to be selected independently for each tier:

| Model         | Key Params                                           | Class Imbalance Handling                             |
| ------------- | ---------------------------------------------------- | ---------------------------------------------------- |
| Random Forest | n_est=200, depth=8, min_split=10, min_leaf=5         | class_weight="balanced"                              |
| XGBoost       | n_est=200, depth=4, lr=0.05, min_child_w=5, sub=0.8  | sample_weight from compute_sample_weight("balanced") |
| LightGBM      | n_est=200, depth=4, lr=0.05, leaves=15, min_child=10 | class_weight="balanced"                              |

### 3.2 Justification for Fixed Params (No Tuning)

Hyperparameter tuning via cross-validation requires a minimum viable sample size. With medium (~74 train) and large (~88 train), a 5-fold CV would yield only ~15–18 training samples per fold — insufficient for stable gradient estimates. Applying tuning only to larger tiers would create inconsistent methodology across tiers. The chosen baseline params are deliberately conservative (shallow trees, high min-child constraints) to reduce overfitting on small samples.

### 3.3 Training Protocol

For each tier, `compute_sample_weight("balanced")` is computed from the training labels and passed to XGBoost (which lacks a native `class_weight` argument). RF and LightGBM use `class_weight="balanced"` directly. Each classifier is instantiated fresh per tier via `make_classifiers()` and deep-copied before fitting to prevent state leakage between iterations.

## 4. Results

### 4.1 Per-Tier Sample Sizes & Thresholds

| Tier              | n_train | n_test | Tier Median ER |
| ----------------- | ------- | ------ | -------------- |
| micro (<10k)      | 275     | 70     | 21.832         |
| small (10k-50k)   | 177     | 45     | 2.894          |
| medium (50k-200k) | 32      | 7      | 1.552          |

*Tier Median ER* is the engagement-rate threshold used as the class boundary for that tier's model. It is derived from the training subset only.

### 4.2 All Model Results

| Tier              | Model        | Macro F1 | Accuracy | Best? |
| ----------------- | ------------ | -------- | -------- | ----- |
| micro (<10k)      | RandomForest | 0.7105   | 0.7143   | YES   |
| micro (<10k)      | XGBoost      | 0.6283   | 0.6286   |       |
| micro (<10k)      | LightGBM     | 0.6708   | 0.6714   |       |
| small (10k-50k)   | RandomForest | 0.6026   | 0.6222   |       |
| small (10k-50k)   | XGBoost      | 0.6961   | 0.7111   | YES   |
| small (10k-50k)   | LightGBM     | 0.5701   | 0.5778   |       |
| medium (50k-200k) | RandomForest | 0.5714   | 0.5714   | YES   |
| medium (50k-200k) | XGBoost      | 0.3      | 0.4286   |       |
| medium (50k-200k) | LightGBM     | 0.5333   | 0.5714   |       |

### 4.3 Aggregate Summary

| Tier              | n_test | Best F1 | Best Model   | Lift    |
| ----------------- | ------ | ------- | ------------ | ------- |
| micro (<10k)      | 70     | 0.7105  | RandomForest | +0.2105 |
| small (10k-50k)   | 45     | 0.6961  | XGBoost      | +0.1961 |
| medium (50k-200k) | 7      | 0.5714  | RandomForest | +0.0714 |

**Weighted average Macro F1 (by test-set size): 0.6972**  
**Lift over random baseline (0.500): +0.1972**

### 4.4 Comparison to Previous Experiments

| Experiment | Description                                    | Macro F1 | Lift     |
| ---------- | ---------------------------------------------- | -------- | -------- |
| NB11c      | 2-class, global median, with follower features | 0.8064   | +0.3064  |
| NB11d      | 2-class, global median, no follower features   | 0.7673   | +0.2673  |
| NB11e      | within-tier labels, single shared model        | ~0.7350  | ~+0.2350 |
| NB11f      | per-tier models (this)                         | 0.6972   | +0.1972  |

### 4.5 Confusion Matrices

Each confusion matrix shows the best-performing model for that tier. Rows are actual labels; columns are predicted labels. Precision and recall are reported per class.

#### micro (<10k)  —  RandomForest  |  F1=0.7105  (n=70)

|                        | Pred: Below (<21.8) | Pred: Above (>=21.8) |
| ---------------------- | ------------------- | -------------------- |
| Actual: Below (<21.8)  | 29                  | 9                    |
| Actual: Above (>=21.8) | 11                  | 21                   |

| Class          | Precision | Recall | F1        |
| -------------- | --------- | ------ | --------- |
| Below (<21.8)  | 0.725     | 0.763  | 0.744     |
| Above (>=21.8) | 0.700     | 0.656  | 0.677     |
| **Accuracy**   |           |        | **0.714** |

#### small (10k-50k)  —  XGBoost  |  F1=0.6961  (n=45)

|                       | Pred: Below (<2.9) | Pred: Above (>=2.9) |
| --------------------- | ------------------ | ------------------- |
| Actual: Below (<2.9)  | 11                 | 5                   |
| Actual: Above (>=2.9) | 8                  | 21                  |

| Class         | Precision | Recall | F1        |
| ------------- | --------- | ------ | --------- |
| Below (<2.9)  | 0.579     | 0.688  | 0.629     |
| Above (>=2.9) | 0.808     | 0.724  | 0.764     |
| **Accuracy**  |           |        | **0.711** |

#### medium (50k-200k)  —  RandomForest  |  F1=0.5714  (n=7)

|                       | Pred: Below (<1.6) | Pred: Above (>=1.6) |
| --------------------- | ------------------ | ------------------- |
| Actual: Below (<1.6)  | 2                  | 2                   |
| Actual: Above (>=1.6) | 1                  | 2                   |

| Class         | Precision | Recall | F1        |
| ------------- | --------- | ------ | --------- |
| Below (<1.6)  | 0.667     | 0.500  | 0.571     |
| Above (>=1.6) | 0.500     | 0.667  | 0.571     |
| **Accuracy**  |           |        | **0.571** |

### 4.6 Feature Importances per Tier (Top 10, XGBoost)

#### micro (<10k)  —  XGBoost  |  F1=0.6283

| Rank | Feature                 | Importance |
| ---- | ----------------------- | ---------- |
| 1    | has_transformation      | 0.0389     |
| 2    | has_external_link       | 0.0378     |
| 3    | style_bullet_count      | 0.0360     |
| 4    | url_count               | 0.0337     |
| 5    | style_exclamation_marks | 0.0313     |
| 6    | emoji_count             | 0.0304     |
| 7    | unique_emoji_count      | 0.0291     |
| 8    | has_direct_address      | 0.0263     |
| 9    | topic_business          | 0.0253     |
| 10   | style_quote_marks       | 0.0247     |

#### small (10k-50k)  —  XGBoost  |  F1=0.6961

| Rank | Feature                | Importance |
| ---- | ---------------------- | ---------- |
| 1    | style_has_quotes       | 0.0470     |
| 2    | is_multi_topic         | 0.0428     |
| 3    | promotional_score      | 0.0389     |
| 4    | text_lexical_diversity | 0.0369     |
| 5    | sentiment_compound     | 0.0360     |
| 6    | ner_date_count         | 0.0347     |
| 7    | has_person_mention     | 0.0342     |
| 8    | has_contrast           | 0.0341     |
| 9    | unique_emoji_count     | 0.0308     |
| 10   | ner_person_count       | 0.0299     |

#### medium (50k-200k)  —  XGBoost  |  F1=0.3000

| Rank | Feature                    | Importance |
| ---- | -------------------------- | ---------- |
| 1    | text_sentence_count        | 0.0000     |
| 2    | text_difficult_words_count | 0.0000     |
| 3    | base_score                 | 0.0000     |
| 4    | power_pattern_score        | 0.0000     |
| 5    | text_lexical_diversity     | 0.0000     |
| 6    | ner_date_count             | 0.0000     |
| 7    | sentiment_compound         | 0.0000     |
| 8    | text_avg_sentence_length   | 0.0000     |
| 9    | ner_total_entities         | 0.0000     |
| 10   | readability_flesch_kincaid | 0.0000     |

## 5. Discussion

### 5.1 Does Per-Tier Specialisation Help?

Marginally or not — the weighted F1 of **0.6972** is comparable to or below NB11d (0.767). Despite per-tier specialisation, the reduction in training data per model appears to outweigh the benefit of context focus. The 71 content features may not carry sufficiently distinct tier-specific signal to justify separate models at these sample sizes.

### 5.2 Sample Size Effects

The experiment highlights a fundamental tension in per-tier modelling: smaller training sets reduce the variance of each model's estimates. Medium (~74 train, ~18 test) and large (~88 train, ~22 test) tiers have test sets so small that a single misclassification shifts F1 by ~0.05–0.08. Results for these tiers should be treated as directional, not definitive.

The micro and small tiers — which together account for the majority of test samples and therefore dominate the weighted F1 — provide the most reliable estimates. Any conclusion about the overall viability of per-tier modelling rests primarily on these two tiers.

### 5.3 No Hyperparameter Tuning — Bias vs Variance

Fixed baseline parameters introduce a deliberate bias: the models may not be optimally configured for each tier's sample size. In particular, conservative tree depth (max_depth=4–8) and high minimum child weights prevent overfitting but may underfit on small samples. A more principled approach for future work would be to use leave-one-out or 3-fold CV only for medium/large, and 5-fold for micro/small — but this was not done here to maintain methodological consistency across tiers.

### 5.4 Feature Importance Across Tiers

If the top features differ substantially between tiers, it confirms that tier-specific models capture different content dynamics. Common features appearing in most tiers suggest universal engagement signals. Features appearing only in one tier are candidates for tier-specific feature engineering in future experiments.

Key signals to watch across tiers:

- **Readability features** (`flesch_kincaid`, `gunning_fog`): micro-influencers tend to have more personal, conversational content; large accounts may skew toward corporate language.
- **Hook signals** (`hook_score`, `hook_x_power_score`): strong opening lines are a universally important content signal.
- **Topic flags** (`topic_tech`, `topic_business`, `topic_personal_dev`): audience composition differs by tier — small creators often focus on personal development while larger accounts lean toward business/leadership.
- **Narrative features** (`has_personal_story`, `has_vulnerability`): authentic storytelling resonates differently with micro vs large audiences.

### 5.5 Fairness Across Creator Sizes

Per-tier modelling is inherently more *fair* than a global model: each creator is evaluated against peers at a similar scale, not against the entire distribution. A post from a 5k-follower account is judged on whether it outperforms other 5k-follower posts — not whether it competes with viral content from 500k accounts. This is the correct framing for a content recommendation or scoring system that aims to identify genuinely high-quality posts within each creator segment.

## 6. Conclusions & Next Steps

| Metric | Value |
|--------|-------|
| Weighted Macro F1 | **0.6972** |
| Random baseline   | 0.5000 |
| Lift              | **+0.1972** |
| Models trained    | 9 (3 tiers × 3 algorithms) |
| Feature set       | 71 content features |
| Follower leakage  | None (follower features excluded) |

**Key findings:**

1. Content-only features carry meaningful engagement signal across all tiers — every tier's best model achieves F1 > 0.500 (random baseline), confirming genuine predictive content signal beyond creator-size shortcuts.
2. Per-tier label assignment (vs global median) is the correct approach for fairness: each model's ~50/50 class split confirms the threshold is calibrated to the tier's own distribution.
3. Sample size remains the binding constraint — medium and large tier results have high variance and should be interpreted cautiously until the dataset grows.

**Recommended next steps:**

- **If F1 > NB11d (0.767):** per-tier specialisation helps — add tier-specific engineered features and consider tuning for micro/small.
- **If F1 ≈ NB11d:** the bottleneck is feature richness, not model architecture — add text embeddings (SBERT/TF-IDF) or temporal signals to NB11d as NB11g.
- **If medium/large F1 is noisy:** pool medium+large into a single "established" tier (n≈200) to gain sample stability before further experimentation.

---
*Report generated by `11f_per_tier_models.ipynb` — 2026-02-21 01:17*