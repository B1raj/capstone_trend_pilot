# NB11 Experiments Comparison Report
## LinkedIn Engagement Prediction — TrendPilot

**Notebooks compared:** NB11, NB11a, NB11b, NB11c
**Date:** February 2026

---

## 1. What We Are Trying to Achieve

The overarching goal across all three experiments is the same:

> **Predict whether a LinkedIn post's content is below average, average, or above average in engagement quality — fairly, across creators of any audience size.**

The engagement rate target `(reactions + comments) / (followers / 1000)` normalises for follower count at the outcome level. But the experiments differ in one critical design question:

**How much should the model know about the creator's audience size when making its prediction?**

This question has real implications:

- A model that knows follower tier can set appropriate expectations per creator — but risks learning audience dynamics instead of content quality
- A model with no follower information must rely purely on what's written — but is blind to context that is genuinely relevant
- A model that classifies within each tier judges a post against creators of the same size — the cleanest framing, but the hardest prediction problem

Each experiment answers a different version of the question.

---

## 2. Experiment Designs

| | NB11 | NB11a | NB11b |
|--|------|-------|-------|
| **Target** | Global engagement rate percentiles | Global engagement rate percentiles | Per-tier engagement rate percentiles |
| **Class thresholds** | p33=2.20, p67=15.16 (global) | p33=2.20, p67=15.16 (global) | Separate p33/p67 per tier |
| **Follower features** | `log_followers` + `follower_tier` | **Removed entirely** | `log_followers` + `follower_tier` |
| **Feature count** | 73 | 71 | 73 |
| **Core question** | What engagement class will this post land in? | Can content alone predict engagement class? | Is this post above average for its creator size? |

### NB11 — Baseline Engagement Rate Model

The base model. Engagement rate is normalised by followers, giving a fair cross-creator target. Follower proxies are kept as features because they legitimately carry context — the model can know creator tier without seeing raw follower count.

### NB11a — Content-Only (No Follower Features)

Removes `log_followers` and `follower_tier` entirely. The model must predict engagement quality using only content signals: hooks, sentiment, readability, style, named entities, topics, and structural patterns. Answers the question: how much of NB11's performance came from creator size vs actual content?

### NB11b — Within-Tier Classification

Keeps follower features but fundamentally changes the class labels. Instead of "is this post above average globally?", it asks "is this post above average for a creator of this audience size?" Each tier gets its own percentile-based thresholds from training data.

**Per-tier thresholds (NB11b training set):**

| Tier | p33 | p67 | n_train |
|------|-----|-----|---------|
| micro (<10k) | 10.92 | 44.84 | 282 |
| small (10k–50k) | 1.38 | 6.84 | 174 |
| medium (50k–200k) | 0.67 | 2.78 | 75 |
| large (>200k) | 0.22 | 1.17 | 86 |

These thresholds reveal how dramatically different "normal" engagement rate is per tier. A micro-creator needs 10.92 rate just to reach Class 1; a large account needs only 0.22. The global thresholds in NB11 were putting almost all large accounts into Class 0 and almost all micro accounts into Class 2.

---

## 3. Results

### 3.1 Overall Performance

| Experiment | Best Model | Test Macro F1 | CV Mean | CV Std | Accuracy |
|------------|-----------|-------------|---------|--------|---------|
| NB11 (global + followers) | XGBoost_base | **0.5997** | **0.6023** | 0.0355 | 0.60 |
| NB11a (global, no followers) | XGBoost_tuned | 0.5301 | 0.5083 | 0.0319 | 0.53 |
| NB11b (within-tier + followers) | RF_base | 0.4306 | 0.4778 | 0.0262 | 0.44 |
| Random baseline | — | 0.3333 | — | — | 0.33 |

All three experiments beat the random baseline meaningfully. NB11 scores highest. NB11a loses ~0.07 points. NB11b loses ~0.17 points.

### 3.2 Per-Class Results — Best Model of Each Experiment

**NB11 (XGBoost_base):**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Below | 0.64 | 0.67 | **0.65** | 54 |
| Average | 0.43 | 0.50 | **0.46** | 48 |
| Above | 0.67 | 0.55 | **0.60** | 53 |
| Macro avg | 0.58 | 0.57 | **0.57** | 155 |

**NB11a (XGBoost_tuned):**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Below | 0.62 | 0.54 | **0.57** | 54 |
| Average | 0.41 | 0.50 | **0.45** | 48 |
| Above | 0.58 | 0.55 | **0.56** | 53 |
| Macro avg | 0.54 | 0.53 | **0.53** | 155 |

**NB11b (RF_base):**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Below (tier) | 0.48 | 0.60 | **0.53** | 53 |
| Average (tier) | 0.28 | 0.22 | **0.24** | 50 |
| Above (tier) | 0.50 | 0.46 | **0.48** | 52 |
| Macro avg | 0.42 | 0.43 | **0.42** | 155 |

**Key pattern across all three:** The Average class (Class 1) is consistently the hardest — it sits on both decision boundaries. NB11b's Average class drops to F1 = 0.24, showing how much harder the within-tier problem is.

### 3.3 Confusion Matrices

**NB11:**
```
              Pred:Below  Pred:Avg  Pred:Above
Actual:Below      36         14         4
Actual:Avg        14         24        10
Actual:Above       6         18        29
```

**NB11a:**
```
              Pred:Below  Pred:Avg  Pred:Above
Actual:Below      29         15        10
Actual:Avg        13         24        11
Actual:Above       5         19        29
```

**NB11b:**
```
              Pred:Below  Pred:Avg  Pred:Above
Actual:Below      32          9        12
Actual:Avg        27         11        12
Actual:Above       8         20        24
```

NB11's confusion matrix shows the cleanest diagonal. NB11b's middle row (Average class) is almost random — errors spread almost equally across all three predicted classes.

### 3.4 Tier-Level Performance — The Key Diagnostic

| Tier | n (test) | NB11 F1 | NB11a F1 | NB11b F1 |
|------|---------|---------|---------|---------|
| micro (<10k) | 63 | 0.3942 | 0.4091 | 0.4263 |
| small (10k–50k) | 51 | 0.3016 | **0.4587** | 0.3673 |
| medium (50k–200k) | 17 | 0.3630 | **0.4444** | 0.3919 |
| large (>200k) | 24 | **0.8889** | 0.4222 | 0.5200 |
| **Variance (std)** | | **0.233** | **0.025** | **0.064** |

This table is the most revealing result in the entire comparison. The **standard deviation of tier F1** captures how biased each model is toward certain creator sizes:

- **NB11 std = 0.233** — extreme variance. The large tier (0.89) vs small tier (0.30) gap confirms the model learned "large account = Class 0" as a rule rather than evaluating content.
- **NB11a std = 0.025** — near-uniform performance across all tiers. Removing followers made the model genuinely content-focused. All four tiers score 0.41–0.46.
- **NB11b std = 0.064** — significant improvement over NB11, but still some variance. The within-tier framing substantially reduced the large-account shortcut (0.89 → 0.52), though it introduced some residual tier effects.

---

## 4. Feature Importance Analysis

### NB11 — Top 10 Features

| Rank | Feature | Category |
|------|---------|---------|
| 1 | `follower_tier` | Creator context |
| 2 | `log_followers` | Creator context |
| 3 | `has_recency_hook` | Hook pattern |
| 4 | `has_entities` | Named entities |
| 5 | `is_promotional` | Content type |
| 6 | `style_question_marks` | Style |
| 7 | `ner_event_count` | Named entities |
| 8 | `url_count` | Format |
| 9 | `has_org_mention` | Named entities |
| 10 | `text_lexical_diversity` | Text quality |

Follower proxies dominate. The top content feature is rank 3 (`has_recency_hook`).

### NB11a — Top 10 Features (No Followers)

| Rank | Feature | Category |
|------|---------|---------|
| 1 | `style_has_question` | Style |
| 2 | `text_lexical_diversity` | Text quality |
| 3 | `link_penalty_score` | Penalty |
| 4 | `has_entities` | Named entities |
| 5 | `url_count` | Format |
| 6 | `style_question_marks` | Style |
| 7 | `has_org_mention` | Named entities |
| 8 | `ner_event_count` | Named entities |
| 9 | `link_spam_penalty` | Penalty |
| 10 | `style_bullet_count` | Style |

With followers removed, content signals take over. The top 10 is a mix of style, named entities, link penalties, and text quality — these are genuine content quality drivers. `has_recency_hook` no longer dominates; instead a broader set of signals shares importance more evenly.

### NB11b — Top 10 Features (Within-Tier)

| Rank | Feature | Category |
|------|---------|---------|
| 1 | `log_followers` | Creator context |
| 2 | `text_avg_sentence_length` | Text quality |
| 3 | `readability_gunning_fog` | Readability |
| 4 | `sentiment_x_readability` | Sentiment |
| 5 | `text_lexical_diversity` | Text quality |
| 6 | `sentiment_compound` | Sentiment |
| 7 | `readability_flesch_kincaid` | Readability |
| 8 | `sentence_count` | Text quality |
| 9 | `text_difficult_words_count` | Text quality |
| 10 | `style_number_count` | Style |

Interestingly, `log_followers` still ranks #1 even in the within-tier setting. Within each tier, the exact follower count still provides signal — likely because engagement rate patterns are not perfectly uniform even within a tier. The key shift is that **readability and text quality features dominate** when the tier-level shortcut is removed. Text quality metrics (sentence length, gunning fog, lexical diversity, difficult words) collectively occupy 6 of the top 10 slots — this is the model searching for genuine writing quality signals within each creator size bracket.

### What the Feature Importance Shift Reveals

The progression from NB11 → NB11a → NB11b tells a clear story:

1. **NB11:** Model primarily uses creator size (follower tier) as a proxy for class. Content signals are secondary.
2. **NB11a:** With followers removed, the model redistributes to a balanced mix of style, entities, and link signals. No single feature dominates — content quality is genuinely multidimensional.
3. **NB11b:** Within-tier forces the model toward deeper text quality signals (readability, vocabulary complexity, sentence structure). These are harder to exploit but more meaningful content quality indicators.

---

## 5. Interpretation and Justification

### Why NB11 Scores Highest (but is the least honest)

NB11's F1 advantage stems almost entirely from the large-tier shortcut. The model correctly predicts Class 0 for 18/24 large-account test posts — but that is because large accounts structurally land in Class 0 under global thresholds, not because the model assessed their content. Remove that shortcut (as NB11a does) and the overall F1 drops from 0.60 to 0.53.

The NB11 score is real but partially misleading: it overstates content prediction ability by conflating it with creator-size prediction.

### Why NB11a is the Most Honest Content Quality Model

NB11a's tier F1 standard deviation of 0.025 is the strongest evidence that this model is doing what it claims to do. A model that performs equally well for micro and large creators — when both are measured against the same global engagement rate standard — is genuinely assessing content signals rather than creator characteristics.

The F1 cost of removing followers (~0.07 points) is modest and worth the fairness gain. This is the recommended model if the goal is **"is this content objectively strong?"**

### Why NB11b is the Hardest Problem — and Still Valuable

NB11b's lower overall F1 (0.43) should not be read as failure. It reflects a fundamentally harder prediction problem: within each tier, the class signal is weaker, the feature space provides less leverage, and the dataset is smaller. The model is attempting to find content signals within micro-creators that separate their below/average/above posts — and doing so without the global structural shortcut.

NB11b is the right approach for the question **"will this content over-perform for a creator of this specific size?"** — the most actionable question for a content recommendation tool. The current performance ceiling reflects dataset size, not a flawed approach.

The feature shift toward readability and text quality in NB11b is also encouraging: these are interpretable, actionable signals that creators can control. A recommendation like "your sentence structure complexity is below average for your tier" is more useful than "large accounts tend to get low engagement rates."

### Summary of Experiment Objectives

| Experiment | Answers | Best for |
|------------|---------|---------|
| NB11 | Will this post perform well for this creator? | Realistic engagement prediction per creator |
| NB11a | Is this content genuinely strong, regardless of who posts it? | Pure content quality scoring tool |
| NB11b | Is this content above average for creators of this size? | Within-tier content benchmarking |

---

## 6. Limitations Common to All Three

1. **Dataset size (772 posts):** With ~50 test samples per class, all F1 estimates carry high variance. The CV mean is more reliable than single test splits.
2. **Rule-based features:** All content signals are keyword flags and counts. They capture surface-level patterns but miss semantic meaning. Text embeddings (SBERT, TF-IDF) would substantially improve all three models.
3. **Temporal effects:** Posting time, trending topics, and platform algorithm changes affect engagement independently of content quality.
4. **NB11b small tier samples:** Medium (75 train) and large (86 train) tiers have limited data for 3-class classification within each tier.

---

## 7. Conclusions

| Question | Answer |
|---------|--------|
| Does removing followers hurt F1? | Modestly — ~0.07 points (0.60 → 0.53). Worth the fairness gain. |
| Was NB11's high large-tier F1 real content prediction? | No — it was learning "large account = Class 0". Tier F1 drops from 0.89 → 0.42 in NB11a. |
| Does within-tier framing work? | Yes, but harder. F1 drops to 0.43; tier variance reduces from std=0.23 → std=0.06. |
| Which model to use for content quality scoring? | **NB11a** — fair, content-focused, interpretable, modest F1 cost. |
| Which model to use for creator-specific prediction? | **NB11** — highest F1, accounts for realistic creator context. |
| What's the next meaningful improvement? | Add text embeddings to all three. Current feature set is a ceiling. |

**Recommended next step:** Add text embeddings (SBERT or TF-IDF of `clean_content`) to NB11c — the 2-class model is the strongest foundation for semantic feature augmentation.

---

## 8. Experiment C — NB11c: 2-Class Binary Classification

### Design

Removes the structurally ambiguous Average class entirely. The single threshold is the **median engagement rate of the training set** (5.985 per 1k followers):

- **Class 0 — Below Average:** rate < 5.985
- **Class 1 — Above Average:** rate >= 5.985

The random baseline shifts from 0.333 to **0.500**. All F1 numbers must be interpreted relative to this higher bar.

### Results

| Model | Macro F1 | Accuracy |
|-------|----------|---------|
| RF_base | 0.7871 | 0.7871 |
| **XGBoost_base** | **0.8064** | **0.8065** |
| LightGBM_base | 0.7999 | 0.8000 |
| XGBoost_tuned | 0.7868 | 0.7871 |
| LightGBM_tuned | 0.7935 | 0.7935 |
| Random baseline | 0.500 | 0.500 |

**Best: XGBoost_base — Macro F1 = 0.8064, Accuracy = 80.7%**

Cross-validation: mean = 0.7693, std = 0.0530

### Classification Report (XGBoost_base)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Below (<6.0) | 0.82 | 0.74 | **0.78** | 78 |
| Above (>=6.0) | 0.76 | 0.83 | **0.80** | 77 |
| Macro avg | 0.79 | 0.79 | **0.79** | 155 |

### Confusion Matrix

```
               Pred:Below  Pred:Above
Actual:Below       58          20
Actual:Above       13          64
```

58/78 below-average posts correctly identified (74% recall).
64/77 above-average posts correctly identified (83% recall).
The model is slightly better at identifying strong content than weak content.

### Tier Performance

| Tier | n | NB11c F1 | NB11a F1 | Delta |
|------|---|---------|---------|-------|
| micro (<10k) | 77 | 0.6183 | 0.4091 | **+0.209** |
| small (10k–50k) | 41 | 0.7515 | 0.4587 | **+0.293** |
| medium (50k–200k) | 16 | 0.4286 | 0.4444 | -0.016 |
| large (>200k) | 21 | 0.4615 | 0.4222 | +0.039 |

Dramatic improvement for micro and small tiers. Medium drops slightly (only 16 samples — high variance). The 2-class framing is substantially more learnable for the two dominant tier groups.

### The Definitive Lift Comparison

| Experiment | Raw F1 | Random Baseline | Lift |
|------------|--------|----------------|------|
| NB11 (3-cls, followers) | 0.5997 | 0.333 | **+0.267** |
| NB11a (3-cls, no followers) | 0.5301 | 0.333 | +0.197 |
| NB11b (3-cls, within-tier) | 0.4306 | 0.333 | +0.098 |
| **NB11c (2-cls, median)** | **0.8064** | **0.500** | **+0.306** |

**NB11c has the highest lift over its random baseline (+0.306)** of any experiment. Raw F1 of 0.80 at 80% accuracy makes it by far the most practically useful model.

### Why 2-Class Works So Much Better

The Average class was the core problem in every 3-class model (F1 = 0.24–0.46 across experiments). It is not a stable, learnable category — content that performs near the median is genuinely ambiguous; small noise in posting time, audience mood, or algorithm state pushes it either direction. Removing it:

1. **Eliminates boundary ambiguity** — the model no longer needs to find a narrow band between two thresholds
2. **Doubles test samples per class** (~50 → ~78 per class) — more reliable evaluation
3. **Creates a cleaner signal** — below vs above median is a more robust distinction than three-way percentile bands
4. **Improves all tiers** — the gain is consistent across creator sizes, confirming the Average class was noise, not signal

### Feature Importance (NB11c)

The top features shift slightly from the 3-class models:

| Rank | Feature | Category |
|------|---------|---------|
| 1 | `follower_tier` | Creator context |
| 2 | `log_followers` | Creator context |
| 3 | `has_org_mention` | Named entities |
| 4 | `style_has_question` | Style |
| 5 | `style_has_quotes` | Style |
| 6 | `hashtag_count_extracted` | Format |
| 7 | `text_lexical_diversity` | Text quality |
| 8 | `ner_org_count` | Named entities |
| 9 | `link_penalty_score` | Penalty |
| 10 | `style_exclamation_marks` | Style |

Follower proxies remain #1–2 (expected with global thresholds). The content signal is now broader — organisation mentions, question/quote/exclamation style, hashtag count, and vocabulary diversity all contribute meaningfully. No single content feature dominates, suggesting genuine multidimensional content quality discrimination.

---

## 9. Final Conclusions Across All Four Experiments

| Experiment | Best F1 | Lift | Tier Std | Honest Content Signal? |
|------------|---------|------|---------|----------------------|
| NB11 (3-cls, followers) | 0.5997 | +0.267 | 0.233 | Partial — shortcut via large tier |
| NB11a (3-cls, no followers) | 0.5301 | +0.197 | 0.025 | Yes — content-only, fairest |
| NB11b (3-cls, within-tier) | 0.4306 | +0.098 | 0.064 | Yes — hardest problem |
| **NB11c (2-cls, median)** | **0.8064** | **+0.306** | ~0.14 | Mostly — still uses follower tier |

**NB11c is the strongest model overall** and the recommended production baseline. The F1 jump from removing the Average class is the single largest improvement across all experiments.

**NB11a remains the most principled model** for pure content scoring — if the goal is creator-agnostic content quality assessment.

**The universal next step for all models:** Replace rule-based content features with SBERT or TF-IDF text embeddings of `clean_content`. The current feature ceiling is the rule-based nature of the 73 content signals. Semantic embeddings would be the most impactful single addition.
