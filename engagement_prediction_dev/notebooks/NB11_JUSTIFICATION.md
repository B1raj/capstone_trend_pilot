
# Engagement Rate Classification — What Was Done and Why

## Core Problem Addressed

Previous approaches to classifying post engagement often failed to account for the influence of audience size. For example:

> **A post with 500 reactions means something completely different depending on whether the author has 2,000 followers or 2,000,000 followers.**

This led to models learning "big accounts get more reactions" — not "this is good content."

---

## Decision 1: Target Variable — Engagement Rate

```python
engagement_rate = (reactions + comments) / (followers / 1000)
```

**Why this formula:**
Dividing by `followers/1000` normalises engagement to a *per-1,000-follower rate*. A nano-influencer with 5k followers getting 50 reactions scores 10.0. A mega-influencer with 500k followers getting 5,000 reactions also scores 10.0. The content performed equally — the rate reflects that.

**Why include comments:**
NB10 dropped comments because they were too noisy to predict in isolation. But as part of a combined engagement metric they add signal — a post that drives discussion is genuinely more engaging.


**Why this approach is superior:**
Engagement rate requires no author history at all, making it applicable to all posts equally. It avoids the pitfalls of methods that depend on author-specific baselines, which are often unavailable for most data.

---


## Decision 2: Split Before Assigning Class Labels

```python
df_train, df_test = train_test_split(df, ...)        # split first
p33 = df_train['engagement_rate'].quantile(1/3)      # thresholds from training only
p67 = df_train['engagement_rate'].quantile(2/3)
# same thresholds applied to test set
```

**Why:** If you compute percentile thresholds on the full dataset and then split, the test set's distribution influences where the class boundaries are drawn. That's a subtle form of data leakage. By deriving thresholds from training data only and applying them to the test set, the model is evaluated under conditions that genuinely simulate deployment.

---


## Decision 3: Drop Leaky Features


Three categories of columns were dropped.


### Direct leakage — columns that contain or are derived from the target

| Column | Reason dropped |
|--------|---------------|
| `reactions`, `comments`, `followers` | Literally the numerator and denominator of the target |
| `reactions_per_word`, `comments_per_word` | Reactions in the numerator — partially the same thing as the target |
| `reactions_per_sentiment` | Same issue — reactions derived |
| `comment_to_reaction_ratio` | Computed from the same post's engagement |
| `base_score_capped` | Composite score derived from engagement signals |


### Influencer history leakage — all `influencer_*` columns

These were aggregated statistics (average reactions, std reactions, median reactions, etc.) computed across the whole dataset. They are problematic because the post being evaluated likely contributed to its own author's averages. Even if it didn't (if they were truly historical), in this dataset there is no guarantee the aggregation window excluded the current post. Safer to remove entirely.

Specifically dropped:
- `influencer_avg_reactions`, `influencer_std_reactions`, `influencer_median_reactions`
- `influencer_avg_comments`, `influencer_std_comments`, `influencer_median_comments`
- `influencer_avg_base_score`, `influencer_avg_sentiment`
- `influencer_post_count`, `influencer_total_engagement`, `influencer_avg_engagement`
- `influencer_consistency_reactions`
- `reactions_vs_influencer_avg`, `comments_vs_influencer_avg`


### Metadata — not learnable features for a content quality model

- `name` — identifier
- `content` — raw text (handled separately if embeddings are added)
- `time_spent`, `location` — session metadata, not post content

---


## Decision 4: Transform Followers Instead of Dropping Entirely

```python
log_followers = np.log1p(followers)

follower_tier = pd.cut(followers,
    bins=[0, 10_000, 50_000, 200_000, np.inf],
    labels=[0, 1, 2, 3]   # micro, small, medium, large
)
```

**Why not drop followers completely:**
Follower count carries legitimate context. A 5-reactions post from a micro-influencer vs. a macro-influencer may have different implications even after rate normalisation (platform algorithm differences, audience engagement norms). The model benefits from knowing the creator context.

**Why not keep raw followers:**
Raw `followers` ranges from 80 to 2.75 million — a 34,000x range. More importantly, raw followers is the exact denominator used to construct the target, which creates a direct correlation that would mislead the model into learning the target's construction formula rather than content quality.

**Why log transform:**
`log1p(followers)` compresses the scale dramatically — the 34,000x range becomes roughly 4x in log space. The model can still distinguish creator sizes without extreme values swamping other features.

**Why tier:**
Provides a clean categorical encoding of creator size that aligns with how the industry actually thinks about influencer categories (micro/small/medium/large). Gives the model a discrete creator-context signal independent of the exact follower number.

---


## Decision 5: Balanced Class Weights + Stratified Split

```python
sample_weight = compute_sample_weight('balanced', y_train)
train_test_split(..., stratify=df['_tmp_class'])
```

**Why:** Percentile-based classes are balanced ~33/33/33 in training by construction, but the test set can drift. Balanced weights ensure no class dominates gradient updates during training. Stratified splitting preserves the approximate class ratio across both splits.

---

## Results

| Model         | Macro F1 Score |
|-------------- |:--------------:|
| Random Forest | 0.502          |
| XGBoost       | 0.566          |
| LightGBM      | 0.568          |
| Random Baseline | 0.333        |

The improvement comes from a better-defined problem, not better modelling:

1. **Cleaner target** — engagement rate separates content quality from audience size, giving the model a genuine signal to learn
2. **No cold-start degradation** — this approach works for all posts equally
3. **More signal in features** — with leaky influencer features removed, the model must learn from content features, which is exactly the goal
4. **Balanced classes** — equal class sizes mean the model cannot win by predicting the majority class

The fundamental insight: **previous models were partially measuring "how big is this creator" rather than "how good is this content."** Engagement rate fixes that.

---


## Class Interpretation

| Class | Threshold | Meaning |
|-------|-----------|---------|
| 0 — Below Average | rate < p33 (~2.2 per 1k followers) | Content under-performing relative to audience size |
| 1 — Average | p33 ≤ rate ≤ p67 (~15.2 per 1k followers) | Content in line with typical performance |
| 2 — Above Average | rate > p67 (~15.2 per 1k followers) | Content significantly over-performing — strong quality signal |

---


## Limitations

1. **772 posts** is a small dataset for 3-class classification — macro F1 ceiling is inherently low
2. **Engagement rate can be noisy** for very small accounts where one viral post creates an extreme rate
3. **Content features are rule-based** — semantic embeddings (SBERT, TF-IDF) would capture meaning more accurately than keyword flags
4. **Temporal effects not modelled** — platform algorithm changes and trending topics affect engagement independently of content quality
