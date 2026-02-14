# Feature Normalization Plan: Addressing High-Magnitude Bias

**Date Created:** February 13, 2026  
**Status:** 📋 PLANNED  
**Priority:** HIGH  
**Goal:** Improve model fairness across all account sizes without sacrificing accuracy

---

## Problem Statement

### Current Issue

The engagement prediction model performs poorly for influencers with small follower counts because high-magnitude aggregate features dominate predictions:

- **`influencer_avg_engagement`**: Can range from 50 (small accounts) to 50,000+ (large accounts)
- **`influencer_total_engagement`**: Even more extreme ranges
- **`influencer_avg_reactions`**, **`influencer_avg_comments`**: Similar scale problems

**Impact:**
- Model learns: "High `influencer_avg_engagement` → predict high engagement"
- This is essentially: "Big accounts get big engagement" (not useful)
- Small accounts (1K-10K followers) receive poor predictions
- Model doesn't learn content quality patterns effectively

### Root Cause

While V3 models successfully excluded the `followers` feature (minimal performance impact: -0.0006 R²), the **influencer aggregate features still carry follower-count bias** because:

1. High-follower accounts naturally have higher historical engagement (absolute numbers)
2. Features aren't normalized to account for audience size
3. Raw counts span 3-4 orders of magnitude (100 vs 100,000)
4. Tree-based models split heavily on high-variance features
5. `influencer_avg_engagement` became #1 most important feature (68% of variance explained)

### Why This Isn't Data Leakage

These features are **legitimate but biased**:
- ✅ Calculated from historical posts only (not current post)
- ✅ No target variable (reactions/comments) from current post used
- ⚠️ BUT they correlate strongly with follower count → proxy for account size, not content quality

---

## Solution Strategy

### Core Approach: Transform, Don't Remove

**Insight from V3_FIXED failure:** Removing influencer features entirely hurts performance significantly.

**Better approach:** Transform features to **preserve information while removing scale bias**

### Multi-Pronged Transformation Strategy

#### 1. **Log Transformation** (Compress Scale)
- Apply `log1p()` to all count-based features
- Effect: 10,000 → 9.21, 100,000 → 11.51
- Captures multiplicative relationships (10x followers → ~10x engagement)
- Benefits: Compresses outliers, more linear relationships, interpretable

**Features to transform:**
```python
log_features = [
    'influencer_avg_engagement',
    'influencer_total_engagement', 
    'influencer_avg_reactions',
    'influencer_avg_comments',
    'influencer_median_reactions',
    'influencer_median_comments',
    'followers'
]
```

#### 2. **Engagement Rate Features** (Per-Capita Normalization)
- Create: `historical_engagement_rate = avg_engagement / follower_count`
- Comparable across all account sizes (e.g., 2% engagement rate)
- No data leakage: uses historical average + current follower count

**New features to create:**
```python
rate_features = {
    'influencer_engagement_rate': influencer_avg_engagement / followers,
    'influencer_reactions_rate': influencer_avg_reactions / followers,
    'influencer_comments_rate': influencer_avg_comments / followers,
    'influencer_engagement_rate_consistency': influencer_std_reactions / followers
}
```

#### 3. **Follower Tier Stratification** (Context-Aware Features)
- Define tiers based on training data distribution
- Create percentile-within-tier features
- Captures: "This account is top 20% among similar-sized accounts"

**Tier definitions** (based on quantiles):
```python
tiers = {
    'micro': (0, 5000),           # <5K followers (bottom 25%)
    'small': (5000, 20000),       # 5K-20K (25-50%)
    'medium': (20000, 100000),    # 20K-100K (50-75%)
    'large': (100000, 500000),    # 100K-500K (75-95%)
    'macro': (500000, float('inf'))  # 500K+ (top 5%)
}
```

**Tier-based features:**
- `follower_tier` (categorical)
- `is_tier_micro`, `is_tier_small`, etc. (one-hot)
- `percentile_engagement_in_tier` (0-1 normalized rank within tier)
- `engagement_vs_tier_median` (ratio to typical performance for tier)

#### 4. **Hybrid Feature Set** (Let Model Learn Weights)
- Include BOTH absolute (log-transformed) AND relative (normalized) features
- Model can learn:
  - For large accounts: absolute matters more
  - For small accounts: rate matters more
  - For medium: combination of both
- Tree models naturally split on appropriate features per context

#### 5. **Robust Scaling** (Handle Outliers)
- Apply `RobustScaler` (median/IQR based) to log-transformed features
- Better than `StandardScaler` (mean/std) for engagement data with viral outliers
- Prevents single viral post from distorting entire scale

---

## Implementation Plan

### Step 1: Create Feature Normalization Notebook

**File:** `05_feature_normalization.ipynb`  
**Input:** `data/feature_engineered_data.csv`  
**Output:** `data/feature_engineered_normalized.csv`

**Operations:**

1. **Log transformation:**
   ```python
   for col in log_features:
       df[f'log_{col}'] = np.log1p(df[col])
   ```

2. **Engagement rate calculation:**
   ```python
   df['influencer_engagement_rate'] = df['influencer_avg_engagement'] / df['followers'].replace(0, 1)
   df['influencer_reactions_rate'] = df['influencer_avg_reactions'] / df['followers'].replace(0, 1)
   df['influencer_comments_rate'] = df['influencer_avg_comments'] / df['followers'].replace(0, 1)
   ```

3. **Follower tier assignment:**
   ```python
   def assign_tier(followers):
       if followers < 5000: return 'micro'
       elif followers < 20000: return 'small'
       elif followers < 100000: return 'medium'
       elif followers < 500000: return 'large'
       else: return 'macro'
   
   df['follower_tier'] = df['followers'].apply(assign_tier)
   df = pd.get_dummies(df, columns=['follower_tier'], prefix='is_tier')
   ```

4. **Percentile-within-tier calculation:**
   ```python
   df['percentile_engagement_in_tier'] = df.groupby('follower_tier')['influencer_avg_engagement'].rank(pct=True)
   df['percentile_reactions_in_tier'] = df.groupby('follower_tier')['influencer_avg_reactions'].rank(pct=True)
   df['percentile_comments_in_tier'] = df.groupby('follower_tier')['influencer_avg_comments'].rank(pct=True)
   ```

5. **Peer-relative features:**
   ```python
   tier_medians = df.groupby('follower_tier')['influencer_avg_engagement'].transform('median')
   df['engagement_vs_tier_median'] = df['influencer_avg_engagement'] / tier_medians
   ```

6. **Save with metadata:**
   ```python
   df.to_csv('data/feature_engineered_normalized.csv', index=False)
   
   # Save tier boundaries for production use
   tier_metadata = {
       'tier_boundaries': tiers,
       'tier_medians': df.groupby('follower_tier')['influencer_avg_engagement'].median().to_dict(),
       'transformation_date': '2026-02-13'
   }
   with open('data/normalization_metadata.json', 'w') as f:
       json.dump(tier_metadata, f, indent=2)
   ```

### Step 2: Update Feature Selection

**File:** `04_feature_selection.ipynb`

**Changes:**

1. **Remove from selection pool:**
   ```python
   EXCLUDE_RAW_MAGNITUDE = [
       'influencer_avg_engagement',     # Use log version
       'influencer_total_engagement',   # Use log version
       'influencer_avg_reactions',      # Use log + rate versions
       'influencer_avg_comments',       # Use log + rate versions
       'influencer_median_reactions',   # Use log version
       'influencer_median_comments',    # Use log version
       'followers'                      # Already excluded in V3
   ]
   ```

2. **Add to selection pool:**
   ```python
   INCLUDE_NORMALIZED = [
       # Log-transformed (absolute, compressed scale)
       'log_influencer_avg_engagement',
       'log_influencer_total_engagement',
       'log_influencer_avg_reactions',
       'log_influencer_avg_comments',
       'log_followers',
       
       # Engagement rates (normalized by audience size)
       'influencer_engagement_rate',
       'influencer_reactions_rate',
       'influencer_comments_rate',
       
       # Tier-based features (context-aware)
       'is_tier_micro', 'is_tier_small', 'is_tier_medium', 'is_tier_large', 'is_tier_macro',
       'percentile_engagement_in_tier',
       'percentile_reactions_in_tier',
       'percentile_comments_in_tier',
       'engagement_vs_tier_median'
   ]
   ```

3. **Re-run feature selection pipeline:**
   - Variance threshold filtering
   - Correlation analysis (may show log and rate versions complement each other)
   - Model-based importance ranking
   - Select top 90 features (same as before)

4. **Save updated selection:**
   ```python
   with open('data/selected_features_normalized.json', 'w') as f:
       json.dump(selected_features_dict, f, indent=2)
   ```

### Step 3: Train V4 Model with Fairness Focus

**File:** `06_model_training_v4.ipynb`

**Key Changes from V3:**

1. **Load normalized features:**
   ```python
   data = pd.read_csv('data/selected_features_data_normalized.csv')
   ```

2. **Stratified train-test split:**
   ```python
   # Ensure all tiers represented in train/test
   from sklearn.model_selection import train_test_split
   
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, 
       test_size=0.2, 
       stratify=data['follower_tier'],  # NEW: stratify by tier
       random_state=42
   )
   ```

3. **Apply RobustScaler:**
   ```python
   from sklearn.preprocessing import RobustScaler
   
   scaler = RobustScaler()
   
   # Only scale log-transformed features (rates already normalized)
   log_cols = [col for col in X_train.columns if col.startswith('log_')]
   X_train[log_cols] = scaler.fit_transform(X_train[log_cols])
   X_test[log_cols] = scaler.transform(X_test[log_cols])
   
   # Save scaler for production
   joblib.dump(scaler, 'models_v4/robust_scaler.pkl')
   ```

4. **Train models (same architectures as V3):**
   - XGBoost
   - LightGBM
   - RandomForest
   - (Optional) Linear Regression for interpretability

5. **Add fairness evaluation section:**

   ```python
   def evaluate_fairness_by_tier(y_true, y_pred, tier_labels):
       """Calculate metrics separately for each follower tier"""
       
       tiers = ['micro', 'small', 'medium', 'large', 'macro']
       fairness_metrics = {}
       
       for tier in tiers:
           mask = (tier_labels == tier)
           if mask.sum() == 0:
               continue
               
           tier_metrics = {
               'count': mask.sum(),
               'mae': mean_absolute_error(y_true[mask], y_pred[mask]),
               'rmse': np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])),
               'r2': r2_score(y_true[mask], y_pred[mask]),
               'mape': safe_mape(y_true[mask], y_pred[mask])
           }
           fairness_metrics[tier] = tier_metrics
       
       # Calculate fairness score
       mapes = [m['mape'] for m in fairness_metrics.values() if m['mape'] is not None]
       fairness_score = max(mapes) / min(mapes) if mapes else None
       
       return fairness_metrics, fairness_score
   
   # Evaluate
   tier_metrics_test, fairness_score = evaluate_fairness_by_tier(
       y_test_reactions, 
       y_pred_reactions,
       test_data['follower_tier']
   )
   
   print(f"\n{'='*60}")
   print("FAIRNESS EVALUATION BY FOLLOWER TIER")
   print(f"{'='*60}\n")
   
   for tier, metrics in tier_metrics_test.items():
       print(f"{tier.upper()} ({metrics['count']} posts):")
       print(f"  MAE:  {metrics['mae']:.2f}")
       print(f"  RMSE: {metrics['rmse']:.2f}")
       print(f"  MAPE: {metrics['mape']:.1f}%")
       print(f"  R²:   {metrics['r2']:.4f}\n")
   
   print(f"Fairness Score: {fairness_score:.2f}")
   print(f"  (<1.5 = Good, >2.0 = Unfair)\n")
   ```

6. **Create fairness visualizations:**

   ```python
   # Plot 1: Predicted vs Actual by Tier
   fig, axes = plt.subplots(1, 5, figsize=(20, 4))
   for idx, tier in enumerate(tiers):
       mask = (test_data['follower_tier'] == tier)
       axes[idx].scatter(y_test[mask], y_pred[mask], alpha=0.3, s=20)
       axes[idx].plot([0, y_test[mask].max()], [0, y_test[mask].max()], 
                      'r--', label='Perfect prediction')
       axes[idx].set_title(f'{tier.upper()} (n={mask.sum()})')
       axes[idx].set_xlabel('Actual')
       axes[idx].set_ylabel('Predicted')
   plt.tight_layout()
   plt.savefig('reports/fairness_pred_vs_actual_by_tier.png', dpi=150)
   
   # Plot 2: Residuals vs Log(Followers)
   residuals = y_test - y_pred
   plt.figure(figsize=(10, 6))
   plt.scatter(test_data['log_followers'], residuals, alpha=0.3, s=20)
   plt.axhline(0, color='red', linestyle='--')
   plt.xlabel('Log(Followers)')
   plt.ylabel('Residuals (Actual - Predicted)')
   plt.title('Residual Pattern Check (should show no trend)')
   plt.savefig('reports/fairness_residuals_vs_followers.png', dpi=150)
   ```

7. **Save comprehensive metadata:**
   ```python
   model_metadata = {
       'version': 'v4',
       'date': '2026-02-13',
       'objective': 'Fairness-improved model with normalized influencer features',
       'features': feature_columns.tolist(),
       'normalization': {
           'log_transformed': log_cols,
           'rate_features': rate_cols,
           'tier_features': tier_cols,
           'scaler': 'RobustScaler'
       },
       'performance': {
           'overall': {
               'mae': overall_mae,
               'rmse': overall_rmse,
               'r2': overall_r2,
               'mape': overall_mape
           },
           'by_tier': tier_metrics_test,
           'fairness_score': fairness_score
       }
   }
   
   with open('models_v4/model_metadata_v4.json', 'w') as f:
       json.dump(model_metadata, f, indent=2)
   ```

### Step 4: Create Fairness Comparison Report

**File:** `reports/V3_VS_V4_FAIRNESS_COMPARISON.md`

**Content:**

```markdown
# V3 vs V4 Fairness Comparison Report

## Overview
- **V3:** Uses raw magnitude features (influencer_avg_engagement, etc.)
- **V4:** Uses normalized features (log-transformed + rates + tier-based)

## Performance Comparison

### Overall Metrics
| Metric | V3 | V4 | Change |
|--------|----|----|--------|
| R² | 0.5946 | [TBD] | [TBD] |
| RMSE | [TBD] | [TBD] | [TBD] |
| MAPE | [TBD] | [TBD] | [TBD] |

### Fairness by Tier (MAPE %)
| Tier | V3 | V4 | Improvement |
|------|----|----|-------------|
| Micro | [TBD] | [TBD] | [TBD] |
| Small | [TBD] | [TBD] | [TBD] |
| Medium | [TBD] | [TBD] | [TBD] |
| Large | [TBD] | [TBD] | [TBD] |
| Macro | [TBD] | [TBD] | [TBD] |

### Fairness Score
- **V3:** [TBD] (max_MAPE / min_MAPE)
- **V4:** [TBD]
- **Target:** <1.5

## Feature Importance Shifts
[Compare top 20 features V3 vs V4]

## Recommendations
[Based on results, recommend V3 or V4 for production]
```

### Step 5: Update Model Testing

**File:** `07_model_testing.ipynb`

**Add fairness test cases:**

```python
test_cases = [
    {
        'name': 'High-quality content, small account',
        'description': 'Well-crafted post from 2K follower creator',
        'content': '[compelling content with hooks, patterns, good readability]',
        'followers': 2000,
        'influencer_avg_engagement': 100,  # Good for size
        'expected_behavior': 'V4 should predict well despite low follower count'
    },
    {
        'name': 'Low-quality content, large account',
        'description': 'Poor content from 500K follower creator',
        'content': '[short, no hooks, promotional]',
        'followers': 500000,
        'influencer_avg_engagement': 25000,
        'expected_behavior': 'V4 should not over-predict just because large account'
    },
    {
        'name': 'Median content, medium account',
        'description': 'Average post from typical account',
        'content': '[standard content]',
        'followers': 50000,
        'influencer_avg_engagement': 2500,
        'expected_behavior': 'Both V3 and V4 should predict reasonably'
    }
]

# For each test case, generate predictions from both V3 and V4
# Compare which model gives more reasonable predictions
```

---

## Success Criteria

### Primary Goal: Fairness Improvement

✅ **Target Achieved If:**

1. **Fairness Score < 1.5**
   - V3 baseline: Expected >2.0 (micro accounts have 2x+ error vs macro)
   - V4 target: <1.5 (max 50% difference in MAPE across tiers)

2. **Small Account MAPE Reduces by ≥20%**
   - If V3 micro/small MAPE = 60%, V4 should achieve ≤48%
   - Indicates model now generalizes better to low-follower creators

3. **Residuals Show No Follower Bias**
   - Plot of residuals vs log(followers) should show random scatter
   - No upward or downward trend line (would indicate systematic bias)

### Secondary Goal: Preserve Overall Accuracy

✅ **Acceptable If:**

1. **Overall R² drop ≤3%**
   - V3 R² = 0.5946
   - V4 R² ≥ 0.577 (acceptable tradeoff for fairness)

2. **Large Account MAPE increase ≤10%**
   - May sacrifice some accuracy on macro accounts to improve micro
   - But shouldn't degrade significantly (still need to serve large creators)

### Feature Importance Evolution

✅ **Desired Shifts:**

1. **Raw magnitude features drop out of top 10**
   - `influencer_avg_engagement` (currently #1) should rank lower
   - No raw count features in top importance

2. **Content features gain prominence**
   - `sentiment_compound`, `readability_*`, `base_score_capped` rank higher
   - Indicates model learning content quality patterns

3. **Normalized features in top 20**
   - `log_influencer_avg_engagement`, `influencer_engagement_rate`, `percentile_engagement_in_tier` appear in top importance
   - Shows transformations are being utilized

### No Data Leakage

✅ **Must Verify:**

1. **All rate features use legitimate inputs**
   - Numerator: Historical engagement only
   - Denominator: Follower count (current, not future)
   - No target variables (reactions/comments from current post)

2. **Tier assignments based on training data**
   - Quantile boundaries calculated on training set
   - Frozen for test/validation/production

3. **Scaler fitted on training only**
   - RobustScaler sees only training data
   - Transform (not fit_transform) applied to test

---

## Key Decisions & Rationale

### Decision 1: Hybrid Feature Set (Absolute + Relative)

**Chose:** Include both log-transformed absolute AND normalized rate features  
**Over:** Removing raw features entirely (V3_FIXED approach)  
**Rationale:**
- V3_FIXED showed significant performance drop when removing all influencer features
- Model needs flexibility to learn when magnitude matters vs when rate matters
- Tree models can naturally split: use absolute for large accounts, rate for small
- Maximizes information available while minimizing bias

### Decision 2: RobustScaler for Log-Transformed Features

**Chose:** RobustScaler (median/IQR based)  
**Over:** StandardScaler (mean/std based)  
**Rationale:**
- Engagement data has outliers (occasional viral posts)
- StandardScaler: One 100K-like post affects scaling for all future data
- RobustScaler: Uses median (50th percentile) and IQR (25th-75th range)
- More stable to extreme values in historical data

### Decision 3: Tier-Based Percentile Features

**Chose:** Calculate percentiles within follower tiers  
**Over:** Global percentiles across all accounts  
**Rationale:**
- "Good engagement" is context-dependent: 200 likes amazing for 2K account, poor for 500K
- 80th percentile engagement for micro ≠ 80th percentile for macro
- Tier-specific percentiles capture: "This creator performs well for their size"
- More actionable: Creators compare themselves to similar accounts

### Decision 4: Stratified Evaluation, Not Stratified Sampling

**Chose:** Evaluate metrics separately by tier; don't oversample tiers in training  
**Over:** Weighted sampling to balance tier distribution in training data  
**Rationale:**
- Dataset reflects real-world distribution (more large accounts in sample)
- Oversampling risks overfitting to underrepresented patterns
- Natural distribution trains on actual data patterns
- Stratified metrics surface fairness issues without distorting training
- If fairness problems persist, can experiment with sampling later

### Decision 5: Keep Followers for Normalization, Exclude from Model

**Chose:** Use `followers` to create rate features; don't input raw `followers` to model  
**Over:** Remove followers column entirely from pipeline  
**Rationale:**
- Followers needed to calculate engagement/followers (rate features)
- V3 already proved raw followers unnecessary for accuracy
- Rate features encode "typical performance" without raw scale bias
- Cleaner than alternatives (e.g., training separate models per tier)

### Decision 6: Log Transformation Before Scaling

**Chose:** Apply log1p first, then RobustScaler  
**Over:** Scale first, then log  
**Rationale:**
- Log addresses inherent distribution skew (exponential → normal)
- Scaling after log normalizes already-compressed values
- Consistent with best practices: transform distributions before standardizing
- Prevents scaler from seeing extreme raw values (10,000 vs 50)

---

## Production Considerations

### Save These Artifacts for Deployment

1. **Transformation metadata:**
   - `data/normalization_metadata.json`
   - Contains: tier boundaries, tier medians, transformation date
   
2. **Scaler object:**
   - `models_v4/robust_scaler.pkl`
   - Must apply same scaling in production

3. **Feature list:**
   - `models_v4/feature_names.json`
   - Ensures correct order and presence of features

4. **Model versioning:**
   - `models_v4/model_metadata_v4.json`
   - Documents transformation pipeline for reproducibility

### Production Pipeline

```python
# 1. Load artifacts
with open('normalization_metadata.json') as f:
    norm_meta = json.load(f)
scaler = joblib.load('robust_scaler.pkl')
model = joblib.load('xgboost_model_v4.pkl')

# 2. Apply same transformations to new post
def predict_engagement(post_data):
    # Extract features from post
    features = extract_features(post_data)
    
    # Log transform
    for col in ['influencer_avg_engagement', 'followers', ...]:
        features[f'log_{col}'] = np.log1p(features[col])
    
    # Calculate rates
    features['influencer_engagement_rate'] = features['influencer_avg_engagement'] / features['followers']
    
    # Assign tier
    features['follower_tier'] = assign_tier(features['followers'], norm_meta['tier_boundaries'])
    features = pd.get_dummies(features, columns=['follower_tier'])
    
    # Calculate percentiles (using training tier medians)
    tier = features['follower_tier_encoded']
    features['engagement_vs_tier_median'] = features['influencer_avg_engagement'] / norm_meta['tier_medians'][tier]
    
    # Scale log features
    log_cols = [col for col in features.columns if col.startswith('log_')]
    features[log_cols] = scaler.transform(features[log_cols])
    
    # Predict
    return model.predict(features)
```

### Monitoring in Production

**Track fairness metrics over time:**

```python
# Weekly fairness audit
def audit_predictions(week_data):
    predictions = week_data['predicted_engagement']
    actuals = week_data['actual_engagement']  # After posts published
    tiers = week_data['follower_tier']
    
    fairness_metrics = evaluate_fairness_by_tier(actuals, predictions, tiers)
    
    # Alert if fairness score degrades
    if fairness_metrics['fairness_score'] > 1.8:
        send_alert("Fairness degradation detected")
    
    # Log for dashboard
    log_metrics_to_dashboard(fairness_metrics)
```

**Monitor for distribution drift:**

```python
# Check if new data differs from training data
def check_distribution_drift(new_data, training_stats):
    for feature in ['log_followers', 'influencer_engagement_rate']:
        new_mean, new_std = new_data[feature].mean(), new_data[feature].std()
        train_mean, train_std = training_stats[feature]['mean'], training_stats[feature]['std']
        
        z_score = abs(new_mean - train_mean) / train_std
        
        if z_score > 3:
            warn(f"Distribution drift in {feature}: {z_score:.2f} std from training")
```

---

## Testing Plan

### Unit Tests

```python
def test_log_transformation():
    """Verify log1p handles zeros and positive values correctly"""
    assert np.log1p(0) == 0
    assert np.isclose(np.log1p(10000), 9.21044, atol=0.001)

def test_rate_calculation():
    """Verify engagement rate calculated correctly"""
    engagement = 1000
    followers = 50000
    expected_rate = 0.02  # 2%
    assert engagement / followers == expected_rate

def test_tier_assignment():
    """Verify tier boundaries work correctly"""
    assert assign_tier(3000, tier_bounds) == 'micro'
    assert assign_tier(5000, tier_bounds) == 'small'
    assert assign_tier(600000, tier_bounds) == 'macro'

def test_no_data_leakage():
    """Verify no target variables used in feature engineering"""
    features = engineer_features(post_data)
    assert 'reactions' not in features.columns
    assert 'comments' not in features.columns
```

### Integration Tests

```python
def test_end_to_end_prediction():
    """Test full pipeline on synthetic data"""
    test_post = {
        'content': 'Never thought I'd share this...',
        'followers': 10000,
        'influencer_avg_engagement': 500
    }
    
    prediction = predict_engagement(test_post)
    
    # Should return reasonable value
    assert 0 < prediction < 10000  # Not negative or absurdly high
    assert isinstance(prediction, (int, float))

def test_extreme_values():
    """Test handling of edge cases"""
    # Very small account
    tiny_account = {'followers': 100, 'influencer_avg_engagement': 5}
    pred1 = predict_engagement(tiny_account)
    
    # Very large account
    huge_account = {'followers': 10000000, 'influencer_avg_engagement': 500000}
    pred2 = predict_engagement(huge_account)
    
    # Both should return valid predictions (no errors/NaN)
    assert not np.isnan(pred1)
    assert not np.isnan(pred2)
    
    # Should reflect magnitude difference (but not linearly due to normalization)
    assert pred2 > pred1
```

### Fairness Validation Tests

```python
def test_fairness_score_improves():
    """V4 fairness score should be better than V3"""
    v3_fairness = calculate_fairness_score(v3_model, test_data)
    v4_fairness = calculate_fairness_score(v4_model, test_data)
    
    assert v4_fairness < v3_fairness, "V4 should have lower (better) fairness score"
    assert v4_fairness < 1.5, "V4 should meet fairness target"

def test_small_account_mape_improves():
    """Small accounts should get better predictions in V4"""
    small_account_mask = (test_data['follower_tier'] == 'small')
    
    v3_mape = mape(test_data.loc[small_account_mask, 'actual'], 
                   test_data.loc[small_account_mask, 'v3_pred'])
    v4_mape = mape(test_data.loc[small_account_mask, 'actual'],
                   test_data.loc[small_account_mask, 'v4_pred'])
    
    improvement = (v3_mape - v4_mape) / v3_mape
    assert improvement >= 0.20, "Should achieve ≥20% MAPE improvement for small accounts"
```

---

## Timeline

| Phase | Task | Duration | Dependencies |
|-------|------|----------|--------------|
| 1 | Create `05_feature_normalization.ipynb` | 2-3 hours | Existing feature_engineered_data.csv |
| 2 | Update `04_feature_selection.ipynb` | 1-2 hours | Phase 1 complete |
| 3 | Create `06_model_training_v4.ipynb` | 3-4 hours | Phase 2 complete |
| 4 | Fairness evaluation & reporting | 2 hours | Phase 3 complete |
| 5 | Update `07_model_testing.ipynb` | 1-2 hours | Phase 3 complete |
| 6 | Documentation & comparison report | 1-2 hours | All phases complete |

**Total Estimated Time:** 10-15 hours

---

## Risks & Mitigations

### Risk 1: Overall Accuracy Drops >5%

**Likelihood:** Medium  
**Impact:** High (V4 wouldn't be production-ready)

**Mitigation:**
- Start with hybrid approach (absolute + relative features)
- If accuracy drops too much, adjust feature mix:
  - Keep more log-transformed absolute features
  - Reduce weight on rate features
  - Use ensemble: V3 for large accounts, V4 for small accounts

**Fallback:** If V4 underperforms, document findings and keep V3 for production with caveat about fairness limitations

### Risk 2: Fairness Doesn't Improve Significantly

**Likelihood:** Low (research shows log + rates effective)  
**Impact:** Medium (effort wasted but learning gained)

**Mitigation:**
- If fairness score >1.5 after normalization, try:
  - More aggressive transformation (percentile ranks instead of log)
  - Stratified sampling (oversample small accounts)
  - Fairness-aware loss function
  - Separate models per tier

### Risk 3: Production Pipeline Becomes Too Complex

**Likelihood:** Medium  
**Impact:** Low (manageable complexity)

**Mitigation:**
- Document transformation pipeline clearly
- Create helper functions for feature engineering
- Package transformations into sklearn Pipeline object:
  ```python
  from sklearn.pipeline import Pipeline
  pipeline = Pipeline([
      ('feature_engineering', CustomTransformer()),
      ('scaling', RobustScaler()),
      ('model', XGBRegressor())
  ])
  ```
- Comprehensive unit tests for each transformation step

### Risk 4: New Accounts Outside Training Distribution

**Likelihood:** High (new mega-influencers emerge)  
**Impact:** Low (edge cases only)

**Mitigation:**
- Use `np.clip()` for extreme values:
  ```python
  features['log_followers'] = np.clip(features['log_followers'], 
                                     TRAINING_MIN, TRAINING_MAX)
  ```
- Set upper tier for 1M+ followers (all treated similarly)
- Monitor for distribution drift in production
- Retrain periodically with new data

---

## Next Steps

### Immediate Actions

1. ☐ Create `05_feature_normalization.ipynb` (see Step 1 implementation)
2. ☐ Run feature normalization on full dataset
3. ☐ Verify no NaN values introduced
4. ☐ Check distribution of new features (histograms, summary stats)

### After Normalization

5. ☐ Update `04_feature_selection.ipynb` with normalized features
6. ☐ Re-run feature selection pipeline
7. ☐ Compare selected features V3 vs V4

### Model Training

8. ☐ Create `06_model_training_v4.ipynb`
9. ☐ Train models with stratified split
10. ☐ Calculate fairness metrics by tier
11. ☐ Generate fairness visualizations

### Evaluation & Decision

12. ☐ Create V3 vs V4 comparison report
13. ☐ Review with stakeholders
14. ☐ **Decision point:** Use V3, V4, or hybrid approach for production?

### If V4 Successful

15. ☐ Update production pipeline with transformations
16. ☐ Create API endpoint with V4 model
17. ☐ Set up fairness monitoring dashboard
18. ☐ Document for future maintainers

---

## References

### Internal Documents
- [FEATURE_ENGINEERING_GUIDE.md](FEATURE_ENGINEERING_GUIDE.md) - Original feature engineering plan
- [MODEL_DEVELOPMENT_PLAN.md](MODEL_DEVELOPMENT_PLAN.md) - Overall modeling strategy
- [notebooks/06_model_training_v3.ipynb](notebooks/06_model_training_v3.ipynb) - Current best model (baseline)
- [notebooks/V2_VS_V3_COMPARISON_REPORT.md](notebooks/V2_VS_V3_COMPARISON_REPORT.md) - Evidence that excluding followers didn't hurt accuracy

### External Research
- Log transformation for skewed distributions: Standard ML best practice
- RobustScaler for outlier-prone data: Scikit-learn documentation
- Fairness metrics in ML: Fairlearn toolkit concepts
- Stratified evaluation: Standard practice for demographic fairness

---

## Appendix: Feature Mapping

### Raw → Normalized Features

| Original Feature | New Features | Transformation Type |
|-----------------|--------------|---------------------|
| `influencer_avg_engagement` | `log_influencer_avg_engagement` | Log transform |
| | `influencer_engagement_rate` | Rate (÷ followers) |
| | `percentile_engagement_in_tier` | Tier percentile |
| | `engagement_vs_tier_median` | Tier-relative |
| `influencer_total_engagement` | `log_influencer_total_engagement` | Log transform |
| `influencer_avg_reactions` | `log_influencer_avg_reactions` | Log transform |
| | `influencer_reactions_rate` | Rate (÷ followers) |
| | `percentile_reactions_in_tier` | Tier percentile |
| `influencer_avg_comments` | `log_influencer_avg_comments` | Log transform |
| | `influencer_comments_rate` | Rate (÷ followers) |
| | `percentile_comments_in_tier` | Tier percentile |
| `influencer_median_reactions` | `log_influencer_median_reactions` | Log transform |
| `influencer_median_comments` | `log_influencer_median_comments` | Log transform |
| `influencer_std_reactions` | (Keep as-is, already relative) | None |
| `influencer_std_comments` | (Keep as-is, already relative) | None |
| `followers` | `log_followers` | Log transform |
| | `follower_tier` | Categorical binning |
| | `is_tier_*` (5 features) | One-hot encoding |

**Total New Features:** ~20 normalized variants replacing 9 raw features

---

## Document Status

**Current Version:** 1.0  
**Last Updated:** February 13, 2026  
**Status:** 📋 READY TO IMPLEMENT

**Reviewers:**
- [ ] Data Scientist (verify no data leakage)
- [ ] ML Engineer (validate production feasibility)
- [ ] Product Owner (confirm fairness requirements)

**Approval Required Before:**
- Starting implementation (estimated 10-15 hours)
- Production deployment (after V4 validation)

---

**Questions or concerns? Contact project lead before proceeding with implementation.**
