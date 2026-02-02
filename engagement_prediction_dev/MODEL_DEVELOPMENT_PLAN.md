# Engagement Prediction Model Development Plan
## TrendPilot LinkedIn Edition - Capstone Project

**Date Created:** February 1, 2026  
**Objective:** Build ML models to predict LinkedIn post engagement (reactions and comments)

---

## 1. EXECUTIVE SUMMARY

### Project Goal
Develop machine learning models that predict:
- **Number of Reactions (Likes)** - Primary engagement metric
- **Number of Comments** - Secondary engagement metric

### Approach
1. Leverage existing base score formula as foundation
2. Enhance with advanced NLP features
3. Build separate models for reactions and comments
4. Integrate content analysis, sentiment, and pattern detection

### Key Success Metrics
- **Reactions Model:** R² > 0.65, RMSE < 150
- **Comments Model:** R² > 0.55, RMSE < 20
- **Base Score Enhancement:** Improve upon algorithmic scoring by 25%+

---

## 2. DATA OVERVIEW

### Dataset Characteristics
- **Source:** Kaggle - LinkedIn Influencers Data
- **Total Records:** 34,012 posts
- **Influencers:** 69 verified professionals
- **Data Completeness:** 86.7% overall, 95.4% for core features

### Key Features Available
✅ **Complete Data (100%):**
- reactions (target variable)
- comments (target variable)
- content text (94.1% available)
- media_type
- num_hashtags

⚠️ **Limited/Missing Data:**
- views (100% missing - cannot use)
- location (6.68% missing)
- timestamps (relative only, not absolute)

### Data Quality Assessment
- **Status:** GOOD - suitable for ML modeling
- **ML-Ready Rows:** ~27,000+ with complete features
- **Target Variables:** Both fully populated, no missing values

---

## 3. FEATURE ENGINEERING STRATEGY

### 3.1 Base Formula Integration
**From `base_score_calculation.txt`:**

Implement algorithmic features that mirror the existing scoring logic:

```python
# Content Length Features
- word_count (100-200 optimal)
- word_count_category (short/optimal/long/too_long)
- length_score (based on base formula ranges)

# Hook Pattern Detection (First Sentence)
- has_never_narrative
- has_specific_time_hook
- has_quote_hook
- has_contrarian_hook
- has_belief_transformation
- hook_type (categorical)
- hook_score (weighted)

# Power Pattern Detection (Full Content)
- has_underdog_story
- has_transformation_narrative
- has_vulnerability
- has_family_story
- has_specific_numbers
- has_list_format
- power_pattern_count
- power_pattern_score

# Visual Content
- media_type_encoded
- has_video (highest engagement)
- has_carousel
- has_image

# Links & Promotion
- has_external_link (penalty factor)
- link_penalty_score
- promotional_intensity (0-10 scale)
- is_heavy_promotion
```

### 3.2 Advanced NLP Features

#### A. Text Statistics
```python
- sentence_count
- avg_sentence_length
- unique_word_ratio
- punctuation_density
- emoji_count
- emoji_types
- question_count
- exclamation_count
```

#### B. Sentiment Analysis
Using VADER or TextBlob:
```python
- sentiment_polarity (-1 to 1)
- sentiment_subjectivity (0 to 1)
- sentiment_category (positive/neutral/negative)
- emotion_intensity
```

#### C. Named Entity Recognition (NER)
Using spaCy:
```python
- person_mentions_count
- organization_mentions_count
- location_mentions_count
- has_celebrity_mention
- entity_diversity_score
```

#### D. Topic Modeling
Using LDA or BERTopic:
```python
- dominant_topic_id
- topic_confidence_score
- topic_label (from Topic_modelling.ipynb)
- is_professional_topic (career/tech/business)
```

#### E. Readability Metrics
```python
- flesch_reading_ease
- flesch_kincaid_grade
- gunning_fog_index
- readability_category (easy/medium/hard)
```

#### F. Engagement Triggers
Pattern matching for viral elements:
```python
- has_call_to_action
- has_storytelling_elements
- has_personal_experience
- has_actionable_advice
- has_controversial_stance
- authenticity_score (0-10)
```

#### G. Hashtag Features
```python
- num_hashtags (already exists)
- avg_hashtag_length
- has_trending_hashtags
- hashtag_relevance_score
```

#### H. TF-IDF or Word Embeddings
```python
- TF-IDF vectors (top 100-200 features)
- OR sentence embeddings using Sentence-BERT
- content_embedding (768-dimensional for BERT)
```

### 3.3 Derived Features
```python
# Engagement rate context
- follower_engagement_ratio
- influencer_avg_reactions (historical)
- influencer_avg_comments (historical)
- post_deviation_from_avg (over/under performance)

# Temporal features (if extractable)
- is_weekday
- time_category (morning/afternoon/evening)

# Combined features
- content_quality_score (length + readability + patterns)
- virality_potential_score (hooks + power patterns + media)
- authenticity_vs_promotion_ratio
```

---

## 4. MODEL DEVELOPMENT PIPELINE

### Phase 1: Data Preparation

#### Step 1.1: Data Loading & Initial Cleaning
```python
Notebook: notebooks/01_data_loading_cleaning.ipynb

Tasks:
- Load influencers_data.csv
- Remove duplicates
- Drop posts with missing content (5.9%)
- Drop posts with missing reactions/comments (if any)
- Validate data types
- Handle outliers (reactions > 99th percentile - cap or log transform)

Outputs:
- cleaned_data.csv
- reports/01_data_loading_cleaning_REPORT.md (detailed justification)

Notebook Requirements:
- Well-commented cells explaining each operation
- Markdown sections describing the rationale
- Visualizations with interpretations
- Statistical summaries with business context
```

#### Step 1.2: Text Preprocessing
```python
Notebook: notebooks/02_text_preprocessing.ipynb

Tasks:
- Lowercase conversion
- URL extraction (for link penalty feature)
- Mention extraction (@username)
- Hashtag extraction and cleaning
- Emoji extraction and counting
- Special character handling
- Create clean_content column

Outputs:
- preprocessed_data.csv
- reports/02_text_preprocessing_REPORT.md (detailed justification)

Notebook Requirements:
- Explain preprocessing choices (why lowercase, how to handle URLs)
- Show before/after examples
- Document regex patterns used
- Justify text normalization decisions
```

#### Step 1.3: Feature Engineering
```python
Notebook: notebooks/03_feature_engineering.ipynb

Tasks:
1. Base Formula Features (from base_score_calculation.txt)
   - Implement all scoring logic as features
   - Calculate base_score_algorithmic
   
2. NLP Features
   - Text statistics
   - Sentiment analysis
   - NER extraction
   - Readability metrics
   - Pattern detection
   
3. Topic Features
   - Load topic labels from Topic_modelling.ipynb output
   - Merge with main dataset
   
4. Derived Features
   - Influencer-level aggregations
   - Engagement ratios

Outputs:
- feature_engineered_data.csv
- reports/03_feature_engineering_REPORT.md (detailed justification)

Notebook Requirements:
- Explain each feature group's purpose and ML benefit
- Show feature distributions and statistics
- Justify why each feature is valuable for predictions
- Document feature engineering decisions with examples
- Include correlation analysis with targets
```

#### Step 1.4: Feature Selection & Encoding
```python
Notebook: notebooks/04_feature_selection_encoding.ipynb

Tasks:
- Encode categorical variables
- Scale numerical features (StandardScaler/RobustScaler)
- Handle multicollinearity (VIF analysis)
- Select top features (correlation, mutual information)
- Create final feature matrix

Outputs:
- model_ready_data.csv
- feature_list.json
- scaler.pkl
- reports/04_feature_selection_encoding_REPORT.md (detailed justification)

Notebook Requirements:
- Justify encoding choices (one-hot vs label vs target)
- Explain scaler selection (Standard vs Robust vs MinMax)
- Document VIF thresholds and multicollinearity decisions
- Show feature selection metrics with rationale
- Visualize feature importance and correlations
```

### Phase 2: Model Development

#### Step 2.1: Baseline Models
```python
Notebook: notebooks/05_baseline_models.ipynb

Models to Test:
1. Linear Regression (interpretable baseline)
2. Ridge Regression (L2 regularization)
3. Lasso Regression (L1 regularization, feature selection)
4. ElasticNet (L1 + L2)

Evaluation:
- R² Score
- RMSE
- MAE
- MAPE

Outputs:
- baseline_results.json
- reports/05_baseline_models_REPORT.md (detailed justification)

Notebook Requirements:
- Explain why each baseline is important
- Document hyperparameter choices
- Compare models with statistical tests
- Analyze coefficient interpretations
- Justify train/validation/test split strategy
```

#### Step 2.2: Tree-Based Models
```python
Notebook: notebooks/06_tree_models.ipynb

Models to Test:
1. Random Forest Regressor
   - Good for non-linear relationships
   - Feature importance analysis
   
2. Gradient Boosting (XGBoost/LightGBM)
   - Often best performance
   - Handle complex interactions
   
3. CatBoost
   - Native categorical encoding
   - Robust to overfitting

Hyperparameter Tuning:
- GridSearchCV or RandomizedSearchCV
- Cross-validation (5-fold)

Outputs:
- tree_model_results.json
- best_model_reactions.pkl
- best_model_comments.pkl
- reports/06_tree_models_REPORT.md (detailed justification)

Notebook Requirements:
- Explain hyperparameter search space choices
- Document cross-validation strategy and folds
- Justify model selection decisions
- Show feature importance comparisons
- Analyze overfitting indicators
```

#### Step 2.3: Advanced Models (Optional)
```python
Notebook: notebooks/07_advanced_models.ipynb

Models to Test:
1. Neural Network (MLP)
   - Deep learning for complex patterns
   - Multiple hidden layers
   
2. Ensemble Model
   - Stacking: combine multiple models
   - Weighted average of predictions

Outputs:
- advanced_model_results.json
- reports/07_advanced_models_REPORT.md (detailed justification)

Notebook Requirements:
- Explain architecture choices (layers, neurons, activation)
- Document training process (epochs, batch size, early stopping)
- Justify ensemble strategies
- Compare with simpler models (complexity vs performance)
```

#### Step 2.4: Model Comparison & Selection
```python
Notebook: notebooks/08_model_comparison.ipynb

Tasks:
- Compare all models side-by-side
- Analyze feature importance
- Check for overfitting (train vs. validation)
- Select best model for each target
- Generate visualizations

Outputs:
- model_comparison_report.html
- feature_importance_charts.png
- reports/08_model_comparison_REPORT.md (detailed justification)

Notebook Requirements:
- Explain model selection criteria and weights
- Justify final model choices with multiple metrics
- Document trade-offs (interpretability vs accuracy)
- Show comprehensive comparison tables
```

### Phase 3: Model Evaluation & Validation

#### Step 3.1: Performance Analysis
```python
Notebook: notebooks/09_model_evaluation.ipynb

Metrics:
- R² Score (train, validation, test)
- RMSE
- MAE
- MAPE
- Residual analysis

Visualizations:
- Actual vs. Predicted scatter plots
- Residual distribution
- Error by feature ranges
- Feature importance

Outputs:
- evaluation_report.html
- reports/09_model_evaluation_REPORT.md (detailed justification)

Notebook Requirements:
- Interpret each metric in business context
- Explain residual patterns and their implications
- Justify success criteria and thresholds
- Document limitations and assumptions
```

#### Step 3.2: Error Analysis
```python
Notebook: notebooks/10_error_analysis.ipynb

Tasks:
- Identify posts with high prediction errors
- Analyze common characteristics of errors
- Check if base formula outperforms for certain types
- Suggest model improvements

Outputs:
- reports/10_error_analysis_REPORT.md (detailed justification)

Notebook Requirements:
- Categorize errors by type (over-prediction vs under-prediction)
- Explain patterns in high-error cases
- Justify potential improvements
- Document edge cases and model weaknesses
```

#### Step 3.3: Model Interpretation
```python
Notebook: notebooks/11_model_interpretation.ipynb

Tasks:
- SHAP values for model explainability
- Partial dependence plots
- Feature interaction analysis
- Generate insights for content creators

Outputs:
- interpretation_report.html
- shap_summary_plots.png
- reports/11_model_interpretation_REPORT.md (detailed justification)

Notebook Requirements:
- Explain SHAP values in layman's terms
- Translate technical findings to actionable insights
- Justify interpretation methodology
- Document feature interactions and their business meaning
```

---

## 5. IMPLEMENTATION ROADMAP

### Week 1: Data Preparation
- [ ] Day 1-2: Data loading, cleaning, validation
- [ ] Day 3-4: Text preprocessing, base formula implementation
- [ ] Day 5-7: NLP feature engineering, topic integration

### Week 2: Model Development
- [ ] Day 1-2: Baseline models (linear, regularized)
- [ ] Day 3-5: Tree-based models (RF, XGBoost, LightGBM)
- [ ] Day 6-7: Hyperparameter tuning, model comparison

### Week 3: Evaluation & Refinement
- [ ] Day 1-2: Comprehensive evaluation, error analysis
- [ ] Day 3-4: Model interpretation (SHAP, feature importance)
- [ ] Day 5-6: Model refinement based on insights
- [ ] Day 7: Final model selection and documentation

### Week 4: Integration & Deployment Prep
- [ ] Day 1-2: Create prediction pipeline script
- [ ] Day 3-4: Build API wrapper for model serving
- [ ] Day 5-6: Integration with Streamlit app
- [ ] Day 7: Final testing and documentation

---

## 6. TECHNICAL STACK

### Core Libraries
```python
# Data Processing
pandas>=2.0.0
numpy>=1.24.0

# NLP
nltk>=3.8
spacy>=3.7.0
textblob>=0.17.0
vaderSentiment>=3.3.2
sentence-transformers>=2.2.0  # for BERT embeddings

# Machine Learning
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
catboost>=1.2.0

# Deep Learning (Optional)
tensorflow>=2.15.0  # or pytorch>=2.0.0

# Model Interpretation
shap>=0.42.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0

# Utilities
joblib>=1.3.0
tqdm>=4.65.0
```

### Environment Setup
```bash
# Create virtual environment
python -m venv venv_model_dev
source venv_model_dev/bin/activate  # Windows: venv_model_dev\Scripts\activate

# Install dependencies
pip install -r requirements_model.txt

# Download NLP models
python -m spacy download en_core_web_sm
python -m nltk.downloader all
```

---

## 7. MODEL OUTPUTS & ARTIFACTS

### Training Artifacts
```
engagement_prediction_dev/
├── models/
│   ├── reactions_model_v1.pkl
│   ├── comments_model_v1.pkl
│   ├── scaler.pkl
│   ├── encoder.pkl
│   └── feature_config.json
│
├── data/
│   ├── cleaned_data.csv
│   ├── preprocessed_data.csv
│   ├── feature_engineered_data.csv
│   ├── model_ready_data.csv
│   ├── train_split.csv
│   ├── val_split.csv
│   └── test_split.csv
│
├── results/
│   ├── baseline_results.json
│   ├── tree_model_results.json
│   ├── model_comparison_report.html
│   ├── evaluation_report.html
│   └── feature_importance_charts.png
│
├── notebooks/                          # All development notebooks
│   ├── 01_data_loading_cleaning.ipynb
│   ├── 02_text_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_feature_selection_encoding.ipynb
│   ├── 05_baseline_models.ipynb
│   ├── 06_tree_models.ipynb
│   ├── 07_advanced_models.ipynb       # Optional
│   ├── 08_model_comparison.ipynb
│   ├── 09_model_evaluation.ipynb
│   ├── 10_error_analysis.ipynb
│   └── 11_model_interpretation.ipynb
│
└── reports/                            # Detailed justification reports
    ├── 01_data_loading_cleaning_REPORT.md
    ├── 02_text_preprocessing_REPORT.md
    ├── 03_feature_engineering_REPORT.md
    ├── 04_feature_selection_encoding_REPORT.md
    ├── 05_baseline_models_REPORT.md
    ├── 06_tree_models_REPORT.md
    ├── 07_advanced_models_REPORT.md
    ├── 08_model_comparison_REPORT.md
    ├── 09_model_evaluation_REPORT.md
    ├── 10_error_analysis_REPORT.md
    └── 11_model_interpretation_REPORT.md
```

### Final Model API
```python
# Prediction interface
class EngagementPredictor:
    def __init__(self, model_dir):
        self.reactions_model = load_model(...)
        self.comments_model = load_model(...)
        self.preprocessor = load_preprocessor(...)
    
    def predict(self, post_content, media_type, num_hashtags, followers):
        """
        Returns:
        {
            'predicted_reactions': int,
            'predicted_comments': int,
            'base_score': int,
            'confidence_interval': tuple,
            'key_features': dict,
            'recommendations': list
        }
        """
        pass
```

---

## 8. RISK MITIGATION

### Identified Risks

| Risk | Impact | Mitigation Strategy |
|------|--------|-------------------|
| 100% missing views data | High | Focus on reactions/comments only; cannot predict reach |
| Relative timestamps only | Medium | Use time_spent patterns, drop time-of-day features |
| Imbalanced data (few high-engagement posts) | High | Use log transformation, stratified sampling |
| Overfitting on specific influencers | High | Train/test split by influencer, cross-validation |
| Topic model drift | Medium | Retrain topic model periodically |
| Base formula already good | Medium | Treat as baseline, enhance with ML |

### Data Quality Checks
- [ ] Remove duplicates (exact content match)
- [ ] Cap outliers at 99th percentile
- [ ] Validate reactions/comments are non-negative
- [ ] Check for data leakage (future info in features)
- [ ] Ensure train/test temporal split if possible

---

## 9. EVALUATION CRITERIA

### Model Performance Thresholds

#### Reactions Model
- **Minimum Acceptable:** R² > 0.50
- **Target:** R² > 0.65
- **Stretch Goal:** R² > 0.75

#### Comments Model
- **Minimum Acceptable:** R² > 0.40
- **Target:** R² > 0.55
- **Stretch Goal:** R² > 0.65

### Business Value Metrics
- Prediction accuracy within ±30% for 70% of posts
- Actionable insights for content improvement
- Outperform base formula by 25% in RMSE

---

## 10. NEXT STEPS

### Immediate Actions (This Week)
1. ✅ Create MODEL_DEVELOPMENT_PLAN.md (this document)
2. ⬜ Set up project structure and requirements.txt
3. ⬜ Implement 01_data_loading.py and 02_text_preprocessing.py
4. ⬜ Build base formula feature extraction (03_feature_engineering.py)
5. ⬜ Start NLP feature engineering

### Follow-up Tasks
- Schedule weekly progress reviews
- Document all decisions and experiments
- Create Jupyter notebooks for exploration
- Prepare presentation of results

---

## 11. SUCCESS DEFINITION

### Project is Successful if:
✅ Both models achieve minimum R² thresholds  
✅ Models outperform base formula on test set  
✅ Feature importance aligns with domain knowledge  
✅ Predictions are explainable (SHAP analysis)  
✅ Integration-ready prediction API is built  
✅ Comprehensive documentation is provided  

---

**Document Status:** Draft v1.0  
**Last Updated:** February 1, 2026  
**Next Review:** After Phase 1 completion
