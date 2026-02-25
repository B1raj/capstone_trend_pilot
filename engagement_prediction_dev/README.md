# Engagement Prediction Model Development

This folder contains all scripts, notebooks, and artifacts for developing machine learning models to predict LinkedIn post engagement (reactions and comments).

## 📁 Project Structure

```
engagement_prediction_dev/
├── notebooks/            # Jupyter notebooks for complete pipeline
│   ├── 01_data_loading_cleaning.ipynb
│   ├── 02_text_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_feature_selection_encoding.ipynb
│   ├── 05_baseline_models.ipynb
│   ├── 06_tree_models.ipynb
│   ├── 07_advanced_models.ipynb
│   ├── 08_model_comparison.ipynb
│   ├── 09_model_evaluation.ipynb
│   ├── 10_error_analysis.ipynb
│   └── 11_model_interpretation.ipynb
│
├── reports/              # Detailed justification reports for each step
│   ├── 01_data_loading_cleaning_REPORT.md
│   ├── 02_text_preprocessing_REPORT.md
│   ├── 03_feature_engineering_REPORT.md
│   ├── 04_feature_selection_encoding_REPORT.md
│   ├── 05_baseline_models_REPORT.md
│   ├── 06_tree_models_REPORT.md
│   ├── 08_model_comparison_REPORT.md
│   ├── 09_model_evaluation_REPORT.md
│   ├── 10_error_analysis_REPORT.md
│   └── 11_model_interpretation_REPORT.md
│
├── data/                 # Processed datasets
│   ├── cleaned_data.csv
│   ├── preprocessed_data.csv
│   ├── feature_engineered_data.csv
│   ├── model_ready_data.csv
│   ├── train_split.csv
│   ├── val_split.csv
│   └── test_split.csv
│
├── models/               # Trained model artifacts
│   ├── reactions_model_v1.pkl
│   ├── comments_model_v1.pkl
│   ├── scaler.pkl
│   ├── encoder.pkl
│   └── feature_config.json
│
├── results/              # Evaluation results and reports
│   ├── baseline_results.json
│   ├── tree_model_results.json
│   ├── model_comparison_report.html
│   ├── evaluation_report.html
│   └── feature_importance_charts.png
│
├── MODEL_DEVELOPMENT_PLAN.md  # Comprehensive development plan
├── requirements_model.txt      # Python dependencies
└── README.md                   # This file
```

## 🎯 Objective

Build two separate regression models:
1. **Reactions Model**: Predict number of likes/reactions
2. **Comments Model**: Predict number of comments

## 📊 Dataset

- **Source**: Kaggle - LinkedIn Influencers Data
- **Size**: 34,012 posts from 69 influencers
- **Target Variables**: reactions (100% complete), comments (100% complete)
- **Key Features**: content text, media_type, hashtags, followers

## 🔧 Setup

### 1. Create Virtual Environment
```bash
python -m venv venv_model_dev
source venv_model_dev/bin/activate  # On Windows: venv_model_dev\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements_model.txt
```

### 3. Download NLP Models
```bash
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt stopwords wordnet vader_lexicon
```

## 🚀 Quick Start

### Development Approach

**We use Jupyter Notebooks for the entire pipeline**, NOT Python scripts. Each notebook:
- Contains detailed comments explaining each operation
- Includes markdown sections with rationale and justifications
- Shows visualizations with interpretations
- Generates a comprehensive report documenting decisions

### Notebook Execution

```bash
# Step 1: Data loading and cleaning
jupyter notebook notebooks/01_data_loading_cleaning.ipynb
# Output: data/cleaned_data.csv + reports/01_data_loading_cleaning_REPORT.md

# Step 2: Text preprocessing
jupyter notebook notebooks/02_text_preprocessing.ipynb
# Output: data/preprocessed_data.csv + reports/02_text_preprocessing_REPORT.md

# Step 3: Feature engineering
jupyter notebook notebooks/03_feature_engineering.ipynb
# Output: data/feature_engineered_data.csv + reports/03_feature_engineering_REPORT.md

# Step 4: Feature selection and encoding
jupyter notebook notebooks/04_feature_selection_encoding.ipynb
# Output: data/model_ready_data.csv + reports/04_feature_selection_encoding_REPORT.md

# Step 5: Train baseline models
jupyter notebook notebooks/05_baseline_models.ipynb
# Output: results/baseline_results.json + reports/05_baseline_models_REPORT.md

# Step 6: Train tree-based models
jupyter notebook notebooks/06_tree_models.ipynb
# Output: models/*.pkl + reports/06_tree_models_REPORT.md

# Step 7: Model comparison
jupyter notebook notebooks/08_model_comparison.ipynb
# Output: results/model_comparison_report.html + reports/08_model_comparison_REPORT.md

# Step 8: Evaluate best models
jupyter notebook notebooks/09_model_evaluation.ipynb
# Output: results/evaluation_report.html + reports/09_model_evaluation_REPORT.md
```

### Report Requirements

Each step produces a detailed Markdown report that includes:
- **Rationale**: Why each decision was made
- **Alternatives Considered**: What other approaches were evaluated
- **Justifications**: Statistical and business reasoning
- **Trade-offs**: Pros and cons of chosen approach
- **Results**: Quantitative outcomes and quality metrics
- **Recommendations**: Next steps and improvements

## 📈 Features

### Base Formula Features (from base_score_calculation.txt)
- Content length categories and scores
- Hook pattern detection (12+ types)
- Power pattern detection (15+ types)
- Media type encoding
- Link penalties
- Promotional content detection

### Advanced NLP Features
- Sentiment analysis (polarity, subjectivity)
- Named Entity Recognition (persons, orgs, locations)
- Topic modeling labels
- Readability metrics (Flesch, Gunning Fog)
- Text statistics (sentence count, unique words, etc.)
- Engagement trigger patterns

### Derived Features
- Influencer-level statistics
- Engagement rates
- Content quality scores
- Virality potential scores

## 🎯 Model Performance Goals

### Reactions Model
- **Minimum**: R² > 0.50
- **Target**: R² > 0.65
- **Stretch**: R² > 0.75

### Comments Model
- **Minimum**: R² > 0.40
- **Target**: R² > 0.55
- **Stretch**: R² > 0.65

## 📝 Development Workflow

### Notebook Documentation Standards

All notebooks must include:

1. **Markdown Headers**: Clear section titles with explanations
2. **Code Comments**: Inline comments for complex operations
3. **Rationale Sections**: Explain WHY decisions were made
4. **Visualizations**: Charts with interpretations
5. **Statistical Summaries**: Metrics with business context
6. **Decision Justifications**: Document alternatives considered

### Timeline

1. **Data Preparation** (Week 1)
   - Clean and validate data (Notebook 01)
   - Preprocess text (Notebook 02)
   - Engineer NLP features (Notebook 03)
   - Select and encode features (Notebook 04)
   - **Deliverables**: 4 notebooks + 4 detailed reports

2. **Model Development** (Week 2)
   - Train baseline models (Notebook 05)
   - Train tree-based models (Notebook 06)
   - Compare models (Notebook 08)
   - **Deliverables**: 3 notebooks + 3 detailed reports

3. **Evaluation** (Week 3)
   - Performance analysis (Notebook 09)
   - Error analysis (Notebook 10)
   - Model interpretation with SHAP (Notebook 11)
   - **Deliverables**: 3 notebooks + 3 detailed reports

4. **Integration** (Week 4)
   - Create prediction API
   - Integrate with Streamlit app
   - Final documentation

## 🔍 Key Insights from EDA

✅ **Strengths**:
- Complete reactions/comments data (100%)
- Rich content text (94.1% available)
- Good data quality overall (86.7% complete)

⚠️ **Limitations**:
- Views data completely missing (100%)
- Timestamps are relative, not absolute
- Cannot optimize for posting time

## 📚 References

- [MODEL_DEVELOPMENT_PLAN.md](MODEL_DEVELOPMENT_PLAN.md) - Comprehensive development plan
- [../eda/eda_report.txt](../eda/eda_report.txt) - EDA findings
- [base_score_calculation.txt](base_score_calculation.txt) - Existing scoring algorithm

## 👥 Contributors

TrendPilot Team - Capstone Project 2026

## 📄 License

MIT License
