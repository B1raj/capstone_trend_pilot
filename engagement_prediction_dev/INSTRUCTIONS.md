# Engagement Prediction — How to Run

## Setup (Git Bash)

```bash
# From the Capstone root
source .venv/Scripts/activate

cd capstone_trend_pilot/engagement_prediction_dev
jupyter notebook
```

## Notebook Order

| # | Notebook | Purpose |
|---|----------|---------|
| 01 | `01_data_loading_cleaning.ipynb` | Load & clean raw LinkedIn post data |
| 02 | `02_text_preprocessing.ipynb` | NLP preprocessing |
| 03 | `03_feature_engineering.ipynb` | Build 72 content features |
| 04 | `04_feature_selection.ipynb` | Select final feature set |
| 05 | `05_exploratory_analysis.ipynb` | EDA & distributions |
| 06 | `06_model_training_v4_FIXED.ipynb` | Baseline models (raw targets, leakage removed) |
| 08 | `08_model_training_loo_relative.ipynb` | **Improved:** LOO relative regression |
| 09 | `09_model_training_classification.ipynb` | **Improved:** 3-class classification |

> Notebooks 01–05 only need to run once to generate `data/selected_features_data.csv`.
> Start from 06 onwards if the data file already exists.

## Data

- Input: `data/selected_features_data.csv` (772 posts × 94 columns)
- Outputs: saved to `data/` (plots, feature importance PNGs)

## Key Design Decisions

- **Influencer features removed** to prevent them from dominating content features
- **NB08 target:** `log(reactions / author_LOO_median)` — predicts relative performance, not absolute counts
- **NB09 classes:** Below / Average / Above author's typical engagement (3-class, balanced)
- See `reports/08_model_improvement_PLAN.md` for full rationale
