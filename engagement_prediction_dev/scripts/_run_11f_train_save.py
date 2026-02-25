# === CELL 1 ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
import xgboost as xgb
import lightgbm as lgb
import copy
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
OUTPUT_DIR = '../data'
os.makedirs(OUTPUT_DIR, exist_ok=True)
print('Libraries loaded.')
print(f'Output directory: {os.path.abspath(OUTPUT_DIR)}')

# === CELL 3 ===
data = pd.read_csv('../data/selected_features_data.csv')
print(f'Shape: {data.shape}')
print(f'Columns ({len(data.columns)}): {list(data.columns)}')

# === CELL 5 ===
df = data.copy()
df['followers'] = df['followers'].clip(lower=1)
df['engagement_rate'] = (df['reactions'] + df['comments']) / (df['followers'] / 1000)

print('Engagement rate summary:')
print(df['engagement_rate'].describe().round(3))
print(f'Global median: {df["engagement_rate"].median():.2f} engagements per 1k followers')
print('Per-tier medians will be computed from training subsets inside the loop.')

# === CELL 6 ===
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(df['engagement_rate'], bins=50, color='steelblue', edgecolor='white')
axes[0].set_title('Engagement Rate (raw)')
axes[0].set_xlabel('Engagements per 1k followers')

axes[1].hist(np.log1p(df['engagement_rate']), bins=50, color='coral', edgecolor='white')
axes[1].set_title('log1p(Engagement Rate)')
axes[1].set_xlabel('log1p scale')

axes[2].hist(np.log10(df['followers']+1), bins=50, color='seagreen', edgecolor='white')
axes[2].set_title('Follower Count (log10)')
axes[2].set_xlabel('log10(followers)')

plt.suptitle('Engagement Rate & Follower Distributions', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/11f_distributions.png', dpi=100, bbox_inches='tight')
plt.show()

# === CELL 8 ===
# Stratify on binary split for balanced train/test classes
df['_tmp_class'] = (df['engagement_rate'] >= df['engagement_rate'].median()).astype(int)

df_train, df_test = train_test_split(
    df, test_size=0.2, random_state=RANDOM_STATE,
    stratify=df['_tmp_class']
)
df_train = df_train.copy().reset_index(drop=True)
df_test  = df_test.copy().reset_index(drop=True)
df_train.drop(columns=['_tmp_class'], inplace=True)
df_test.drop(columns=['_tmp_class'], inplace=True)

print(f'Train: {len(df_train)} posts | Test: {len(df_test)} posts')
print(f'Train authors: {df_train["name"].nunique()}')

# === CELL 10 ===
def add_follower_features(df_):
    df_ = df_.copy()
    df_['log_followers'] = np.log1p(df_['followers'])
    df_['follower_tier'] = pd.cut(
        df_['followers'],
        bins=[0, 10_000, 50_000, 200_000, np.inf],
        labels=[0, 1, 2, 3],   # micro, small, medium, large
        include_lowest=True
    ).astype(int)
    return df_

df_train = add_follower_features(df_train)
df_test  = add_follower_features(df_test)

print('Follower tier distribution (full dataset):')
tier_labels = {0: 'micro (<10k)', 1: 'small (10k-50k)', 2: 'medium (50k-200k)', 3: 'large (>200k)'}
tier_counts = pd.concat([df_train, df_test])['follower_tier'].value_counts().sort_index()
for t, n in tier_counts.items():
    print(f'  {tier_labels[t]:22s}: {n:4d} ({n/772*100:.1f}%)')

# === CELL 12 ===
# ── Columns to DROP (leakage or direct engagement signals) ──────────────────
DROP_COLS = [
    # Raw targets / used in target construction
    'reactions', 'comments', 'followers', 'engagement_rate',
    # Derived directly from reactions/comments (same-post leakage)
    'base_score_capped',
    'reactions_per_word', 'comments_per_word', 'reactions_per_sentiment',
    'comment_to_reaction_ratio',
    # Influencer-history features (aggregated from same dataset — data leakage)
    'influencer_avg_reactions', 'influencer_std_reactions', 'influencer_median_reactions',
    'influencer_avg_comments', 'influencer_std_comments', 'influencer_median_comments',
    'influencer_avg_base_score', 'influencer_avg_sentiment',
    'influencer_post_count', 'influencer_total_engagement', 'influencer_avg_engagement',
    'influencer_consistency_reactions',
    'reactions_vs_influencer_avg', 'comments_vs_influencer_avg',
    # Metadata / text / identifiers — not ML features
    'name', 'content', 'time_spent', 'location',
    # Follower proxy features — constant within each tier model; excluded from features
    'log_followers',
    'follower_tier',
]

# Only drop cols that actually exist
drop_existing = [c for c in DROP_COLS if c in df_train.columns]
print(f'Dropping {len(drop_existing)} columns (leakage/metadata):')
for c in drop_existing:
    print(f'  {c}')

# ── Keep all remaining numeric columns as features ───────────────────────────
all_cols = df_train.columns.tolist()
feature_cols = [
    c for c in all_cols
    if c not in drop_existing and c not in ('engagement_class',)
]

# Verify all are numeric
non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df_train[c])]
if non_numeric:
    print(f'\nNon-numeric columns removed from features: {non_numeric}')
    feature_cols = [c for c in feature_cols if c not in non_numeric]

print(f'\nTotal features: {len(feature_cols)}')
print('Feature columns:', feature_cols)

# === CELL 13 ===
# Feature matrices — labels are computed per-tier inside the training loop
X_train = df_train[feature_cols].fillna(0)
X_test  = df_test[feature_cols].fillna(0)

print(f'X_train: {X_train.shape}  |  X_test: {X_test.shape}')
print(f'Per-tier binary labels (y) will be computed inside the training loop.')
print(f'Each tier uses its own training-subset median as the class boundary.')

# === CELL 15 ===
from tqdm.auto import tqdm
print('tqdm loaded.')

# === CELL 17 ===
def make_classifiers():
    return {
        'RandomForest': RandomForestClassifier(
            n_estimators=200, max_depth=8,
            min_samples_split=10, min_samples_leaf=5,
            class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
            eval_metric='logloss', random_state=RANDOM_STATE,
            n_jobs=-1, verbosity=0
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            num_leaves=15, min_child_samples=10,
            class_weight='balanced', random_state=RANDOM_STATE,
            n_jobs=-1, verbose=-1
        ),
    }

print('Classifier factory defined: RandomForest | XGBoost | LightGBM')
print('Fixed baseline params — no tuning (small per-tier sample sizes).')

# === CELL 19 ===
tier_labels_map = {0: 'micro (<10k)', 1: 'small (10k-50k)',
                   2: 'medium (50k-200k)', 3: 'large (>200k)'}
all_tier_results = {}

for tier_id in tqdm([0, 1, 2, 3], desc='Tiers', unit='tier'):
    train_mask = df_train['follower_tier'] == tier_id
    test_mask  = df_test['follower_tier']  == tier_id

    X_tr = X_train[train_mask.values]
    X_te = X_test[test_mask.values]

    # Per-tier median label from training subset
    tier_med = df_train.loc[train_mask, 'engagement_rate'].median()
    y_tr = (df_train.loc[train_mask, 'engagement_rate'] >= tier_med).astype(int)
    y_te = (df_test.loc[test_mask,   'engagement_rate'] >= tier_med).astype(int)

    n_tr, n_te = len(y_tr), len(y_te)
    tier_name = tier_labels_map[tier_id]

    print(f'\n{"="*60}')
    print(f'{tier_name}  |  train={n_tr}  test={n_te}  median={tier_med:.3f}')
    print(f'  Train Class 0: {(y_tr==0).sum()}  Class 1: {(y_tr==1).sum()}')

    if y_te.nunique() < 2:
        print('  Skipped — only 1 class in test set')
        continue

    sw = compute_sample_weight('balanced', y_tr)
    tier_clf_results = []

    for clf_name, clf in tqdm(make_classifiers().items(),
                              desc=f'  {tier_name}', leave=False):
        clf_fit = copy.deepcopy(clf)
        if 'XGBoost' in clf_name:
            clf_fit.fit(X_tr, y_tr, sample_weight=sw)
        else:
            clf_fit.fit(X_tr, y_tr)

        y_pred = clf_fit.predict(X_te)
        mf1 = f1_score(y_te, y_pred, average='macro', zero_division=0)
        acc = accuracy_score(y_te, y_pred)
        tier_clf_results.append({
            'model': clf_name,
            'macro_f1': round(mf1, 4),
            'acc': round(acc, 4),
            'clf': clf_fit
        })
        print(f'  {clf_name:15s}  F1={mf1:.4f}  Acc={acc:.4f}')

    all_tier_results[tier_id] = {
        'name': tier_name, 'n_train': n_tr, 'n_test': n_te,
        'median_er': tier_med, 'results': tier_clf_results,
        'y_te': y_te, 'X_te': X_te
    }

# === CELL 20 ===
# ── Save best model per tier ──────────────────────────────────────────────────
import joblib

MODEL_DIR = '../models/11f_per_tier'
os.makedirs(MODEL_DIR, exist_ok=True)

saved_models = {}

for tier_id, info in all_tier_results.items():
    best = max(info['results'], key=lambda r: r['macro_f1'])
    safe_name = (info['name']
                 .replace('(', '').replace(')', '')
                 .replace('<', 'lt').replace('>', 'gt')
                 .replace('/', '_').replace(' ', '_'))
    fname = f"tier_{tier_id}_{safe_name}_{best['model']}.joblib"
    fpath = os.path.join(MODEL_DIR, fname)
    joblib.dump(best['clf'], fpath)
    saved_models[tier_id] = {
        'model':     best['model'],
        'macro_f1':  best['macro_f1'],
        'file':      fname,
        'median_er': info['median_er'],
        'tier_name': info['name'],
    }
    print(f"  Saved tier {tier_id} ({info['name']})  "
          f"model={best['model']}  F1={best['macro_f1']:.4f}  -> {fname}")

# ── Save metadata JSON (feature list, tier medians, model paths) ──────────────
metadata = {
    'feature_cols': feature_cols,
    'n_features':   len(feature_cols),
    'tiers': {
        str(tier_id): {
            'tier_name': v['tier_name'],
            'median_er': v['median_er'],
            'best_model': v['model'],
            'macro_f1':  v['macro_f1'],
            'model_file': v['file'],
        }
        for tier_id, v in saved_models.items()
    },
}

import json as _json
metadata_path = os.path.join(MODEL_DIR, 'metadata.json')
with open(metadata_path, 'w') as _f:
    _json.dump(metadata, _f, indent=2)

print("")
print(f"Metadata saved: {metadata_path}")
print(f"Models directory: {os.path.abspath(MODEL_DIR)}")

