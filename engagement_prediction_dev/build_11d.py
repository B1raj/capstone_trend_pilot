"""Build NB11d: clone NB11c and remove log_followers + follower_tier from features."""
import json, copy

SRC = 'notebooks/11c_two_class_classification.ipynb'
DST = 'notebooks/11d_no_followers_two_class.ipynb'

with open(SRC, encoding='utf-8') as f:
    nb = json.load(f)

nb = copy.deepcopy(nb)
cells = nb['cells']

# ── Cell 0: title / motivation ───────────────────────────────────────────────
cells[0]['source'] = [
    "# Experiment D: 2-Class Engagement Rate Classification (No Follower Features)\n",
    "## LinkedIn Engagement Prediction — TrendPilot\n",
    "\n",
    "**Experiment:** Binary classification — Below Average vs Above Average, with follower\n",
    "features (`log_followers`, `follower_tier`) removed entirely.\n",
    "\n",
    "**Question:** Can content-only signals distinguish above-median from below-median\n",
    "engagement rate, without any audience-size shortcut?\n",
    "\n",
    "---\n",
    "\n",
    "## Motivation\n",
    "\n",
    "NB11c (2-class binary model) achieved the best overall Macro F1 across all experiments\n",
    "(0.8064, lift +0.306 over random). However, the two most important features were:\n",
    "\n",
    "| Rank | Feature | Importance |\n",
    "|------|---------|------------|\n",
    "| 1 | follower_tier | 0.098 |\n",
    "| 2 | log_followers | 0.067 |\n",
    "\n",
    "Together these account for ~16.5% of total model importance. This raises a concern:\n",
    "is the model learning genuine content quality, or is it still exploiting creator-size\n",
    "patterns to determine which side of the median a post falls on?\n",
    "\n",
    "This experiment removes both follower proxy features entirely, leaving only\n",
    "pure content and timing signals (71 features).\n",
    "\n",
    "## What to look for\n",
    "- How much does Macro F1 drop when follower features are removed?\n",
    "- Which content features rise to the top of importance?\n",
    "- Does per-tier performance become more consistent (std of tier F1 lower)?\n",
    "- Is the remaining lift (+F1 above 0.500 random) driven by real content signal?\n",
]

# ── Cell 11: follower tier markdown — update to note exclusion ───────────────
cells[11]['source'] = [
    "## 5. Follower Tier Features — Excluded\n",
    "\n",
    "In NB11c, `log_followers` and `follower_tier` were the top two features by\n",
    "importance (0.098 and 0.067 respectively). To test whether the model can learn\n",
    "from content alone, both are **excluded from the feature set in this experiment**.\n",
    "\n",
    "The `add_follower_features` function still runs (columns are created) but both\n",
    "columns are added to `DROP_COLS` in the next cell, so they are never seen by\n",
    "the models.\n",
    "\n",
    "Raw `followers` is already dropped (it is in the target denominator).\n",
    "No audience-size information is available to the classifier.\n",
]

# ── Cell 14: DROP_COLS — add log_followers and follower_tier ─────────────────
cells[14]['source'] = [
    "# ── Columns to DROP (leakage or direct engagement signals) ──────────────────\n",
    "DROP_COLS = [\n",
    "    # Raw targets / used in target construction\n",
    "    'reactions', 'comments', 'followers', 'engagement_rate', 'engagement_class',\n",
    "    # Derived directly from reactions/comments (same-post leakage)\n",
    "    'base_score_capped',\n",
    "    'reactions_per_word', 'comments_per_word', 'reactions_per_sentiment',\n",
    "    'comment_to_reaction_ratio',\n",
    "    # Influencer-history features (aggregated from same dataset — data leakage)\n",
    "    'influencer_avg_reactions', 'influencer_std_reactions', 'influencer_median_reactions',\n",
    "    'influencer_avg_comments', 'influencer_std_comments', 'influencer_median_comments',\n",
    "    'influencer_avg_base_score', 'influencer_avg_sentiment',\n",
    "    'influencer_post_count', 'influencer_total_engagement', 'influencer_avg_engagement',\n",
    "    'influencer_consistency_reactions',\n",
    "    'reactions_vs_influencer_avg', 'comments_vs_influencer_avg',\n",
    "    # Metadata / text / identifiers — not ML features\n",
    "    'name', 'content', 'time_spent', 'location',\n",
    "    # Follower proxy features — excluded to test pure content signal (NB11d)\n",
    "    'log_followers', 'follower_tier',\n",
    "]\n",
    "\n",
    "# Only drop cols that actually exist\n",
    "drop_existing = [c for c in DROP_COLS if c in df_train.columns]\n",
    "print(f'Dropping {len(drop_existing)} columns (leakage/metadata):')  \n",
    "for c in drop_existing:\n",
    "    print(f'  {c}')\n",
    "\n",
    "# ── Keep all remaining numeric / binary columns as features ─────────────────\n",
    "all_cols = df_train.columns.tolist()\n",
    "feature_cols = [c for c in all_cols if c not in drop_existing and c != 'engagement_class']\n",
    "\n",
    "# Verify all are numeric\n",
    "non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df_train[c])]\n",
    "if non_numeric:\n",
    "    print(f'\\nNon-numeric columns removed from features: {non_numeric}')\n",
    "    feature_cols = [c for c in feature_cols if c not in non_numeric]\n",
    "\n",
    "print(f'\\nTotal features: {len(feature_cols)}')\n",
    "print('Feature columns:', feature_cols)\n",
]

# ── Cell 29: feature importance — update labels to NB11d ────────────────────
old29 = ''.join(cells[29]['source'])
new29 = old29.replace(
    "f'Top 25 Features -- {best_row[\"model\"]} (NB11c 2-Class)'",
    "f'Top 25 Features -- {best_row[\"model\"]} (NB11d 2-Class, No Followers)'"
).replace(
    "'../data/11c_feature_importance.png'",
    "'../data/11d_feature_importance.png'"
).replace(
    "'Saved: ../data/11c_feature_importance.png'",
    "'Saved: ../data/11d_feature_importance.png'"
)
cells[29]['source'] = [new29]

# ── Cell 31: tier evaluation — compare vs NB11c ─────────────────────────────
cells[31]['source'] = [
    "tier_labels_map = {0: 'micro (<10k)', 1: 'small (10k-50k)',\n",
    "                   2: 'medium (50k-200k)', 3: 'large (>200k)'}\n",
    "\n",
    "# NB11c (2-class with follower proxies) tier F1 for reference\n",
    "nb11c_tier_f1 = {0: 0.6176, 1: 0.7519, 2: 0.4286, 3: 0.4615}\n",
    "\n",
    "tier_results = []\n",
    "print('=' * 70)\n",
    "print('Macro F1 by Follower Tier -- NB11d (no followers) vs NB11c (with followers)')\n",
    "print('=' * 70)\n",
    "print(f'{\"Tier\":22s}  {\"n\":>4}  {\"NB11d no-flw\":>12}  {\"NB11c 2cls\":>10}  {\"Delta\":>7}')\n",
    "print('-' * 70)\n",
    "\n",
    "for tier_id, tier_name in tier_labels_map.items():\n",
    "    mask = df_test['follower_tier'] == tier_id\n",
    "    n = mask.sum()\n",
    "    if n == 0 or y_test[mask].nunique() < 2:\n",
    "        print(f'  {tier_name}: {n} posts -- skipped')\n",
    "        continue\n",
    "    yp_t = best_model.predict(X_test[mask])\n",
    "    mf1  = f1_score(y_test[mask], yp_t, average='macro', zero_division=0)\n",
    "    ref  = nb11c_tier_f1.get(tier_id, float('nan'))\n",
    "    delta = mf1 - ref\n",
    "    d_str = f'+{delta:.4f}' if delta >= 0 else f'{delta:.4f}'\n",
    "    print(f'  {tier_name:22s}  {n:>4}  {mf1:>12.4f}  {ref:>10.4f}  {d_str:>7}')\n",
    "    tier_results.append({'tier': tier_name, 'n': n, 'macro_f1': round(mf1,4),\n",
    "                         'macro_f1_ref': ref, 'delta': round(delta,4)})\n",
    "\n",
    "print('-' * 70)\n",
    "overall_delta = best_row['macro_f1'] - 0.8064\n",
    "d_str2 = f'+{overall_delta:.4f}' if overall_delta >= 0 else f'{overall_delta:.4f}'\n",
    "print(f'  {\"Overall\":22s}  {len(y_test):>4}  {best_row[\"macro_f1\"]:>12.4f}  {0.8064:>10.4f}  {d_str2:>7}')\n",
    "print()\n",
    "print('Random baseline: 0.500 (2-class)')\n",
    "print(f'NB11c lift: 0.8064 - 0.500 = +0.306')\n",
    "lift_d = round(best_row['macro_f1'] - 0.5, 4)\n",
    "print(f'NB11d lift: {best_row[\"macro_f1\"]} - 0.500 = +{lift_d}')\n",
]

# ── Cell 32: tier plot — update labels to NB11d ─────────────────────────────
old32 = ''.join(cells[32]['source'])
new32 = old32.replace(
    "label='NB11c (2-class)'",
    "label='NB11d (no followers)'"
).replace(
    "label='NB11a (3-class)'",
    "label='NB11c (with followers)'"
).replace(
    "'Tier F1: NB11c vs NB11a'",
    "'Tier F1: NB11d vs NB11c'"
).replace(
    "'Delta vs NB11a (2-class gain per tier)'",
    "'Delta vs NB11c (no-follower effect per tier)'"
).replace(
    "'2-Class vs 3-Class: Tier Fairness (NB11c)'",
    "'2-Class No Followers: Tier Fairness (NB11d)'"
).replace(
    "'../data/11c_tier_evaluation.png'",
    "'../data/11d_tier_evaluation.png'"
).replace(
    "'Saved: ../data/11c_tier_evaluation.png'",
    "'Saved: ../data/11d_tier_evaluation.png'"
)
cells[32]['source'] = [new32]

# ── Cell 34: summary printout — update to NB11d ─────────────────────────────
cells[34]['source'] = [
    "print('=' * 65)\n",
    "print('NB11d -- 2-CLASS ENGAGEMENT RATE, NO FOLLOWER FEATURES')\n",
    "print('Experiment D: Binary median split, content-only features')\n",
    "print('=' * 65)\n",
    "results_df = pd.DataFrame(results)\n",
    "print(results_df.to_string(index=False))\n",
    "\n",
    "print(f'\\nClass threshold (training median): {median_rate:.3f}')\n",
    "\n",
    "print(f'\\nComparison (raw F1):')\n",
    "print(f'  NB11  (3-class, global, with followers):   0.5997')\n",
    "print(f'  NB11a (3-class, global, no followers):     0.5301')\n",
    "print(f'  NB11b (3-class, within-tier):              0.4306')\n",
    "print(f'  NB11c (2-class, with followers):           0.8064')\n",
    "print(f'  NB11d (2-class, no followers):             {best_row[\"macro_f1\"]}')\n",
    "\n",
    "print(f'\\nLift over random baseline (comparable across class counts):')\n",
    "print(f'  NB11:  0.5997 - 0.333 = +0.267')\n",
    "print(f'  NB11a: 0.5301 - 0.333 = +0.197')\n",
    "print(f'  NB11b: 0.4306 - 0.333 = +0.098')\n",
    "print(f'  NB11c: 0.8064 - 0.500 = +0.306')\n",
    "lift = round(best_row['macro_f1'] - 0.5, 4)\n",
    "print(f'  NB11d: {best_row[\"macro_f1\"]} - 0.500 = +{lift}')\n",
]

# ── Cell 35: conclusions — update to Experiment D ───────────────────────────
cells[35]['source'] = [
    "## 14. Conclusions -- Experiment D (2-Class, No Follower Features)\n",
    "\n",
    "### What This Experiment Tests\n",
    "NB11d removes `log_followers` and `follower_tier` from the feature set of\n",
    "NB11c. The model must classify above/below-median engagement using only\n",
    "content, timing, and text-structure signals (71 features).\n",
    "\n",
    "### Why This Matters\n",
    "In NB11c, follower proxies ranked #1 and #2 by feature importance (0.098 and\n",
    "0.067), contributing ~16.5% of total model signal. The concern is that the\n",
    "global median does not sit uniformly across tiers — larger accounts tend to\n",
    "cluster below the global median, so follower_tier can still act as a shortcut\n",
    "even in a binary setting.\n",
    "\n",
    "### Interpreting the F1 Drop\n",
    "Removing follower proxies will likely reduce overall Macro F1. The drop\n",
    "quantifies how much of NB11c's performance was follower-driven:\n",
    "\n",
    "```\n",
    "NB11c (with followers):  F1 ~ 0.8064  lift = +0.306\n",
    "NB11d (no followers):    F1 = [result] lift = [F1 - 0.500]\n",
    "Drop = NB11c_F1 - NB11d_F1  (follower contribution)\n",
    "```\n",
    "\n",
    "### Practical Interpretation\n",
    "- If the drop is small (< 0.05): content features carry most of the signal.\n",
    "  The model is genuinely learning content quality.\n",
    "- If the drop is large (> 0.10): follower proxies were a major driver.\n",
    "  The model's performance in NB11c was partly due to creator-size shortcutting.\n",
    "\n",
    "### Recommended Next Step\n",
    "Compare tier F1 standard deviations across NB11c and NB11d. If NB11d has\n",
    "more consistent tier F1 (lower std), it is fairer across creator sizes.\n",
    "Add text embeddings (SBERT/TF-IDF) to NB11d to further improve content signal.\n",
]

with open(DST, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'Written: {DST}')
print(f'Total cells: {len(nb["cells"])}')
