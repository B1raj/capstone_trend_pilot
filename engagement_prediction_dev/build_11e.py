import json, copy

SRC = 'notebooks/11c_two_class_classification.ipynb'
DST = 'notebooks/11e_within_tier_two_class.ipynb'

with open(SRC, encoding='utf-8') as f:
    nb = json.load(f)

nb = copy.deepcopy(nb)
cells = nb['cells']

for cell in cells:
    cell['outputs'] = []
    cell['execution_count'] = None

# Cell 0: title
cells[0]['source'] = [
    "# Experiment E: 2-Class Within-Tier Engagement Rate Classification\n",
    "## LinkedIn Engagement Prediction - TrendPilot\n",
    "\n",
    "**Experiment:** Binary classification using per-tier median thresholds.\n",
    "Is this post above average *for a creator of this size*?\n",
    "\n",
    "**Question:** When the class boundary is tier-relative, does `follower_tier`\n",
    "lose its dominance? Can the model find genuine content signals?\n",
    "\n",
    "---\n",
    "\n",
    "## Motivation\n",
    "\n",
    "NB11c (2-class, global median) achieved Macro F1 = 0.8064 but the top two\n",
    "features were follower proxies (follower_tier=0.098, log_followers=0.067).\n",
    "\n",
    "**Root cause:** The global median (5.985) does not sit at the same relative\n",
    "position within each tier. Large accounts produce lower ER at scale, so\n",
    "`follower_tier = large` almost always maps to Class 0. This is a structural\n",
    "shortcut, not content learning.\n",
    "\n",
    "**Fix:** Compute a separate median per tier from training data. Within each\n",
    "tier ~50% become Class 0 and ~50% Class 1. `follower_tier` can no longer\n",
    "predict class directly - the model must find content signals.\n",
    "\n",
    "| Experiment | Threshold | Followers | Shortcut possible? |\n",
    "|------------|-----------|-----------|--------------------|\n",
    "| NB11c | Global median | Yes (73) | Yes |\n",
    "| NB11d | Global median | No (71) | N/A |\n",
    "| NB11e | Per-tier median | Yes (73) | No |\n",
    "\n",
    "## What to look for\n",
    "- Does `follower_tier` importance drop significantly?\n",
    "- Is tier F1 std lower than NB11c (approx 0.136)?\n",
    "- How does overall F1 compare to NB11c and NB11d?\n",
]

# Cell 9: label markdown
cells[9]['source'] = [
    "## 4. Assign 2-Class Labels (Per-Tier Training Median)\n",
    "\n",
    "Compute the median engagement rate separately within each follower tier\n",
    "from training data only.\n",
    "\n",
    "- Class 0: rate < tier median (Below average for this tier)\n",
    "- Class 1: rate >= tier median (Above average for this tier)\n",
    "\n",
    "Each tier will have ~50/50 class balance by design.\n",
    "Thresholds are derived from training data only -- no leakage.\n",
]

# Cell 10: per-tier label assignment — needs follower_tier, so create it first
cells[10]['source'] = [
    "# follower_tier is needed for per-tier thresholds; add it here before labels.\n",
    "# Cell 12 will run add_follower_features again (adds log_followers too) -- idempotent.\n",
    "df_train['follower_tier'] = pd.cut(\n",
    "    df_train['followers'], bins=[0, 10_000, 50_000, 200_000, np.inf],\n",
    "    labels=[0, 1, 2, 3], include_lowest=True\n",
    ").astype(int)\n",
    "df_test['follower_tier'] = pd.cut(\n",
    "    df_test['followers'], bins=[0, 10_000, 50_000, 200_000, np.inf],\n",
    "    labels=[0, 1, 2, 3], include_lowest=True\n",
    ").astype(int)\n",
    "\n",
    "tier_labels_map_local = {0: 'micro (<10k)', 1: 'small (10k-50k)',\n",
    "                          2: 'medium (50k-200k)', 3: 'large (>200k)'}\n",
    "\n",
    "tier_medians = {}\n",
    "\n",
    "print('Per-tier training medians:')\n",
    "hdr = f'{\"Tier\":22s}  {\"Median ER\":>10}  {\"Train n\":>8}  {\"Class 0\":>8}  {\"Class 1\":>8}'\n",
    "print(hdr)\n",
    "print('-' * 68)\n",
    "\n",
    "for tier_id in [0, 1, 2, 3]:\n",
    "    train_mask = df_train['follower_tier'] == tier_id\n",
    "    test_mask  = df_test['follower_tier']  == tier_id\n",
    "\n",
    "    tier_med = df_train.loc[train_mask, 'engagement_rate'].median()\n",
    "    tier_medians[tier_id] = tier_med\n",
    "\n",
    "    df_train.loc[train_mask, 'engagement_class'] = (\n",
    "        df_train.loc[train_mask, 'engagement_rate'] >= tier_med\n",
    "    ).astype(int)\n",
    "    df_test.loc[test_mask, 'engagement_class'] = (\n",
    "        df_test.loc[test_mask, 'engagement_rate'] >= tier_med\n",
    "    ).astype(int)\n",
    "\n",
    "    n_train = train_mask.sum()\n",
    "    c0 = (df_train.loc[train_mask, 'engagement_class'] == 0).sum()\n",
    "    c1 = (df_train.loc[train_mask, 'engagement_class'] == 1).sum()\n",
    "    row = f'  {tier_labels_map_local[tier_id]:22s}  {tier_med:>10.3f}  {n_train:>8}  {c0:>8}  {c1:>8}'\n",
    "    print(row)\n",
    "\n",
    "LABEL_NAMES = {0: 'Below-tier-median', 1: 'Above-tier-median'}\n",
    "\n",
    "print('\\nClass distribution - TRAINING:')\n",
    "for c in [0, 1]:\n",
    "    n = (df_train['engagement_class'] == c).sum()\n",
    "    print(f'  Class {c} {LABEL_NAMES[c]:25s}: {n:4d} ({n/len(df_train)*100:.1f}%)')\n",
    "\n",
    "print('\\nClass distribution - TEST:')\n",
    "for c in [0, 1]:\n",
    "    n = (df_test['engagement_class'] == c).sum()\n",
    "    print(f'  Class {c} {LABEL_NAMES[c]:25s}: {n:4d} ({n/len(df_test)*100:.1f}%)')\n",
    "\n",
    "print('\\nRandom baseline: 0.500 (2-class)')\n",
    "print('Each tier has ~50/50 split -- follower_tier cannot shortcut to class label')\n",
]

# Cell 11: follower markdown
cells[11]['source'] = [
    "## 5. Follower Tier Features -- Kept (Labels Are Now Tier-Relative)\n",
    "\n",
    "Both `log_followers` and `follower_tier` are kept as features (73 total).\n",
    "However, since class labels are tier-relative, `follower_tier` can no longer\n",
    "predict class directly:\n",
    "\n",
    "- NB11c: large account -> ER < global median -> Class 0 (shortcut works)\n",
    "- NB11e: large account -> ER vs large-account median -> 50/50 chance\n",
    "\n",
    "The proxies can still provide scale context, but the structural shortcut\n",
    "is broken. The model must learn content signals.\n",
    "\n",
    "| Tier | Range | Rationale |\n",
    "|------|-------|----------|\n",
    "| micro | < 10k | Personal/niche audiences |\n",
    "| small | 10k-50k | Growing creators |\n",
    "| medium | 50k-200k | Established voices |\n",
    "| large | > 200k | Macro influencers |\n",
]

# Cell 27: confusion matrix paths
s27 = ''.join(cells[27]['source'])
s27 = s27.replace("'../data/11c_confusion_matrix.png'", "'../data/11e_confusion_matrix.png'")
s27 = s27.replace("'Saved: ../data/11c_confusion_matrix.png'", "'Saved: ../data/11e_confusion_matrix.png'")
s27 = s27.replace("(NB11c)", "(NB11e)")
cells[27]['source'] = [s27]

# Cell 29: feature importance paths
s29 = ''.join(cells[29]['source'])
s29 = s29.replace("(NB11c 2-Class)", "(NB11e Within-Tier 2-Class)")
s29 = s29.replace("'../data/11c_feature_importance.png'", "'../data/11e_feature_importance.png'")
s29 = s29.replace("'Saved: ../data/11c_feature_importance.png'", "'Saved: ../data/11e_feature_importance.png'")
cells[29]['source'] = [s29]

# Cell 31: tier evaluation
cells[31]['source'] = [
    "tier_labels_map = {0: 'micro (<10k)', 1: 'small (10k-50k)',\n",
    "                   2: 'medium (50k-200k)', 3: 'large (>200k)'}\n",
    "\n",
    "nb11c_tier_f1 = {0: 0.6183, 1: 0.7515, 2: 0.4286, 3: 0.4615}\n",
    "\n",
    "tier_results = []\n",
    "print('=' * 72)\n",
    "print('Macro F1 by Follower Tier -- NB11e (within-tier) vs NB11c (global)')\n",
    "print('=' * 72)\n",
    "hdr2 = f'{\"Tier\":22s}  {\"n\":>4}  {\"NB11e tier\":>10}  {\"NB11c global\":>12}  {\"Delta\":>7}'\n",
    "print(hdr2)\n",
    "print('-' * 72)\n",
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
    "    d_str = ('+' if delta >= 0 else '') + f'{delta:.4f}'\n",
    "    print(f'  {tier_name:22s}  {n:>4}  {mf1:>10.4f}  {ref:>12.4f}  {d_str:>7}')\n",
    "    tier_results.append({'tier': tier_name, 'n': n, 'macro_f1': round(mf1,4),\n",
    "                         'macro_f1_ref': ref, 'delta': round(delta,4)})\n",
    "\n",
    "print('-' * 72)\n",
    "overall_delta = best_row['macro_f1'] - 0.8064\n",
    "d_str2 = ('+' if overall_delta >= 0 else '') + f'{overall_delta:.4f}'\n",
    "print(f'  {\"Overall\":22s}  {len(y_test):>4}  {best_row[\"macro_f1\"]:>10.4f}  {0.8064:>12.4f}  {d_str2:>7}')\n",
    "print()\n",
    "print('Random baseline: 0.500')\n",
    "print('NB11c lift (global): 0.8064 - 0.500 = +0.306')\n",
    "lift_e = round(best_row['macro_f1'] - 0.5, 4)\n",
    "print(f'NB11e lift (tier):   {best_row[\"macro_f1\"]} - 0.500 = +{lift_e}')\n",
    "\n",
    "if tier_results:\n",
    "    e_vals = [r['macro_f1'] for r in tier_results]\n",
    "    c_vals = [nb11c_tier_f1[i] for i in sorted(nb11c_tier_f1)]\n",
    "    print(f'\\nTier F1 std  NB11c (global): {np.std(c_vals):.4f}')\n",
    "    print(f'Tier F1 std  NB11e (tier):   {np.std(e_vals):.4f}')\n",
    "    print('(Lower std = more consistent across creator sizes)')\n",
]

# Cell 32: tier plot paths/labels
s32 = ''.join(cells[32]['source'])
s32 = s32.replace("label='NB11c (2-class)'", "label='NB11e (within-tier)'")
s32 = s32.replace("label='NB11a (3-class)'", "label='NB11c (global median)'")
s32 = s32.replace("'Tier F1: NB11c vs NB11a'", "'Tier F1: NB11e vs NB11c'")
s32 = s32.replace("'Delta vs NB11a (2-class gain per tier)'", "'Delta vs NB11c (within-tier effect)'")
s32 = s32.replace("Tier Fairness (NB11c)", "Fairness (NB11e)")
s32 = s32.replace("'../data/11c_tier_evaluation.png'", "'../data/11e_tier_evaluation.png'")
s32 = s32.replace("'Saved: ../data/11c_tier_evaluation.png'", "'Saved: ../data/11e_tier_evaluation.png'")
cells[32]['source'] = [s32]

# Cell 34: summary
cells[34]['source'] = [
    "print('=' * 65)\n",
    "print('NB11e -- 2-CLASS WITHIN-TIER ENGAGEMENT RATE CLASSIFICATION')\n",
    "print('Experiment E: Per-tier median split')\n",
    "print('=' * 65)\n",
    "results_df = pd.DataFrame(results)\n",
    "print(results_df.to_string(index=False))\n",
    "\n",
    "print('\\nPer-tier training thresholds:')\n",
    "tier_label_map_print = {0: 'micro (<10k)', 1: 'small (10k-50k)',\n",
    "                        2: 'medium (50k-200k)', 3: 'large (>200k)'}\n",
    "for tid, med in tier_medians.items():\n",
    "    print(f'  {tier_label_map_print[tid]:22s}: median = {med:.3f}')\n",
    "\n",
    "print('\\nComparison (raw F1):')\n",
    "print('  NB11  (3-class, global, with followers):   0.5997')\n",
    "print('  NB11a (3-class, global, no followers):     0.5301')\n",
    "print('  NB11b (3-class, within-tier):              0.4306')\n",
    "print('  NB11c (2-class, global median):            0.8064')\n",
    "print('  NB11d (2-class, no followers):             0.6758')\n",
    "print(f'  NB11e (2-class, within-tier median):       {best_row[\"macro_f1\"]}')\n",
    "\n",
    "print('\\nLift over random (0.500 for all 2-class):')\n",
    "print('  NB11c: 0.8064 - 0.500 = +0.306')\n",
    "print('  NB11d: 0.6758 - 0.500 = +0.176')\n",
    "lift = round(best_row['macro_f1'] - 0.5, 4)\n",
    "print(f'  NB11e: {best_row[\"macro_f1\"]} - 0.500 = +{lift}')\n",
]

# Cell 35: conclusions
cells[35]['source'] = [
    "## 14. Conclusions -- Experiment E (2-Class Within-Tier)\n",
    "\n",
    "### What This Experiment Tests\n",
    "NB11e uses per-tier median thresholds. Within each follower tier ~50% of\n",
    "posts are Class 0 and ~50% are Class 1. The question is: Is this post\n",
    "above average for a creator of this size?\n",
    "\n",
    "### Why This Breaks the Structural Shortcut\n",
    "In NB11c, the global median did not sit at the same relative position within\n",
    "each tier -- large accounts clustered below it. Per-tier medians remove this\n",
    "alignment: regardless of tier, a post has ~50% chance of being above or below\n",
    "its relevant threshold. `follower_tier` must now earn importance through\n",
    "content-level context, not structural position.\n",
    "\n",
    "### Reading the Results\n",
    "- `follower_tier` drops in rank: the structural shortcut is broken\n",
    "- Overall F1 < NB11c: that shortcut was carrying real signal\n",
    "- Tier F1 std < NB11c (0.136): fairer across creator sizes\n",
    "- Tier F1 std approaches NB11d (0.021): content features doing the work\n",
    "\n",
    "### Comparison Summary\n",
    "| Experiment | Threshold | Followers | F1 | Lift | Question |\n",
    "|------------|-----------|-----------|-----|------|----------|\n",
    "| NB11c | Global median | Yes | 0.8064 | +0.306 | Above global median? |\n",
    "| NB11d | Global median | No  | 0.6758 | +0.176 | Content-only above global? |\n",
    "| NB11e | Per-tier median | Yes | [result] | [lift] | Above median for this tier? |\n",
    "\n",
    "### Recommended Next Step\n",
    "If NB11e shows better tier consistency without a large F1 penalty, it is\n",
    "the most balanced 2-class model. Add text embeddings (SBERT/TF-IDF) as\n",
    "the next feature engineering step.\n",
]

with open(DST, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'Written: {DST}  ({len(nb["cells"])} cells)')
