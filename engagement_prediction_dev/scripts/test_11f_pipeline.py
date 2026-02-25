"""
test_11f_pipeline.py
====================
End-to-end test: raw rows -> pipeline -> load 11f tier models -> predicted vs actual.

Run from the project root or any directory:
    python scripts/test_11f_pipeline.py

Expects:
  data/linkedin_posts_new.csv
  models/11f_per_tier/metadata.json
  models/11f_per_tier/tier_*.joblib
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

# Allow importing 11f_pipeline from the scripts directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)

# Paths relative to project root
RAW_DATA_PATH  = os.path.join(_PROJECT_DIR, "data", "linkedin_posts_new.csv")
MODEL_DIR      = os.path.join(_PROJECT_DIR, "models", "11f_per_tier")
METADATA_PATH  = os.path.join(MODEL_DIR, "metadata.json")

TIER_NAMES = {0: "micro (<10k)", 1: "small (10k-50k)", 2: "medium (50k-200k)", 3: "large (>200k)"}

# ── 1. Load metadata (tier medians + model file names) ────────────────────────
print("=" * 65)
print("Loading model metadata...")
with open(METADATA_PATH) as f:
    meta = json.load(f)

tier_medians = {int(k): v["median_er"] for k, v in meta["tiers"].items()}
tier_model_files = {int(k): v["model_file"] for k, v in meta["tiers"].items()}
feature_cols = meta["feature_cols"]

print(f"  Features expected: {meta['n_features']}")
for t, v in meta["tiers"].items():
    print(f"  Tier {t} ({v['tier_name']}): median_er={v['median_er']:.3f}  "
          f"best_model={v['best_model']}  macro_f1={v['macro_f1']:.4f}")

# ── 2. Load all 4 tier models ─────────────────────────────────────────────────
print("\nLoading saved models...")
models = {}
for tier_id, fname in tier_model_files.items():
    fpath = os.path.join(MODEL_DIR, fname)
    models[tier_id] = joblib.load(fpath)
    print(f"  Tier {tier_id}: {fname}")

# ── 3. Load raw data and sample rows across all tiers ─────────────────────────
print(f"\nLoading raw data from: {RAW_DATA_PATH}")
raw_all = pd.read_csv(RAW_DATA_PATH)
print(f"  Total rows: {len(raw_all)}")

# Compute follower tier for sampling purposes (same bins as pipeline)
TIER_BINS = [0, 10_000, 50_000, 200_000, float("inf")]
raw_all["_tier_tmp"] = pd.cut(raw_all["followers"],
                               bins=TIER_BINS, labels=[0, 1, 2, 3],
                               include_lowest=True).astype(int)

# Pick 2 rows per tier (fixed seed for reproducibility)
sample_frames = []
for t in [0, 1, 2, 3]:
    tier_rows = raw_all[raw_all["_tier_tmp"] == t]
    n = min(2, len(tier_rows))
    sample_frames.append(tier_rows.sample(n, random_state=7))

df_sample = pd.concat(sample_frames).drop(columns=["_tier_tmp"]).copy()
print(f"  Sampled {len(df_sample)} rows (2 per tier)\n")

# ── 4. Run the pipeline ────────────────────────────────────────────────────────
print("=" * 65)
print("Running 11f pipeline on sampled rows...")
from Uppass import run_pipeline   # noqa: E402  (import after path setup)

X_sample, y_sample = run_pipeline(df_sample, verbose=True)

# ── 5. Predict and compare ────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Per-row predictions vs actuals")
print("=" * 65)

results = []
for idx in range(len(y_sample)):
    tier = int(y_sample.iloc[idx]["follower_tier"])
    er   = float(y_sample.iloc[idx]["engagement_rate"])
    med  = tier_medians[tier]
    actual_class = int(er >= med)     # 1 = above median, 0 = below

    x_row = X_sample.iloc[[idx]][feature_cols]
    pred_class = int(models[tier].predict(x_row)[0])
    correct = actual_class == pred_class

    results.append({
        "row":          idx,
        "tier":         tier,
        "tier_name":    TIER_NAMES[tier],
        "er":           round(er, 3),
        "tier_median":  round(med, 3),
        "actual_class": actual_class,
        "actual_label": "above" if actual_class == 1 else "below",
        "pred_label":   "above" if pred_class == 1 else "below",
        "correct":      correct,
    })

    label_actual = "ABOVE" if actual_class == 1 else "BELOW"
    label_pred   = "ABOVE" if pred_class == 1 else "BELOW"
    tick = "OK" if correct else "WRONG"
    print(f"  [{tick}] Tier {tier} ({TIER_NAMES[tier]})")
    print(f"        ER={er:.3f}  (tier median={med:.3f})")
    print(f"        Actual={label_actual}  Predicted={label_pred}")
    print()

# ── 6. Summary ────────────────────────────────────────────────────────────────
n_correct = sum(r["correct"] for r in results)
n_total   = len(results)
print("=" * 65)
print(f"Summary: {n_correct}/{n_total} correct ({100*n_correct/n_total:.0f}%)")
print("=" * 65)

res_df = pd.DataFrame(results)
print(res_df[["tier_name", "er", "tier_median", "actual_label",
              "pred_label", "correct"]].to_string(index=False))
