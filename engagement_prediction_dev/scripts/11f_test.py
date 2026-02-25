"""
11f_test.py
===========
End-to-end test of the 11f per-tier engagement prediction pipeline.

Steps:
  1. Load raw LinkedIn posts (linkedin_posts_new.csv)
  2. Sample ~3 rows from each of the 4 follower tiers
  3. Run through the full pipeline (clean -> preprocess -> engineer)
  4. Load the saved 11f tier models + tier medians from metadata.json
  5. Predict above/below median engagement for each post
  6. Show predicted vs actual side-by-side

Run from project root or scripts/ directory:
    python scripts/11f_test.py
"""

import os
import sys
import json
import importlib.util

import joblib
import numpy as np
import pandas as pd

# ── Resolve paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models", "11f_per_tier")

# ── Load pipeline module (name starts with digit, so use importlib) ───────────
_spec = importlib.util.spec_from_file_location(
    "pipeline_11f", os.path.join(SCRIPT_DIR, "11f_pipeline.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

run_pipeline = _mod.run_pipeline
TIER_NAMES   = _mod.TIER_NAMES     # {0: "micro (<10k)", ...}

# ── Load metadata + tier models ───────────────────────────────────────────────
meta_path = os.path.join(MODELS_DIR, "metadata.json")
if not os.path.exists(meta_path):
    sys.exit(f"ERROR: metadata.json not found at {meta_path}\n"
             "Run 11f_per_tier_models.ipynb (or _run_11f_train_save.py) first.")

with open(meta_path) as f:
    metadata = json.load(f)

tier_models  = {}   # tier_id (int) -> fitted classifier
tier_medians = {}   # tier_id (int) -> float ER median (from training set)

print("Loading trained models...")
for tier_str, info in metadata["tiers"].items():
    t = int(tier_str)
    tier_medians[t] = info["median_er"]
    model_path = os.path.join(MODELS_DIR, info["model_file"])
    tier_models[t] = joblib.load(model_path)
    print(f"  Tier {t} ({info['tier_name']:20s})  "
          f"{info['best_model']:14s}  train macro-F1={info['macro_f1']:.4f}")

# ── Load raw data and sample across tiers ─────────────────────────────────────
raw_path = os.path.join(DATA_DIR, "linkedin_posts_new.csv")
raw = pd.read_csv(raw_path)
print(f"\nRaw dataset loaded: {len(raw)} rows, {len(raw.columns)} columns")

# Assign rough tiers to enable balanced sampling
_followers = pd.to_numeric(raw["followers"], errors="coerce").fillna(0).clip(lower=0)
raw["_tier_tmp"] = pd.cut(
    _followers,
    bins=[0, 10_000, 50_000, 200_000, float("inf")],
    labels=[0, 1, 2, 3],
    include_lowest=True,
).astype(float).astype("Int64")

rng = np.random.default_rng(42)
sampled_indices = []
for t in [0, 1, 2, 3]:
    pool = raw.index[raw["_tier_tmp"] == t].tolist()
    k    = min(3, len(pool))
    chosen = rng.choice(pool, size=k, replace=False).tolist()
    sampled_indices.extend(chosen)
    print(f"  Tier {t} pool={len(pool):3d}  sampled={k}")

sample_raw = (
    raw.loc[sampled_indices]
       .drop(columns=["_tier_tmp"])
       .reset_index(drop=True)
)
print(f"\nTotal sampled: {len(sample_raw)} posts")

# ── Run the pipeline ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Running pipeline...")
print("=" * 60)
X, y = run_pipeline(sample_raw, verbose=True)
print("=" * 60)

if len(X) == 0:
    sys.exit("ERROR: Pipeline returned 0 rows — check input data.")

# ── Predict per row and compare ───────────────────────────────────────────────
rows = []
for i in range(len(X)):
    tier = int(y.loc[i, "follower_tier"])
    er   = float(y.loc[i, "engagement_rate"])

    if tier not in tier_models:
        rows.append({
            "tier": tier, "tier_name": TIER_NAMES.get(tier, "?"),
            "engagement_rate": er, "tier_median": None,
            "actual": None, "predicted": None, "correct": None,
            "note": "no model for tier",
        })
        continue

    med          = tier_medians[tier]
    actual_class = int(er >= med)
    pred_class   = int(tier_models[tier].predict(X.iloc[[i]])[0])

    rows.append({
        "tier":            tier,
        "tier_name":       TIER_NAMES[tier],
        "engagement_rate": round(er, 3),
        "tier_median":     round(med, 3),
        "actual":          actual_class,   # 0=below, 1=above
        "predicted":       pred_class,
        "correct":         actual_class == pred_class,
        "note":            "",
    })

results = pd.DataFrame(rows)

# ── Display results table ─────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("RESULTS")
print("=" * 75)

header = (f"{'#':>2}  {'Tier':<22}  {'ER':>8}  {'Median':>8}  "
          f"{'Actual':>8}  {'Pred':>8}  {'OK?':>5}")
print(header)
print("-" * 75)

label = lambda c: ("above" if c == 1 else "below") if c is not None else "n/a"

for idx, r in results.iterrows():
    mark = ("OK" if r["correct"] else "MISS") if r["correct"] is not None else "n/a"
    print(f"{idx:>2}  {r['tier_name']:<22}  {r['engagement_rate']:>8.3f}  "
          f"{(r['tier_median'] if r['tier_median'] else 0):>8.3f}  "
          f"{label(r['actual']):>8}  {label(r['predicted']):>8}  {mark:>5}")

print("-" * 75)
n_correct = int(results["correct"].sum())
n_total   = int(results["correct"].notna().sum())
print(f"Sample accuracy: {n_correct}/{n_total} = {n_correct/n_total:.1%}")
print()

# ── Per-tier breakdown ────────────────────────────────────────────────────────
print("Per-tier breakdown:")
for t in sorted(results["tier"].unique()):
    sub = results[results["tier"] == t]
    c   = int(sub["correct"].sum())
    n   = int(sub["correct"].notna().sum())
    print(f"  Tier {t} ({TIER_NAMES.get(t, '?'):22s})  {c}/{n} correct")

print()
