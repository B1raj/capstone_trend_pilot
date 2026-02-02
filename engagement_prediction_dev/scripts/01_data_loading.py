"""
Step 1.1: Data Loading & Initial Cleaning
==========================================
This script loads the LinkedIn influencers dataset and performs initial cleaning:
- Load influencers_data.csv
- Remove duplicates
- Drop posts with missing content
- Drop posts with missing reactions/comments
- Validate data types
- Handle outliers (cap at 99th percentile or log transform)
- Basic data quality checks

Output: data/cleaned_data.csv
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# Configuration
INPUT_FILE = '../eda/influencers_data.csv'
OUTPUT_DIR = '../data'
OUTPUT_FILE = 'cleaned_data.csv'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("STEP 1.1: DATA LOADING & INITIAL CLEANING")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/7] Loading data...")
df = pd.read_csv(INPUT_FILE)
initial_rows = len(df)
initial_cols = len(df.columns)
print(f"   ✓ Loaded: {initial_rows:,} rows × {initial_cols} columns")

# Display basic info
print("\n   Dataset Info:")
print(f"   - Unique influencers: {df['name'].nunique()}")
print(f"   - Date range: {df['time_spent'].value_counts().head(3).to_dict() if 'time_spent' in df.columns else 'N/A'}")

# ============================================================================
# 2. REMOVE DUPLICATE ROWS
# ============================================================================
print("\n[2/7] Removing duplicate rows...")
duplicates_total = df.duplicated().sum()
print(f"   - Found {duplicates_total} exact duplicate rows")

if duplicates_total > 0:
    df = df.drop_duplicates()
    print(f"   ✓ Removed {duplicates_total} duplicates")
else:
    print("   ✓ No duplicates found")

# Check for duplicate content (same post text)
if 'content' in df.columns:
    content_duplicates = df['content'].dropna().duplicated().sum()
    print(f"   - Found {content_duplicates} posts with duplicate content")
    if content_duplicates > 0:
        print("   ℹ Keeping duplicate content (may be reposts by different influencers)")

# ============================================================================
# 3. HANDLE MISSING CRITICAL DATA
# ============================================================================
print("\n[3/7] Handling missing critical data...")

# Check missing values in key columns
critical_columns = ['content', 'reactions', 'comments']
missing_summary = {}

for col in critical_columns:
    if col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        missing_summary[col] = {'count': missing_count, 'pct': missing_pct}
        print(f"   - {col}: {missing_count:,} missing ({missing_pct:.2f}%)")

# Drop rows with missing content
if 'content' in df.columns:
    before = len(df)
    df = df[df['content'].notna()].copy()
    removed = before - len(df)
    print(f"   ✓ Removed {removed:,} rows with missing content")

# Drop rows with missing target variables (reactions or comments)
targets_missing_before = len(df)
if 'reactions' in df.columns and 'comments' in df.columns:
    df = df[(df['reactions'].notna()) & (df['comments'].notna())].copy()
    removed = targets_missing_before - len(df)
    if removed > 0:
        print(f"   ✓ Removed {removed:,} rows with missing reactions/comments")
    else:
        print(f"   ✓ All rows have reactions and comments data")

# ============================================================================
# 4. VALIDATE AND CONVERT DATA TYPES
# ============================================================================
print("\n[4/7] Validating and converting data types...")

# Convert numeric columns
numeric_cols = ['reactions', 'comments', 'num_hashtags', 'followers', 'hashtag_followers']

for col in numeric_cols:
    if col in df.columns:
        # Convert to numeric, coercing errors
        original_type = df[col].dtype
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Report any conversion issues
        nulls_after = df[col].isnull().sum()
        if nulls_after > 0:
            print(f"   ⚠ {col}: {nulls_after} values couldn't be converted to numeric")
        else:
            print(f"   ✓ {col}: converted to numeric ({original_type} → {df[col].dtype})")

# Ensure reactions and comments are integers and non-negative
if 'reactions' in df.columns:
    df['reactions'] = df['reactions'].fillna(0).astype(int)
    df.loc[df['reactions'] < 0, 'reactions'] = 0
    
if 'comments' in df.columns:
    df['comments'] = df['comments'].fillna(0).astype(int)
    df.loc[df['comments'] < 0, 'comments'] = 0

print(f"   ✓ Data type conversion complete")

# ============================================================================
# 5. HANDLE OUTLIERS
# ============================================================================
print("\n[5/7] Handling outliers...")

def detect_and_handle_outliers(df, column, method='cap', percentile=99):
    """
    Detect and handle outliers in a numeric column.
    
    Parameters:
    - method: 'cap' (cap at percentile) or 'log' (log transform)
    - percentile: percentile threshold for capping
    """
    if column not in df.columns or df[column].isnull().all():
        return df, {}
    
    values = df[column].dropna()
    threshold = np.percentile(values, percentile)
    outliers = (values > threshold).sum()
    
    stats = {
        'min': values.min(),
        'max': values.max(),
        'mean': values.mean(),
        'median': values.median(),
        f'{percentile}th_percentile': threshold,
        'outliers_count': outliers,
        'outliers_pct': (outliers / len(values)) * 100
    }
    
    if method == 'cap' and outliers > 0:
        df[column] = df[column].clip(upper=threshold)
        print(f"   - {column}: Capped {outliers:,} outliers at {threshold:.0f} ({stats['outliers_pct']:.1f}%)")
    elif method == 'log':
        # We'll add a log-transformed column for modeling later
        print(f"   - {column}: {outliers:,} outliers detected ({stats['outliers_pct']:.1f}%), will use log transform in modeling")
    
    return df, stats

# Handle outliers in engagement metrics
outlier_stats = {}

for col in ['reactions', 'comments']:
    if col in df.columns:
        df, stats = detect_and_handle_outliers(df, col, method='cap', percentile=99)
        outlier_stats[col] = stats

# Handle follower outliers
if 'followers' in df.columns:
    df, stats = detect_and_handle_outliers(df, 'followers', method='cap', percentile=99)
    outlier_stats['followers'] = stats

print(f"   ✓ Outlier handling complete")

# ============================================================================
# 6. DATA QUALITY CHECKS
# ============================================================================
print("\n[6/7] Performing data quality checks...")

quality_issues = []

# Check for negative values
for col in ['reactions', 'comments', 'followers', 'num_hashtags']:
    if col in df.columns:
        negatives = (df[col] < 0).sum()
        if negatives > 0:
            quality_issues.append(f"{col} has {negatives} negative values")
            df[col] = df[col].clip(lower=0)

# Check for empty content
if 'content' in df.columns:
    empty_content = (df['content'].str.strip() == '').sum()
    if empty_content > 0:
        quality_issues.append(f"{empty_content} posts have empty content")
        df = df[df['content'].str.strip() != '']

# Check for unrealistic values
if 'reactions' in df.columns and 'followers' in df.columns:
    # Flag posts with reactions > 10x followers (likely data error)
    unrealistic = (df['reactions'] > df['followers'] * 10).sum()
    if unrealistic > 0:
        quality_issues.append(f"{unrealistic} posts have reactions > 10x followers")

if quality_issues:
    print("   ⚠ Quality issues found:")
    for issue in quality_issues:
        print(f"      - {issue}")
else:
    print("   ✓ No critical quality issues found")

# ============================================================================
# 7. SAVE CLEANED DATA
# ============================================================================
print("\n[7/7] Saving cleaned data...")

final_rows = len(df)
rows_removed = initial_rows - final_rows
removal_pct = (rows_removed / initial_rows) * 100

output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
df.to_csv(output_path, index=False)

print(f"   ✓ Saved to: {output_path}")
print(f"   - Final dataset: {final_rows:,} rows × {len(df.columns)} columns")
print(f"   - Rows removed: {rows_removed:,} ({removal_pct:.1f}%)")

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print("\n" + "="*80)
print("CLEANING SUMMARY")
print("="*80)

print(f"\nData Transformation:")
print(f"  Initial rows:     {initial_rows:,}")
print(f"  Duplicates:       -{duplicates_total:,}")
print(f"  Missing content:  -{missing_summary.get('content', {}).get('count', 0):,}")
print(f"  Missing targets:  -{(targets_missing_before - len(df)):,}")
print(f"  Quality issues:   -{len(quality_issues)}")
print(f"  {'─'*30}")
print(f"  Final rows:       {final_rows:,} ({(final_rows/initial_rows)*100:.1f}% retained)")

print(f"\nTarget Variable Statistics:")
if 'reactions' in df.columns:
    print(f"  Reactions:")
    print(f"    - Min: {df['reactions'].min():,}")
    print(f"    - Max: {df['reactions'].max():,}")
    print(f"    - Mean: {df['reactions'].mean():.0f}")
    print(f"    - Median: {df['reactions'].median():.0f}")
    print(f"    - Std: {df['reactions'].std():.0f}")

if 'comments' in df.columns:
    print(f"  Comments:")
    print(f"    - Min: {df['comments'].min():,}")
    print(f"    - Max: {df['comments'].max():,}")
    print(f"    - Mean: {df['comments'].mean():.0f}")
    print(f"    - Median: {df['comments'].median():.0f}")
    print(f"    - Std: {df['comments'].std():.0f}")

print(f"\nData Quality:")
print(f"  ✓ No duplicate rows")
print(f"  ✓ No missing target variables")
print(f"  ✓ All numeric columns validated")
print(f"  ✓ Outliers handled (capped at 99th percentile)")
print(f"  ✓ Non-negative values enforced")

print("\n" + "="*80)
print("✓ STEP 1.1 COMPLETE - Ready for Step 1.2 (Text Preprocessing)")
print("="*80)

# Save statistics for reference
stats_summary = {
    'initial_rows': initial_rows,
    'final_rows': final_rows,
    'rows_removed': rows_removed,
    'removal_percentage': removal_pct,
    'duplicates_removed': duplicates_total,
    'missing_content': missing_summary.get('content', {}).get('count', 0),
    'outlier_stats': outlier_stats,
    'quality_issues': quality_issues
}

import json
stats_path = os.path.join(OUTPUT_DIR, 'cleaning_stats.json')
with open(stats_path, 'w') as f:
    json.dump(stats_summary, f, indent=2, default=str)

print(f"\n📊 Statistics saved to: {stats_path}")
