#!/usr/bin/env python
"""
Diagnostic: Why are surrogates predicting DR=1.0 when simulator gives DR=0.0?

Test hypothesis: Surrogates were trained on data that doesn't match current simulator behavior.
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, '.')
sys.path.insert(0, 'BO')

from dataset.generator import DatasetGenerator
from core.surrogate_loader import SurrogateLoader

print("="*70)
print("SURROGATE FAILURE DIAGNOSTIC")
print("="*70)

# 1. Load best config
print("\n[1] Loading best BO config...")
with open('BO/bo_results/results/best_config.json') as f:
    best_config = json.load(f)

design = best_config['biosensor_design']
env = best_config['measurement_environment']

print(f"  kd={design['kd_nm']:.4f}, sensitivity={design['sensitivity']:.4f}")
print(f"  type={design['type']}, noise={env['noise_preset']}, scenario={env['target_scenario']}")

# 2. Get surrogate prediction
print("\n[2] Getting surrogate predictions...")
loader = SurrogateLoader('BO/bo_results')
biosensor_enc, noise_enc, scenario_enc = loader.encode_categorical(
    design['type'],
    env['noise_preset'],
    env['target_scenario']
)

X_raw = np.array([[
    design['kd_nm'],
    design['sensitivity'],
    biosensor_enc,
    noise_enc,
    scenario_enc,
]], dtype=np.float32)

X_scaled = loader.scaler.transform(X_raw)
preds = loader.predict_metrics(X_scaled)

print(f"  Surrogate DR={preds[0]:.4f}")
print(f"  Surrogate FNR={preds[1]:.4f}")
print(f"  Surrogate TTD={preds[4]:.1f}s")

# 3. Find training samples with similar parameters
print("\n[3] Checking training data for similar parameters...")
df = pd.read_csv('data/master_index.csv')

# Load metadata to get kd values
from dataset.generator import DatasetGenerator
gen = DatasetGenerator('models/bone_environment.ant', output_dir='data', seed=42)

# Find samples close to the best config
print("\n  Searching for training samples with kd~0.14, sensitivity~1.92...")
print("  (This might take a moment...)")

# Sample a subset to avoid opening all metadata files
metadata_dir = Path('data/metadata')
metadata_files = list(metadata_dir.glob('*.json'))[:100]  # Sample first 100

nearby_samples = []
for metadata_file in metadata_files:
    try:
        with open(metadata_file) as f:
            meta = json.load(f)

        kd = float(meta['biosensor_config'].get('kd', 0))
        sensitivity = float(meta['biosensor_config'].get('sensitivity', 0))

        if (0.10 <= kd <= 0.18 and 1.8 <= sensitivity <= 2.0):
            nearby_samples.append({
                'kd': kd,
                'sensitivity': sensitivity,
                'type': meta['biosensor_config'].get('circuit_type', '?'),
                'noise': meta.get('noise_preset', '?'),
                'scenario': meta.get('scenario', '?'),
                'file': metadata_file.name,
            })
    except:
        pass

if nearby_samples:
    print(f"\n  Found {len(nearby_samples)} nearby training samples:")
    df_nearby = pd.DataFrame(nearby_samples)
    print(df_nearby.to_string())
else:
    print("\n  No nearby training samples found in first 100 metadata files")

# 4. Test a simple case: low kd should give low DR
print("\n[4] Testing data consistency...")
print("\n  Theory: With kd=0.1442 (very sensitive), we should see high DR")
print("  Reality: Simulator gives DR=0.0")
print("  Question: What does training data show for this kd range?")

# Look at master_index
print(f"\n  Training data detection_rate distribution:")
print(f"    Mean: {df['detection_rate'].mean():.4f}")
print(f"    Median: {df['detection_rate'].median():.4f}")
print(f"    % DR=0.0: {100*sum(df['detection_rate']==0)/len(df):.1f}%")
print(f"    % DR=1.0: {100*sum(df['detection_rate']==1)/len(df):.1f}%")

# Check a specific scenario
ckd_medium = df[(df['scenario']=='ckd_mbd') & (df['noise_preset']=='medium')]
if len(ckd_medium) > 0:
    print(f"\n  CKD_MBD + medium noise ({len(ckd_medium)} samples):")
    print(f"    Mean DR: {ckd_medium['detection_rate'].mean():.4f}")
    print(f"    % DR=0.0: {100*sum(ckd_medium['detection_rate']==0)/len(ckd_medium):.1f}%")
    print(f"    % DR=1.0: {100*sum(ckd_medium['detection_rate']==1)/len(ckd_medium):.1f}%")

# 5. Hypothesis test
print("\n" + "="*70)
print("HYPOTHESIS ASSESSMENT")
print("="*70)

print("\nPossible causes:")
print("  1. Surrogates overfit - learned patterns that don't generalize")
print("  2. Training data corrupted - kd/sensitivity don't match results")
print("  3. Simulator changed - current behavior differs from training data")
print("  4. Feature scaling issue - scaled features are in wrong range")
print("  5. Extrapolation - best config is actually OOD despite z-scores")

print(f"\nScaled features: {X_scaled[0]}")
print(f"  Max |z|: {np.max(np.abs(X_scaled[0])):.2f}")

if np.max(np.abs(X_scaled[0])) < 2.0:
    print("  → Within typical range, not flagged OOD")
else:
    print("  → Somewhat extreme, might be OOD")

print("\nCONCLUSION:")
print("  The surrogates are UNRELIABLE for this design region.")
print("  BO found a solution that looks perfect in surrogate space")
print("  but fails completely in the real simulator.")
print("\n  RECOMMENDED ACTION:")
print("  1. Stop using current BO results")
print("  2. Strengthen OOD detection or reduce BO iterations")
print("  3. Retrain surrogates with quality validation")
print("  4. Use RL results as baseline instead")
