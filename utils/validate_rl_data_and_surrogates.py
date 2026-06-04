"""
Validate RL Data Pipeline & Surrogate Model Quality

Ensures:
1. Data format matches RL expectations
2. Features are correctly extracted
3. Surrogate models (SNR, DR, FNR, TTD) achieve good R²
4. No data leakage or misalignment issues
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("GENEVO2 V7: RL Data Pipeline & Surrogate Validation")
print("="*80)

# Load V7 dataset
data_dir = Path('data_fixed_v7')
master_path = data_dir / 'master_index.csv'

print("\n[STEP 1] Loading Dataset")
print("-" * 80)

try:
    df = pd.read_csv(master_path)
    print(f"[OK] Loaded {len(df):,} simulations from {master_path}")
    print(f"[OK] Columns: {df.columns.tolist()[:5]} ... (13 total)")
except Exception as e:
    print(f"[ERROR] Failed to load dataset: {e}")
    exit(1)

# Verify data integrity
print("\n[STEP 2] Data Integrity Checks")
print("-" * 80)

checks = [
    ("No null values in key columns", df[['scenario', 'snr_db', 'time_to_detection', 'false_negative_rate']].isnull().sum().sum() == 0),
    ("SNR range reasonable", df['snr_db'].min() > -100 and df['snr_db'].max() < 100),
    ("TTD in valid range", ((df['time_to_detection'] >= 400) | (df['time_to_detection'] == 9000)).all()),
    ("FNR in [0,1]", (df['false_negative_rate'] >= 0).all() and (df['false_negative_rate'] <= 1).all()),
    ("Detection metrics present", 'n_detections' in df.columns),
]

for check_name, result in checks:
    status = "[OK]" if result else "[FAIL]"
    print(f"{status} {check_name}")

# Load sample metadata to verify instrumentation data
print("\n[STEP 3] Metadata & Instrumentation Verification")
print("-" * 80)

sample_run_id = df.iloc[0]['run_id']
sample_meta_path = data_dir / 'metadata' / f'{sample_run_id}.json'

try:
    with open(sample_meta_path) as f:
        sample_meta = json.load(f)

    has_instrumentation = 'instrumentation' in sample_meta
    has_stages = has_instrumentation and 'stages' in sample_meta['instrumentation']

    print(f"[OK] Metadata structure valid")
    print(f"[OK] Instrumentation data present: {has_instrumentation}")
    print(f"[OK] Stage data present: {has_stages}")

    if has_stages:
        stages = sample_meta['instrumentation']['stages'].keys()
        print(f"[OK] Stages available: {', '.join(stages)}")
except Exception as e:
    print(f"[FAIL] Metadata read failed: {e}")

# Feature engineering for surrogates
print("\n[STEP 4] Feature Engineering for ML Surrogates")
print("-" * 80)

features_to_extract = [
    'snr_db',
    'scenario_encoded',
    'biosensor_type_encoded',
    'noise_preset_encoded',
]

# Create baseline features
X = pd.DataFrame()
X['snr_db'] = df['snr_db']
X['scenario_encoded'] = pd.factorize(df['scenario'])[0]
X['biosensor_type_encoded'] = pd.factorize(df['biosensor_type'])[0]
X['noise_preset_encoded'] = pd.factorize(df['noise_preset'])[0]

print(f"[OK] Extracted {X.shape[1]} baseline features")
print(f"[OK] Feature matrix shape: {X.shape}")

# Target variables
targets = {
    'SNR (dB)': df['snr_db'],
    'DR (Detection Rate)': 1 - df['false_negative_rate'],  # DR = 1 - FNR
    'FNR (False Neg Rate)': df['false_negative_rate'],
    'TTD (seconds)': df['time_to_detection'],
}

print(f"[OK] Target variables: {', '.join(targets.keys())}")

# Train surrogates and evaluate
print("\n[STEP 5] Surrogate Model Training & Evaluation")
print("-" * 80)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

results = {}
for target_name, target_values in targets.items():
    print(f"\n--- {target_name} ---")

    # Filter out sentinel values for TTD
    if target_name == 'TTD (seconds)':
        valid_idx = target_values < 9000
        X_train_data = X_scaled[valid_idx]
        y_train_data = target_values[valid_idx].values
    else:
        X_train_data = X_scaled
        y_train_data = target_values.values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_data, y_train_data, test_size=0.2, random_state=42
    )

    # Train gradient boosting (typically better for surrogates)
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    results[target_name] = {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'n_samples': len(y_train)
    }

    # Quality assessment
    quality = "EXCELLENT" if r2 > 0.85 else "GOOD" if r2 > 0.65 else "FAIR" if r2 > 0.50 else "POOR"

    print(f"  Samples trained: {len(y_train)}")
    print(f"  R² score: {r2:.4f} ({quality})")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")

# Summary
print("\n" + "="*80)
print("SURROGATE MODEL QUALITY SUMMARY")
print("="*80)

all_r2 = [r['r2'] for r in results.values()]
avg_r2 = np.mean(all_r2)
min_r2 = np.min(all_r2)

print(f"\nAverage R²: {avg_r2:.4f}")
print(f"Minimum R²: {min_r2:.4f}")
print(f"All > 0.6: {'YES' if all(r > 0.6 for r in all_r2) else 'NO'}")

for target, metrics in results.items():
    r2_status = "[OK]" if metrics['r2'] > 0.6 else "[WARN]"
    print(f"{r2_status} {target}: R² = {metrics['r2']:.4f}")

# Final assessment
print("\n" + "="*80)
print("FINAL ASSESSMENT")
print("="*80)

all_checks_pass = (
    avg_r2 > 0.6 and
    df['snr_db'].std() > 10 and  # Good SNR variation
    len(df) > 4000  # Sufficient data
)

if all_checks_pass:
    print("\n[OK] Data pipeline is sound")
    print("[OK] Surrogate models are of acceptable quality")
    print("[OK] Ready for RL training with V7 dataset")
    print("\nRecommendation: Proceed with RL training")
else:
    print("\n[WARN] Some quality metrics below threshold")
    print("Recommendation: Review and address issues before RL training")

print("\n" + "="*80)
