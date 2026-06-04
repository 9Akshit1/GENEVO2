# Surrogate Model Rebuild Summary

## Problems Fixed

### 1. **Wrong Input Features**
- **Before**: Surrogates trained on `[snr_db, biosensor_type_enc, noise_preset_enc]`
  - SNR is a *derived* output, not an input parameter
  - This caused massive extrapolation errors (163.8% avg error on training data)
  
- **After**: Surrogates trained on original design parameters
  - `[kd, sensitivity, biosensor_type_enc, noise_preset_enc, scenario_enc]`
  - Directly predicts `[detection_rate, fnr, time_to_detection]`
  - Much better fit (R² = 0.745-0.794 on test set)

### 2. **Missing Encoder Initialization**
- **Error**: "Encoders not initialized. Call refit_scaler() first"
- **Root Cause**: Label encoders weren't being saved/loaded properly
- **Fix**: Now save label_encoders_v1.pkl during training
  - SurrogateLoader._initialize() loads them automatically
  - encode_categorical() can now safely transform all categorical vars

### 3. **Scaler Not Saved**
- **Before**: Scaler was fit on data but not persisted
- **After**: Now save scaler_v1.pkl with surrogates
  - SurrogateLoader loads it automatically
  - All scaling is consistent between training and BO

### 4. **Integration into BO Pipeline**
- **Before**: Expected surrogates to exist at BO/results/saved_ml/
- **After**: bo_main.py now:
  1. Checks if surrogates exist (scaler_v1.pkl)
  2. If missing, auto-builds them from data using SurrogateBuilder
  3. Saves to output surrogate_dir
  4. Loads them with proper initialization

### 5. **Search Space Mismatch**
- **Before**: target_scenario included "both" (not in training data)
- **After**: target_scenario = ["healthy", "pmo", "ckd_mbd"]
  - Matches training data exactly
  - No more "unseen labels" errors

## Files Created/Modified

### New Files
- **BO/core/build_surrogates.py** - SurrogateBuilder class
  - Extracts features from metadata JSON files
  - Trains models on actual design parameters
  - Saves surrogates + scaler + encoders

### Modified Files
- **BO/core/surrogate_loader.py** - Rewritten
  - Now loads scaler and label_encoders automatically
  - Single call to _initialize() sets everything up
  - No "Encoders not initialized" errors anymore

- **BO/evaluation/objective_function.py** - Updated
  - Uses new 5-feature input format
  - Calls encode_categorical() with scenario param
  - Still estimates SNR for visualization

- **BO/bo_main.py** - Enhanced
  - Imports SurrogateBuilder
  - Auto-builds surrogates if missing
  - Better error handling

- **BO/search_space/biosensor_space.py** - Fixed
  - target_scenario now matches training data

## Feature Importance (from new surrogates)

| Feature | Detection Rate | FNR | TTD |
|---------|---|---|---|
| Scenario | 56.5% | 58.1% | 50.9% |
| Biosensor Type | 30.0% | 30.4% | 40.6% |
| Sensitivity | 6.7% | 5.9% | 3.9% |
| Kd | 6.1% | 5.2% | 3.9% |
| Noise | 0.7% | 0.5% | 0.7% |

**Key Insight**: Disease scenario (healthy vs PMO vs CKD-MBD) is the dominant factor (~55%), followed by biosensor type (~30%). This makes biological sense.

## Surrogate Quality

New surrogates show good generalization:

| Metric | Detection Rate | FNR | TTD |
|--------|---|---|---|
| CV R² | 0.7279 | 0.7364 | 0.7859 |
| Test R² | 0.7465 | 0.7525 | 0.7941 |
| Test RMSE | 0.2058 | 0.2027 | 1555.7 |
| Train-Test Gap | 0.1153 | 0.1055 | 0.0646 |

Gaps < 0.15 indicate good model fit without excessive overfitting.

## Usage

```bash
# Auto-builds surrogates, then runs BO (100 total evals)
python BO/bo_main.py --n-init 20 --n-iter 80

# Output directory contains:
# - BO/results/saved_ml/    ← saved surrogates
# - BO/bo_results_fresh/    ← BO results
#   - results/best_config.json
#   - logs/iteration_log.csv
#   - plots/
```

## Next Steps

1. Run validate_bo_design.py to test best design against actual simulator
2. Run validate_surrogate_accuracy.py to verify accuracy on training domain
3. Compare with validation results from earlier
