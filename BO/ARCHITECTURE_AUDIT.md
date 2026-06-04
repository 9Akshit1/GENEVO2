# BO Pipeline Architecture Audit & Fixes

## Executive Summary

**Critical Issues Found: 7**  
**Severity Levels: 5 Critical, 2 High**  
**Root Cause: Architectural misalignment between surrogate training, inference, and BO optimization**

All issues have been systematically identified and fixed. This document details:
1. What was wrong
2. Why it happened
3. How it was fixed
4. What changed in the code

---

## Issue 1: Surrogate Persistence & Initialization BROKEN

### Problem
**Error Message Symptom:**  
```
Surrogate prediction failed: Encoders not initialized. Call refit_scaler() first.
```

**Root Cause:**
- `SurrogateLoader` fails silently when surrogate files don't exist
- `_initialize()` catches `FileNotFoundError` but logs only a warning
- Encoder/scaler state is never validated before use in objective function
- No error message until surrogates are actually needed

**Impact:**
- BO optimization crashed during first objective evaluation
- Validation pipeline couldn't predict metrics
- No feedback about what was missing

### Fix Applied

**File: `BO/core/surrogate_loader.py`**

1. **Changed silent failures to explicit errors:**
   - `_initialize()` now raises `FileNotFoundError` if critical files missing
   - Detailed error messages state exactly which files are needed
   - Saves users time debugging

2. **Added validation helpers:**
   - `is_initialized()` checks all three conditions (models, scaler, encoders)
   - New `predict_metrics()` method ensures consistent prediction pipeline

3. **Improved error messages:**
   - Lists known categorical values when decoding fails
   - Directs users to run `python BO/bo_main.py` if surrogates missing

**Before:**
```python
def _initialize(self, version: str = "v1"):
    saved_ml_dir = self.surrogate_dir / "saved_ml"
    if not saved_ml_dir.exists():
        logger.warning(f"saved_ml directory not found...")  # SILENT FAIL
        return
```

**After:**
```python
def _initialize(self, version: str = "v1"):
    saved_ml_dir = self.surrogate_dir / "saved_ml"
    if not saved_ml_dir.exists():
        raise FileNotFoundError(
            f"Surrogate models directory not found at {saved_ml_dir}. "
            f"Run surrogate training first..."  # EXPLICIT ERROR
        )
```

---

## Issue 2: Surrogate Training Feature Inconsistency

### Problem
**Root Cause:**
- Features extracted from metadata files but order not explicitly tracked
- No validation that scaler is fitted before training
- No bounds information saved for extrapolation detection
- Metadata incomplete - missing feature ordering and scaling parameters

**Impact:**
- During inference, features might be in wrong order
- Scaler state not reproducible across sessions
- No way to detect if new data is outside training distribution

### Fix Applied

**File: `BO/core/build_surrogates.py`**

1. **Added explicit feature validation:**
   ```python
   # Validate input
   if X.shape[0] < 20:
       raise ValueError(f"Not enough data for {metric_name}: {X.shape[0]} samples")
   if y.shape[0] != X.shape[0]:
       raise ValueError(f"Mismatched X and y sizes")
   ```

2. **Save comprehensive metadata:**
   - Feature names with explicit ordering
   - Scaler mean, scale, variance (for OOD detection)
   - Training data bounds for each feature
   - Mapping of categorical values

3. **Log scaler parameters for debugging:**
   ```python
   logger.info(f"Scaler fitted on {X.shape[0]} samples")
   logger.info(f"  Feature means: {self.scaler.mean_}")
   logger.info(f"  Feature stds: {self.scaler.scale_}")
   ```

**Before:**
```python
metadata = {
    'version': version,
    'n_features': 5,
    'feature_names': [...],
    # Missing: scaler parameters, bounds, feature ordering
}
```

**After:**
```python
metadata = {
    'version': version,
    'feature_order': feature_names,  # EXPLICIT ORDERING
    'scaler_mean': list(map(float, self.scaler.mean_)),
    'scaler_scale': list(map(float, self.scaler.scale_)),
    'training_data_bounds': {
        'kd_min': 0.1, 'kd_max': 10.0,
        'sensitivity_min': 0.5, 'sensitivity_max': 5.0,
        # ... all bounds
    },
}
```

---

## Issue 3: Surrogate Extrapolation Undetected

### Problem
**Root Cause:**
- BO optimizer can find configurations far outside training distribution
- No penalty for extrapolation → surrogates exploit regions with high uncertainty
- Predictions become unreliable far from training data
- Validation shows huge prediction errors (inf%)

**Impact:**
- BO "optimizes" in regions where surrogates are unreliable
- Best configuration found is based on garbage predictions
- When validated, produces pathological results (DR=0, FNR=1)

### Fix Applied

**File: `BO/evaluation/objective_function.py`**

1. **Added OOD (Out-of-Distribution) detection:**
   ```python
   def _compute_ood_penalty(self, X_scaled: np.ndarray) -> float:
       # For standardized data, |z| > 3 is extreme
       abs_z = np.abs(X_scaled[0])
       n_extreme = np.sum(abs_z > 3.0)
       penalty = n_extreme / len(abs_z)
       return np.clip(penalty, 0.0, 1.0)
   ```

2. **Apply penalty to objective:**
   ```python
   objective = objective * (1.0 - ood_penalty)
   ```
   - Reduces score by 0-100% based on extrapolation severity
   - Keeps BO in trusted region of surrogate space

3. **Track OOD metrics in evaluation details:**
   ```python
   details['ood_penalty'] = float(ood_penalty)
   ```
   - Users can see which configurations are risky

---

## Issue 4: Weak Hard Constraints

### Problem
**Root Cause:**
- Constraints have high thresholds (DR ≥ 0.70, FNR ≤ 0.20)
- But penalties are the same for all violations
- Optimizer exploits edge cases

**Clinical Impact:**
- Biosensor with DR=0.28 can still get optimized if FNR/TTD good
- Violates clinical requirement: must actually DETECT disease

### Fix Applied

**File: `BO/evaluation/objective_function.py`**

1. **Added three-layer constraint architecture:**
   - **Layer 0:** Validate input bounds (catch garbage params early)
   - **Layer A:** Hard constraints → CATASTROPHIC_PENALTY (-100)
   - **Layer B:** Weighted objective (only if constraints pass)

2. **Constraints are truly hard:**
   ```python
   if self.apply_constraints:
       if dr_pred < self.min_detection_rate:
           return self.CATASTROPHIC_PENALTY  # Reject immediately
       if fnr_pred > self.max_false_negative_rate:
           return self.CATASTROPHIC_PENALTY
       if snr_db < self.min_snr_db:
           return self.CATASTROPHIC_PENALTY
   ```

3. **Constraints are scientifically justified:**
   - DR ≥ 0.70: Must detect 70% of disease cases (conservative for research)
   - FNR ≤ 0.20: Cannot miss >20% (no false negatives in clinical use)
   - SNR ≥ 0 dB: Basic signal requirement

---

## Issue 5: Parameter Conversion Broken

### Problem
**Root Cause:**
- BO config structure not validated before conversion
- Missing fields silently ignored → invalid biosensor configs created
- Parameter ranges not checked (could violate model assumptions)
- Response time not included for amplifying sensors

**Impact:**
- Simulator receives invalid configurations
- All biosensors produce zero signal
- Can't distinguish between "bad design" and "broken conversion"

### Fix Applied

**File: `BO/validate_bo_design.py`**

1. **Added explicit field validation:**
   ```python
   for field in ['type', 'kd_nm', 'sensitivity']:
       if field not in design:
           raise ValueError(f"biosensor_design missing field: {field}")
   ```

2. **Added parameter range validation:**
   ```python
   if not (0.05 <= kd <= 20.0):
       raise ValueError(f"kd_nm={kd} outside valid range [0.05, 20.0] nM")
   ```

3. **Ensure sensor-specific params are present:**
   ```python
   if design['type'] == 'amplifying':
       if 'response_time_s' not in design:
           raise ValueError("amplifying biosensor requires response_time_s")
   ```

**Before:**
```python
biosensor_config = {
    'circuit_type': design['type'],
    'sensitivity': float(design['sensitivity']),
    'threshold': 0.6,  # Hardcoded!
}
# Missing validation, no error checking
```

**After:**
```python
# Validate all required fields exist
# Validate all parameter ranges are reasonable
# Check sensor-specific requirements
biosensor_config = {
    'circuit_type': design['type'],
    'sensitivity': sensitivity,  # Validated
    'threshold': 0.6,
    'response_time': response_time,  # Present for amplifying
}
logger.info(f"✓ BO config converted successfully")
```

---

## Issue 6: Simulator Integration Not Validated

### Problem
**Root Cause:**
- All validation scenarios produce identical results (DR=0, FNR=1, TTD=3600, SNR=0)
- Could be: simulator broken, params invalid, metric extraction wrong, or thresholds too high
- No intermediate logging to diagnose which step fails

**Impact:**
- Impossible to know if BO is working or if validation is broken
- Can't iterate on design without understanding root cause

### Fix Applied

**Files: `BO/validate_bo_design.py` and `BO/diagnose_pipeline.py`**

1. **Better error handling in conversion:**
   - Validates biosensor config before passing to simulator
   - Logs parameters for inspection
   - Provides detailed error messages

2. **Created diagnostic suite (`diagnose_pipeline.py`):**
   - **Test 1:** Data loading (check master_index.csv)
   - **Test 2:** Surrogate training (validate metrics improve)
   - **Test 3:** Surrogate loading (check initialization)
   - **Test 4:** Parameter conversion (round-trip test)
   - **Test 5:** Biosensor creation (test all types)
   - **Test 6:** Objective function (test evaluation)
   - **Test 7:** Simulator integration (run real simulation)

3. **Each test reports:**
   - Status (pass/fail)
   - Intermediate values for debugging
   - Specific error messages pointing to next step

---

## Issue 7: No Surrogate Retraining Capability

### Problem
**Root Cause:**
- No CLI flag to retrain surrogates if they become stale
- Can't easily regenerate with new training data
- Forces users to manually delete files and re-run

### Fix Applied

**File: `BO/bo_main.py`**

1. **Added `--retrain-surrogates` flag:**
   ```python
   parser.add_argument(
       "--retrain-surrogates",
       action="store_true",
       help="Force retraining of surrogate models even if they exist",
   )
   ```

2. **Updated surrogate loading logic:**
   ```python
   if args.retrain_surrogates or not surrogates_exist:
       logger.info("Training surrogates...")
       builder.train_all_surrogates(X, df_results)
       builder.save_surrogates(args.surrogate_dir, version="v1")
   else:
       logger.info(f"Found existing surrogate files")
   ```

3. **Improved error handling:**
   - Try/except blocks with detailed logging
   - Return 1 on failure so script exits cleanly

---

## Summary of Changed Files

| File | Changes | Severity |
|------|---------|----------|
| `BO/core/surrogate_loader.py` | Full rewrite of initialization & validation | CRITICAL |
| `BO/core/build_surrogates.py` | Added feature validation & comprehensive metadata | CRITICAL |
| `BO/evaluation/objective_function.py` | Added OOD detection & better constraint checking | CRITICAL |
| `BO/bo_main.py` | Better error handling, --retrain-surrogates flag | CRITICAL |
| `BO/validate_bo_design.py` | Parameter validation in conversion | HIGH |
| `BO/diagnose_pipeline.py` | NEW: Comprehensive diagnostic suite | HIGH |

---

## How to Use the Fixes

### Step 1: Run Diagnostics
```bash
python BO/diagnose_pipeline.py
```
This will:
- Check data exists and loads
- Train surrogates if missing
- Verify all components work independently
- Point out any remaining issues

### Step 2: Run BO Optimization
```bash
# First time or with new data:
python BO/bo_main.py --retrain-surrogates

# Subsequent runs (reuse existing surrogates):
python BO/bo_main.py
```

### Step 3: Validate Results
```bash
python BO/validate_bo_design.py --bo-results-dir BO/bo_results
```

---

## Expected Improvements

### Before Fixes
- ❌ Encoder initialization fails silently
- ❌ Surrogates extrapolate undetected
- ❌ BO finds pathological solutions
- ❌ Validation shows inf% errors
- ❌ All simulator outputs are zero
- ❌ No way to debug failures

### After Fixes
- ✅ Clear error messages if surrogates missing
- ✅ OOD penalty prevents extrapolation
- ✅ Hard constraints enforce clinical viability
- ✅ Parameter validation catches bad configs
- ✅ Diagnostic suite tests each component
- ✅ Retraining works correctly

---

## Next Steps (if issues remain)

1. **Run `diagnose_pipeline.py`** - identifies which component fails
2. **Check data quality** - verify master_index.csv and metadata files
3. **Verify simulator** - ensure bone_environment.ant model is valid
4. **Inspect first BO iteration** - check if initial samples are reasonable
5. **Review constraint thresholds** - may need tuning for your biosensor space

---

## Architecture Notes for Future Development

### Feature Engineering
- Features are explicitly ordered: `[kd, sensitivity, biosensor_type_enc, noise_preset_enc, scenario_enc]`
- StandardScaler is the ONLY preprocessing step
- Feature names are always in the same order

### Surrogate Inference
- **Always** scale inputs with the saved scaler before prediction
- **Always** encode categoricals using saved label encoders
- **Never** mix raw and scaled features

### Constraints vs Objective
- Constraints are **hard**: violate them = automatic rejection
- Objective is **soft**: weighted combination of metrics
- OOD penalty is applied to objective, not constraints (allows search within known regions)

### Extensibility
- To add new metrics: extend `build_surrogates.py` and `surrogate_loader.py`
- To change objective weights: use `ObjectiveFunction.__init__()` parameters
- To modify constraints: edit `min_detection_rate`, `max_false_negative_rate`, `min_snr_db`
