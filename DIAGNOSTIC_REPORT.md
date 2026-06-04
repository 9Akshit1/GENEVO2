# GENEVO2 BO DIAGNOSTIC REPORT

## CRITICAL ISSUES FOUND

### 1. **METRICS ACCESS BUG** (bo_main.py:204-206)
**Severity**: CRITICAL - Breaks entire BO pipeline
**Location**: `BO/bo_main.py` lines 204-206
**Issue**: Detection Rate Classifier returns classification metrics (ROC-AUC, Brier) NOT R² metrics
- Line 204 tries to access `metrics['detection_rate']['r2_test']` which doesn't exist
- `train_dr_classifier()` returns: `train_auc`, `test_auc`, `train_brier`, `test_brier`, `cv_auc_mean`, `cv_auc_std`
- Quantile regressors correctly return: `r2_train`, `r2_test`, `rmse_test`, `mae_test`
**Fix**: Update logging to use correct metric keys for each model type

---

### 2. **FEATURE ARRAY CONSTRUCTION BUG** (objective_function.py:300-320)
**Severity**: CRITICAL - Produces incorrect surrogate inputs
**Location**: `BO/evaluation/objective_function.py` lines 300-320
**Issue**: Feature array is constructed in wrong order
```python
# WRONG - what code currently does:
X = [[kd, sensitivity, biosensor_enc, noise_enc, scenario_enc, sensitivity]]
# Note: sensitivity is repeated, order is wrong
```
**Should be**: `[kd, sensitivity, biosensor_type_enc, noise_preset_enc, scenario_enc]`
**Extra Issues**:
- `encode_categorical()` is called 3x inefficiently
- Feature order contradicts surrogate training which uses: `[kd, sensitivity, biosensor_type_enc, noise_preset_enc, scenario_enc]`
- This causes massive surrogate mispredictions!
**Fix**: Refactor to build X_raw correctly, reuse encoder output

---

### 3. **SURROGATE PREDICTION RETURN BUG** (diagnose_pipeline.py:111)
**Severity**: MEDIUM - Test code incompatible with surrogate format
**Location**: `BO/diagnose_pipeline.py` line 111
**Issue**: Code unpacks 3 values from `predict_metrics()` but it returns 7
```python
dr, fnr, ttd = loader.predict_metrics(X_scaled)  # WRONG - returns 7 values!
```
**Should be**:
```python
dr, fnr_median, fnr_lower, fnr_upper, ttd_median, ttd_lower, ttd_upper = loader.predict_metrics(X_scaled)
```
**Fix**: Update test to handle v2 quantile regression outputs

---

### 4. **SEARCH SPACE INCONSISTENCY** (biosensor_space.py:104)
**Severity**: MEDIUM - Design issue
**Location**: `BO/search_space/biosensor_space.py` line 104
**Issue**: `target_scenario` includes "healthy" but surrogates are trained only on disease states
- Data shows "healthy" has detection_rate=0.0 for almost all cases
- BO should target only diseased scenarios: `["pmo", "ckd_mbd"]`
**Impact**: Half of random samples will be noise (healthy=0.0 DR always)
**Fix**: Remove "healthy" from target_scenario options

---

### 5. **MODEL PERFORMANCE ISSUE - FNR QUANTILE REGRESSION** 
**Severity**: HIGH - Reduces BO effectiveness
**Current Performance**:
```
FNR median CV R²: -0.4558 (+/- 0.0123)  ← Worse than mean baseline!
FNR lower CV R²: -0.4558 (+/- 0.0123)
FNR upper CV R²: 0.4375 (+/- 0.0285)
```
**Issue**: Negative R² means FNR predictions are unreliable
**Root Cause**: 
- FNR distribution is heavily skewed (mostly 0.0 with rare high values)
- Quantile regression struggles with extreme imbalance
- Training data shows FNR ≈ 0.0 for 99% of amplifying sensors
**Implications**: BO cannot trust FNR predictions, relies only on DR
**Fix Options**:
1. Use different quantile levels (0.25, 0.5, 0.75) instead of (0.1, 0.5, 0.9)
2. Use isotonic regression for FNR instead of GradientBoosting
3. Add FNR stratification to data loading
4. Reduce FNR weight in objective function

---

### 6. **DETECTION RATE CLASSIFIER CALIBRATION ISSUE**
**Severity**: MEDIUM - May overconfident in predictions
**Current Performance**:
```
Train Brier: 0.0230 (2.3% MSE on probabilities)
Test Brier: 0.0282 (2.8% MSE on probabilities)
```
**Issue**: Brier scores are very low but calibration may be overfitting
**Observation**: Training data shows DR ∈ {0.0, 1.0} (hard binary)
- Almost no intermediate values like DR=0.5
- Calibration on all-binary data may not generalize well
**Fix**: Evaluate model on disease vs. healthy subsets separately

---

### 7. **PHYSICS MODEL SNR BOUNDS** (physics_forward_model.py)
**Severity**: LOW - May produce unrealistic estimates
**Issue**: SNR clipped to [-50, 50] dB (lines 164, 287-296)
**Impact**: Unrealistic processing delays for very low SNR
**Observation**: Some test data has SNR < -20 dB (highly unfavorable)
**Fix**: Review SNR normalization in objective (line 153 of objective_function.py)

---

## DATASET QUALITY ISSUES

### Detection Rate Distribution
```
Detection Rate: min=0.0000, max=1.0000, mean=0.5585
- ~44% of dataset has DR=0.0 (no detection)
- ~56% of dataset has DR=1.0 (perfect detection)
- Few intermediate values
→ Makes regression harder, but binary classification appropriate
```

### False Negative Rate Distribution
```
FNR: min=0.0000, max=1.0000, mean=0.2568
- Heavily skewed to 0.0 (amplifying sensors rarely miss)
- Rare high FNR values (direct_binding sometimes fails)
- Median near 0.0, but mean=0.257 due to outliers
→ Quantile regression inappropriate; use classification instead
```

### Time-to-Detection Distribution
```
TTD: min=400.0, max=9000.0, mean=4358.7
- Range matches design (min=400s processing delay, max=timeout at 9000s)
- Skewed toward 400-1000s for high-SNR sensors
- Distribution suggests good feature separation
→ Quantile regression appropriate; model performs well (R²=0.81)
```

### Scenario Imbalance
```
Amplifying sensors: mostly 400-700s TTD (good)
Direct binding: mostly 700-9000s TTD (poor)
Healthy scenario: DR≈0.0 (always fails) → BAD TRAINING DATA
```

---

## ACCURACY/OPTIMIZATION IMPACT

### Problems Harming Accuracy:
1. **Feature order bug** → Surrogates receive scrambled inputs → predictions worthless
2. **FNR model poor performance** → BO ignores FNR constraint
3. **Search space includes "healthy"** → 50% of samples predict DR≈0.0
4. **Diagnostic code broken** → Can't validate models before BO run

### Problems Harming Optimization:
1. **Hard constraint MIN_DR=0.70** → 56% of training data fails (all healthy)
2. **OOD penalty may be too harsh** → (lines 134-138, 249-250)
3. **Weights biased to DR** → (45% weight on DR vs 15% on TTD)
4. **No scenario diversity** → BO can't optimize robustness across PMO/CKD-MBD

---

## RECOMMENDED FIX ORDER

### Phase 1: CRITICAL FIXES (blocks execution)
1. ✅ Fix metrics access in bo_main.py
2. ✅ Fix feature array construction in objective_function.py
3. ✅ Fix surrogate prediction unpacking in diagnose_pipeline.py

### Phase 2: HIGH PRIORITY (breaks correctness)
4. ✅ Remove "healthy" from target_scenario
5. ✅ Improve FNR model (use different approach)
6. ✅ Add data quality validation

### Phase 3: OPTIMIZATION IMPROVEMENTS
7. Adjust hard constraint thresholds based on achievable distribution
8. Tune objective weights based on importance
9. Add robustness analysis across scenarios

---

## VALIDATION CHECKLIST

After fixes:
- [ ] Run diagnose_pipeline.py with all tests passing
- [ ] Train surrogates and verify metrics are logged correctly
- [ ] Evaluate objective_function with test configs
- [ ] Run 5-iteration BO and check convergence
- [ ] Verify best_config is physically feasible
- [ ] Check output directory structure

