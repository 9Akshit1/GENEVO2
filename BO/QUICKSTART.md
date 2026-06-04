# BO Pipeline Quick Start Guide

## Overview

The Bayesian Optimization (BO) pipeline has been comprehensively fixed and now includes:

1. **Robust surrogate model training** with proper state persistence
2. **Out-of-distribution detection** to prevent extrapolation
3. **Hard clinical constraints** to ensure realistic solutions
4. **Comprehensive diagnostics** to identify any remaining issues
5. **Proper parameter validation** throughout the pipeline

## Quick Start (5 Steps)

### Step 1: Run Diagnostics

First, verify that everything is working:

```bash
python BO/diagnose_pipeline.py
```

This will:
- Check that data exists and loads correctly
- Train surrogates (if needed)
- Verify surrogate persistence
- Test parameter conversion
- Test biosensor creation
- Test objective function evaluation
- Test simulator integration

All 7 tests should pass. If any fail, fix the specific issue (see ARCHITECTURE_AUDIT.md for details).

### Step 2: Train Surrogates (if needed)

If diagnostics show surrogates are missing:

```bash
python BO/bo_main.py --retrain-surrogates --n-init 20 --n-iter 80
```

This will:
- Load training data from `data/master_index.csv`
- Train three surrogate models (DR, FNR, TTD)
- Save models to `BO/bo_results/saved_ml/`
- Run BO optimization for 20 initial + 80 iterations

**Retraining flag:** Use `--retrain-surrogates` to force retraining even if models exist.

### Step 3: Run Optimization

For quick tests:

```bash
python BO/bo_main.py --n-init 5 --n-iter 10
```

For full optimization:

```bash
python BO/bo_main.py --n-init 20 --n-iter 80
```

### Step 4: Validate Results

Test the best configuration from BO on the actual simulator:

```bash
python BO/validate_bo_design.py --bo-results-dir BO/bo_results
```

This will:
- Load the best BO configuration
- Run actual simulations for each disease scenario
- Compare surrogate predictions vs actual results
- Report prediction accuracy

**Expected output:** Detection rates should be realistic (>50% for good designs), not pathological (0%).

### Step 5: Inspect Results

Results are saved to `BO/bo_results/`:

- `results/best_config.json` - Best biosensor design found
- `results/optimization_results.json` - Summary of optimization
- `logs/iteration_log.csv` - Per-iteration scores and parameters
- `plots/convergence.png` - Convergence curve
- `validation_report.json` - Comparison of predictions vs actual

## Configuration Parameters

### BO Parameters

```bash
python BO/bo_main.py \
  --n-init 20 \           # Number of initial random samples (default: 20)
  --n-iter 80 \           # Number of BO iterations (default: 80)
  --random-state 42 \     # Random seed for reproducibility
  --retrain-surrogates    # Force retrain (default: reuse if exist)
```

### Paths

```bash
python BO/bo_main.py \
  --data-dir data \                    # Input data directory
  --surrogate-dir BO/bo_results \      # Where to save surrogates
  --output-dir BO/bo_results           # Where to save BO results
```

### Optional RL Comparison

```bash
python BO/bo_main.py \
  --compare-rl \                       # Enable RL comparison
  --rl-dir rl_results_v7               # Path to RL results
```

## Understanding the Output

### Diagnostic Output

```
✓ Data loaded: 1000 samples
✓ Surrogates trained: DR_R²=0.85, FNR_R²=0.78, TTD_R²=0.82
✓ Surrogates loaded: 3 models initialized
✓ Parameters converted: biosensor created successfully
✓ Objective evaluated: score=0.725
✓ Simulator works: DR=0.80, FNR=0.10, TTD=500s, SNR=15dB
```

**Interpretation:**
- R² > 0.7 is good
- Score 0-1 where 1 is best
- DR > 0.5 is clinically relevant
- SNR > 0 dB is necessary

### BO Optimization Output

```
[1/5] Building/Loading Surrogate Models
  ✓ Loaded 3 surrogate models

[2/5] Initializing BO Components
  ✓ Initialized all components

[3/5] Running Bayesian Optimization
  Sample 1/100: biosensor=amplifying, Kd=2.5, score=0.45
  Sample 2/100: biosensor=direct_binding, Kd=1.2, score=0.52
  ...
  Sample 100/100: biosensor=amplifying, Kd=3.1, score=0.78

[4/5] Comparing with RL Baseline (optional)
  ✓ Comparison saved

[5/5] Summary
  Best composite score: 0.7823
  Predicted Detection Rate: 0.8500
  Predicted False Negative Rate: 0.0800
  Predicted Time to Detection: 425.3 s
  Estimated SNR: 18.50 dB

  Best Biosensor Design:
    Type: amplifying
    Kd: 3.14 nM
    Sensitivity: 2.78
    Response time: 289 s
    Noise preset: medium
    Target scenario: pmo
```

### Validation Output

```
[1/5] Loading best BO configuration...
  ✓ Loaded best_config.json

[2/5] Converting BO parameters to biosensor configuration...
  ✓ Biosensor type: amplifying
    Kd: 3.1400 nM
    Sensitivity: 2.7800

[3/5] Testing biosensor disease separability...
  ✓ Separability check: H=0.234, P=0.567, C=1.123 — PASS

[4/5] Running actual simulator for all scenarios...
  Running: healthy...
    DR=0.0000, FNR=1.0000, TTD=3600.0s, SNR=0.00dB
  Running: pmo...
    DR=0.8500, FNR=0.1500, TTD=425.3s, SNR=18.50dB
  Running: ckd_mbd...
    DR=0.9200, FNR=0.0800, TTD=380.1s, SNR=22.40dB

[5/5] Comparing surrogate predictions vs actual simulation results...

  PREDICTED (from surrogates):
    Detection Rate:       0.850000
    False Negative Rate:  0.080000
    Time-to-Detection:    425.3 s
    Estimated SNR:        18.50 dB

  ACTUAL (from simulator):
    Detection Rate:       0.866667
    False Negative Rate:  0.133333
    Time-to-Detection:    401.7 s
    Estimated SNR:        20.45 dB

  PREDICTION ERRORS:
    Detection Rate error:      2.0%
    False Negative Rate error: 66.7%
    Time-to-Detection error:   5.5%
    SNR error:                10.5%

  ✓ GOOD: Prediction error 21.0% — Surrogates are trustworthy
  ✓ Design is clinically viable (DR > 50%)
```

**Interpretation:**
- Healthy scenario always has DR≈0 (normal - not disease)
- Disease scenarios should have DR > 50% (good design)
- Prediction errors < 20% are excellent
- Prediction errors 20-50% are acceptable
- Prediction errors > 50% indicate surrogates are unreliable

## Troubleshooting

### Issue: "Surrogate models directory not found"

**Solution:** Run diagnostics which will train surrogates:
```bash
python BO/diagnose_pipeline.py
```

### Issue: Data loading fails

**Check:** Does `data/master_index.csv` exist?

```bash
ls data/master_index.csv
```

If not, generate data first:
```bash
python dataset/generator.py --n-simulations 1000
```

### Issue: All simulator outputs are zero

**Likely cause:** Invalid biosensor parameters

**Debug:** Run diagnostics test 5 and 7
```bash
python BO/diagnose_pipeline.py
```

Look for failures in:
- "TEST 5: Biosensor Creation"
- "TEST 7: Simulator Integration"

### Issue: BO results show pathological solutions (DR=0, score=-100)

**Likely cause:** Hard constraints too strict

**Fix:** Edit constraint thresholds in `BO/evaluation/objective_function.py`:
```python
MIN_DETECTION_RATE = 0.50  # Lower threshold
MAX_FALSE_NEGATIVE_RATE = 0.30  # Higher tolerance
MIN_SNR_DB = -5.0  # More lenient
```

### Issue: "Feature names" warning in IDE

**Not a problem:** This is safe to ignore (unused variable in refactored code).

## Advanced Usage

### Custom Objective Weights

Edit objective function to weight metrics differently:

```python
# In BO/evaluation/objective_function.py
weight_dr = 0.50    # Emphasize detection rate
weight_fnr = 0.20   # Less weight to false negatives
weight_ttd = 0.15   # Less weight to speed
weight_snr = 0.15   # Bonus for signal quality
```

### Disable Constraints for Testing

If you want to understand the unconstrained optimum:

```python
# In BO/bo_main.py, when creating ObjectiveFunction:
objective_fn = ObjectiveFunction(
    physics_model, surrogate_loader,
    apply_constraints=False  # Disable hard constraints
)
```

⚠️ WARNING: This may find unrealistic solutions. Use only for research.

### Use RL Results as Warmstart

To compare BO with RL baseline:

```bash
python BO/bo_main.py --compare-rl --rl-dir rl_results_v7
```

This loads RL-optimized designs and shows how BO compares.

## Expected Runtime

- Diagnostics: ~2-5 minutes (depends on data size)
- Surrogate training: ~1-3 minutes
- BO optimization (20 init + 80 iter): ~10-30 minutes
- Validation: ~2-5 minutes

Total time: ~15-45 minutes for full pipeline

## File Structure After Running

```
BO/
├── bo_main.py                  # Entry point
├── validate_bo_design.py        # Validation script
├── diagnose_pipeline.py         # Diagnostic suite (NEW)
├── ARCHITECTURE_AUDIT.md        # Detailed audit report (NEW)
├── QUICKSTART.md               # This file (NEW)
├── bo_results/                 # Output directory
│   ├── saved_ml/               # Surrogate models
│   │   ├── surrogate_detection_rate_v1.pkl
│   │   ├── surrogate_fnr_v1.pkl
│   │   ├── surrogate_ttd_v1.pkl
│   │   ├── scaler_v1.pkl
│   │   ├── label_encoders_v1.pkl
│   │   └── metadata_v1.json
│   ├── results/
│   │   ├── best_config.json    # Best design
│   │   └── optimization_results.json
│   ├── logs/
│   │   ├── iteration_log.csv   # Per-iteration details
│   │   └── bo_optimization.log
│   ├── plots/
│   │   ├── convergence.png
│   │   └── robustness_heatmap.png
│   ├── models/
│   │   └── gp_surrogate.pkl    # Fitted GP model
│   └── validation_report.json   # Validation results
└── ... (other modules)
```

## Next Steps

1. **Run diagnostics** to identify any issues
2. **Train surrogates** if needed
3. **Run optimization** with desired parameters
4. **Validate results** with actual simulator
5. **Inspect best design** in `BO/bo_results/results/best_config.json`

---

**For detailed technical information, see ARCHITECTURE_AUDIT.md**
