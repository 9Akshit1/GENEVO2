# GENEVO2 -- Complete Agent Context Document

**Purpose:** This document gives a fresh AI agent everything it needs to understand the
GENEVO2 project end-to-end: scientific goals, biological model, code architecture, full
development history, all bugs found and fixed, all results, active constraints, and what
to do vs. what NOT to do. Written as of 2026-06-25.

---

## 1. WHAT THIS PROJECT IS

GENEVO2 is a computational biosensor optimization project. The goal is to design an
implantable/wearable multi-biomarker biosensor that can:

1. **Detect bone disease** -- specifically postmenopausal osteoporosis (PMO) and
   Chronic Kidney Disease - Mineral and Bone Disorder (CKD-MBD)
2. **Trigger targeted drug release** -- autonomously deliver the right dose of a
   sclerostin-inhibiting drug (romosozumab analog) to improve bone mineral density

The biosensor simultaneously detects three analytes in the bone microenvironment:
- **Sclerostin (SOST)**: the primary biomarker -- elevated in PMO and CKD-MBD
- **CTX (C-terminal telopeptide of type I collagen)**: bone resorption marker
- **P1NP (Procollagen type I N-terminal propeptide)**: bone formation marker

The project uses four main optimization/modeling approaches:
1. **Bayesian Optimization (BO)** -- the main approach, actively developed, GP-EI
2. **Multi-Objective BO (MOBO)** -- Pareto-front extension with EHVI acquisition
3. **Reinforcement Learning (RL)** -- secondary approach, abandoned at 0.55 reward (SNR leakage)
4. **GNN Surrogate** -- graph neural network alternative to GBM, message-passing over biosensor graph

The project is run on **Windows 11**, so all Python scripts must use ASCII-only output
(no Unicode box-drawing or symbol characters -- these crash the Windows cp1252 console).

---

## 2. BIOLOGICAL MODEL

### 2.1 ODE Model -- `models/bone_environment.ant`

The core simulation is an Antimony-format ODE model representing the bone microenvironment.
It is loaded by `libRoadRunner` (a systems biology simulator).

**Key species:**
- `Sclerostin_bone`, `Sclerostin_sensor`: sclerostin concentration in bone vs. sensor chamber
- `RANKL_bone`, `OPG_bone`: osteoclast activator/inhibitor pair
- `CTX_bone`, `CTX_sensor`: resorption marker
- `P1NP_bone`, `P1NP_sensor`: formation marker
- `$Estrogen`, `$PTH`: boundary species (hormone drivers, not evolved by ODEs)
- `OB` (osteoblasts), `OC` (osteoclasts): cellular populations

**Two compartments:**
- `bone` (volume 1.0): the actual bone tissue
- `sensor_chamber` (volume 0.1): the implanted sensor's sampling chamber

**Diffusion equilibrium:**
Biomarkers diffuse from bone to sensor chamber at 25x amplification.

**Nominal sensor concentrations (nM):**
| Biomarker | Healthy | PMO-mild | PMO   | CKD-MBD |
|-----------|---------|----------|-------|---------|
| SOST      | 0.375   | 0.5625   | 0.875 | 1.125   |
| CTX       | 0.200   | 0.300    | 0.500 | 0.500   |
| P1NP      | 0.350   | 0.385    | 0.525 | 0.625   |

**CRITICAL ODE BUG (now fixed):** Early versions called `roadrunner.reset()` after the
600s equilibration, reverting all disease ICs to healthy defaults. This meant PMO and CKD
scenarios were indistinguishable from healthy -- the entire simulator was producing wrong
data for months. Fix: removed the `reset()` call in `simulation/simulator.py`.

### 2.2 Disease Scenarios -- `models/environment_configs.py`

Seven scenarios are defined, but only four are used in BO optimization:
- `healthy`: premenopausal women reference (Estrogen=1.0 nM, PTH=45 pg/mL)
- `pmo_mild`: perimenopause onset, 1-3 years post-menopause (SOST 1.5x)
- `pmo`: established PMO, 5+ years post-menopause (SOST 2.3x, CTX 2.5x)
- `ckd_mbd`: CKD Stage 3-5 (SOST 3.0x, CTX 2.5x, P1NP 1.8x)

Other scenarios exist (pmo_severe, ckd_stage3, ckd_stage5d) but are NOT used in current
BO training data.

**Patient variability** is modeled via lognormal noise applied to each parameter:
- Rate constants (k_prod, k_deg): sigma=0.10-0.15
- Biomarker ICs: sigma=0.15-0.50 (disease-specific)
- PMO sigma raised in v5.0 audit: Sclerostin_bone sigma=0.35 (was 0.15)
  because literature (Mirza 2010, d=2.35) shows PMO patients span a wide range,
  with some established osteoporosis patients having LOWER sclerostin than healthy
  (osteocyte apoptosis reduces SOST-producing cells)
- CTX sigma raised in v5.1: 0.30->0.45 (Garnero 1994: inter-individual CV 45-55%)
- CKD sigma=0.50 for SOST (wide heterogeneity across Stage 3 to dialysis)

**data_v18 addition -- correlated biomarker sampling:**
In data_v18, PMO patient biomarkers are sampled with moderate correlation (r=0.17
from Kirmani 2010) rather than independently. This is scientifically more accurate
but does NOT materially affect surrogate performance because the rank correlation
at the population level is preserved.

**Boundary conditions:** Sensor ICs are ALWAYS derived from bone ICs after variability:
`CTX_sensor = 25 x CTX_bone`. This maintains diffusion equilibrium.

### 2.3 The Detection Logic -- `simulation/biosensor_engine.py`

The `BiosensorEngine` class applies a biosensor signal model to the ODE time series.

**Array biosensor signal formula:**
```
signal(t) = sensitivity x (w_scl x occ(scl, kd) + w_ctx x occ(ctx, kd_ctx) + w_p1np x occ(p1np, kd_p1np))
```
where `occ(c, kd) = c / (kd + c)` is the Langmuir occupancy.

**Threshold calibration (CRITICAL -- V5.0 fix):**
Old code: `threshold = sensitivity x occupancy(kd, C_thresh)`
This made sensitivity cancel out of the detection equation entirely:
```
detection = signal >= threshold
          = sensitivity x occ(S) >= sensitivity x occ(C)
          -> occ(S) >= occ(C)    # sensitivity divides out!
```
This meant sensitivity had 0% feature importance -- a catastrophic scientific error.

New code: threshold is calibrated against a REFERENCE sensor with sensitivity=1.0,
making the detection condition actually depend on sensitivity.

**Detection algorithm:**
- 5 consecutive timepoints above threshold -> detected
- `time_to_detection` = time of 5th consecutive point
- `detection_rate` = fraction of `n_trials` runs that detect
- `false_negative_rate` = 1 - detection_rate for disease scenarios
- TTD deterministic: 25s (SNR>25dB), 75s (SNR>15dB), 175s (SNR>5dB), 400s (SNR<=5dB)

**Locked detector config (v15):**
- Detector: simple persistence (no EWMA, no CUSUM, no smoothing)
- margin = 1.25 (threshold = healthy_signal x 1.25)
- detection_window = 10 (number of consecutive points required)
- rolling_window = 1 (no rolling average)

---

## 3. CODE ARCHITECTURE

```
GENEVO2/
|
|-- models/
|   |-- bone_environment.ant          # Antimony ODE model
|   |-- biosensors.py                 # Biosensor class hierarchy (Array, DirectBinding, etc.)
|   |-- environment_configs.py        # EnvironmentConfig: scenario definitions + variability
|   |-- noise.py                      # NoiseModel: adds sensor noise (realistic=13.6 dB)
|   +-- instrumentation_utilities.py
|
|-- simulation/
|   |-- simulator.py                  # BoneEnvironmentSimulator: wraps libRoadRunner ODE
|   |-- biosensor_engine.py           # BiosensorEngine: applies biosensor model to ODE output
|   |-- dataset/                      # Dataset generation utilities
|   |   |-- generator.py             # Full pipeline: sim + biosensor + metrics
|   +-- run_dataset.py                # CLI for batch generation
|
|-- data_v18/                         # ACTIVE training data (1,500 configs, 28 cols)
|   |-- master_index.csv              # One row per config-scenario run
|   |-- metadata/                     # Per-run JSON metadata files
|   +-- timeseries/                   # Per-run time series (optional)
|
|-- BO/                               # Bayesian Optimization (main branch)
|   |-- bo_main.py                    # UNIFIED entry point: --mode dispatch
|   |-- bo_closed_loop.py             # Real-sim closed-loop BO with surrogate refinement
|   |-- bo_converged.py               # Multi-seed convergence test (5+ independent runs)
|   |-- bo_patient_subtypes.py        # Per-subtype BO (4 clinical subtypes)
|   |-- bo_topology_search.py         # 2-channel vs 3-channel topology comparison
|   |-- inspect_surrogates.py         # Quick surrogate QA check
|   |
|   |-- search_space/
|   |   +-- biosensor_space.py        # 10-dim search space + bounds
|   |
|   |-- evaluation/
|   |   |-- therapeutic_objective_v6.py  # ACTIVE objective (relative threshold, no sens bias)
|   |   |-- pkpd_closed_loop.py          # PK/PD multi-dose romosozumab simulation
|   |   |-- drug_interactions.py         # Drug interaction model (thiazide, calcium, VitD)
|   |   |-- safety_constraints.py        # Hard clinical safety constraint checker
|   |   |-- safety_validation_final.py   # Full safety feasibility audit
|   |   |-- rl_adaptive_dosing.py        # RL-based adaptive dosing (experimental)
|   |   |-- robustness_analyzer.py       # Robustness under parameter perturbation
|   |   +-- physics_forward_model.py     # Thin wrapper (passes to surrogate)
|   |
|   |-- core/
|   |   |-- surrogate_loader_v3.py       # ACTIVE surrogate loader (10-feature, v18 models)
|   |   |-- build_surrogates_v3.py       # Retrain surrogates from CSV data
|   |   +-- ensemble_surrogate.py        # GBM + GNN weighted ensemble (v18 + GNN)
|   |
|   |-- mobo/                            # Multi-Objective Bayesian Optimization
|   |   |-- mobo_pipeline.py             # Full MOBO loop with EHVI acquisition
|   |   |-- mobo_objectives.py           # 3-objective decomposition (DR, therapeutic, specificity)
|   |   |-- pareto.py                    # Pareto front + hypervolume computation
|   |   +-- ehvi_acquisition.py          # Monte Carlo EHVI acquisition function
|   |
|   |-- surrogates/
|   |   +-- gnn/
|   |       |-- graph_biosensor.py       # Biosensor-as-graph representation
|   |       |-- gnn_surrogate.py         # NumpyMPNN (inference) + TorchMPNN (training)
|   |       +-- train_gnn.py             # Training loop (requires PyTorch)
|   |
|   |-- optimizer/
|   |   |-- gaussian_process_bo.py       # GP-based BO with EI acquisition
|   |   +-- bo_pipeline.py               # Orchestrates the BO loop
|   |
|   |-- acquisition/
|   |   +-- acquisition_functions.py     # EI, UCB implementations
|   |
|   |-- benchmarks/
|   |   |-- multi_optimizer_benchmark.py  # BO vs DE vs CMA-ES vs NSGA2-SE vs Random
|   |   +-- topology_comparison.py        # 2ch vs 3ch head-to-head comparison
|   |
|   |-- analysis/
|   |   |-- rank_rho_v18.py              # Spearman rank-rho validation on data_v18
|   |   |-- bo_multistart.py             # Multi-start BO analysis (50 seeds)
|   |   |-- kd_ctx_scan.py               # 1D kd_ctx scan with real simulator
|   |   +-- debug_pmo_mild_variance.py   # PMO-mild DR variance analysis
|   |
|   |-- validation/
|   |   |-- validate_top_designs.py       # Surrogate vs real simulator comparison
|   |   |-- sim_to_real_calibration.py   # Sim-to-real gap calibration check
|   |   +-- therapeutic_clinical_validation.py  # Vs romosozumab trial data (Cosman, McClung)
|   |
|   |-- diagnostics/
|   |   |-- score_distribution.py        # Landscape analysis (v3 vs v6 correlation)
|   |   |-- robustness_stress_test.py    # Stress test under perturbation
|   |   +-- parameter_landscape.py
|   |
|   |-- design_model/
|   |   |-- design_retrieval.py          # K-NN retrieval (replaces MLP)
|   |   |-- train_design_model.py        # MLP (deprecated -- R^2 = -0.0006)
|   |   +-- diagnose_mlp_failure.py      # Evidence for why MLP fails
|   |
|   |-- data_expansion/
|   |   |-- generate_optimized_dataset.py  # Generate 50k-100k optimized designs
|   |   +-- expand_dataset.py              # General dataset expansion utilities
|   |
|   +-- bo_results/                       # All BO run outputs
|       |-- saved_ml/                     # Surrogate pkl files (v18 is current)
|       |-- results/                      # best_config.json, optimization_results.json
|       |-- diagnostics/                  # baseline AUC, landscape, rank-rho results
|       |-- benchmark/                    # Multi-optimizer benchmark results
|       |-- convergence/                  # Multi-seed convergence report
|       |-- subtypes/                     # Per-subtype best configs
|       +-- topology/                     # 2ch vs 3ch topology results
|
|-- RL/                               # Reinforcement Learning branch (ABANDONED)
|   |-- rl_environment.py             # Gymnasium-compatible RL environment
|   |-- rl_model.py                   # PPO policy network
|   |-- rl_trainer.py                 # Training loop
|   |-- rl_pipeline.py                # Full pipeline: data -> surrogate -> RL -> eval
|   |-- surrogate_trainer.py          # Trains surrogates for RL (v2_rl, 3-feature)
|   +-- [diagnostic scripts]
|
|-- docs/
|   |-- AGENT_CONTEXT.md             # This file
|   |-- COMMANDS.md                  # Pipeline reference commands
|   |-- METHODS_PAPER_DRAFT.md       # Draft manuscript
|   +-- HONEST_ASSESSMENT_FINAL.md   # Honest capabilities/limitations assessment
|
+-- data_v18/                        # 1,500 training configs (see Section 4)
```

---

## 4. THE DATA -- data_v18

**Location:** `data_v18/master_index.csv`
**Size:** 1,500 rows, 28 columns

**Scenario distribution:**
| Scenario | Count |
|----------|-------|
| ckd_mbd  | 450   |
| pmo_mild | 375   |
| pmo      | 375   |
| healthy  | 300   |

**28 columns:**
```
run_id, timestamp, scenario, biosensor_type, noise_preset, topology,
correlated_sampling, kd, sensitivity, threshold, response_time, kd_scl,
kd_ctx, kd_p1np, w_scl, w_ctx, w_p1np, snr_db, n_detections, detection_rate,
time_to_detection, false_negative_rate, sclerostin_mean, sclerostin_std,
ctx_mean, p1np_mean, metadata_file, timeseries_file
```

**Key differences from data_v16:**
- **Correlated biomarker sampling**: PMO biomarkers have r=0.17 inter-correlation (Kirmani 2010)
- **Patient-specific PK/PD**: Drug response parameters vary per patient (not just per scenario)
- **Drug interactions included**: Calcium supplements, thiazide diuretics, Vitamin D interactions
- **4 patient subtypes**: young_pmo, elderly_pmo, ckd_controlled, ckd_advanced distinguished
- **Column name changes**: `kd_nm` -> `kd`, `kd_ctx_nm` -> `kd_ctx`, `kd_p1np_nm` -> `kd_p1np`
  (SurrogateLoaderV3 handles both old and new column names)

**IMPORTANT for code:** When reading data_v18 vs data_v16, the column names differ.
The GNN's `graph_biosensor.py:dataset_from_csv()` handles both naming conventions.

---

## 5. THE SURROGATE MODELS -- v18 (CURRENT)

Surrogates are GBM models that approximate the ODE simulator output without running
the expensive simulation. BO evaluates thousands of configs; surrogates take <1ms vs ~3s.

**Current active version: v18** (located in `BO/bo_results/saved_ml/`)

**Three separate models:**
1. `surrogate_detection_rate_v18.pkl` -- predicts detection rate [0,1] for a scenario
2. `surrogate_fnr_v18.pkl` -- predicts false negative rate [0,1]
3. `surrogate_ttd_v18.pkl` -- predicts time to detection [0, 9000]
4. `scaler_v18.pkl` -- StandardScaler for feature normalization
5. `metadata_v18.json` -- feature names, training stats, version info

**10 input features (FEATURE_NAMES in SurrogateLoaderV3):**
```
log_kd, log_sensitivity, log_response_time,
biosensor_type_enc, noise_preset_enc, scenario_enc,
log_kd_ctx, log_kd_p1np, w_ctx, w_p1np
```

**v18 surrogate performance (rank_rho_results.json, 2026-06-25):**
| Metric | v18 | v4 baseline | Threshold |
|--------|-----|-------------|-----------|
| Overall rank-rho | **0.8312** | 0.517 | >0.70 = good |
| PMO-mild rank-rho | 0.730 | n/a | |
| PMO rank-rho | 0.817 | n/a | |
| CKD rank-rho | 0.742 | n/a | |
| Bias | -0.016 | +0.046 | |
| RMSE | 0.248 | n/a | |

**Why v18 >> v4:** v18 was trained on 1,500 configs with patient subtypes, correlated
sampling, and drug interactions. v4 was trained on data_v10 (~37,500 configs but without
the subtype diversity). The subtype diversity -- especially the wide range of young_pmo
vs ckd_advanced patient profiles -- gives v18 dramatically better ranking.

**Critical: SNR is intentionally excluded.** Early surrogates (v1, v2_rl) included
`snr_db` as a feature. Since SNR is computed from the same simulation that produces
DR/FNR/TTD, it's a proxy that collapses all other feature importances to zero -- SNR
accounted for 89-90% of importance. Without SNR, feature importances are physically
interpretable: scenario_enc=44-53%, log_sensitivity=23-25%, log_kd=7-14%.

**How surrogates are called:**
```python
loader = SurrogateLoaderV3(results_dir="BO/bo_results")
dr, fnr, ttd = loader.predict(
    kd_nm=1.0, sensitivity=3.0, response_time=500.0,
    biosensor_type="array", noise_preset="realistic", scenario="pmo",
    kd_ctx=0.316, kd_p1np=0.2, w_ctx=0.35, w_p1np=0.15
)
```

**Retrain surrogates from new data:**
```powershell
python BO/core/build_surrogates_v3.py --data-dir data_v18 --out-dir BO/bo_results --version v18
```

---

## 6. THE OBJECTIVE FUNCTION -- TherapeuticObjectiveV6

**File:** `BO/evaluation/therapeutic_objective_v6.py`
**Constructor:** `TherapeuticObjectiveV6(physics_model, surrogate_loader_v3, apply_constraints=True)`

This is the ACTIVE optimization objective. V6 is sensitivity-independent in its
therapeutic term because it uses relative signal ratios (sensitivity cancels).

### 6.1 Therapeutic Term (40% weight) -- SENSITIVITY INDEPENDENT

Uses analytical Langmuir physics, not the surrogate.

```
R_scenario = composite_signal(scenario) / composite_signal(healthy)
           = (w_scl x occ_ratio_scl + w_ctx x occ_ratio_ctx + w_p1np x occ_ratio_p1np)
```

Drug dose: `dose = K_RELEASE x max(0, (R - DRUG_THRESHOLD_FRAC) / DRUG_THRESHOLD_FRAC)`
- DRUG_THRESHOLD_FRAC = 1.08
- K_RELEASE = 1.0, D_HALF = 0.15

Overdose penalty (quadratic when dose > D_SAFE=0.50):
```
overdose = max(0, dose/D_SAFE - 1)^2 x ALPHA_OVERDOSE (3.0)
```
**KNOWN ISSUE:** D_SAFE=0.50 is below the average dose/cycle for PMO (0.57) and CKD (0.81).
This means overdose_pct=100% for these scenarios. This is a model calibration problem --
D_SAFE needs raising to ~0.80, but this touches therapeutic parameters near ABSOLUTE
CONSTRAINTS. Do NOT change without explicit instruction.

Therapeutic weights: mild 55%, PMO 25%, CKD 20% (mild-centric by design).

### 6.2 Detection Terms (surrogate-based)

- DR term (25%): `0.5 x dr_mean + 0.5 x dr_min` (penalizes worst-case scenario)
- FNR term (15%): `1 - fnr_mean`
- TTD term (5%): `1 - ttd_mean/9000`
- FP penalty (15%): `-dr_healthy` (false positives on healthy patients)

### 6.3 Hard Constraints

- DR_ckd < 0.50 or DR_pmo < 0.50 -> penalty proportional to deficit
- FNR_max > 0.60 -> penalty
- DR_healthy > 0.15 -> FP penalty

### 6.4 Objective Ceiling

Max achievable composite score ~0.82-0.85. After v18 BO, best surrogate score ~0.72.
This is lower than the v4-era 0.793 because v18 surrogates evaluate configs more
strictly (diverse patient profiles, including hard ckd_advanced cases that pull DR down).

### 6.5 Key Output Keys (evaluate_with_details returns)

```
dr_mean, dr_ckd, dr_pmo, dr_mild, dr_min,
fnr_mean, ttd_mean, dr_healthy,
R_healthy, R_pmo_mild, R_pmo, R_ckd,
dose_mild, dose_pmo, dose_ckd,
overdose_mild, overdose_pmo, overdose_ckd,
bmd_net_mild, bmd_net_pmo, bmd_net_ckd,
therapeutic_mean, infeas_penalty, soft_penalty
```

**NEVER use old v3 key names:** `dr_pred`, `fnr_pred`, `ttd_pred_s`, `snr_db_est`,
`dr_ckd_pred`, `dr_pmo_pred`. These no longer exist and will cause KeyErrors.

---

## 7. THE SEARCH SPACE -- BiosensorSearchSpace

**File:** `BO/search_space/biosensor_space.py`

10-dimensional search space:

| Parameter | Type | Range | Scale |
|-----------|------|-------|-------|
| biosensor_type | categorical | ["array"] | -- (single value, dead dim) |
| kd_nm | continuous | [0.1, 10.0] nM | log |
| sensitivity | continuous | [0.5, 5.0] | log |
| response_time_s | continuous | [100, 3600] s | log |
| noise_preset | categorical | ["realistic"] | -- (single value, dead dim) |
| target_scenario | categorical | ["pmo", "ckd_mbd"] | -- |
| kd_ctx_nm | continuous | [0.1, 10.0] nM | log |
| kd_p1np_nm | continuous | [0.1, 10.0] nM | log |
| w_ctx | continuous | [0.01, 0.49] | linear |
| w_p1np | continuous | [0.01, 0.49] | linear |

**Known search space issues (NOT yet fixed):**
- sensitivity=5.0 always hits the upper bound (BO wants more sensitivity). Recommend
  expanding to [0.5, 10.0] but requires dataset regeneration.
- w_p1np=0.49 may be a secondary ridge -- landscape audit found a DIFFERENT global
  optimum at w_p1np=0.084 (score=0.728) vs BO's w_p1np=0.49 ridge (score=0.72).
  The true global optimum may be at low w_p1np.

---

## 8. BAYESIAN OPTIMIZATION PIPELINE

### 8.1 Main Entry Point -- `BO/bo_main.py` (UNIFIED, --mode dispatch)

```powershell
# Standard BO (default)
python BO/bo_main.py

# Multi-seed convergence test
python BO/bo_main.py --mode converged --n-runs 5 --n-init 50 --n-iter 150

# Patient subtype designs (4 subtypes)
python BO/bo_main.py --mode subtypes --n-init 30 --n-iter 100

# Multi-objective BO (Pareto front)
python BO/bo_main.py --mode mobo --n-init 32 --n-iter 100

# Full benchmark: BO vs DE vs CMA-ES vs NSGA-II vs Random
python BO/bo_main.py --mode benchmark --n-runs 20 --n-init 50 --n-iter 150

# Diagnostic: is 0.967 DR hard? (Priority 1)
python BO/bo_main.py --mode baseline

# Validate surrogate quality (Priority 4)
python BO/bo_main.py --mode validate

# Global landscape audit (Priority 3)
python BO/bo_main.py --mode landscape --n-samples 10000

# 2ch vs 3ch topology comparison
python BO/bo_main.py --mode topology

# Closed-loop BO with real simulator
python BO/bo_main.py --mode closed-loop --n-rounds 5 --n-inner 50
```

**IMPORTANT:** The old `--use-v6` flag no longer exists. All runs use `--mode standard`
(or just `python BO/bo_main.py` with no flags). Any script that spawns bo_main.py
subprocesses must use `--mode standard`, NOT `--use-v6`.

### 8.2 The GP -- `BO/optimizer/gaussian_process_bo.py`

- Kernel: RBF with active dimension selection
- Acquisition: Expected Improvement (EI)
- Initial samples: Sobol quasi-random (or LHS)
- Active dimensions: 6/10 (excludes scenario_enc, biosensor_type_enc, noise_preset_enc,
  response_time since they're effectively constant)

### 8.3 Current Best Results (2026-06-25)

**Standard BO (v18 surrogate) -- Quick run (n_init=20, n_iter=50):**
- Best composite score: 0.7205

**Full benchmark results (n_runs=5, budget=100 each):**
| Optimizer | Mean | Std | Wilcoxon p vs Random |
|-----------|------|-----|----------------------|
| BO | 0.7210 | 0.0112 | p=0.0625 (n.s. -- needs n=20) |
| CMA-ES | 0.7152 | 0.0099 | p=0.0625 (n.s.) |
| NSGA2-SE | 0.6673 | 0.0551 | p=0.3125 (n.s.) |
| DE | 0.6576 | 0.0233 | p=0.3125 (n.s.) |
| Random | 0.6327 | 0.0314 | -- |

**NOTE:** n=5 runs per optimizer is underpowered for Wilcoxon significance. The prior
publication-grade benchmark (n=20) gave BO p=0.0032. Rerun with `--n-runs 20` for paper.

**Global landscape audit results (2026-06-25, 10,000 LHS samples):**
- LHS mean: -0.027 (most random configs are poor)
- LHS P95: 0.5605
- LHS max: 0.7280 (10k random samples)
- BO best: 0.7205 (quick) / ~0.73 (full run)
- BO is in top ~0.1% of landscape -- globally competitive

**Multi-modality finding (IMPORTANT):**
The LHS global maximum (score=0.7280) uses w_p1np=0.084 and kd_ctx=2.56 -- very
different from BO's ridge (w_p1np=0.49, kd_ctx=3.72). The landscape has at least two
high-scoring ridges. BO found one; the true global optimum may be on the other.

### 8.4 Closed-Loop BO -- `BO/bo_closed_loop.py`

```powershell
python BO/bo_closed_loop.py --n-rounds 5 --n-inner 50 --top-k 10 --n-trials 5 --early-stop
```

Round flow:
1. BO finds top-K configs using current surrogate
2. Real simulator evaluates each config (n_trials per scenario)
3. Real measurements added to augmented_rows list
4. Surrogate retrained on original data + augmented data
5. Repeat

**Closed-loop results (pre-2026-06-25, with v4 surrogate):**
- After 5 rounds: real DR = 0.967 (n=10), corrected to 0.960 at n=50
- FP rate: ~5% [2%, 11%] (NOT 0% -- need n>=50 to estimate properly)
- kd_ctx corrected from 0.130 to 0.316 nM (+2pp DR, Langmuir-validated)

**Note on surrogate bias metric:**
`surrogate_bias` = composite_score - raw_DR is NOT a proper calibration metric. The
composite score is capped at ~0.82; raw DR grows toward 1.0. Growing gap is expected.
True calibration: `predicted_DR - actual_DR` (use validate_top_designs.py).

---

## 9. MULTI-OBJECTIVE BO (MOBO)

### 9.1 Overview -- `BO/mobo/`

MOBO decomposes the V6 composite objective into THREE SEPARATE objectives that a
Pareto-based optimizer can trade off without arbitrary weighting:

| Objective | Definition | Direction |
|-----------|-----------|-----------|
| f1: DR_mean | Mean detection rate across pmo_mild, pmo, ckd_mbd | Maximize |
| f2: therapeutic_mean | Weighted BMD net gain (V6 therapeutic term) | Maximize |
| f3: specificity | 1 - FP_rate on healthy patients | Maximize |

**Why this is better than V6 weighted sum:**
V6's weights (40% therapeutic, 25% DR, ...) are arbitrary choices. MOBO exposes the
FULL Pareto front -- a clinician can then choose:
- DR=0.97, therapeutic=0.60, specificity=0.90 (detection-first design)
- DR=0.92, therapeutic=0.85, specificity=0.95 (therapeutic-first design)
- etc.

**Feasibility constraints:**
- DR per disease scenario >= 0.85 (hard floor)
- FP rate <= 0.10 (hard ceiling)

**Reference point for hypervolume:** [0.0, -0.5, 0.0] (worst-case scenario)

### 9.2 EHVI Acquisition -- `BO/mobo/ehvi_acquisition.py`

Monte Carlo Expected Hypervolume Improvement. Fits one GP per objective, then uses
MC sampling to estimate the expected improvement in hypervolume. Acquisition maximization
uses L-BFGS-B from multiple random restarts.

**Parameters:**
- n_mc_samples = 64 (MC samples for EHVI estimation)
- n_restarts = 10 (L-BFGS-B restart count)

### 9.3 Running MOBO

```powershell
python BO/bo_main.py --mode mobo --n-init 32 --n-iter 100 --seed 42
```

Output: `BO/bo_results/mobo/mobo_results.json` with:
- pareto_configs: list of non-dominated configs
- pareto_objectives: [DR, therapeutic, specificity] for each Pareto point
- hypervolume_curve: hypervolume over iterations (convergence diagnostic)
- best_per_objective: single best config per objective (for comparison)

**Status:** MOBO code is complete and functional. Has NOT been run in production yet
(no mobo_results.json exists in bo_results/mobo/). This is a pending task.

---

## 10. GNN SURROGATE

### 10.1 Graph Representation -- `BO/surrogates/gnn/graph_biosensor.py`

A biosensor array is represented as a directed graph:
- **Nodes (3):** one per detection channel (SOST, CTX, P1NP)
- **Node features (7 per node):** [log_kd, weight, log_conc_healthy, log_conc_disease,
  occ_healthy, occ_disease, occ_ratio]
- **Edges (6):** all directed pairs (i->j AND j->i for undirected coverage)
- **Edge features (2 per edge):** [weight_product, log_kd_ratio]
- **Global features (5):** [log_sensitivity, log_response_time, noise_enc, scenario_enc, btype_enc]
- **Targets (3):** [DR, FNR, TTD_normalized]

**Why graph representation is valuable:**
- Node features capture each channel's discrimination capability independently
- Edge features model cross-channel correlations (CTX+SOST synergy)
- Global features handle disease context and sensing conditions
- Variable topology: zero-weight nodes model 1ch/2ch subsets of the 3ch array

### 10.2 MPNN Architecture -- `BO/surrogates/gnn/gnn_surrogate.py`

3-layer message-passing neural network:
- Layer 0: NODE_FEATURE_DIM -> HIDDEN_DIM=64 (no input projection, matches NumpyMPNN)
- Layers 1-2: HIDDEN_DIM -> HIDDEN_DIM
- Readout: mean pooling + global features -> HIDDEN_DIM
- Per-output heads: HIDDEN_DIM -> 32 -> 1 (three heads: DR, FNR, TTD)

**Two implementations:**
1. `NumpyMPNN` -- pure numpy, inference only, always available
2. `TorchMPNN` -- PyTorch, required for training; exports to NumpyMPNN .npz format

```python
# Inference (no PyTorch needed)
from BO.surrogates.gnn.gnn_surrogate import GNNSurrogate
gnn = GNNSurrogate(weights_path="BO/surrogates/gnn/weights/gnn_best.npz")
dr, fnr, ttd = gnn.predict(kd_nm=1.0, sensitivity=3.0, ..., scenario="pmo")

# Training (requires: pip install torch)
python BO/surrogates/gnn/train_gnn.py --data-csv data_v18/master_index.csv
```

**Status:** GNN code is complete. Weights file (`gnn_best.npz`) does NOT yet exist --
the GNN has not been trained. Training is a pending task (requires PyTorch install).

### 10.3 Ensemble Surrogate -- `BO/core/ensemble_surrogate.py`

Drop-in replacement for SurrogateLoaderV3 that weights GBM and GNN by their rank-rho:

```python
from BO.core.ensemble_surrogate import EnsembleSurrogate
ens = EnsembleSurrogate(
    surrogate_dir="BO/bo_results",
    gnn_weights="BO/surrogates/gnn/weights/gnn_best.npz"
)
dr, fnr, ttd = ens.predict(kd_nm=1.0, sensitivity=3.0, ...)
```

Weight calculation: `w_gbm = rho_gbm / (rho_gbm + rho_gnn)`, `w_gnn = 1 - w_gbm`.
GBM baseline rank-rho = 0.517 (v4). Ensemble automatically degrades to GBM-only if
GNN weights file not found.

---

## 11. PK/PD CLOSED-LOOP SIMULATION

### 11.1 Overview -- `BO/evaluation/pkpd_closed_loop.py`

Models a multi-dose feedback-controlled therapeutic cycle. This is NOT reinforcement
learning -- it is a simple threshold rule applied repeatedly.

**What it models:**
- t=0: Sensor measures SOST/CTX/P1NP -> composite signal R > threshold -> Drug released
- t=7d: Drug partially clears (romosozumab t_1/2 = 6.9 days, Padhi 2011 Bone)
  SOST suppressed by -27% (McClung 2014 JBMR at steady state)
- t=7d: Sensor re-measures -> if R still > threshold -> next dose
- t=14d, 21d, ...: Repeat up to max_cycles

**PK constants (Padhi 2011 Bone, romosozumab 210mg SC):**
- T_HALF_DAYS = 6.9 days
- K_EL = ln(2) / 6.9 = 0.100/day

**Biomarker suppression fractions at Cmax (McClung 2014 JBMR):**
- F_SUPP_SOST = 0.27 (SOST -27% at month 3)
- F_SUPP_CTX = 0.55 (CTX -55% at month 1 peak)
- F_SUPP_P1NP = 0.20 (P1NP -20% at month 3 trough)

**BMD gain model (calibrated to Cosman 2016 NEJM ARCH trial):**
- BMD_MAX_PER_DOSE = 0.06/4.0 g/cm2 per biweekly cycle
- BMD_T_RECOVERY = 30.0 days (time constant)
- BMD_BASELINE = 0.775 g/cm2 (ARCH trial mean baseline LS BMD)

### 11.2 Drug Interactions -- `BO/evaluation/drug_interactions.py`

Models common co-medication effects:
- **Thiazide diuretics:** Increase calcium reabsorption -> hypercalcemia risk amplified
- **Calcium supplements:** Additive with romosozumab BMD effect but raises serum Ca
- **Vitamin D deficiency:** Reduces romosozumab efficacy by 15-25%

Used in data_v18 generation to create realistic clinical heterogeneity.

---

## 12. SAFETY CONSTRAINTS

### 12.1 Hard Constraints -- `BO/evaluation/safety_constraints.py`

Non-negotiable clinical limits. Infeasible configs are EXCLUDED, not just penalized:

| Constraint | Threshold | Basis |
|------------|-----------|-------|
| Dosing ceiling | max 1.0 dose/cycle | FDA romosozumab label 2019 |
| BMD gain rate | <= 15% per 6 months | 2x ARCH trial rate (safety ceiling) |
| eGFR decline | <= 15% from baseline | Nephrotoxicity threshold |
| Serum calcium | <= 10.5 mg/dL | Hypercalcemia threshold |
| FP rate | <= 15% | Unnecessary dosing burden |
| Min DR | >= 50% | Detection floor |

**Safety feasibility (from data_v18 audit):**
- 95.7% of configs are feasible
- 4.3% violate hypercalcemia constraint (concentrated in CKD + thiazide diuretic group)
- Main infeasibility driver: high-SOST patient + thiazide + aggressive dosing

### 12.2 Safety Validation -- `BO/evaluation/safety_validation_final.py`

Full safety audit script for a candidate biosensor config:
```powershell
python BO/evaluation/safety_validation_final.py --config BO/bo_results/results/best_config.json
```

---

## 13. PATIENT SUBTYPES

### 13.1 Four Clinical Subtypes -- `BO/bo_patient_subtypes.py`

The population-average V6 objective is SATURATED for established PMO/CKD
(DR > 0.90 for nearly all kd_ctx values). The real design challenge is optimization
for clinically distinct patient subtypes:

| Subtype | Clinical Profile | Objective Re-weighting |
|---------|-----------------|----------------------|
| young_pmo | Age < 55, early PMO, good compliance | DR weight up, FP tolerance higher |
| elderly_pmo | Age > 70, established PMO + comorbidities | Safety-first, FP very dangerous |
| ckd_controlled | CKD Stage 3 (eGFR 30-60) | Kidney-safe dosing, CKD-BMD balance |
| ckd_advanced | CKD Stage 4-5 (eGFR < 30 or dialysis) | Strict safety ceiling, lowest DR threshold |

**V6 is NOT modified.** A `SubtypeObjective` wrapper calls `V6.evaluate_with_details()`
and re-combines the components with subtype-specific weights.

**Running all 4 subtypes:**
```powershell
python BO/bo_main.py --mode subtypes --n-init 30 --n-iter 100
```

**Output:** `BO/bo_results/subtypes/{subtype_name}/best_config.json` + comparison JSON.

**Four-class classification baseline:** LR on biomarkers achieves 67.1% 4-class accuracy
(vs 96.0% binary PMO/CKD detection). Subtype discrimination is the hard problem.

---

## 14. TOPOLOGY SEARCH

### 14.1 2-Channel vs 3-Channel -- `BO/bo_topology_search.py`

Answers the question: is CTX worth the manufacturing complexity?

- **2-channel:** SOST + P1NP only (w_ctx forced to 0)
- **3-channel:** SOST + CTX + P1NP (all weights free, current design)

**Results (Phase 2.1, 2026-06-20):**
- 2-channel mean score: 0.737 (±0.08)
- 3-channel mean score: 0.745 (±0.09)
- Mann-Whitney p = 0.19 (NOT statistically significant)

**Conclusion:** 2-channel SOST+P1NP is scientifically defensible as a simplification.
CTX adds marginal value at the V6 objective level. However, for disease SUBTYPE
discrimination (ckd vs pmo), CTX provides the key discriminating signal (CTX is equally
elevated in PMO and CKD, while P1NP differs). The 3-channel design was kept frozen.

---

## 15. CLINICAL VALIDATION

### 15.1 Romosozumab Trial Benchmarks -- `BO/validation/therapeutic_clinical_validation.py`

Compares GENEVO2 therapeutic predictions to published Phase 2/3 romosozumab trials:

| Trial | Key Data Used |
|-------|--------------|
| Cosman 2016 NEJM (ARCH, n=3591) | 6mo and 12mo BMD, lumbar spine + total hip |
| McClung 2014 JBMR (Phase 2, n=419) | CTX/P1NP suppression at months 1/3/6 |
| Padhi 2011 Bone (Phase 1) | PK: t_1/2 = 6.9 days (used for K_EL) |
| Genant 2017 Lancet (FRAME, n=7180) | BMD + fracture data |

**Sim-to-real calibration:** `BO/validation/sim_to_real_calibration.py`
Validates that simulator PK/PD predictions match clinical data within acceptable range.
Result: 0 FAILs (fixed in Phase 2 clinical foundation work 2026-06-18).

### 15.2 Baseline AUC (Priority 1 -- COMPLETED)

**File:** `BO/bo_results/diagnostics/baseline_auc_results.json`
**Result (from data_v18, 2026-06-25):**

| Feature Set | Binary AUC (disease vs healthy) |
|-------------|--------------------------------|
| SOST only | **0.960** |
| SOST + CTX | 0.959 |
| SOST + P1NP | 0.959 |
| SOST + CTX + P1NP | 0.958 |
| 4-class accuracy | 0.671 |

**Verdict (from results file):** "TRIVIALLY EASY (SOST alone AUC > 0.92). Biosensor
optimization primarily adds safety/efficiency value, not detection capability."

**CRITICAL IMPLICATION FOR PAPERS:** Do NOT frame the project as "BO improves
disease detection." Binary detection is trivially easy with SOST alone. BO's value is:
1. Therapeutic calibration (correct drug dosing for each patient)
2. Safety (minimizing overdose/false-positive dosing)
3. Disease SUBTYPE discrimination (4-class: 67% vs single-class trivial)

---

## 16. MULTI-OPTIMIZER BENCHMARK

### 16.1 Results -- `BO/bo_results/benchmark/benchmark_results.json`

**Latest run (2026-06-25, n=5 per optimizer, budget=100):**
See Section 8.3 for full table.

**Prior publication-grade run (n=20, budget=200, using v6 objective):**
- BO mean: 0.748, Random mean: 0.734
- BO lift: +0.014, p=0.0032 (BO wins)
- BO wins on 15/20 runs

**CMA-ES nearly ties BO:** CMA-ES mean=0.7152 vs BO=0.7210 (difference=0.006 < 1 std dev).
Both achieve similar quality at same budget. BO's advantage is sample efficiency
(explicit uncertainty modeling), not necessarily higher asymptotic score.

**Why V6 is the first objective where BO beats random:**
V3/V4/V5 showed BO=random (p>0.4) because the landscape was effectively 1D
(sensitivity-dominated, r=0.994 correlation). V6 breaks this with r(v3,v6)=0.860,
creating genuine multi-dimensional structure that GP can exploit.

---

## 17. MLP FAILURE AND K-NN RETRIEVAL

### 17.1 Why MLP Failed

The `train_design_model.py` script trained an MLP to predict "given patient biomarkers
(SOST, CTX, P1NP), what biosensor design is optimal?" This is a one-to-many mapping
problem -- many different biosensor configs achieve equally good scores (96x range in
kd_nm for the top 20% of configs). An MLP predicts a point (mean of this wide distribution)
which is never actually a good design -> R^2 = -0.0006.

Evidence from `diagnose_mlp_failure.py` (n=2000 LHS):
- kd_nm spans 96x range in top 20% (kd_cv >> 0.3) -- one-to-many confirmed
- 4 clusters, all with identical mean score 0.637-0.648 -- multimodal landscape

### 17.2 K-NN Retrieval -- `BO/design_model/design_retrieval.py`

Replaces MLP. Instead of predicting one design, retrieves a SET of good designs.

```powershell
# Build retrieval index
python BO/design_model/design_retrieval.py --build-index --use-existing-data

# Query: what biosensor for this patient profile?
python BO/design_model/design_retrieval.py --retrieve --sost 0.875 --ctx 0.500 --p1np 0.525 --disease pmo --severity moderate
```

---

## 18. REINFORCEMENT LEARNING BRANCH (ABANDONED)

### 18.1 Architecture -- `RL/`

- **Environment (`RL/rl_environment.py`):** State=[SNR, scenario_enc, biosensor_enc, noise_enc] (4D), Action=[delta_snr, delta_biosensor, delta_noise] (3D continuous), Reward=surrogate_DR - FNR_penalty
- **Agent:** PPO policy network (`RL/rl_model.py`)
- **Surrogates used:** OLD v2_rl surrogates (3-feature: SNR, biosensor_type, noise_preset)

### 18.2 Why RL Was Abandoned

The RL environment used v2_rl surrogates which had SNR leakage -- SNR encodes the answer
(DR is computed from the same simulation that produces SNR). The RL agent learned a
degenerate policy: "increase SNR" regardless of disease scenario. Different disease
scenarios (PMO vs CKD) got identical parameter adjustments.

**RL plateau:** ~0.55 reward, corresponding to ~55% DR -- well below BO's 0.967 real DR.

**v2_rl surrogates still exist** in `BO/bo_results/saved_ml/` as `surrogate_*_v2_rl.pkl`
but must NOT be used for new optimization work.

---

## 19. COMPLETE BUG HISTORY

### 19.1 Critical ODE Bug (Pre-v3 data)
**Bug:** `simulation/simulator.py` called `roadrunner.reset()` after equilibration,
reverting all disease-specific ICs to healthy defaults.
**Effect:** All scenarios indistinguishable. All data before v3 is corrupted.
**Fix:** Removed the `reset()` call.

### 19.2 Amplifying Sensor Threshold Bug
**Bug:** Threshold calibrated at t=0. Amplifying sensors output 0 at t=0 -> threshold=0.
**Fix:** Calibrate at t=1800s.

### 19.3 Sensitivity Cancellation Bug (Critical)
**Bug:** `threshold = sensitivity x occupancy(kd, C_thresh)` -> sensitivity divides out.
**Effect:** Sensitivity had 0% feature importance. BO could not exploit it.
**Fix:** Use reference sensitivity=1.0 for threshold calibration (v5.0 biosensor fix).

### 19.4 SNR Leakage in v1/v2 Surrogates
**Bug:** Surrogate features included snr_db. SNR accounted for 89-90% importance.
**Fix:** Completely excluded SNR from v3+ surrogate feature set.

### 19.5 Scenario Feature Excluded from Surrogate
**Bug:** v2_rl surrogate excluded scenario "to prevent data leakage" (incorrect reasoning).
**Fix:** Scenario included as scenario_enc in v3 feature set.

### 19.6 Wrong CKD Calibration Baseline
**Bug:** CKD baseline hardcoded to 0.424 nM (corrupted value from Bug 19.1).
**Fix:** Correct baseline used after removing reset().

### 19.7 Stale P1NP IC (v14 fix)
**Bug:** PMO P1NP sensor IC was 0.700 nM (2.0x healthy) but literature says 1.5x = 0.525 nM.
**Fix:** Changed to 0.525 nM in v14. Data regenerated.

### 19.8 Surrogate Loader Version Mismatch
**Bug:** bo_main.py hardcoded surrogate_version="v3" but DEFAULT_VERSION="v4".
**Fix:** Changed to SurrogateLoaderV3.DEFAULT_VERSION.

### 19.9 Old V3 Key Names in bo_pipeline.py
**Bug:** bo_pipeline.py used stale v3 key names (dr_pred, fnr_pred, etc.).
**Fix:** Updated to v6 key names (dr_mean, fnr_mean, etc.) with .get() fallback chains.

### 19.10 TherapeuticObjectiveV6 Constructor Missing Argument
**Bug:** Phase 2 scripts called TherapeuticObjectiveV6(physics_model) with one argument.
**Fix:** Constructor always requires both positional arguments.

### 19.11 Unicode Encoding Crashes on Windows
**Bug:** Multiple scripts used Unicode characters in print() statements.
**Fix:** Replaced with ASCII equivalents: [OK], [FAIL], [!!], ->, rho, 2.

### 19.12 generate_optimized_dataset.py Phase C Overflow
**Bug:** Phase C n_vicinity was pre-computed, overflowed n_total.
**Fix:** Compute n_vicinity = args.n_total - rows_written dynamically.

### 19.13 Objective Landscape Collapse V3-V5 (r=0.994)
**Bug:** V3/V4/V5 objectives all had r=0.994 correlation with each other.
**Effect:** BO found no signal above random (p=0.46). Landscape effectively 1D.
**Fix:** V6 uses relative threshold (sensitivity cancels), overdose penalty.
Result: r(v3,v6)=0.860, BO wins with p=0.0032.

### 19.14 run_random_search.py Comparison Bug
**Bug:** Script hardcoded noise_preset="medium" when re-evaluating BO best config
(optimized at "realistic"). Produced spurious 0.136 gap.
**Fix:** Corrected BO vs random lift = +0.0012.

### 19.15 --use-v6 Flag Silently Breaking Multi-Seed Runs (2026-06-25 fix)
**Bug:** bo_converged.py and bo_multistart.py spawned bo_main.py subprocesses with
obsolete --use-v6 flag (removed when bo_main.py was refactored to --mode dispatch).
**Effect:** All multi-seed runs silently failed; convergence_report.json had n_successful=1.
**Fix:** Changed to --mode standard in all subprocess cmd lists.

### 19.16 Landscape BO Comparison KeyError (2026-06-25 fix)
**Bug:** best_config.json stores "type": "array" under biosensor_design, but V6 expects
"biosensor_type". Made BO score appear as 0.0 -> wrong "stuck in local optimum" verdict.
**Fix:** Added dict flattening and key rename logic before scoring in bo_main.py landscape mode.

### 19.17 robustness_analyzer.py Dead Import Crash (2026-06-25 fix)
**Bug:** robustness_analyzer.py imported from evaluation.objective_function_v3 which
no longer exists. Crashed all robustness analyses, returning all-zero metrics.
**Fix:** Removed dead import; added inline Langmuir threshold for single-channel case.

### 19.18 Dosing Unit Bug in Safety Constraints (2026-06-18 fix)
**Bug:** Safety constraint checker compared dose in mg to fractional units -> all
PMO/CKD configs falsely flagged as overdose.
**Fix:** Unified units to fractional dose throughout safety_constraints.py.

### 19.19 BMD Rate Limit Too Tight (2026-06-18 fix)
**Bug:** MAX_BMD_GAIN_RATE_6MO was set to 3% -- well below romosozumab's actual
+6.7%/6mo in ARCH trial. Most therapeutic configs falsely flagged as infeasible.
**Fix:** Raised to 15% (2x ARCH trial safety ceiling). 4.3% infeasibility rate (hypercalcemia only).

---

## 20. FROZEN DECISIONS -- DO NOT CHANGE THESE

1. **Biology is frozen**: Do NOT change biomarker concentrations, sigma values, ODE
   rate constants, or scenario definitions. The v5.x calibration (PMO sigma=0.35,
   CTX sigma=0.45, CKD SOST=3.0x) is based on literature (Mirza 2010, Garnero 1994,
   Cejka 2011).

2. **Array architecture is frozen**: The 3-biomarker array (SOST + CTX + P1NP) is
   the final architecture. Do NOT add biomarkers or switch to single-channel.

3. **No new detector studies**: The detector is locked at simple persistence with
   margin=1.25, p=10. The hybrid detector (EWMA+persistence) was explicitly REJECTED
   because it amplified drift artifacts.

4. **No MLP for design model**: MLP has R^2=-0.0006 due to the one-to-many problem.
   K-NN retrieval is the replacement.

5. **V6 is the active objective**: Do NOT use V3, V4, or V5 for new BO runs.
   V7 exists (true PK/PD dynamics) but is NOT the active objective.

6. **--mode standard, not --use-v6**: The --use-v6 flag no longer exists. All
   bo_main.py invocations use --mode.

---

## 21. CURRENT STATUS (2026-06-25)

### What's Complete and Verified:
- ODE model: correct, all bugs fixed
- Biosensor array model (3-channel): working, sensitivity bug fixed
- Surrogate v18 (10-feature, no SNR): working, rank-rho=0.8312 (vs v4=0.517)
- V6 objective: working, sensitivity-independent, landscape r=0.86 vs V3
- data_v18: 1,500 configs with correlated sampling, patient subtypes, drug interactions
- Closed-loop BO: COMPLETED (v4 era), best real DR=0.960 at n=50, FP~5%
- kd_ctx corrected: 0.130->0.316 nM (Langmuir-validated, +2pp DR)
- MOBO code: COMPLETE but not yet run (no mobo_results.json)
- GNN surrogate code: COMPLETE but not yet trained (no gnn_best.npz weights)
- EnsembleSurrogate: COMPLETE, falls back to GBM-only until GNN trained
- PK/PD closed-loop sim: COMPLETE, calibrated to Cosman 2016 ARCH trial
- Safety constraints: COMPLETE, 95.7% of configs feasible
- Patient subtypes: COMPLETE, prior results from 2026-06-24
- Clinical validation: COMPLETE, 0 FAILs vs romosozumab trial benchmarks
- Topology search: COMPLETE, 2ch vs 3ch not significantly different (p=0.19)
- Baseline AUC (Priority 1): DONE, SOST-only=0.960 (trivially easy binary)
- rank-rho v18 (Priority 4): DONE, overall=0.8312 (GOOD)
- Landscape audit (Priority 3): DONE, BO in top ~0.1% of 10k LHS
- Multi-optimizer benchmark: DONE (n=5, underpowered); prior n=20 gave BO p=0.0032
- Bug fixes (6 bugs): ALL FIXED as of 2026-06-25

### What's Pending:
1. **Train GNN surrogate** (requires PyTorch):
   ```powershell
   pip install torch; python BO/surrogates/gnn/train_gnn.py --data-csv data_v18/master_index.csv
   ```

2. **Run MOBO** (first ever run):
   ```powershell
   python BO/bo_main.py --mode mobo --n-init 32 --n-iter 100
   ```

3. **Run publication-grade multi-optimizer benchmark** (n=20):
   ```powershell
   python BO/bo_main.py --mode benchmark --n-runs 20 --n-init 50 --n-iter 150
   ```

4. **Run convergence test** (was broken until 2026-06-25, now fixed):
   ```powershell
   python BO/bo_main.py --mode converged --n-runs 5 --n-init 50 --n-iter 150
   ```

5. **Fix D_SAFE (overdose calibration):** D_SAFE=0.50 is below avg dose/cycle for
   PMO (0.57) and CKD (0.81). Needs raising to ~0.80 but touches therapeutic parameters.

6. **Expand sensitivity upper bound** [0.5->10.0] and w_p1np upper bound [0.49->0.70]:
   BO consistently hits these ceilings. Requires dataset regeneration.

7. **Update Methods Paper** (`docs/METHODS_PAPER_DRAFT.md`): Revise scientific framing --
   BO optimizes therapeutic calibration/safety, NOT detection capability.

---

## 22. HOW WE USE CLAUDE CODE

Claude Code (Sonnet 4.6) is used via VS Code extension (Windows). Key constraints:

- **Bash tool fails on Windows paths**: Use PowerShell tool for running Python scripts.
  Never use Bash for `python ...` commands -- it can't navigate Windows paths properly.

- **Always use PowerShell for Python execution:**
  ```powershell
  cd C:\Users\eruku\Akshith\GENEVO2
  python BO/bo_main.py --mode standard
  ```

- **ASCII-only console output**: All print() statements in scripts must use ASCII only.
  Windows cp1252 encoding crashes on Unicode symbols. Use [OK], [FAIL], [!!], ->, rho, 2.

- **No autonomous changes to frozen parameters**: Any suggestion to change biomarker
  sigmas, concentrations, detector settings, or the objective function should be flagged
  as contradicting the freeze decision.

- **Memory system**: Claude maintains persistent memory at
  `C:\Users\eruku\.claude\projects\c--Users-eruku-Akshith-GENEVO2\memory\`
  Always check MEMORY.md index before diving into code.

- **--mode dispatch is the only valid interface**: bo_main.py always uses --mode.
  Never generate commands with --use-v6, --use-v4, or any other obsolete flag.

---

## 23. REFERENCE COMMANDS

All commands from project root (`C:\Users\eruku\Akshith\GENEVO2`). Full reference: `docs/COMMANDS.md`.

```powershell
# Standard BO (v18 surrogate, v6 objective, ~5 min)
python BO/bo_main.py

# Full standard BO (publication-grade, ~20 min)
python BO/bo_main.py --mode standard --n-init 50 --n-iter 150

# Multi-seed convergence test (5 seeds, now FIXED)
python BO/bo_main.py --mode converged --n-runs 5 --n-init 50 --n-iter 150

# Benchmark (publication-grade, ~90 min)
python BO/bo_main.py --mode benchmark --n-runs 20 --n-init 50 --n-iter 150

# Patient subtypes (4 subtypes, ~20 min)
python BO/bo_main.py --mode subtypes --n-init 30 --n-iter 100

# Multi-objective BO (first run)
python BO/bo_main.py --mode mobo --n-init 32 --n-iter 100

# 2ch vs 3ch topology
python BO/bo_main.py --mode topology

# Baseline AUC diagnostic (Priority 1, ~1 min)
python BO/bo_main.py --mode baseline

# Surrogate rank-rho validation (Priority 4, ~2 min)
python BO/bo_main.py --mode validate

# Global landscape audit (Priority 3, ~40 min)
python BO/bo_main.py --mode landscape --n-samples 10000

# Closed-loop BO (real simulator, ~2-4 hours)
python BO/bo_main.py --mode closed-loop --n-rounds 5 --n-inner 50 --top-k 10

# Retrain surrogates from data_v18
python BO/core/build_surrogates_v3.py --data-dir data_v18 --out-dir BO/bo_results --version v18

# Train GNN (requires torch install first)
pip install torch
python BO/surrogates/gnn/train_gnn.py --data-csv data_v18/master_index.csv

# Validate surrogate vs real simulator (30-60 min)
python BO/validation/validate_top_designs.py --n-lhs 200 --top-k 50 --n-trials 5

# Clinical validation vs romosozumab trials
python BO/validation/therapeutic_clinical_validation.py

# Full safety audit
python BO/evaluation/safety_validation_final.py

# K-NN design retrieval
python BO/design_model/design_retrieval.py --retrieve --sost 0.875 --ctx 0.500 --p1np 0.525 --disease pmo
```

---

## 24. KEY NUMBERS TO KNOW

| Metric | Value | Context |
|--------|-------|---------|
| SOST-only binary AUC | 0.960 | Baseline LR on data_v18 -- detection is trivially easy |
| 4-class subtype accuracy | 67.1% | The hard problem -- BO adds value here |
| v18 surrogate rank-rho | 0.8312 | GOOD (threshold: >0.70) -- vs v4=0.517 |
| v18 surrogate bias | -0.016 | Well-calibrated (slightly pessimistic) |
| Best BO score (quick run) | 0.7205 | v18 surrogate, n_init=20, n_iter=50 |
| LHS max score (10k) | 0.7280 | BO is in top ~0.1% of landscape |
| BO Wilcoxon p (n=20) | 0.0032 | Publication-grade significance vs Random |
| BO vs CMA-ES difference | 0.006 | < 1 std dev -- essentially tied |
| Best real DR (pmo_mild) | 0.900 [0.786, 0.957] | n=50, kd_ctx=0.316 |
| Best real DR (pmo) | 1.000 | n=50 |
| Best real DR (ckd) | 0.980 [0.895, 0.996] | n=50 |
| FP rate (healthy) | ~5% [2%, 11%] | n=100 combined; NOT 0% -- report honestly |
| Surrogate score ceiling | ~0.82-0.85 | Mathematical max given weights |
| kd_nm optimal | ~1.0-1.1 nM | Langmuir sweet spot for SOST |
| kd_ctx optimal | 0.316 nM | Langmuir geometric mean; BO's 0.130 was surrogate error |
| w_ctx optimal | ~0.40-0.45 | CTX channel carries most discrimination signal |
| w_p1np optimal | ~0.01 (or 0.084?) | BO ridge: 0.01; LHS global max: 0.084 |
| sensitivity bound | 5.0 (always hit) | Recommend expanding to 10.0 + regen data |
| Safety feasibility | 95.7% | 4.3% fail hypercalcemia (CKD + thiazide) |
| 2ch vs 3ch difference | p=0.19 | Not significant -- 2ch defensible simplification |
| Topology search | p=0.19 | 2-channel ≈ 3-channel at V6 objective level |
| GNN status | Untrained | Code complete, needs PyTorch + training run |
| MOBO status | Not run | Code complete, no results yet |

---

## 25. SCIENTIFIC INTERPRETATION OF RESULTS

### Why sensitivity hits the upper bound (5.0):
Sensitivity amplifies the composite signal linearly. The surrogate correctly learned
that higher sensitivity -> stronger SNR -> better DR. The objective's therapeutic term
is sensitivity-independent (by design in V6). So BO freely maximizes sensitivity to
maximize DR, and runs into the search space ceiling of 5.0. **BO wants more sensitivity;
the ceiling is artificial.** Expand to [0.5, 10.0] when regenerating data.

### Why P1NP is (mostly) disabled:
P1NP discrimination ratios (disease/healthy) are inherently lower than SOST or CTX.
At any kd_p1np, P1NP adds less signal separation. BO correctly sets w_p1np to minimum.
HOWEVER: the landscape audit found a global max at w_p1np=0.084 (not minimum 0.01).
The true optimal may be a small non-zero P1NP weight for disease subtype discrimination.

### Why kd_ctx was wrong at 0.13 (now corrected to 0.316):
Langmuir theory predicts optimal kd for discrimination = sqrt(C_healthy x C_disease).
For CTX: sqrt(0.200 x 0.500) = 0.316 nM. BO found 0.13 nM (below theoretical optimum).
A 9-point 1D scan with n=20 real simulator trials confirmed kd_ctx=0.316 gives
DR_mean=1.000 vs kd_ctx=0.130 gives DR_mean=0.967. Fix: surrogate interpolation error
in the 0.10-0.20 nM range due to sparse training data in that region.

### Why 0.967 DR is not impressive (but still has value):
SOST-alone logistic regression achieves AUC=0.960. Binary detection (disease vs healthy)
is trivially easy -- SOST concentrations differ by 2.3x-3.0x between groups, far
exceeding noise levels. BO's REAL value is:
1. Therapeutic calibration: correct drug dose per patient profile
2. Safety: minimizing overdose and false-positive dosing burden
3. Subtype discrimination: 4-class accuracy 67% is the hard problem

---

*End of AGENT_CONTEXT.md -- Last updated 2026-06-25*
