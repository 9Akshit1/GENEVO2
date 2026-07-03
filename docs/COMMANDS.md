# GENEVO2 — Pipeline Command Reference

All commands run from the **project root** (`c:\Users\eruku\Akshith\GENEVO2`).

**Active pipeline (v19):** Corrected dataset with `response_time=600s` (was NaN in v18).
Analytical FP rate now used in objective (replaces surrogate-predicted dr_healthy).
Previous: `data_v18/` (1,500 configs, rank-rho=0.8326). Target: `data_v19/`.

---

## STEP 0: Environment Setup

```powershell
# Activate virtual environment (Windows)
.\venv\Scripts\Activate.ps1

# Verify Python path
python -c "import sklearn, scipy, numpy; print('deps OK')"
```

---

## STEP 1: Generate data_v19 (REQUIRED — replaces data_v18)

Regenerates dataset with `response_time=600s` recorded (not NaN).
This fixes the surrogate PI=0 artefact for response_time.

```powershell
# Larger dataset for better surrogate coverage
python simulation/dataset/run_generation.py --n-configs 2000 --out data_v19
```

**Verify output:**
```powershell
python -c "
import pandas as pd
df = pd.read_csv('data_v19/master_index.csv')
print(f'Rows: {len(df)}  Cols: {len(df.columns)}')
print(df['scenario'].value_counts().to_string())
print('DR mean (pmo):', df[df['scenario']=='pmo']['detection_rate'].mean().round(3))
print('response_time NaN:', df['response_time'].isna().sum(), '(should be 0)')
"
```

---

## STEP 2: Train Surrogates on data_v19

```powershell
# Train v19 surrogates (saves to BO/bo_results/saved_ml/)
python BO/core/build_surrogates.py --data-dir data_v19 --out-dir BO/bo_results

# Expected output: DR AUC ~0.87, rank-rho ~0.83
```

---

## STEP 3: Validate Surrogate Quality

```powershell
# Saves JSON automatically to BO/bo_results/diagnostics/rank_rho_results.json
python BO/analysis/rank_rho.py --data-dir data_v19
```

**Current result (data_v19, n=1615):**
- Overall rank-rho = **0.8351** (GOOD — exceeds 0.70 threshold)
- PMO-mild: rho=0.7352  |  PMO: rho=0.7812  |  CKD-MBD: rho=0.7455

---

## STEP 4: Baseline Task Difficulty Check

**Run before claiming any DR result is meaningful.**

```powershell
python BO/bo_main.py --mode baseline --data-dir data_v19
# Results: BO/bo_results/diagnostics/baseline_auc_results.json
```

**Current result (data_v18):**
- SOST alone binary AUC = **0.9666** (trivially easy)
- Interpretation: BO optimizes safety and dosing precision, NOT detection per se

---

## STEP 5: Standard BO Run (Core Optimization)

```powershell
# Full run — ~10-20 min on CPU
python BO/bo_main.py --mode standard --n-init 50 --n-iter 150

# Results: BO/bo_results/results/best_config.json
#          BO/bo_results/results/optimization_results.json
```

**Note:** V6 objective now uses analytical FP rate (not surrogate).
Expect sensitivity optimum to shift lower (~3.0 instead of ~4.4) due to FP penalty.

---

## STEP 6: Multi-Start Convergence Test

```powershell
# 10 seeds, ~40 min
python BO/bo_main.py --mode converged --n-runs 10 --n-init 50 --n-iter 150
# Results: BO/bo_results/convergence/
```

---

## STEP 7: Validate Top-5 Configs with Real Simulator

After BO, run the top 5 configs through the real simulator (not surrogates) to confirm results.

```powershell
python BO/analysis/validate_best_config.py --n-trials 20
# Results: BO/bo_results/diagnostics/top5_validation.json
```

**Acceptance criteria:** Real-sim DR(pmo) >= 0.85 and FP(healthy) <= 0.25

---

## STEP 8: Topology Comparison (2ch vs 3ch) — ≥5 Seeds Required

```powershell
# More seeds for publication-grade stats (n=10, ~2 hours)
python BO/bo_main.py --mode topology --data-dir data_v19 --n-seeds 10 --n-init 30 --n-iter 100

# Results: BO/bo_results_topology/topology_comparison.json
# Reports: mean ± std per topology, Mann-Whitney p-value
```

**Previous result (data_v18, n=1 seed — insufficient):**
- 2ch: 0.737, 3ch: 0.745, p=0.19 (inconclusive — only 1 seed)
- v19 target: ≥5 seeds with 95% CI

---

## STEP 8.5: Landscape Audit
```powershell
python BO/bo_main.py --mode landscape --n-lhs 10000
# Results: BO/bo_results/diagnostics/landscape_audit.json
```

---

## STEP 9: Sobol Sensitivity Analysis

```powershell
python BO/analysis/sobol_sensitivity.py --N 512
# Results: BO/bo_results/diagnostics/sobol_results.json
```

---

## STEP 9.5: Distribution Shift Robustness

Tests the validated config against 7 perturbation scenarios (concentration shift, variability scaling, sensor drift, channel failure, population shift).

```powershell
python BO/bo_main.py --mode dist-shift
# Results: BO/bo_results/diagnostics/distribution_shift_results.json
```

---

## STEP 9.6: Surrogate Permutation Importance

Computes permutation importance and partial dependence for all 3 GBM surrogates.

```powershell
python BO/bo_main.py --mode shap --data-dir data_v19
# Results: BO/bo_results/diagnostics/surrogate_interpretability.json
```

---

## STEP 9.7: Generate Publication Figures

Creates 10 publication-quality PNG figures from existing JSON result files (fast, no simulations).

```powershell
python BO/analysis/generate_plots.py
# Results: BO/bo_results/diagnostics/plots/
#   kd_ctx_scan_plot.png        — DR per scenario vs K_{d,CTX}
#   benchmark_boxplot.png       — Score distribution per optimizer
#   sobol_barplot.png           — Sobol sensitivity indices
#   realsim_validation.png      — Surrogate vs real-sim DR per scenario
#   convergence_summary.png     — Final score across 10 seeds
#   langmuir_isotherms.png      — Binding isotherms per aptamer channel (analytical)
#   robustness_analysis.png     — Missing-channel + gain drift sensitivity
#   closed_loop_learning.png    — Surrogate score and real DR across AL rounds
#   scenario_rank_rho.png       — Per-scenario surrogate rank correlation
#   # patient_subtypes.png — REMOVED (patient subtype analysis retired)
```

**Expected change from v18→v19:**
- v18: sensitivity ST=0.856 (artificially high — FP penalty was near-constant at 3.7%)
- v19: sensitivity ST should decrease as analytical FP creates real tradeoff

---

## STEP 10: Multi-Objective BO (Pareto Front)

```powershell
python BO/bo_main.py --mode mobo --n-init 30 --n-iter 100
# Results: BO/bo_results/mobo/mobo_results.json
```

**Root cause fix (2026-06-28):** MOBO was finding 0/130 feasible because the surrogate predicts
FP~18% for the warm-start config (vs real 5%), failing the `MAX_FP_RATE=0.10` hard constraint.
FP is NOT constrained in `is_feasible()` anymore — it is already optimised as the `specificity`
objective (f3=1-FP). The DR constraints remain (PMO-mild>=0.55, PMO>=0.80, CKD>=0.80).
With this fix, the warm start (DR_mean=0.858 predicted) should be feasible from iteration 0.

**Quick test before full run (30 seconds):**
```powershell
python BO/bo_main.py --mode mobo --n-init 5 --n-iter 3
# Should show: "Warm start: y=[0.858 0.903 0.814] feasible=True"
```

---

## STEP 11: kd_ctx Scan (Priority 2 from CLAUDE.md)

```powershell
python BO/bo_main.py --mode kd-scan
# WARNING: Uses real simulator. ~2-4 hours.
# Results: BO/bo_results/diagnostics/kd_ctx_scan_results.json
```

---

## STEP 12: Closed-Loop BO (if rank-rho < 0.70 on v19)

```powershell
python BO/bo_main.py --mode closed-loop --n-init 30 --n-iter 100
# WARNING: Very slow (~2-4 hours). Uses real simulator.
```

---

## STEP 13: Benchmarking

```powershell
python BO/bo_main.py --mode benchmark --n-runs 20 --n-init 50 --n-iter 150
# Results: BO/bo_results/benchmark/benchmark_results.json

# ⚠️ DO NOT USE: subtypes mode has been retired.
# The per-subtype BO used hand-specified objective weights with no empirical grounding.
# Use MOBO Pareto front (Step 10) instead — it provides the full trade-off space
# without unjustifiable per-patient weight assumptions.
# python BO/bo_main.py --mode subtypes   <-- DEPRECATED, do not run
```

---

## Key Results Files

| File | Contents |
|------|----------|
| `data_v19/master_index.csv` | Ground-truth simulation data (response_time=600s) |
| `BO/bo_results/results/best_config.json` | Best biosensor design from last BO run |
| `BO/bo_results/results/optimization_results.json` | Full BO trajectory |
| `BO/bo_results/diagnostics/baseline_auc_results.json` | LR baseline AUC (Priority 1) |
| `BO/bo_results/diagnostics/rank_rho_results.json` | Surrogate rank-rho (Priority 4) |
| `BO/bo_results/diagnostics/landscape_audit.json` | Global landscape audit (Priority 3) |
| `BO/bo_results/diagnostics/kd_ctx_scan_results.json` | kd_ctx scan (Priority 2) |
| `BO/bo_results/diagnostics/sobol_results.json` | Sobol sensitivity indices |
| `BO/bo_results/benchmark/benchmark_results.json` | BO vs DE vs CMA-ES vs Random |
| `BO/bo_results/targeted/` | Best biosensor design from targeted BO run |
| `BO/bo_results/convergence/` | Multi-start convergence stats |
| `BO/bo_results_topology/topology_comparison.json` | 2ch vs 3ch with CIs |
| `data/patient_cohort_n1000.csv` | Synthetic patient cohort (n=1,000) |
| `docs/METHODS_PAPER_DRAFT.md` | Methods paper draft |
| `docs/RESULTS_AUDIT.md` | Audit trail and known issues |

---

## Interpreting Results

### Biosensor is good IF:
- DR_pmo >= 0.85 (detects PMO reliably)
- DR_mild >= 0.70 (catches mild cases)
- DR_healthy <= 0.25 (analytical FP target — previous 0.05 was unrealistic for margin=1.25)
- FNR_mean <= 0.20 (low miss rate)
- surrogate rank-rho >= 0.70 (trustworthy optimization guidance)

### FP rate expectations (REVISED):
- margin=1.25 with SOST σ=0.30, CTX σ=0.45 → composite CV ≈ 21%
- Analytical FP (seed-888 config): ~21.7% (vs real-sim 35%, vs surrogate 3.7%)
- Target: reduce margin or reduce sensitivity to bring FP below 20%

### BO is effective IF:
- Wilcoxon p < 0.05 vs random search
- BO score > random by >= 0.01 composite score units
- BO score ranks in top 5% of LHS landscape

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `FileNotFoundError: scaler_v1.pkl` | Surrogates not trained | Run `build_surrogates_v3.py --data-dir data_v19` |
| `FileNotFoundError: master_index.csv` | Wrong `--data-dir` | Check path points to `data_v19/` |
| `ModuleNotFoundError` | Wrong working directory | Always run from project root |
| `AttributeError: _analytical_healthy_fp_rate` | Old `therapeutic_objective_v6.py` | Pull latest from `BO/evaluation/` |
| `response_time NaN in data` | Using old data_v18 | Regenerate with `run_generation_v19.py` |
