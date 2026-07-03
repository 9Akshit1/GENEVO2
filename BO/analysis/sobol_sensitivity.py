#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sobol variance-based global sensitivity analysis of the V6 objective.

Uses the Saltelli sampling scheme (Saltelli 2002; Saltelli et al. 2010)
to compute first-order (S1) and total-order (ST) Sobol indices for all
6 continuous biosensor design parameters.

Reference:
  Saltelli A. (2002). Making best use of model evaluations to compute
  sensitivity indices. CMAME 192: 280-297.
  Saltelli A. et al. (2010). Variance based sensitivity analysis of model
  output. Design and estimator for the total sensitivity index. CMAME 200:
  845-877.

Why this matters:
  Sobol indices decompose total objective variance into additive contributions
  per parameter, revealing which design knobs matter most and whether parameter
  interactions are important.  This is methodologically stronger than a 1D
  kd_ctx scan because it operates globally over the full parameter space.

Runtime: ~2 min (N=512 → 512×(6+2)=4096 surrogate evaluations, ~1ms each)

Usage:
    python BO/analysis/sobol_sensitivity.py
    python BO/analysis/sobol_sensitivity.py --N 1024 --out BO/bo_results/diagnostics/sobol_results.json

Output:
    BO/bo_results/diagnostics/sobol_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

# Parameter definitions: (name, lower, upper, scale) — all SOST-optimal range
PARAMS: List[Tuple[str, float, float, str]] = [
    ("kd_nm",       0.1,  10.0, "log"),   # SOST affinity [nM]
    ("kd_ctx_nm",   0.1,  10.0, "log"),   # CTX affinity [nM]
    ("kd_p1np_nm",  0.1,  10.0, "log"),   # P1NP affinity [nM]
    ("sensitivity", 0.5,   5.0, "log"),   # Signal amplification
    ("w_ctx",       0.01,  0.49, "linear"), # CTX weight
    ("w_p1np",      0.01,  0.49, "linear"), # P1NP weight
]
N_PARAMS = len(PARAMS)

FIXED = {
    "biosensor_type":   "array",
    "noise_preset":     "realistic",
    "target_scenario":  "pmo",
    "response_time_s":  600.0,
}


def _unit_to_param(x_unit: np.ndarray) -> dict:
    """Map a [0,1]^D unit vector to a biosensor config dict."""
    cfg = dict(FIXED)
    for j, (name, lo, hi, scale) in enumerate(PARAMS):
        u = float(x_unit[j])
        if scale == "log":
            cfg[name] = float(10.0 ** (np.log10(lo) + u * (np.log10(hi) - np.log10(lo))))
        else:
            cfg[name] = float(lo + u * (hi - lo))

    # Enforce weight sum constraint: w_scl = 1 - w_ctx - w_p1np >= 0.01
    w_ctx  = cfg["w_ctx"]
    w_p1np = cfg["w_p1np"]
    if w_ctx + w_p1np > 0.98:
        total = w_ctx + w_p1np
        cfg["w_ctx"]  = w_ctx  / total * 0.98
        cfg["w_p1np"] = w_p1np / total * 0.98
    return cfg


def _saltelli_sample(N: int, D: int, rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate two independent Saltelli base matrices A and B, each (N, D).
    """
    A = rng.random((N, D))
    B = rng.random((N, D))
    return A, B


def _evaluate_batch(
    X: np.ndarray,
    objective_fn,
) -> np.ndarray:
    """Evaluate objective_fn for each row of X (unit-space matrix). Returns (N,) array."""
    scores = np.zeros(X.shape[0])
    for i, row in enumerate(X):
        cfg = _unit_to_param(row)
        try:
            scores[i] = float(objective_fn(cfg))
        except Exception:
            scores[i] = 0.0
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{X.shape[0]} evaluated ...", flush=True)
    return scores


def compute_sobol_indices(
    f_A:     np.ndarray,
    f_B:     np.ndarray,
    f_AB_list: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Saltelli (2002) estimators for S1 and ST.

    S1_i  = (1/N) * sum(f_B * (f_AB_i - f_A)) / Var[Y]
    ST_i  = (1/(2N)) * sum((f_A - f_AB_i)^2)  / Var[Y]

    All arrays are (N,). f_AB_list[i] is f evaluated with column i from B,
    all other columns from A.
    """
    N = len(f_A)
    # Total variance estimated from pooled A and B samples
    f_all = np.concatenate([f_A, f_B])
    var_Y = float(np.var(f_all, ddof=1))
    if var_Y < 1e-10:
        return np.zeros(len(f_AB_list)), np.zeros(len(f_AB_list))

    S1  = np.zeros(len(f_AB_list))
    ST  = np.zeros(len(f_AB_list))

    for i, f_AB_i in enumerate(f_AB_list):
        # First-order: Jansen/Saltelli 2002 estimator
        S1[i]  = float(np.mean(f_B * (f_AB_i - f_A))) / var_Y
        # Total-order: Homma & Saltelli 1996
        ST[i]  = float(np.mean((f_A - f_AB_i) ** 2) / 2.0) / var_Y

    return np.clip(S1, -0.5, 1.5), np.clip(ST, 0.0, 1.5)


def main():
    parser = argparse.ArgumentParser(description="Sobol sensitivity analysis (surrogate-based)")
    parser.add_argument("--N",             type=int, default=512,
                        help="Base sample size (total evaluations = N*(D+2), default: 512)")
    parser.add_argument("--surrogate-dir", type=Path, default=Path("BO/bo_results"),
                        help="Surrogate models directory")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--out",           type=Path,
                        default=Path("BO/bo_results/diagnostics/sobol_results.json"))
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    print("=" * 70)
    print("GENEVO2 — Sobol Variance-Based Sensitivity Analysis")
    print("=" * 70)
    print(f"  N (base)    : {args.N}")
    print(f"  D (params)  : {N_PARAMS}")
    print(f"  Total evals : {args.N * (N_PARAMS + 2)}")
    print(f"  Parameters  : {[p[0] for p in PARAMS]}")
    print()

    from BO.core.surrogate_loader import SurrogateLoaderV3
    from BO.evaluation.physics_forward_model import PhysicsForwardModel
    from BO.evaluation.therapeutic_objective_v6 import TherapeuticObjectiveV6

    loader  = SurrogateLoaderV3(args.surrogate_dir)
    physics = PhysicsForwardModel()
    v6      = TherapeuticObjectiveV6(physics, loader)

    rng = np.random.RandomState(args.seed)

    # Step 1: Generate Saltelli matrices
    print("  Generating Saltelli base matrices A and B ...")
    A, B = _saltelli_sample(args.N, N_PARAMS, rng)

    # Step 2: Evaluate f(A), f(B), and f(A_B^i) for each parameter i
    print(f"  Evaluating f(A) [{args.N} evals] ...")
    f_A = _evaluate_batch(A, v6)

    print(f"  Evaluating f(B) [{args.N} evals] ...")
    f_B = _evaluate_batch(B, v6)

    f_AB_list = []
    for i, (param_name, *_) in enumerate(PARAMS):
        # A_B^i: all columns from A except column i, which comes from B
        AB_i = A.copy()
        AB_i[:, i] = B[:, i]
        print(f"  Evaluating f(A_B^{i}) [{args.N} evals] — param: {param_name} ...")
        f_AB_list.append(_evaluate_batch(AB_i, v6))

    # Step 3: Compute Sobol indices
    S1, ST = compute_sobol_indices(f_A, f_B, f_AB_list)

    # Descriptive statistics
    f_all = np.concatenate([f_A, f_B])
    var_Y  = float(np.var(f_all, ddof=1))
    mean_Y = float(np.mean(f_all))

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Objective statistics across full parameter space:")
    print(f"    mean = {mean_Y:.4f}  std = {float(np.std(f_all)):.4f}"
          f"  min = {float(f_all.min()):.4f}  max = {float(f_all.max()):.4f}")
    print()
    print(f"  {'Parameter':<18}  {'S1 (first-order)':>17}  {'ST (total-order)':>17}  {'Interaction':>12}")
    print(f"  {'-'*18}  {'-'*17}  {'-'*17}  {'-'*12}")
    for i, (name, *_) in enumerate(PARAMS):
        interaction = float(ST[i] - S1[i])
        print(f"  {name:<18}  {S1[i]:>17.4f}  {ST[i]:>17.4f}  {interaction:>+12.4f}")

    print()
    print(f"  Sum of S1 : {S1.sum():.4f}  (expected <= 1.0; >1 indicates negative first-order estimates)")
    print(f"  Sum of ST : {ST.sum():.4f}  (expected >= 1.0; measures total interactions)")

    # Interpretation
    print()
    print("  --- Interpretation ---")
    ranked = sorted(range(N_PARAMS), key=lambda i: ST[i], reverse=True)
    print(f"  Most influential parameter (ST): {PARAMS[ranked[0]][0]} (ST={ST[ranked[0]]:.4f})")
    print(f"  Least influential             : {PARAMS[ranked[-1]][0]} (ST={ST[ranked[-1]]:.4f})")
    top3 = [PARAMS[ranked[j]][0] for j in range(min(3, N_PARAMS))]
    print(f"  Top 3 by total-order          : {top3}")
    interaction_sum = float(np.sum(ST) - np.sum(np.clip(S1, 0, None)))
    if interaction_sum > 0.05:
        print(f"  Parameter interactions present: sum(ST-S1)={interaction_sum:.4f} > 0.05")
    else:
        print(f"  Parameter interactions negligible: sum(ST-S1)={interaction_sum:.4f} <= 0.05")

    # Save
    results = {
        "N_base":       args.N,
        "N_total_evals": args.N * (N_PARAMS + 2),
        "D":            N_PARAMS,
        "seed":         args.seed,
        "objective_stats": {
            "mean":  mean_Y,
            "std":   float(np.std(f_all)),
            "var":   var_Y,
            "min":   float(f_all.min()),
            "max":   float(f_all.max()),
        },
        "parameters": [p[0] for p in PARAMS],
        "S1":  {PARAMS[i][0]: float(S1[i]) for i in range(N_PARAMS)},
        "ST":  {PARAMS[i][0]: float(ST[i]) for i in range(N_PARAMS)},
        "interaction": {PARAMS[i][0]: float(ST[i] - S1[i]) for i in range(N_PARAMS)},
        "ranking_by_ST": [PARAMS[r][0] for r in ranked],
        "interpretation": {
            "most_influential": PARAMS[ranked[0]][0],
            "top3": top3,
            "interaction_present": interaction_sum > 0.05,
            "interaction_sum": float(interaction_sum),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Sobol results saved: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
