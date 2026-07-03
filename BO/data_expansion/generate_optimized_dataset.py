#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate 50k–100k optimized biosensor designs for design-model training.

Strategy
--------
Three complementary sources of design diversity:

  A. RANDOM SEARCH (40% of budget)
     Uniform Latin Hypercube sampling across the full 9D parameter space.
     Captures the marginal distribution of configs and their v6 scores —
     essential for the design model to learn that most of the space is poor.

  B. MULTI-POINT BO EXPLOITS (30% of budget)
     Run N_BO_RUNS independent BO campaigns (20+80 budget each) and collect
     ALL evaluated points (not just the best).  Each BO run generates points
     from a different GP posterior path — together they densely sample the
     top-scoring regions.

  C. VICINITY SAMPLING (30% of budget)
     For each of the K_VICINITY best configs found by BO, generate
     VICINITY_PER_BEST new configs by perturbing within ±SIGMA_PERTURB
     of the best parameter values (log-space for kd parameters).
     Ensures the training dataset has high density near the optimum.

Output
------
CSV at --out (default: BO/data_expansion/optimized_designs.csv).

Columns:
  kd_nm, kd_ctx_nm, kd_p1np_nm, sensitivity, response_time_s,
  w_ctx, w_p1np, biosensor_type, noise_preset, target_scenario,
  score_v6, score_v3, dr_pmo, dr_mild, dr_ckd, fnr_mean, ttd_mean,
  bmd_net_mild, bmd_net_pmo, bmd_net_ckd, dr_healthy, source

The 'source' column is one of: random, bo_exploit, vicinity.

Usage:
    python BO/data_expansion/generate_optimized_dataset.py
    python BO/data_expansion/generate_optimized_dataset.py --n-total 50000 --n-bo-runs 25
    python BO/data_expansion/generate_optimized_dataset.py --n-total 100000 --n-bo-runs 50
"""

import argparse
import csv
import sys
import logging
from pathlib import Path
from time import time

import numpy as np
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from BO.core.surrogate_loader import SurrogateLoaderV3
from evaluation.physics_forward_model import PhysicsForwardModel
from evaluation.objective_function_v3 import ObjectiveFunctionV3
from evaluation.therapeutic_objective_v6 import TherapeuticObjectiveV6

CONSOLE = logging.getLogger("dataset_gen")
CONSOLE.setLevel(logging.INFO)
CONSOLE.propagate = False
if not CONSOLE.handlers:
    _ch = logging.StreamHandler(sys.stdout)
    _ch.setFormatter(logging.Formatter("%(message)s"))
    CONSOLE.addHandler(_ch)

# Parameter space bounds (log-space for kd values).
# response_time_s is removed — fixed at 600s for all array biosensors
# (surrogate PI=0 because training data had constant/NaN response_time).
_BOUNDS = {
    "log_kd_nm":     (np.log10(0.1),   np.log10(10.0)),
    "log_kd_ctx_nm": (np.log10(0.05),  np.log10(10.0)),
    "log_kd_p1np_nm":(np.log10(0.05),  np.log10(10.0)),
    "log_sensitivity":(np.log10(0.5),  np.log10(5.5)),
    "w_ctx":         (0.0,   0.60),
    "w_p1np":        (0.0,   0.60),
}
_PARAM_KEYS = list(_BOUNDS.keys())
_NDIM = len(_PARAM_KEYS)
_RESPONSE_TIME_S = 600.0  # fixed, not optimized

_BIOSENSOR_TYPE = "array"
_NOISE_PRESET   = "realistic"
_TARGET_SCENARIO = "pmo"


def _unit_to_config(x: np.ndarray) -> dict:
    """Map [0,1]^6 unit hypercube sample to a config dict."""
    lo = np.array([_BOUNDS[k][0] for k in _PARAM_KEYS])
    hi = np.array([_BOUNDS[k][1] for k in _PARAM_KEYS])
    scaled = lo + x * (hi - lo)
    # Ensure w_ctx + w_p1np <= 0.80 (sensible weight sum)
    w_ctx  = float(scaled[4])
    w_p1np = float(scaled[5])
    if w_ctx + w_p1np > 0.80:
        scale = 0.80 / (w_ctx + w_p1np)
        w_ctx  *= scale
        w_p1np *= scale
    return {
        "kd_nm":           float(10 ** scaled[0]),
        "kd_ctx_nm":       float(10 ** scaled[1]),
        "kd_p1np_nm":      float(10 ** scaled[2]),
        "sensitivity":     float(10 ** scaled[3]),
        "response_time_s": _RESPONSE_TIME_S,
        "w_ctx":           w_ctx,
        "w_p1np":          w_p1np,
        "biosensor_type":  _BIOSENSOR_TYPE,
        "noise_preset":    _NOISE_PRESET,
        "target_scenario": _TARGET_SCENARIO,
    }


def _config_to_unit(cfg: dict) -> np.ndarray:
    """Map a config dict back to [0,1]^6 for perturbation."""
    lo = np.array([_BOUNDS[k][0] for k in _PARAM_KEYS])
    hi = np.array([_BOUNDS[k][1] for k in _PARAM_KEYS])
    raw = np.array([
        np.log10(max(cfg.get("kd_nm", 1.0), 1e-6)),
        np.log10(max(cfg.get("kd_ctx_nm", 1.0), 1e-6)),
        np.log10(max(cfg.get("kd_p1np_nm", 1.0), 1e-6)),
        np.log10(max(cfg.get("sensitivity", 1.0), 1e-6)),
        cfg.get("w_ctx", 0.1),
        cfg.get("w_p1np", 0.1),
    ])
    return np.clip((raw - lo) / (hi - lo + 1e-12), 0.0, 1.0)


def _evaluate(cfg: dict, obj_v6: TherapeuticObjectiveV6,
              obj_v3: ObjectiveFunctionV3) -> dict:
    """Evaluate a config and return the full metrics dict."""
    score_v6, detail = obj_v6.evaluate_with_details(cfg)
    score_v3 = obj_v3(cfg)
    return {
        "score_v6":    round(float(score_v6), 6),
        "score_v3":    round(float(score_v3), 6),
        "dr_pmo":      round(float(detail.get("dr_pmo", 0)), 5),
        "dr_mild":     round(float(detail.get("dr_mild", 0)), 5),
        "dr_ckd":      round(float(detail.get("dr_ckd", 0)), 5),
        "fnr_mean":    round(float(detail.get("fnr_mean", 0)), 5),
        "ttd_mean":    round(float(detail.get("ttd_mean", 0)), 1),
        "bmd_net_mild":round(float(detail.get("bmd_net_mild", 0)), 6),
        "bmd_net_pmo": round(float(detail.get("bmd_net_pmo", 0)), 6),
        "bmd_net_ckd": round(float(detail.get("bmd_net_ckd", 0)), 6),
        "dr_healthy":  round(float(detail.get("dr_healthy", 0)), 5),
    }


def _cfg_to_row(cfg: dict, metrics: dict, source: str) -> dict:
    row = {
        "kd_nm":           cfg.get("kd_nm"),
        "kd_ctx_nm":       cfg.get("kd_ctx_nm"),
        "kd_p1np_nm":      cfg.get("kd_p1np_nm"),
        "sensitivity":     cfg.get("sensitivity"),
        "response_time_s": cfg.get("response_time_s"),
        "w_ctx":           cfg.get("w_ctx"),
        "w_p1np":          cfg.get("w_p1np"),
        "biosensor_type":  cfg.get("biosensor_type"),
        "noise_preset":    cfg.get("noise_preset"),
        "target_scenario": cfg.get("target_scenario"),
        "source":          source,
    }
    row.update(metrics)
    return row


_FIELDNAMES = [
    "kd_nm", "kd_ctx_nm", "kd_p1np_nm", "sensitivity", "response_time_s",
    "w_ctx", "w_p1np", "biosensor_type", "noise_preset", "target_scenario",
    "score_v6", "score_v3", "dr_pmo", "dr_mild", "dr_ckd", "fnr_mean",
    "ttd_mean", "bmd_net_mild", "bmd_net_pmo", "bmd_net_ckd", "dr_healthy", "source",
]


def run_bo_collect_all(obj_v6, n_init=20, n_iter=80, seed=0) -> list:
    """
    Run one BO campaign and return ALL evaluated configs (not just the best).

    Reuses the same GP-EI loop as bo_main.py to avoid code duplication.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern

    rng = np.random.RandomState(seed)

    # Initial DoE (Latin Hypercube)
    lhs = qmc.LatinHypercube(d=_NDIM, seed=seed)
    X = lhs.random(n_init)
    y = np.array([obj_v6(_unit_to_config(x)) for x in X])

    gp = GaussianProcessRegressor(
        kernel=Matern(nu=2.5),
        n_restarts_optimizer=5,
        normalize_y=True,
    )

    all_X = list(X.copy())
    all_y = list(y.copy())

    for _ in range(n_iter):
        gp.fit(np.array(all_X), np.array(all_y))
        mu_best = max(all_y)

        # Candidate batch via random + EI
        candidates = rng.rand(500, _NDIM)
        mu_c, sigma_c = gp.predict(candidates, return_std=True)
        from scipy.stats import norm as _norm
        z = (mu_c - mu_best) / (sigma_c + 1e-9)
        ei = sigma_c * (_norm.cdf(z) * z + _norm.pdf(z))
        best_idx = np.argmax(ei)
        x_next = candidates[best_idx]

        y_next = obj_v6(_unit_to_config(x_next))
        all_X.append(x_next)
        all_y.append(y_next)

    configs_and_scores = [
        (_unit_to_config(x), y) for x, y in zip(all_X, all_y)
    ]
    return configs_and_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--surrogate-dir", type=Path, default=Path("BO/bo_results"))
    parser.add_argument("--n-total",    type=int, default=50000,
                        help="Target total dataset size")
    parser.add_argument("--n-bo-runs",  type=int, default=25,
                        help="Number of independent BO campaigns")
    parser.add_argument("--k-vicinity", type=int, default=200,
                        help="Top configs to use for vicinity sampling")
    parser.add_argument("--sigma-perturb", type=float, default=0.05,
                        help="Unit-cube perturbation sigma for vicinity sampling")
    parser.add_argument("--out", type=Path,
                        default=Path("BO/data_expansion/optimized_designs.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time()
    CONSOLE.info("=" * 70)
    CONSOLE.info("GENEVO2 — Optimized Design Dataset Generator")
    CONSOLE.info("=" * 70)
    CONSOLE.info(f"  Target size  : {args.n_total:,}")
    CONSOLE.info(f"  BO campaigns : {args.n_bo_runs}")
    CONSOLE.info(f"  Vicinity top : {args.k_vicinity}")
    CONSOLE.info(f"  Output       : {args.out}")
    CONSOLE.info("")

    surrogate = SurrogateLoaderV3(args.surrogate_dir)
    physics   = PhysicsForwardModel()
    obj_v6    = TherapeuticObjectiveV6(physics, surrogate)
    obj_v3    = ObjectiveFunctionV3(physics, surrogate)

    rng = np.random.RandomState(args.seed)

    n_random   = int(args.n_total * 0.40)
    n_bo_total = int(args.n_total * 0.30)
    n_vicinity = args.n_total - n_random - n_bo_total

    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    all_bo_configs = []  # (cfg, score_v6) — collected for vicinity phase

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        writer.writeheader()

        # ── Phase A: Random / LHS ─────────────────────────────────────────────
        CONSOLE.info(f"[Phase A] Random LHS sampling — {n_random:,} configs")
        lhs_sampler = qmc.LatinHypercube(d=_NDIM, seed=args.seed)
        batch_size = 1000
        n_done = 0
        while n_done < n_random:
            batch = min(batch_size, n_random - n_done)
            X_batch = lhs_sampler.random(batch)
            for x in X_batch:
                cfg = _unit_to_config(x)
                metrics = _evaluate(cfg, obj_v6, obj_v3)
                writer.writerow(_cfg_to_row(cfg, metrics, "random"))
                rows_written += 1
            n_done += batch
            CONSOLE.info(f"  A: {n_done:,}/{n_random:,} random configs  "
                         f"(total={rows_written:,}  "
                         f"t={time()-t0:.0f}s)")

        # ── Phase B: BO exploit ───────────────────────────────────────────────
        CONSOLE.info(f"\n[Phase B] BO exploit — {args.n_bo_runs} campaigns")
        n_per_bo = n_bo_total // max(args.n_bo_runs, 1)
        bo_budget_total = 0
        for run in range(args.n_bo_runs):
            seed_run = args.seed + run + 1
            CONSOLE.info(f"  BO run {run+1}/{args.n_bo_runs}  seed={seed_run}")
            bo_results = run_bo_collect_all(obj_v6, n_init=20, n_iter=80, seed=seed_run)
            # Collect all evaluated configs (not just best)
            all_bo_configs.extend(bo_results)
            # Write a random subsample from this run to avoid overweighting BO
            indices = rng.choice(len(bo_results), min(n_per_bo, len(bo_results)), replace=False)
            for i in indices:
                cfg, _ = bo_results[i]
                metrics = _evaluate(cfg, obj_v6, obj_v3)
                writer.writerow(_cfg_to_row(cfg, metrics, "bo_exploit"))
                rows_written += 1
            bo_budget_total += len(indices)
            CONSOLE.info(f"    -> {len(indices)} samples written  (total={rows_written:,}  "
                         f"t={time()-t0:.0f}s)")

        # ── Phase C: Vicinity ─────────────────────────────────────────────────
        # Recompute how many vicinity configs are needed to reach n_total,
        # since Phase B is capped by n_bo_runs*(n_init+n_iter) not n_bo_total.
        n_vicinity = args.n_total - rows_written
        CONSOLE.info(f"\n[Phase C] Vicinity sampling — {n_vicinity:,} configs (filling to {args.n_total:,} total)")
        # Find top-K configs from BO runs by v6 score
        all_bo_configs.sort(key=lambda x: x[1], reverse=True)
        top_configs = [cfg for cfg, _ in all_bo_configs[:args.k_vicinity]]

        if not top_configs:
            CONSOLE.info("  No BO configs available — falling back to random")
            lhs_extra = qmc.LatinHypercube(d=_NDIM, seed=args.seed + 99)
            X_extra = lhs_extra.random(n_vicinity)
            for x in X_extra:
                cfg = _unit_to_config(x)
                metrics = _evaluate(cfg, obj_v6, obj_v3)
                writer.writerow(_cfg_to_row(cfg, metrics, "vicinity_fallback"))
                rows_written += 1
        else:
            n_per_top = max(1, n_vicinity // len(top_configs))
            done = 0
            for i, base_cfg in enumerate(top_configs):
                if done >= n_vicinity:
                    break
                base_unit = _config_to_unit(base_cfg)
                perturb_count = min(n_per_top, n_vicinity - done)
                for _ in range(perturb_count):
                    noise = rng.normal(0, args.sigma_perturb, _NDIM)
                    x_new = np.clip(base_unit + noise, 0.0, 1.0)
                    cfg = _unit_to_config(x_new)
                    metrics = _evaluate(cfg, obj_v6, obj_v3)
                    writer.writerow(_cfg_to_row(cfg, metrics, "vicinity"))
                    rows_written += 1
                    done += 1
                if (i + 1) % 50 == 0:
                    CONSOLE.info(f"  C: {done:,}/{n_vicinity:,} vicinity configs  "
                                 f"(total={rows_written:,}  t={time()-t0:.0f}s)")

    CONSOLE.info("\n" + "=" * 70)
    CONSOLE.info(f"COMPLETE: {rows_written:,} configs written in {time()-t0:.0f}s")
    CONSOLE.info(f"Output: {args.out}")
    CONSOLE.info("=" * 70)


if __name__ == "__main__":
    main()
