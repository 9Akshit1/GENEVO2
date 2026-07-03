#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 2.1: 20-run statistical benchmark — 2-channel vs 3-channel topology.

Runs N_RUNS independent BO campaigns for each topology (separate random seeds).
Reports:
  - Best score per run
  - Mean ± std across runs
  - Mann-Whitney U test (is 3ch significantly better?)
  - Convergence curve (median best-so-far vs iteration)

This answers: "Is the 2-channel advantage/disadvantage statistically robust,
or is it seed-dependent?"

Usage:
  python BO/benchmarks/topology_comparison.py
  python BO/benchmarks/topology_comparison.py --n-runs 20 --n-init 20 --n-iter 80
  python BO/benchmarks/topology_comparison.py --n-runs 5 --n-init 10 --n-iter 30  # quick
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from BO.core.surrogate_loader import SurrogateLoaderV3
from search_space.biosensor_space import BiosensorSearchSpace
from evaluation.physics_forward_model import PhysicsForwardModel
from evaluation.therapeutic_objective_v6 import TherapeuticObjectiveV6

logger = logging.getLogger(__name__)


def _run_single_bo(objective_fn, search_space, topology, n_init, n_iter, seed):
    """
    Single BO run for one topology.  Returns (best_score, best_so_far_curve).
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel
    from scipy.stats import qmc
    from scipy.stats import norm as _norm
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    rng = np.random.RandomState(seed)

    _BOUNDS = [
        (np.log10(0.1),  np.log10(10.0)),
        (np.log10(0.5),  np.log10(5.0)),
        (np.log10(0.1),  np.log10(10.0)),
        (np.log10(0.1),  np.log10(10.0)),
        (0.0,   0.49),
        (0.001, 0.49),
    ]
    ndim = len(_BOUNDS)

    def _unit_to_config(x):
        lo = np.array([b[0] for b in _BOUNDS])
        hi = np.array([b[1] for b in _BOUNDS])
        s = lo + x * (hi - lo)
        cfg = {
            "biosensor_type": "array", "noise_preset": "realistic",
            "target_scenario": "pmo", "response_time_s": 600.0,
            "kd_nm": float(10**s[0]), "sensitivity": float(10**s[1]),
            "kd_ctx_nm": float(10**s[2]), "kd_p1np_nm": float(10**s[3]),
            "w_ctx": float(s[4]), "w_p1np": float(s[5]),
        }
        return search_space.enforce_topology(cfg, topology)

    lhs = qmc.LatinHypercube(d=ndim, seed=seed)
    X_init = lhs.random(n_init)
    if topology == "2ch":
        X_init[:, 4] = 0.0

    all_X = list(X_init)
    all_Y = [float(objective_fn(_unit_to_config(x))) for x in X_init]

    active_dims = [0, 1, 2, 3, 5] if topology == "2ch" else list(range(ndim))
    kernel = Matern(nu=2.5, length_scale_bounds=(0.01, 100.0)) + WhiteKernel(
        noise_level_bounds=(1e-8, 1.0)
    )
    gp = GaussianProcessRegressor(
        kernel=kernel, alpha=0.0, normalize_y=True,
        n_restarts_optimizer=2, random_state=seed,
    )

    best_so_far = []
    running_best = max(all_Y)
    best_so_far.extend([running_best] * n_init)

    for _ in range(n_iter):
        gp.fit(np.array(all_X)[:, active_dims], np.array(all_Y))
        y_best = float(np.max(all_Y))
        cands = rng.rand(500, ndim)
        if topology == "2ch":
            cands[:, 4] = 0.0
        mu_c, sigma_c = gp.predict(cands[:, active_dims], return_std=True)
        z = (mu_c - y_best - 0.01) / (sigma_c + 1e-9)
        ei = sigma_c * (_norm.cdf(z) * z + _norm.pdf(z))
        x_next = cands[int(np.argmax(ei))]
        y_next = float(objective_fn(_unit_to_config(x_next)))
        all_X.append(x_next)
        all_Y.append(y_next)
        running_best = max(running_best, y_next)
        best_so_far.append(running_best)

    return float(max(all_Y)), best_so_far


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2.1: 20-run topology benchmark (2ch vs 3ch)",
    )
    parser.add_argument("--surrogate-dir", type=Path, default=Path("BO/bo_results"))
    parser.add_argument("--output-dir",    type=Path, default=Path("BO/bo_results_topology"))
    parser.add_argument("--n-runs",    type=int, default=20, help="BO runs per topology")
    parser.add_argument("--n-init",    type=int, default=20)
    parser.add_argument("--n-iter",    type=int, default=80)
    parser.add_argument("--base-seed", type=int, default=100)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Phase 2.1 Topology Benchmark")
    logger.info(f"  {args.n_runs} runs × (2ch + 3ch), {args.n_init}+{args.n_iter} evals each")
    logger.info("=" * 70)

    surrogate_loader = SurrogateLoaderV3(args.surrogate_dir)
    physics_model    = PhysicsForwardModel()
    objective_fn     = TherapeuticObjectiveV6(physics_model, surrogate_loader)
    search_space     = BiosensorSearchSpace()

    results_by_topo = {"2ch": [], "3ch": []}
    curves_by_topo  = {"2ch": [], "3ch": []}

    n_total = args.n_runs * 2
    done = 0

    for topo in ["2ch", "3ch"]:
        logger.info(f"\n--- Topology: {topo} ---")
        for run_i in range(args.n_runs):
            seed = args.base_seed + run_i * 7 + (0 if topo == "2ch" else 1000)
            t0   = time.time()
            best, curve = _run_single_bo(
                objective_fn, search_space, topo,
                args.n_init, args.n_iter, seed,
            )
            results_by_topo[topo].append(best)
            curves_by_topo[topo].append(curve)
            done += 1
            elapsed = time.time() - t0
            logger.info(
                f"  [{topo}] run {run_i+1}/{args.n_runs}  "
                f"best={best:.4f}  t={elapsed:.0f}s  "
                f"({done}/{n_total} total)"
            )

    # ── Statistical test ──────────────────────────────────────────────────────
    scores_2ch = np.array(results_by_topo["2ch"])
    scores_3ch = np.array(results_by_topo["3ch"])

    stat, p_value = mannwhitneyu(scores_3ch, scores_2ch, alternative="greater")

    logger.info("\n" + "=" * 70)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 70)
    logger.info(f"\n{'Metric':<28} {'2-channel':>12} {'3-channel':>12}  {'Delta':>10}")
    logger.info("-" * 66)
    logger.info(
        f"  {'Mean best score':<26} {scores_2ch.mean():>12.4f} "
        f"{scores_3ch.mean():>12.4f}  {scores_3ch.mean()-scores_2ch.mean():>+10.4f}"
    )
    logger.info(
        f"  {'Std best score':<26} {scores_2ch.std():>12.4f} "
        f"{scores_3ch.std():>12.4f}"
    )
    logger.info(
        f"  {'Median best score':<26} {np.median(scores_2ch):>12.4f} "
        f"{np.median(scores_3ch):>12.4f}  "
        f"{np.median(scores_3ch)-np.median(scores_2ch):>+10.4f}"
    )
    logger.info(
        f"  {'Max score':<26} {scores_2ch.max():>12.4f} "
        f"{scores_3ch.max():>12.4f}"
    )
    logger.info(f"\n  Mann-Whitney U stat : {stat:.1f}")
    logger.info(f"  p-value (3ch>2ch)   : {p_value:.4f}")

    alpha = 0.05
    if p_value < alpha:
        verdict = (
            f"3-channel is SIGNIFICANTLY better than 2-channel (p={p_value:.4f} < {alpha}). "
            "CTX channel provides statistically meaningful discrimination."
        )
    else:
        gap_rel = abs(scores_3ch.mean() - scores_2ch.mean()) / max(abs(scores_3ch.mean()), 1e-9)
        if gap_rel < 0.05:
            verdict = (
                f"2-channel matches 3-channel within 5% (p={p_value:.4f} >= {alpha}). "
                "Simplification to 2-channel is scientifically justified."
            )
        else:
            verdict = (
                f"3-channel trends higher but NOT significantly so (p={p_value:.4f} >= {alpha}). "
                "More runs needed to resolve. Consider manufacturing complexity trade-off."
            )

    logger.info(f"\n  VERDICT: {verdict}")

    # Convergence: median best-so-far across runs
    min_len = min(len(c) for c in curves_by_topo["2ch"] + curves_by_topo["3ch"])
    curve_2ch = np.median([c[:min_len] for c in curves_by_topo["2ch"]], axis=0)
    curve_3ch = np.median([c[:min_len] for c in curves_by_topo["3ch"]], axis=0)

    logger.info("\n  Convergence (median best-so-far, every 20 evals):")
    logger.info(f"  {'Eval':>6}  {'2ch':>8}  {'3ch':>8}  {'Gap':>8}")
    for i in range(0, min_len, 20):
        logger.info(
            f"  {i+1:>6}  {curve_2ch[i]:>8.4f}  {curve_3ch[i]:>8.4f}  "
            f"{curve_3ch[i]-curve_2ch[i]:>+8.4f}"
        )

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        "timestamp":      datetime.now().isoformat(),
        "n_runs":         args.n_runs,
        "n_init":         args.n_init,
        "n_iter":         args.n_iter,
        "scores_2ch":     scores_2ch.tolist(),
        "scores_3ch":     scores_3ch.tolist(),
        "mean_2ch":       float(scores_2ch.mean()),
        "mean_3ch":       float(scores_3ch.mean()),
        "std_2ch":        float(scores_2ch.std()),
        "std_3ch":        float(scores_3ch.std()),
        "mann_whitney_U": float(stat),
        "p_value":        float(p_value),
        "significant":    bool(p_value < alpha),
        "verdict":        verdict,
        "convergence_2ch": curve_2ch.tolist(),
        "convergence_3ch": curve_3ch.tolist(),
    }

    out_path = args.output_dir / "topology_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved to: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
