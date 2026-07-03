#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 2.1: Topology Search via Bayesian Optimization.

Runs BO over two discrete sensor topologies and compares results:
  - 2-channel: SOST + P1NP only (w_ctx forced to 0)
  - 3-channel: SOST + CTX + P1NP (all weights free)

Each topology gets n_init + n_iter function evaluations.
Output: JSON report + console table showing best scores and configs.

Scientific question: Can a simpler 2-channel sensor match 3-channel?
  If 2ch score ≈ 3ch score → simplification is justified (lower cost,
    fewer failure modes, less complex manufacturing)
  If 3ch score >> 2ch score → CTX provides discriminating signal worth
    the added complexity

Uses existing surrogates (data_v16). After generating data_v17, re-run
with --data-dir data_v17 --retrain-surrogates for topology-aware models.

Usage:
  python BO/bo_topology_search.py                          # quick run
  python BO/bo_topology_search.py --n-init 20 --n-iter 80 # full run
  python BO/bo_topology_search.py --data-dir data_v17 --retrain-surrogates
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.stats import qmc, mannwhitneyu

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from BO.core.surrogate_loader import SurrogateLoaderV3
from BO.core.build_surrogates import SurrogateBuilderV3
from search_space.biosensor_space import BiosensorSearchSpace
from evaluation.physics_forward_model import PhysicsForwardModel
from evaluation.therapeutic_objective_v6 import TherapeuticObjectiveV6

logger = logging.getLogger(__name__)


# ── Internal LHS-then-GP BO loop ──────────────────────────────────────────────

def _run_topology_bo(
    objective_fn,
    search_space: BiosensorSearchSpace,
    topology: str,
    n_init: int,
    n_iter: int,
    seed: int,
) -> dict:
    """
    Run a single BO campaign for a given topology.

    Returns a dict with keys: best_score, best_config, all_scores, all_configs.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel
    from scipy.stats import norm as _norm

    rng = np.random.RandomState(seed)

    # ── Parameter vector: 6 continuous dims that affect V6 score ─────────────
    # [log_kd_nm, log_sensitivity, log_kd_ctx_nm, log_kd_p1np_nm, w_ctx, w_p1np]
    # For 2-channel: dim 4 (w_ctx) is always 0 → effectively 5 active dims
    _P_NAMES = ["kd_nm", "sensitivity", "kd_ctx_nm", "kd_p1np_nm", "w_ctx", "w_p1np"]
    _BOUNDS = [
        (np.log10(0.1),  np.log10(10.0)),   # log_kd_nm
        (np.log10(0.5),  np.log10(5.0)),    # log_sensitivity
        (np.log10(0.1),  np.log10(10.0)),   # log_kd_ctx_nm
        (np.log10(0.1),  np.log10(10.0)),   # log_kd_p1np_nm
        (0.0,  0.49),                         # w_ctx
        (0.001, 0.49),                        # w_p1np
    ]
    ndim = len(_P_NAMES)

    def _unit_to_config(x: np.ndarray) -> dict:
        lo = np.array([b[0] for b in _BOUNDS])
        hi = np.array([b[1] for b in _BOUNDS])
        scaled = lo + x * (hi - lo)
        cfg = {
            "biosensor_type":  "array",
            "noise_preset":    "realistic",
            "target_scenario": "pmo",
            "kd_nm":      float(10 ** scaled[0]),
            "sensitivity": float(10 ** scaled[1]),
            "kd_ctx_nm":  float(10 ** scaled[2]),
            "kd_p1np_nm": float(10 ** scaled[3]),
            "w_ctx":  float(scaled[4]),
            "w_p1np": float(scaled[5]),
            "response_time_s": 600.0,
        }
        cfg = search_space.enforce_topology(cfg, topology)
        return cfg

    def _evaluate(x: np.ndarray) -> float:
        cfg = _unit_to_config(x)
        return float(objective_fn(cfg))

    # Initial LHS
    lhs = qmc.LatinHypercube(d=ndim, seed=seed)
    X_init = lhs.random(n_init)
    # For 2ch: zero out w_ctx dimension
    if topology == "2ch":
        X_init[:, 4] = 0.0

    Y_init = np.array([_evaluate(x) for x in X_init])

    all_X = list(X_init)
    all_Y = list(Y_init)

    logger.info(f"  [{topology}] Init: best={max(all_Y):.4f} mean={np.mean(all_Y):.4f}")

    kernel = Matern(nu=2.5, length_scale_bounds=(0.01, 100.0)) + WhiteKernel(
        noise_level_bounds=(1e-8, 1.0)
    )

    # Active dims: for 2ch, w_ctx column is always 0 → exclude from GP
    active_dims = [0, 1, 2, 3, 5] if topology == "2ch" else list(range(ndim))

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=0.0,
        normalize_y=True,
        n_restarts_optimizer=3,
        random_state=seed,
    )

    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    for it in range(n_iter):
        X_arr = np.array(all_X)[:, active_dims]
        Y_arr = np.array(all_Y)
        gp.fit(X_arr, Y_arr)
        y_best = float(np.max(Y_arr))

        # Candidate batch: 1000 random + EI selection
        cands = rng.rand(1000, ndim)
        if topology == "2ch":
            cands[:, 4] = 0.0
        mu_c, sigma_c = gp.predict(cands[:, active_dims], return_std=True)
        xi = 0.01
        z = (mu_c - y_best - xi) / (sigma_c + 1e-9)
        from scipy.stats import norm as _norm
        ei = sigma_c * (_norm.cdf(z) * z + _norm.pdf(z))
        best_idx = int(np.argmax(ei))
        x_next = cands[best_idx]

        y_next = _evaluate(x_next)
        all_X.append(x_next)
        all_Y.append(y_next)

        if (it + 1) % 20 == 0:
            logger.info(
                f"  [{topology}] iter {it+1}/{n_iter}  "
                f"best={max(all_Y):.4f}  last={y_next:.4f}"
            )

    best_idx = int(np.argmax(all_Y))
    best_x   = all_X[best_idx]
    best_cfg = _unit_to_config(best_x)
    best_score = float(all_Y[best_idx])

    return {
        "topology":     topology,
        "best_score":   best_score,
        "best_config":  best_cfg,
        "all_scores":   [float(y) for y in all_Y],
        "n_evals":      len(all_Y),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2.1: Topology Search BO (2ch vs 3ch)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data-dir",       type=Path, default=Path("data_v19"))
    parser.add_argument("--surrogate-dir",  type=Path, default=Path("BO/bo_results"))
    parser.add_argument("--output-dir",     type=Path, default=Path("BO/bo_results_topology"))
    parser.add_argument("--n-init",         type=int,  default=20)
    parser.add_argument("--n-iter",         type=int,  default=80)
    parser.add_argument("--n-seeds",        type=int,  default=5,
                        help="Independent BO seeds per topology (≥5 for CI; reviewer requirement)")
    parser.add_argument("--random-state",   type=int,  default=42,
                        help="Base random seed (seeds = base, base+1, ..., base+n_seeds-1)")
    parser.add_argument("--retrain-surrogates", action="store_true")
    parser.add_argument("--verbose", "-v",  action="store_true")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if not args.data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.surrogate_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("GENEVO2 Phase 2.1 — Topology Search BO")
    logger.info("=" * 70)
    logger.info(f"  Data dir      : {args.data_dir}")
    logger.info(f"  Surrogate dir : {args.surrogate_dir}")
    logger.info(f"  Output dir    : {args.output_dir}")
    logger.info(f"  n_init        : {args.n_init}")
    logger.info(f"  n_iter        : {args.n_iter}")
    logger.info(f"  n_seeds       : {args.n_seeds} (per topology)")
    logger.info("")

    # ── Surrogates ────────────────────────────────────────────────────────────
    saved_ml_dir = args.surrogate_dir / "saved_ml"
    scaler_path = saved_ml_dir / "scaler.pkl"
    surrogates_exist = scaler_path.exists()

    if args.retrain_surrogates or not surrogates_exist:
        logger.info("[1/4] Retraining surrogates...")
        builder = SurrogateBuilderV3(logger)
        X_raw, df_results = builder.load_and_prepare_data(args.data_dir)
        X = builder.fit_scaler(X_raw)
        y_dr  = df_results["detection_rate"].values.astype(np.float32)
        y_fnr = df_results["false_negative_rate"].values.astype(np.float32)
        y_ttd = df_results["time_to_detection"].values.astype(np.float32)
        metrics = builder.train_all(X, y_dr, y_fnr, y_ttd)
        builder.save(args.surrogate_dir)
        logger.info(
            f"  DR AUC={metrics['detection_rate']['test_roc_auc']:.4f}  "
            f"FNR R²={metrics['fnr']['test_r2']:.4f}  "
            f"TTD R²={metrics['ttd']['test_r2']:.4f}"
        )
    else:
        logger.info("[1/4] Loading existing surrogates...")

    try:
        surrogate_loader = SurrogateLoaderV3(args.surrogate_dir)
        logger.info("  Surrogates loaded")
    except Exception as e:
        logger.error(f"Failed to load surrogates: {e}")
        return 1

    physics_model = PhysicsForwardModel()
    objective_fn  = TherapeuticObjectiveV6(physics_model, surrogate_loader)
    search_space  = BiosensorSearchSpace()

    # ── Run topology BO (multi-seed) ──────────────────────────────────────────
    results = {}
    topologies = ["2ch", "3ch"]
    seeds = [args.random_state + i for i in range(args.n_seeds)]

    for topo in topologies:
        logger.info(f"\n[{topologies.index(topo)+2}/4] Running BO for topology={topo} "
                    f"({args.n_seeds} seeds)...")
        seed_results = []
        for seed in seeds:
            t0 = time.time()
            res = _run_topology_bo(
                objective_fn=objective_fn,
                search_space=search_space,
                topology=topo,
                n_init=args.n_init,
                n_iter=args.n_iter,
                seed=seed,
            )
            elapsed = time.time() - t0
            res["elapsed_s"] = round(elapsed, 1)
            res["seed"] = seed
            seed_results.append(res)
            logger.info(f"  [{topo}] seed={seed} done in {elapsed:.0f}s — "
                        f"best score = {res['best_score']:.4f}")

        best_scores = [r["best_score"] for r in seed_results]
        best_idx = int(np.argmax(best_scores))
        results[topo] = {
            "best_score":   float(np.max(best_scores)),
            "mean_score":   float(np.mean(best_scores)),
            "std_score":    float(np.std(best_scores, ddof=1)),
            "all_seed_scores": best_scores,
            "best_config":  seed_results[best_idx]["best_config"],
            "seed_results": seed_results,
            "n_seeds":      args.n_seeds,
        }
        logger.info(f"  [{topo}] multi-seed summary: "
                    f"mean={results[topo]['mean_score']:.4f} ± "
                    f"{results[topo]['std_score']:.4f} "
                    f"(n={args.n_seeds})")

    # ── Detailed evaluation of overall-best configs ───────────────────────────
    logger.info("\n[4/4] Detailed evaluation of best configs...")
    for topo, res in results.items():
        score, detail = objective_fn.evaluate_with_details(res["best_config"])
        res["details"] = {k: float(v) if isinstance(v, (int, float)) else v
                          for k, v in detail.items()}
        res["v6_score_detail"] = float(score)

    # ── Print comparison table ────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("TOPOLOGY COMPARISON RESULTS")
    logger.info("=" * 70)
    logger.info(f"{'Metric':<30} {'2-channel':>12} {'3-channel':>12}  {'Delta':>10}")
    logger.info("-" * 70)

    metrics_to_show = [
        ("V6 best score",       "best_score",     None),
        ("DR mean (surrogate)", "details.dr_mean", None),
        ("DR_CKD",              "details.dr_ckd",  None),
        ("DR_PMO",              "details.dr_pmo",  None),
        ("DR_mild",             "details.dr_mild", None),
        ("FNR mean",            "details.fnr_mean",None),
        ("Therapeutic BMD",     "details.therapeutic_mean", None),
        ("BMD net (PMO)",       "details.bmd_net_pmo", None),
        ("BMD net (mild)",      "details.bmd_net_mild", None),
        ("FP rate (healthy)",   "details.dr_healthy", None),
    ]

    def _get(res, key):
        if "." in key:
            parts = key.split(".", 1)
            sub = res.get(parts[0], {})
            return sub.get(parts[1], float("nan")) if isinstance(sub, dict) else float("nan")
        return res.get(key, float("nan"))

    for label, key, _ in metrics_to_show:
        v2 = _get(results["2ch"], key)
        v3 = _get(results["3ch"], key)
        delta = v3 - v2 if (isinstance(v2, float) and isinstance(v3, float)) else float("nan")
        logger.info(f"  {label:<28} {v2:>12.4f} {v3:>12.4f}  {delta:>+10.4f}")

    logger.info("\nBest 2-channel config:")
    c2 = results["2ch"]["best_config"]
    logger.info(f"  kd_nm={c2['kd_nm']:.3f}  sensitivity={c2['sensitivity']:.3f}")
    logger.info(f"  kd_p1np={c2['kd_p1np_nm']:.3f}  w_p1np={c2['w_p1np']:.3f}  w_ctx={c2['w_ctx']:.3f}")

    logger.info("\nBest 3-channel config:")
    c3 = results["3ch"]["best_config"]
    logger.info(f"  kd_nm={c3['kd_nm']:.3f}  sensitivity={c3['sensitivity']:.3f}")
    logger.info(f"  kd_ctx={c3['kd_ctx_nm']:.3f}  kd_p1np={c3['kd_p1np_nm']:.3f}")
    logger.info(f"  w_ctx={c3['w_ctx']:.3f}  w_p1np={c3['w_p1np']:.3f}")

    # Statistical significance (Mann-Whitney U — non-parametric, ≥5 seeds)
    scores_2ch = results["2ch"]["all_seed_scores"]
    scores_3ch = results["3ch"]["all_seed_scores"]
    mwu_stat, mwu_p = mannwhitneyu(scores_3ch, scores_2ch, alternative="greater")

    mean_2ch = results["2ch"]["mean_score"]
    mean_3ch = results["3ch"]["mean_score"]
    std_2ch  = results["2ch"]["std_score"]
    std_3ch  = results["3ch"]["std_score"]
    gap      = mean_3ch - mean_2ch
    rel_gap  = abs(gap) / max(abs(mean_3ch), 1e-9)

    logger.info("\n" + "-" * 70)
    logger.info("MULTI-SEED STATISTICAL COMPARISON:")
    logger.info(f"  2ch: {mean_2ch:.4f} +/- {std_2ch:.4f} (n={args.n_seeds})")
    logger.info(f"  3ch: {mean_3ch:.4f} +/- {std_3ch:.4f} (n={args.n_seeds})")
    logger.info(f"  Gap (3ch - 2ch): {gap:+.4f} ({rel_gap*100:.1f}%)")
    logger.info(f"  Mann-Whitney U p={mwu_p:.4f} "
                f"({'significant at alpha=0.05' if mwu_p < 0.05 else 'NOT significant'})")

    if mwu_p >= 0.05:
        verdict = (
            f"2-channel vs 3-channel: gap={gap:+.4f} ({rel_gap*100:.1f}%), "
            f"Mann-Whitney p={mwu_p:.3f} (not significant). "
            "Simplification to 2-channel is statistically defensible."
        )
    elif rel_gap < 0.05:
        verdict = (
            f"3-channel statistically outperforms (p={mwu_p:.3f}) but gap "
            f"({rel_gap*100:.1f}%) is clinically negligible (<5%)."
        )
    else:
        verdict = (
            f"3-channel significantly outperforms 2-channel: "
            f"gap={rel_gap*100:.1f}%, p={mwu_p:.3f}. CTX channel justified."
        )
    logger.info(f"\n  VERDICT: {verdict}")
    logger.info("-" * 70)

    # ── Save JSON results ─────────────────────────────────────────────────────
    output = {
        "timestamp":        datetime.now().isoformat(),
        "data_dir":         str(args.data_dir),
        "surrogate_dir":    str(args.surrogate_dir),
        "n_init":           args.n_init,
        "n_iter":           args.n_iter,
        "n_seeds":          args.n_seeds,
        "random_state":     args.random_state,
        "results":          results,
        "verdict":          verdict,
        "score_gap_abs":    float(gap),
        "score_gap_rel":    float(rel_gap),
        "mwu_p":            float(mwu_p),
        "mwu_stat":         float(mwu_stat),
    }

    out_path = args.output_dir / "topology_comparison.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved to: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
