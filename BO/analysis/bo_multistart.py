#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-start BO runner: runs BO N times with different seeds and reports
mean ± std of the best composite score.

Each run uses the same surrogate (pre-trained on data_v18) but a different
random LHS initial design, so the results quantify BO convergence stability.

If std(best_score) < 0.01: BO has converged — one run was enough.
If std(best_score) > 0.03: BO landscape is rugged — need more evaluations
                           or a global search (LHS baseline).

Usage:
    python BO/analysis/bo_multistart.py
    python BO/analysis/bo_multistart.py --n-runs 10 --n-iter 80
    python BO/analysis/bo_multistart.py --n-runs 5 --out BO/analysis/multistart_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Seeds for independent BO runs
_DEFAULT_SEEDS = [42, 7, 123, 999, 2024, 31415, 1337, 888, 555, 77]


def run_single_bo(
    seed: int,
    n_init: int,
    n_iter: int,
    data_dir: str,
    surrogate_dir: str,
    output_dir: str,
    verbose: bool,
) -> dict | None:
    """
    Run one BO execution as a subprocess and return the result dict.

    Returns None if the run fails.
    """
    out_path = Path(output_dir) / f"run_seed{seed:04d}"
    out_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "BO/bo_main.py",
        "--mode", "standard",
        "--data-dir",       data_dir,
        "--surrogate-dir",  surrogate_dir,
        "--output-dir",     str(out_path),
        "--n-init",         str(n_init),
        "--n-iter",         str(n_iter),
        "--random-state",   str(seed),
    ]

    logger.info("Starting BO run seed=%d ...", seed)
    try:
        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
            timeout=3600,
        )
        if result.returncode != 0:
            logger.error("Run seed=%d failed (exit %d)", seed, result.returncode)
            if not verbose and result.stderr:
                logger.error(result.stderr[-500:])
            return None
    except subprocess.TimeoutExpired:
        logger.error("Run seed=%d timed out (>1h)", seed)
        return None
    except Exception as e:
        logger.error("Run seed=%d exception: %s", seed, e)
        return None

    # Read results
    result_file = out_path / "results" / "optimization_results.json"
    config_file = out_path / "results" / "best_config.json"

    if not result_file.exists():
        logger.error("Run seed=%d: result file missing: %s", seed, result_file)
        return None

    with open(result_file) as f:
        opt_result = json.load(f)

    best_score = opt_result.get("best_y") or opt_result.get("best_y_observed", 0.0)

    best_config = None
    if config_file.exists():
        with open(config_file) as f:
            best_config = json.load(f)

    logger.info("  Seed=%d => best_score=%.4f", seed, best_score)
    return {
        "seed":        seed,
        "best_score":  float(best_score),
        "n_init":      n_init,
        "n_iter":      n_iter,
        "output_dir":  str(out_path),
        "best_config": best_config,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Multi-start BO: run BO N times to estimate convergence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--n-runs", dest="n_runs", type=int, default=5,
        help="Number of independent BO runs (default: 5)",
    )
    parser.add_argument(
        "--n-init", dest="n_init", type=int, default=20,
        help="LHS initial samples per run (default: 20)",
    )
    parser.add_argument(
        "--n-iter", dest="n_iter", type=int, default=80,
        help="BO iterations per run (default: 80)",
    )
    parser.add_argument(
        "--data-dir", dest="data_dir", default="data_v19",
        help="Dataset directory (default: data_v19)",
    )
    parser.add_argument(
        "--surrogate-dir", dest="surrogate_dir", default="BO/bo_results",
        help="Surrogate directory (default: BO/bo_results)",
    )
    parser.add_argument(
        "--output-dir", dest="output_dir", default="BO/bo_results_multistart",
        help="Output directory for all runs (default: BO/bo_results_multistart)",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Explicit seed list (overrides --n-runs; e.g. --seeds 42 7 123)",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output JSON path for summary (optional)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    seeds = args.seeds or _DEFAULT_SEEDS[: args.n_runs]

    logger.info("=" * 64)
    logger.info("MULTI-START BO  (%d runs, n_init=%d, n_iter=%d)", len(seeds), args.n_init, args.n_iter)
    logger.info("Seeds: %s", seeds)
    logger.info("=" * 64)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    run_results = []
    for seed in seeds:
        r = run_single_bo(
            seed=seed,
            n_init=args.n_init,
            n_iter=args.n_iter,
            data_dir=args.data_dir,
            surrogate_dir=args.surrogate_dir,
            output_dir=args.output_dir,
            verbose=args.verbose,
        )
        if r is not None:
            run_results.append(r)

    if not run_results:
        logger.error("All BO runs failed.")
        return 1

    scores = [r["best_score"] for r in run_results]
    mean_s = float(np.mean(scores))
    std_s  = float(np.std(scores))
    best   = max(run_results, key=lambda x: x["best_score"])

    print("\n" + "=" * 64)
    print("MULTI-START BO SUMMARY")
    print("=" * 64)
    print(f"  Runs completed      : {len(run_results)} / {len(seeds)}")
    print(f"  Best score (mean)   : {mean_s:.4f} +/- {std_s:.4f}")
    print(f"  Best score (max)    : {max(scores):.4f}  (seed={best['seed']})")
    print(f"  Best score (min)    : {min(scores):.4f}")
    print(f"  95% CI (approx)     : [{mean_s - 2*std_s:.4f}, {mean_s + 2*std_s:.4f}]")
    print()
    if std_s < 0.01:
        verdict = "CONVERGED — BO is stable; single-run results are reliable"
    elif std_s < 0.03:
        verdict = "ACCEPTABLE — moderate variance; 5-run average is trustworthy"
    else:
        verdict = "HIGH VARIANCE — BO is not converging; increase n-iter or use random restarts"
    print(f"  Convergence verdict : {verdict}")
    print()
    print("  Per-run scores:")
    for r in sorted(run_results, key=lambda x: -x["best_score"]):
        print(f"    seed={r['seed']:<6d}  score={r['best_score']:.4f}  → {r['output_dir']}")
    print("=" * 64)

    summary = {
        "n_runs_completed": len(run_results),
        "n_runs_requested": len(seeds),
        "scores": {
            "mean": mean_s,
            "std":  std_s,
            "max":  max(scores),
            "min":  min(scores),
            "all":  scores,
        },
        "best_run_seed": best["seed"],
        "best_run_dir":  best["output_dir"],
        "runs":          run_results,
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Strip large best_config fields to keep summary compact
        for r in summary["runs"]:
            r.pop("best_config", None)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Summary saved: %s", out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
