#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-start BO convergence runner.

Usage:
    python BO/bo_converged.py [--n-runs 10] [--n-init 50] [--n-iter 150]

Output: BO/bo_results_converged/convergence_report.json
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

_DEFAULT_SEEDS = [42, 7, 123, 999, 2024, 31415, 1337, 888, 555, 77]


def run_single(
    seed: int,
    n_init: int,
    n_iter: int,
    surrogate_dir: str,
    output_root: Path,
    verbose: bool,
) -> dict | None:
    """Run one BO subprocess; return parsed result dict or None on failure."""
    out_dir = output_root / f"run_seed{seed:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "BO/bo_main.py",
        "--mode", "standard",
        "--surrogate-dir", surrogate_dir,
        "--output-dir", str(out_dir),
        "--n-init", str(n_init),
        "--n-iter", str(n_iter),
        "--random-state", str(seed),
    ]

    logger.info("  seed=%d  n_init=%d  n_iter=%d ...", seed, n_init, n_iter)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
            timeout=7200,
        )
    except subprocess.TimeoutExpired:
        logger.error("  seed=%d TIMEOUT (>2h)", seed)
        return None
    except Exception as exc:
        logger.error("  seed=%d EXCEPTION: %s", seed, exc)
        return None

    if proc.returncode != 0:
        stderr_tail = (proc.stderr or "")[-400:]
        logger.error("  seed=%d FAILED (exit %d)\n%s", seed, proc.returncode, stderr_tail)
        return None

    result_file = out_dir / "results" / "optimization_results.json"
    config_file = out_dir / "results" / "best_config.json"

    if not result_file.exists():
        logger.error("  seed=%d: result file missing", seed)
        return None

    with open(result_file) as f:
        opt = json.load(f)

    best_y = float(opt.get("best_y") or opt.get("best_y_observed") or 0.0)

    best_config = None
    if config_file.exists():
        with open(config_file) as f:
            best_config = json.load(f)

    return {"seed": seed, "best_score": best_y, "best_config": best_config, "output_dir": str(out_dir)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="High-convergence BO: n_init=50 n_iter=200 multi-start",
    )
    parser.add_argument("--n-init",      type=int, default=50,
                        help="LHS initial samples (default 50; was 20)")
    parser.add_argument("--n-iter",      type=int, default=200,
                        help="BO iterations after init (default 200; was 80)")
    parser.add_argument("--n-runs",      type=int, default=1,
                        help="Number of independent seeds (default 1; use 10 for convergence test)")
    parser.add_argument("--surrogate-dir", default="BO/bo_results",
                        help="Surrogate directory (default BO/bo_results)")
    parser.add_argument("--output-dir", type=Path, default=Path("BO/bo_results_converged"),
                        help="Root output directory (default BO/bo_results_converged)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Explicit seed list (overrides --n-runs)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    seeds = args.seeds if args.seeds else _DEFAULT_SEEDS[:args.n_runs]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 70)
    print("HIGH-CONVERGENCE BO")
    print(f"  n_init = {args.n_init}  (was 20)")
    print(f"  n_iter = {args.n_iter}  (was 80)")
    print(f"  total evaluations per run = {args.n_init + args.n_iter}")
    print(f"  seeds  = {seeds}")
    print(f"  output = {args.output_dir}")
    print("=" * 70)

    results = []
    for seed in seeds:
        r = run_single(
            seed=seed,
            n_init=args.n_init,
            n_iter=args.n_iter,
            surrogate_dir=args.surrogate_dir,
            output_root=args.output_dir,
            verbose=args.verbose,
        )
        if r is not None:
            results.append(r)
            logger.info("  seed=%d => %.4f", seed, r["best_score"])

    if not results:
        print("[FAIL] All BO runs failed.")
        return 1

    scores = np.array([r["best_score"] for r in results])
    best_idx = int(np.argmax(scores))
    best_run = results[best_idx]

    print()
    print("-" * 70)
    print("CONVERGENCE REPORT")
    print("-" * 70)
    if len(scores) > 1:
        print(f"  Runs completed : {len(scores)}/{len(seeds)}")
        print(f"  Best score     : {scores.max():.4f}  (seed {best_run['seed']})")
        print(f"  Mean score     : {scores.mean():.4f}")
        print(f"  Std score      : {scores.std():.4f}  (target < 0.01)")
        print(f"  Min score      : {scores.min():.4f}")

        verdict = "CONVERGED" if scores.std() < 0.01 else (
            "ACCEPTABLE" if scores.std() < 0.02 else "NEEDS MORE RUNS"
        )
        print(f"  Verdict        : {verdict}")
    else:
        print(f"  Single run score: {scores[0]:.4f}")

    print()
    if best_run["best_config"]:
        cfg = best_run["best_config"]
        print("BEST CONFIGURATION:")
        for k, v in cfg.items():
            if isinstance(v, float):
                print(f"  {k:<20} {v:.4f}")
            else:
                print(f"  {k:<20} {v}")

    # Save convergence report
    report = {
        "n_init": args.n_init,
        "n_iter": args.n_iter,
        "seeds": seeds,
        "runs": results,
        "summary": {
            "n_successful": len(scores),
            "best_score": float(scores.max()),
            "mean_score": float(scores.mean()),
            "std_score": float(scores.std()),
            "min_score": float(scores.min()),
        },
        "best_run": best_run,
    }
    report_path = args.output_dir / "convergence_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Also copy best config to standard location
    if best_run["best_config"]:
        results_dir = args.output_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "best_config.json", "w") as f:
            json.dump(best_run["best_config"], f, indent=2)

    print(f"\nReport saved: {report_path}")

    if len(scores) > 1 and scores.std() >= 0.01:
        print()
        print("[!!] CONVERGENCE NOT ACHIEVED: std = %.4f >= 0.01" % scores.std())
        print("     Try: --n-init 64 --n-iter 300 --n-runs 10")
        print("     Or reduce search space dimensionality (freeze response_time, noise_preset)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
