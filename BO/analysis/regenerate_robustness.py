#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regenerate robustness metrics for all 10 convergence seeds.

The existing convergence_report.json has disease_mean_dr=-100.0 for every
seed because the robustness evaluation was run before the ODE path bug was
fixed.  This script:
  1. Loads each seed's best_config from convergence_report.json
  2. Runs RobustnessAnalyzer (3 noise × 3 scenario × n_trials stochastic runs)
  3. Patches the convergence_report.json in-place with real metrics
  4. Saves a side-file: convergence_robustness_patch.json for audit

Usage:
    python BO/analysis/regenerate_robustness.py
    python BO/analysis/regenerate_robustness.py --n-trials 10 --seeds 888 1337
    python BO/analysis/regenerate_robustness.py --dry-run   (preview, no writes)

Runtime:  ~30 min for all 10 seeds at n_trials=10 (9 cells × 10 trials × 10 seeds × ~2s/sim)
For a quick check on seed 888 only:
    python BO/analysis/regenerate_robustness.py --seeds 888 --n-trials 10
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _bo_cfg_to_robustness_biosensor(cfg_bd: dict) -> dict:
    """Convert biosensor_design sub-dict to the format RobustnessAnalyzer expects."""
    kd      = float(cfg_bd.get("kd_nm", 1.0))
    kd_ctx  = float(cfg_bd.get("kd_ctx_nm", kd))
    kd_p1np = float(cfg_bd.get("kd_p1np_nm", kd))
    w_ctx   = float(cfg_bd.get("w_ctx", 0.0))
    w_p1np  = float(cfg_bd.get("w_p1np", 0.0))
    w_scl   = max(0.0, 1.0 - w_ctx - w_p1np)
    sens    = float(cfg_bd.get("sensitivity", 1.0))
    noise   = cfg_bd.get("noise_preset", "realistic")
    btype   = cfg_bd.get("type", cfg_bd.get("biosensor_type", "array"))

    # Threshold calibrated to margin=1.25 (same formula as in robustness_analyzer.py)
    H = {"scl": 0.375, "ctx": 0.200, "p1np": 0.350}
    P = {"scl": 0.875, "ctx": 0.500, "p1np": 0.525}

    def _occ(c, kd_v): return c / (kd_v + c + 1e-12)

    ref_scl  = _occ(H["scl"],  kd)
    ref_ctx  = _occ(H["ctx"],  kd_ctx)
    ref_p1np = _occ(H["p1np"], kd_p1np)

    def _composite(cd):
        n_s = _occ(cd["scl"],  kd)     / (ref_scl  + 1e-12)
        n_c = _occ(cd["ctx"],  kd_ctx) / (ref_ctx  + 1e-12)
        n_p = _occ(cd["p1np"], kd_p1np) / (ref_p1np + 1e-12)
        return sens * (w_scl * n_s + w_ctx * n_c + w_p1np * n_p)

    sig_h = _composite(H)
    sig_p = _composite(P)
    hp_gap_ref = (sig_p / sens) - 1.0
    threshold = float(sig_h + 1.25 * hp_gap_ref)

    return {
        "kd_nm":           kd,
        "kd_ctx_nm":       kd_ctx,
        "kd_p1np_nm":      kd_p1np,
        "w_ctx":           w_ctx,
        "w_p1np":          w_p1np,
        "sensitivity":     sens,
        "noise_preset":    noise,
        "biosensor_type":  btype,
        "response_time_s": float(cfg_bd.get("response_time_s", 600.0)),
        "_threshold":      threshold,  # for reference only
    }


def evaluate_robustness_raw(cfg: dict, n_trials: int) -> dict:
    """
    Run real simulator across 3 scenarios × 3 noise presets.
    Returns a flat dict with mean/std/n_trials for each (scenario, noise) cell,
    plus aggregate metrics matching RobustnessAnalyzer output format.
    """
    from simulation.dataset.generator import DatasetGenerator

    kd      = float(cfg.get("kd_nm", 1.0))
    kd_ctx  = float(cfg.get("kd_ctx_nm", kd))
    kd_p1np = float(cfg.get("kd_p1np_nm", kd))
    w_ctx   = float(cfg.get("w_ctx", 0.0))
    w_p1np  = float(cfg.get("w_p1np", 0.0))
    w_scl   = max(0.0, 1.0 - w_ctx - w_p1np)
    sens    = float(cfg.get("sensitivity", 1.0))

    H = {"scl": 0.375, "ctx": 0.200, "p1np": 0.350}
    P = {"scl": 0.875, "ctx": 0.500, "p1np": 0.525}

    def _occ(c, kd_v): return c / (kd_v + c + 1e-12)

    ref_scl  = _occ(H["scl"],  kd)
    ref_ctx  = _occ(H["ctx"],  kd_ctx)
    ref_p1np = _occ(H["p1np"], kd_p1np)

    def _composite(cd):
        n_s = _occ(cd["scl"],  kd)     / (ref_scl  + 1e-12)
        n_c = _occ(cd["ctx"],  kd_ctx) / (ref_ctx  + 1e-12)
        n_p = _occ(cd["p1np"], kd_p1np) / (ref_p1np + 1e-12)
        return sens * (w_scl * n_s + w_ctx * n_c + w_p1np * n_p)

    sig_h = _composite(H)
    sig_p = _composite(P)
    threshold = float(sig_h + 1.25 * ((sig_p / sens) - 1.0))

    biosensor_config = {
        "circuit_type": "array",
        "kd_scl":   kd,
        "kd_ctx":   kd_ctx,
        "kd_p1np":  kd_p1np,
        "w_scl":    w_scl,
        "w_ctx":    w_ctx,
        "w_p1np":   w_p1np,
        "sensitivity": sens,
        "threshold": threshold,
        "dynamic_range": (0.0, sens * 3.0),
        "kd": kd,
    }

    noise_presets = ["low", "medium", "high"]
    scenarios     = ["healthy", "pmo", "ckd_mbd"]

    cell_results: Dict[str, dict] = {}
    all_disease_drs: List[float] = []
    all_healthy_drs: List[float] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        gen = DatasetGenerator(
            antimony_model_path="simulation/models/bone_environment.ant",
            output_dir=tmpdir,
            seed=None,
        )

        for scenario in scenarios:
            for noise in noise_presets:
                drs = []
                for _ in range(n_trials):
                    try:
                        result = gen.generate_single_simulation_instrumented(
                            scenario_name=scenario,
                            biosensor_config=biosensor_config,
                            noise_preset=noise,
                            duration=3600.0,
                            num_points=361,
                            apply_variability=True,
                            instrument=False,
                        )
                        if result is not None:
                            drs.append(float(result["measurement"]["detection_rate"]))
                    except Exception:
                        pass

                if drs:
                    cell_results[f"{scenario}_{noise}"] = {
                        "dr_mean": float(np.mean(drs)),
                        "dr_std":  float(np.std(drs)),
                        "n_trials": len(drs),
                    }
                    if scenario in ("pmo", "ckd_mbd"):
                        all_disease_drs.extend(drs)
                    elif scenario == "healthy":
                        all_healthy_drs.extend(drs)

    if not all_disease_drs:
        return {
            "disease_mean_dr":       -1.0,
            "disease_worst_case_dr": -1.0,
            "disease_best_case_dr":  -1.0,
            "disease_dr_std":        0.0,
            "healthy_fp_rate":       0.0,
            "robustness_index":      0.0,
            "evaluation_method": "ACTUAL SIMULATOR (no valid trials)",
            "cell_results": cell_results,
        }

    disease_arr = np.array(all_disease_drs)
    healthy_arr = np.array(all_healthy_drs) if all_healthy_drs else np.array([0.0])

    mean_dr    = float(np.mean(disease_arr))
    min_dr     = float(np.min(disease_arr))
    max_dr     = float(np.max(disease_arr))
    std_dr     = float(np.std(disease_arr))
    fp_rate    = float(np.mean(healthy_arr))

    robustness_index = float(np.clip(
        0.60 * min_dr + 0.20 * mean_dr + 0.10 * (1.0 - min_dr) + 0.10 * (1.0 - fp_rate),
        0.0, 1.0
    ))

    return {
        "disease_mean_dr":       mean_dr,
        "disease_worst_case_dr": min_dr,
        "disease_best_case_dr":  max_dr,
        "disease_dr_std":        std_dr,
        "healthy_fp_rate":       fp_rate,
        "robustness_index":      robustness_index,
        "evaluation_method": "ACTUAL SIMULATOR, real ODE runs",
        "cell_results": cell_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate robustness metrics for convergence seeds"
    )
    parser.add_argument(
        "--convergence-report", type=Path,
        default=Path("BO/bo_results/convergence/convergence_report.json"),
    )
    parser.add_argument("--n-trials", type=int, default=10,
                        help="Simulator trials per (scenario, noise) cell (default: 10)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Which seeds to update (default: all). Example: --seeds 888 1337")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without writing files")
    parser.add_argument("--out-patch", type=Path,
                        default=Path("BO/bo_results/convergence/convergence_robustness_patch.json"),
                        help="Side-file for the patch data (audit trail)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    print("=" * 70)
    print("GENEVO2 — Robustness Regeneration for Convergence Seeds")
    print("=" * 70)
    print(f"  Report    : {args.convergence_report}")
    print(f"  n_trials  : {args.n_trials} per (scenario, noise) cell")
    print(f"  Seeds     : {args.seeds or 'ALL'}")
    print(f"  Dry run   : {args.dry_run}")
    print()

    if not args.convergence_report.exists():
        print(f"[ERROR] Report not found: {args.convergence_report}")
        return 1

    with open(args.convergence_report) as f:
        report = json.load(f)

    updated_report = copy.deepcopy(report)
    patch_data: Dict[int, dict] = {}

    target_seeds = set(args.seeds) if args.seeds else None

    for i, run in enumerate(updated_report["runs"]):
        seed = run["seed"]
        if target_seeds and seed not in target_seeds:
            continue

        bd = run["best_config"]["biosensor_design"]
        me = run["best_config"].get("measurement_environment", {})

        cfg = {**bd, **me, "biosensor_type": bd.get("type", "array")}

        print(f"\n--- Seed {seed:5d} (score={run['best_score']:.4f}) ---")
        print(f"  kd={cfg.get('kd_nm', 0):.3f}  kd_ctx={cfg.get('kd_ctx_nm', 0):.3f}"
              f"  sens={cfg.get('sensitivity', 0):.3f}")

        if args.dry_run:
            print("  [DRY RUN] Would run robustness evaluation here.")
            continue

        try:
            rob = evaluate_robustness_raw(cfg, args.n_trials)
        except Exception as e:
            print(f"  [ERROR] {type(e).__name__}: {e}")
            continue

        print(f"  disease_mean_dr       = {rob['disease_mean_dr']:.3f}")
        print(f"  disease_worst_case_dr = {rob['disease_worst_case_dr']:.3f}")
        print(f"  healthy_fp_rate       = {rob['healthy_fp_rate']:.3f}")
        print(f"  robustness_index      = {rob['robustness_index']:.3f}")

        # Patch in-place
        updated_report["runs"][i]["best_config"]["robustness_analysis"] = {
            "disease_mean_dr":       rob["disease_mean_dr"],
            "disease_worst_case_dr": rob["disease_worst_case_dr"],
            "disease_best_case_dr":  rob["disease_best_case_dr"],
            "disease_dr_std":        rob["disease_dr_std"],
            "healthy_fp_rate":       rob["healthy_fp_rate"],
            "robustness_index":      rob["robustness_index"],
            "evaluation_method":     rob["evaluation_method"],
        }

        # Update best_run if this is the best seed
        if updated_report.get("best_run", {}).get("seed") == seed:
            updated_report["best_run"]["best_config"]["robustness_analysis"] = \
                updated_report["runs"][i]["best_config"]["robustness_analysis"]

        patch_data[seed] = rob

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return 0

    if not patch_data:
        print("\n[WARN] No seeds were processed.")
        return 0

    # Write patched report
    with open(args.convergence_report, "w") as f:
        json.dump(updated_report, f, indent=2)
    print(f"\n[OK] Patched convergence report: {args.convergence_report}")

    # Write audit patch file
    args.out_patch.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_patch, "w") as f:
        json.dump({str(k): v for k, v in patch_data.items()}, f, indent=2, default=float)
    print(f"[OK] Patch audit saved: {args.out_patch}")

    # Summary table
    print("\nRobustness Summary (updated seeds):")
    print(f"  {'Seed':>6}  {'Mean DR':>8}  {'Min DR':>8}  {'FP Rate':>8}  {'Rob Index':>10}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}")
    for seed_key, rob in patch_data.items():
        print(f"  {seed_key:>6}  {rob['disease_mean_dr']:>8.3f}  "
              f"{rob['disease_worst_case_dr']:>8.3f}  "
              f"{rob['healthy_fp_rate']:>8.3f}  "
              f"{rob['robustness_index']:>10.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
