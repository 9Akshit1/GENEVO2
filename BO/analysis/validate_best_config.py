#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Real-simulator validation of the best BO configuration.

Runs the full ODE + biosensor pipeline on seed 888's best config (composite
score = 0.722) with n=20 trials per scenario.  Reports actual detection rates
and compares them to the surrogate predictions embedded in convergence_report.json.

Purpose:
  - Verify surrogate predictions translate to real ODE outcomes
  - Quantify surrogate bias at the optimal configuration point
  - Provide simulator-validated DR values for the methods paper

Usage:
    python BO/analysis/validate_best_config.py
    python BO/analysis/validate_best_config.py --n-trials 30 --seed-config 1337
    python BO/analysis/validate_best_config.py --config-json path/to/config.json

Output:
    BO/bo_results/diagnostics/best_config_validation.json
"""

from __future__ import annotations

import argparse
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

SCENARIOS = ["pmo_mild", "pmo", "ckd_mbd", "healthy"]

# Scenario display labels for output
SCENARIO_LABELS = {
    "pmo_mild": "PMO-mild",
    "pmo":      "PMO",
    "ckd_mbd":  "CKD-MBD",
    "healthy":  "Healthy (FP rate)",
}


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _load_seed_config(convergence_path: Path, seed: int) -> Optional[dict]:
    """Load best_config from convergence report for a given seed."""
    with open(convergence_path) as f:
        report = json.load(f)
    for run in report["runs"]:
        if run["seed"] == seed:
            bd = run["best_config"]["biosensor_design"]
            me = run["best_config"]["measurement_environment"]
            return {**bd, **me, "biosensor_type": bd.get("type", "array")}
    return None


def _build_biosensor_config(cfg: dict) -> dict:
    """Convert BO config dict to the format expected by DatasetGenerator."""
    kd      = float(cfg.get("kd_nm", 1.0))
    kd_ctx  = float(cfg.get("kd_ctx_nm", kd))
    kd_p1np = float(cfg.get("kd_p1np_nm", kd))
    w_ctx   = float(cfg.get("w_ctx", 0.0))
    w_p1np  = float(cfg.get("w_p1np", 0.0))
    w_scl   = max(0.0, 1.0 - w_ctx - w_p1np)
    sens    = float(cfg.get("sensitivity", 1.0))

    # Threshold from margin=1.25 calibrated against healthy PMO signal gap
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


def run_validation(
    cfg: dict,
    n_trials: int,
    surrogate_predictions: Optional[dict] = None,
) -> dict:
    """
    Run real simulator n_trials times per scenario. Return detailed results.
    """
    from simulation.dataset.generator import DatasetGenerator

    biosensor_cfg = _build_biosensor_config(cfg)
    print(f"\n  Biosensor config:")
    print(f"    kd_SOST    = {cfg.get('kd_nm', 0):.4f} nM")
    print(f"    kd_CTX     = {cfg.get('kd_ctx_nm', 0):.4f} nM")
    print(f"    kd_P1NP    = {cfg.get('kd_p1np_nm', 0):.4f} nM")
    print(f"    w_CTX      = {cfg.get('w_ctx', 0):.4f}")
    print(f"    w_P1NP     = {cfg.get('w_p1np', 0):.4f}")
    print(f"    sensitivity= {cfg.get('sensitivity', 0):.4f}")
    print(f"    threshold  = {biosensor_cfg['threshold']:.4f}")
    print(f"  n_trials per scenario: {n_trials}\n")

    results: Dict[str, dict] = {}

    for scenario in SCENARIOS:
        trial_drs, trial_ttds = [], []
        print(f"  [{scenario:<10}]  running {n_trials} trials ...", end=" ", flush=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            gen = DatasetGenerator(
                antimony_model_path="simulation/models/bone_environment.ant",
                output_dir=tmpdir,
                seed=None,
            )

            for trial_idx in range(n_trials):
                try:
                    result = gen.generate_single_simulation_instrumented(
                        scenario_name=scenario,
                        biosensor_config=biosensor_cfg,
                        noise_preset="realistic",
                        duration=3600.0,
                        num_points=361,
                        apply_variability=True,
                        instrument=False,
                    )
                    if result is None:
                        continue
                    trial_drs.append(float(result["measurement"]["detection_rate"]))
                    trial_ttds.append(float(result["measurement"]["time_to_detection"]))
                except Exception as e:
                    logger.debug(f"    trial {trial_idx} error: {e}")

        if not trial_drs:
            print("FAILED (no successful trials)")
            results[scenario] = {"dr_mean": None, "dr_std": None, "n_trials": 0}
            continue

        dr_mean = float(np.mean(trial_drs))
        dr_std  = float(np.std(trial_drs))
        ttd_mean = float(np.mean(trial_ttds)) if trial_ttds else None
        ci_half = 1.96 * dr_std / np.sqrt(len(trial_drs))

        print(f"DR={dr_mean:.3f}±{dr_std:.3f}  CI95=[{dr_mean-ci_half:.3f},{dr_mean+ci_half:.3f}]  n={len(trial_drs)}")

        results[scenario] = {
            "dr_mean":  dr_mean,
            "dr_std":   dr_std,
            "dr_ci95_low":  float(dr_mean - ci_half),
            "dr_ci95_high": float(dr_mean + ci_half),
            "ttd_mean": ttd_mean,
            "n_trials": len(trial_drs),
        }

    # Surrogate vs real comparison
    comparison = {}
    if surrogate_predictions:
        print("\n  --- Surrogate vs Real-Simulator Comparison ---")
        print(f"  {'Scenario':<14}  {'Surrogate':>10}  {'Sim DR':>10}  {'Bias':>8}")
        print(f"  {'-'*14}  {'-'*10}  {'-'*10}  {'-'*8}")
        for scenario in SCENARIOS:
            surr_key = f"detection_rate_{scenario.replace('ckd_mbd', 'ckd')}"
            if scenario == "pmo_mild":
                surr_key = "detection_rate_min"
            if scenario == "pmo":
                surr_key = "detection_rate_pmo"
            if scenario == "ckd_mbd":
                surr_key = "detection_rate_ckd"
            if scenario == "healthy":
                surr_key = "healthy_fp_rate"

            surr_val = surrogate_predictions.get(surr_key)
            real_val = results.get(scenario, {}).get("dr_mean")
            if surr_val is not None and real_val is not None:
                bias = real_val - surr_val
                print(f"  {scenario:<14}  {surr_val:>10.3f}  {real_val:>10.3f}  {bias:>+8.3f}")
                comparison[scenario] = {
                    "surrogate": float(surr_val),
                    "real_sim":  float(real_val),
                    "bias":      float(bias),
                }

    return {"by_scenario": results, "comparison": comparison}


def main():
    parser = argparse.ArgumentParser(description="Real-simulator validation of best BO config")
    parser.add_argument("--n-trials",    type=int, default=20,
                        help="Simulator runs per scenario (default: 20)")
    parser.add_argument("--seed-config", type=int, default=888,
                        help="Which convergence seed's best config to validate (default: 888)")
    parser.add_argument("--config-json", type=Path, default=None,
                        help="Path to a best_config.json (overrides --seed-config)")
    parser.add_argument("--convergence-report", type=Path,
                        default=Path("BO/bo_results/convergence/convergence_report.json"),
                        help="Path to convergence_report.json")
    parser.add_argument("--out", type=Path,
                        default=Path("BO/bo_results/diagnostics/best_config_validation.json"),
                        help="Output JSON path")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    print("=" * 70)
    print("GENEVO2 — Real-Simulator Validation of Best BO Configuration")
    print("=" * 70)

    surrogate_preds = None

    if args.config_json and args.config_json.exists():
        print(f"  Loading config from: {args.config_json}")
        with open(args.config_json) as f:
            raw = json.load(f)
        if "biosensor_design" in raw:
            cfg = {**raw["biosensor_design"], **raw.get("measurement_environment", {})}
            cfg["biosensor_type"] = cfg.pop("type", cfg.get("biosensor_type", "array"))
            surrogate_preds = raw.get("predicted_performance")
        else:
            cfg = raw
    else:
        if not args.convergence_report.exists():
            print(f"[ERROR] Convergence report not found: {args.convergence_report}")
            return 1
        print(f"  Loading seed {args.seed_config} config from: {args.convergence_report}")
        cfg = _load_seed_config(args.convergence_report, args.seed_config)
        if cfg is None:
            print(f"[ERROR] Seed {args.seed_config} not found in convergence report.")
            return 1

        # Also extract surrogate predictions for comparison
        with open(args.convergence_report) as f:
            report = json.load(f)
        for run in report["runs"]:
            if run["seed"] == args.seed_config:
                surrogate_preds = run["best_config"].get("predicted_performance")
                break

    print(f"  Config: {cfg}")
    print()

    results = run_validation(cfg, args.n_trials, surrogate_predictions=surrogate_preds)

    # Summary
    by_sc = results["by_scenario"]
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    disease_drs = [
        by_sc[sc]["dr_mean"] for sc in ("pmo_mild", "pmo", "ckd_mbd")
        if by_sc.get(sc, {}).get("dr_mean") is not None
    ]
    fp_rate = by_sc.get("healthy", {}).get("dr_mean")

    if disease_drs:
        print(f"  Disease DR mean  : {np.mean(disease_drs):.3f}")
        print(f"  Disease DR min   : {np.min(disease_drs):.3f}")
    if fp_rate is not None:
        print(f"  FP rate (healthy): {fp_rate:.3f}")

    # Surrogate assessment
    comp = results.get("comparison", {})
    if comp:
        biases = [v["bias"] for v in comp.values() if "bias" in v]
        if biases:
            mean_bias = np.mean(biases)
            print(f"\n  Mean surrogate bias: {mean_bias:+.3f}")
            if abs(mean_bias) < 0.05:
                print("  Assessment: SURROGATE WELL-CALIBRATED at optimal config")
            elif abs(mean_bias) < 0.10:
                print("  Assessment: MODEST SURROGATE BIAS — acceptable for ranking")
            else:
                print("  Assessment: SIGNIFICANT SURROGATE BIAS — interpret predictions with caution")

    # Save
    output = {
        "seed_config": args.seed_config,
        "n_trials": args.n_trials,
        "config_used": cfg,
        "surrogate_predictions": surrogate_preds,
        "validation_results": results,
        "summary": {
            "disease_dr_mean": float(np.mean(disease_drs)) if disease_drs else None,
            "disease_dr_min":  float(np.min(disease_drs)) if disease_drs else None,
            "fp_rate":         float(fp_rate) if fp_rate is not None else None,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\n[OK] Results saved: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
