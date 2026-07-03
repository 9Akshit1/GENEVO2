#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Top-Design Validator

Answers the single most important question: do the configurations BO finds
remain excellent when evaluated by the REAL simulator rather than the surrogate?

Method
------
1. Sample N configs from the search space via LHS (diverse, space-filling).
2. Evaluate each with the v4 surrogate → surrogate_score, dr_surr, fnr_surr, ttd_surr
3. Take the top-K high-scoring configs (surrogate-selected high-value region).
4. Evaluate those K configs with the REAL simulator (n_trials per scenario).
5. Report: R², rank correlation, bias between surrogate and reality.

If surrogate R² >> 0 and rank correlation > 0.7 → BO is finding real good designs.
If surrogate is systematically optimistic → BO is gaming the surrogate.

Usage
-----
    cd c:\\Users\\eruku\\Akshith\\GENEVO2
    python BO/validation/validate_top_designs.py [--n-lhs 200] [--top-k 50] [--n-trials 5]
"""

import argparse
import json
import logging
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "BO"))

from BO.core.surrogate_loader import SurrogateLoaderV3
from search_space.biosensor_space import BiosensorSearchSpace
from evaluation.physics_forward_model import PhysicsForwardModel
from evaluation.therapeutic_objective_v6 import TherapeuticObjectiveV6
from acquisition.acquisition_functions import ExpectedImprovement

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Suppress per-simulation spam (ODE solver, biosensor engine, dataset generator)
for _noisy in ["simulation.simulator", "simulation.biosensor_engine", "dataset.generator"]:
    logging.getLogger(_noisy).setLevel(logging.WARNING)

SCENARIOS = ["healthy", "pmo_mild", "pmo", "ckd_mbd"]
DISEASE_SCENARIOS = ["pmo_mild", "pmo", "ckd_mbd"]
ANT_MODEL = str(ROOT / "simulation" / "models" / "bone_environment.ant")

# Healthy concentrations for threshold computation
_H = {"scl": 0.375, "ctx": 0.200, "p1np": 0.350}
_P = {"scl": 0.875, "ctx": 0.500, "p1np": 0.525}


def _array_biosensor_config(cfg: dict) -> dict:
    """Build the biosensor config dict for the real simulator from a BO search-space dict."""
    kd = cfg["kd_nm"]
    kd_ctx = cfg.get("kd_ctx_nm", kd)
    kd_p1np = cfg.get("kd_p1np_nm", kd)
    sensitivity = cfg["sensitivity"]
    w_ctx = cfg.get("w_ctx", 0.0)
    w_p1np = cfg.get("w_p1np", 0.0)
    w_scl = max(0.0, 1.0 - w_ctx - w_p1np)

    def _occ(c, k): return c / (k + c + 1e-12)

    ref_scl = _occ(_H["scl"], kd)
    ref_ctx = _occ(_H["ctx"], kd_ctx)
    ref_p1np = _occ(_H["p1np"], kd_p1np)

    def _composite(d):
        return sensitivity * (
            w_scl * _occ(d["scl"], kd) / (ref_scl + 1e-12) +
            w_ctx * _occ(d["ctx"], kd_ctx) / (ref_ctx + 1e-12) +
            w_p1np * _occ(d["p1np"], kd_p1np) / (ref_p1np + 1e-12)
        )

    sig_h = _composite(_H)
    sig_p = _composite(_P)
    hp_gap_ref = (sig_p / sensitivity) - 1.0
    threshold = float(sig_h + 1.25 * hp_gap_ref)

    return {
        "circuit_type": "array",
        "kd": kd,
        "kd_scl": kd,
        "kd_ctx": kd_ctx,
        "kd_p1np": kd_p1np,
        "w_scl": w_scl,
        "w_ctx": w_ctx,
        "w_p1np": w_p1np,
        "sensitivity": sensitivity,
        "threshold": threshold,
        "dynamic_range": (0.0, sensitivity * 3.0),
    }


def evaluate_with_real_simulator(cfg: dict, n_trials: int = 5) -> dict:
    """Run the real ODE simulator for a config across all scenarios and noise=realistic.

    Returns a dict with mean DR (disease), FNR, TTD per scenario plus overall averages.
    Returns None if the simulator import fails.
    """
    try:
        from simulation.dataset.generator import DatasetGenerator
    except Exception as e:
        logger.error(f"Cannot import DatasetGenerator: {e}")
        return None

    biosensor_config = _array_biosensor_config(cfg)

    results = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        gen = DatasetGenerator(
            antimony_model_path=ANT_MODEL,
            output_dir=tmpdir,
            seed=None,
        )
        for scenario in SCENARIOS:
            drs, fnrs, ttds = [], [], []
            for _ in range(n_trials):
                r = gen.generate_single_simulation_instrumented(
                    scenario_name=scenario,
                    biosensor_config=biosensor_config,
                    noise_preset="realistic",
                    duration=3600.0,
                    num_points=361,
                    apply_variability=True,
                    instrument=False,
                )
                if r is None:
                    continue
                m = r["measurement"]
                drs.append(float(m["detection_rate"]))
                fnrs.append(float(m["false_negative_rate"]))
                ttds.append(float(m["time_to_detection"]))

            if drs:
                results[scenario] = {
                    "dr": float(np.mean(drs)),
                    "fnr": float(np.mean(fnrs)),
                    "ttd": float(np.mean(ttds)),
                    "n": len(drs),
                }

    if not results:
        return None

    disease_drs = [results[s]["dr"] for s in DISEASE_SCENARIOS if s in results]
    disease_fnrs = [results[s]["fnr"] for s in DISEASE_SCENARIOS if s in results]
    disease_ttds = [results[s]["ttd"] for s in DISEASE_SCENARIOS if s in results]

    return {
        "per_scenario": results,
        "dr_mean": float(np.mean(disease_drs)) if disease_drs else 0.0,
        "fnr_mean": float(np.mean(disease_fnrs)) if disease_fnrs else 1.0,
        "ttd_mean": float(np.mean(disease_ttds)) if disease_ttds else 9000.0,
        "dr_healthy": results.get("healthy", {}).get("dr", 0.0),
    }


def main():
    parser = argparse.ArgumentParser(description="Validate top BO designs against real simulator")
    parser.add_argument("--n-lhs", type=int, default=200,
                        help="Number of LHS samples to evaluate with surrogate (default: 200)")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-K configs to validate with real simulator (default: 50)")
    parser.add_argument("--n-trials", type=int, default=5,
                        help="Real simulator trials per (config × scenario) (default: 5)")
    parser.add_argument("--surrogate-dir", type=Path, default=ROOT / "BO" / "bo_results",
                        help="Surrogate directory (default: BO/bo_results)")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "BO" / "validation" / "results",
                        help="Output directory for results")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SURROGATE ACCURACY VALIDATION")
    print("=" * 70)
    print(f"  LHS samples:    {args.n_lhs}")
    print(f"  Top-K to test:  {args.top_k}")
    print(f"  Trials/sim:     {args.n_trials}")
    print()

    # Load surrogate + objective
    print(f"[1/4] Loading surrogate from {args.surrogate_dir}...")
    loader = SurrogateLoaderV3(results_dir=args.surrogate_dir)
    physics_model = PhysicsForwardModel()
    obj = TherapeuticObjectiveV6(physics_model, loader)
    search_space = BiosensorSearchSpace()
    n_params = search_space.n_params

    # LHS sampling
    print(f"[2/4] Generating {args.n_lhs} LHS samples...")
    from scipy.stats import qmc
    sampler = qmc.LatinHypercube(d=n_params, seed=0)
    X_lhs = sampler.random(n=args.n_lhs)

    # Evaluate all with surrogate
    print("[3/4] Evaluating with surrogate...")
    surrogate_scores = []
    configs = []
    for x in X_lhs:
        cfg = search_space.vector_to_dict(x)
        score = obj(cfg)
        surrogate_scores.append(score)
        configs.append(cfg)

    surrogate_scores = np.array(surrogate_scores)
    top_idx = np.argsort(surrogate_scores)[-args.top_k:][::-1]

    print(f"\n  Surrogate score stats over {args.n_lhs} configs:")
    print(f"    mean={surrogate_scores.mean():.4f}  "
          f"std={surrogate_scores.std():.4f}  "
          f"max={surrogate_scores.max():.4f}  "
          f"min={surrogate_scores.min():.4f}")
    print(f"  Top-{args.top_k} surrogate score range: "
          f"{surrogate_scores[top_idx].min():.4f} - {surrogate_scores[top_idx].max():.4f}")

    # Evaluate top-K with real simulator
    print(f"\n[4/4] Evaluating top-{args.top_k} with REAL simulator "
          f"({args.n_trials} trials/scenario)...")
    print("  This may take 10-30 minutes depending on hardware.\n")

    records = []
    for rank, i in enumerate(top_idx):
        cfg = configs[i]
        surr_score = float(surrogate_scores[i])

        # Get surrogate DR/FNR/TTD
        score_v6, detail = obj.evaluate_with_details(cfg)
        surr_dr = detail.get("dr_mean", 0.0)
        surr_fnr = detail.get("fnr_mean", 1.0)
        surr_ttd = detail.get("ttd_mean", 9000.0)
        surr_dr_healthy = detail.get("dr_healthy", 0.0)

        # Real simulator
        real = evaluate_with_real_simulator(cfg, n_trials=args.n_trials)
        if real is None:
            print(f"  Config {rank+1:3d}/{args.top_k}: SIMULATOR FAILED")
            continue

        real_dr = real["dr_mean"]
        real_fnr = real["fnr_mean"]
        real_ttd = real["ttd_mean"]
        real_dr_healthy = real["dr_healthy"]

        records.append({
            "rank": rank + 1,
            "surr_score": surr_score,
            "surr_dr": surr_dr,
            "surr_fnr": surr_fnr,
            "surr_ttd": surr_ttd,
            "surr_dr_healthy": surr_dr_healthy,
            "real_dr": real_dr,
            "real_fnr": real_fnr,
            "real_ttd": real_ttd,
            "real_dr_healthy": real_dr_healthy,
            "dr_error": real_dr - surr_dr,
            "fnr_error": real_fnr - surr_fnr,
            "ttd_error": real_ttd - surr_ttd,
            **{f"cfg_{k}": v for k, v in cfg.items() if isinstance(v, (int, float))},
        })

        print(f"  [{rank+1:3d}/{args.top_k}] surr_score={surr_score:.4f} | "
              f"DR: surr={surr_dr:.3f} real={real_dr:.3f} (d={real_dr-surr_dr:+.3f}) | "
              f"FNR: surr={surr_fnr:.3f} real={real_fnr:.3f} | "
              f"TTD: surr={surr_ttd:.0f}s real={real_ttd:.0f}s")

    if len(records) < 5:
        print("\nInsufficient data for analysis. Check simulator setup.")
        return

    df = pd.DataFrame(records)

    # Compute statistics
    from scipy.stats import spearmanr, pearsonr

    dr_r2 = pearsonr(df.surr_dr, df.real_dr)[0] ** 2
    fnr_r2 = pearsonr(df.surr_fnr, df.real_fnr)[0] ** 2
    ttd_r2 = pearsonr(df.surr_ttd, df.real_ttd)[0] ** 2

    dr_rho, dr_p = spearmanr(df.surr_dr, df.real_dr)
    fnr_rho, _ = spearmanr(df.surr_fnr, df.real_fnr)

    dr_bias = float(df.dr_error.mean())
    fnr_bias = float(df.fnr_error.mean())
    ttd_bias = float(df.ttd_error.mean())

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n  Surrogate vs Real Simulator (n={len(df)} configs)")
    print(f"  {'Metric':<8} {'R2':>7} {'Rank-rho':>9} {'Bias (real-surr)':>18}")
    print(f"  {'-'*50}")
    print(f"  {'DR':<8} {dr_r2:>7.4f} {dr_rho:>9.4f} {dr_bias:>+18.4f}")
    print(f"  {'FNR':<8} {fnr_r2:>7.4f} {fnr_rho:>9.4f} {fnr_bias:>+18.4f}")
    print(f"  {'TTD':<8} {ttd_r2:>7.4f} {'n/a':>9} {ttd_bias:>+18.1f}s")

    if dr_bias > 0.02:
        verdict = "SURROGATE PESSIMISTIC  (real DR > predicted -- BO undershooting)"
    elif dr_bias < -0.02:
        verdict = "SURROGATE OPTIMISTIC   (real DR < predicted -- BO gaming surrogate)"
    else:
        verdict = "SURROGATE WELL-CALIBRATED"

    print(f"\n  Bias verdict:   {verdict}")
    print(f"  Rank verdict:   {'Good ranking (rho>0.6)' if dr_rho > 0.6 else 'Poor ranking -- surrogate misleading'}")

    print()
    print("  BOTTOM LINE:")
    if dr_r2 > 0.5 and dr_rho > 0.6:
        print("  [OK]  Surrogate faithfully guides BO. Top designs are REAL top designs.")
    elif dr_rho > 0.5:
        print("  [~]   Surrogate ranking partially correct. Closed-loop BO will help.")
    else:
        print("  [!!]  Surrogate ranking not reliable. Closed-loop BO is ESSENTIAL.")
    print()

    # Save results
    out_csv = args.out_dir / "surrogate_vs_real.csv"
    df.to_csv(out_csv, index=False)

    out_json = args.out_dir / "surrogate_accuracy.json"
    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_lhs": args.n_lhs,
        "top_k": args.top_k,
        "n_trials_per_sim": args.n_trials,
        "n_validated": len(df),
        "dr_r2": float(dr_r2),
        "fnr_r2": float(fnr_r2),
        "ttd_r2": float(ttd_r2),
        "dr_rank_correlation": float(dr_rho),
        "dr_rank_p_value": float(dr_p),
        "dr_bias": float(dr_bias),
        "fnr_bias": float(fnr_bias),
        "ttd_bias_s": float(ttd_bias),
        "verdict": verdict,
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Results -> {out_csv}")
    print(f"  Summary -> {out_json}")


if __name__ == "__main__":
    main()
