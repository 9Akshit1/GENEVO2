#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Closed-Loop Bayesian Optimization

The fundamental limitation of surrogate-only BO is that it optimizes a
PROXY for the real objective.  The surrogate may be inaccurate exactly
in the regions BO finds promising — leading it to exploit surrogate
artifacts rather than real biosensor performance.

This script implements closed-loop BO with iterative surrogate refinement:

  Round 0  ─►  BO on base surrogate (v4)
            ─►  Top-K configs validated with REAL simulator
            ─►  Real measurements added to training data
  Round 1  ─►  Surrogate retrained on original + augmented data
            ─►  BO on refined surrogate → better exploration of true landscape
            ─►  Top-K validated → more augmented data
  ...
  Round N  ─►  Final surrogate accurate in the high-scoring region
            ─►  Final BO → best design by real metrics

Why this gets much better:
  - Each real evaluation CORRECTS surrogate errors in the promising region
  - The surrogate bias identified in validate_top_designs.py is eliminated
  - BO is no longer limited by fixed surrogate accuracy
  - 5 rounds × 10 configs × 4 scenarios = 200 new real evaluations focused on
    exactly where it matters most

Usage
-----
    cd c:\\Users\\eruku\\Akshith\\GENEVO2
    python BO/bo_closed_loop.py [options]

    --n-rounds 5        refinement rounds (default: 5)
    --n-inner 50        BO iterations per round (default: 50)
    --n-init 20         initial random samples per round (default: 20)
    --top-k 10          configs validated with real sim per round (default: 10)
    --n-trials 5        real simulator trials per (config × scenario) (default: 5)
    --data-dir data_v16 training data directory (default: data_v16)
    --out-dir BO/bo_results_closed_loop  output directory
"""

import argparse
import json
import logging
import shutil
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "BO"))

from BO.core.surrogate_loader import SurrogateLoaderV3
from BO.core.build_surrogates import SurrogateBuilderV3
from search_space.biosensor_space import BiosensorSearchSpace
from evaluation.physics_forward_model import PhysicsForwardModel
from evaluation.therapeutic_objective_v6 import TherapeuticObjectiveV6
from acquisition.acquisition_functions import ExpectedImprovement
from optimizer.gaussian_process_bo import GaussianProcessBO

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Suppress per-simulation spam (ODE solver, biosensor engine, dataset generator)
for _noisy in ["simulation.simulator", "simulation.biosensor_engine", "dataset.generator"]:
    logging.getLogger(_noisy).setLevel(logging.WARNING)

ANT_MODEL = str(ROOT / "simulation" / "models" / "bone_environment.ant")
SCENARIOS = ["healthy", "pmo_mild", "pmo", "ckd_mbd"]
DISEASE_SCENARIOS = ["pmo_mild", "pmo", "ckd_mbd"]

_H = {"scl": 0.375, "ctx": 0.200, "p1np": 0.350}
_P = {"scl": 0.875, "ctx": 0.500, "p1np": 0.525}


def _array_biosensor_config(cfg: dict) -> dict:
    """Convert BO search-space dict → simulator biosensor config with correct threshold."""
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
    threshold = float(sig_h + 1.25 * ((sig_p / sensitivity) - 1.0))

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


def simulate_config(cfg: dict, n_trials: int, gen) -> list[dict]:
    """Evaluate one biosensor config across all scenarios. Returns list of CSV rows."""
    biosensor_cfg = _array_biosensor_config(cfg)
    kd = cfg["kd_nm"]
    kd_ctx = cfg.get("kd_ctx_nm", kd)
    kd_p1np = cfg.get("kd_p1np_nm", kd)
    w_ctx = cfg.get("w_ctx", 0.0)
    w_p1np = cfg.get("w_p1np", 0.0)
    w_scl = max(0.0, 1.0 - w_ctx - w_p1np)
    sensitivity = cfg["sensitivity"]
    threshold = biosensor_cfg["threshold"]

    rows = []
    for scenario in SCENARIOS:
        for _ in range(n_trials):
            r = gen.generate_single_simulation_instrumented(
                scenario_name=scenario,
                biosensor_config=biosensor_cfg,
                noise_preset="realistic",
                duration=3600.0,
                num_points=361,
                apply_variability=True,
                instrument=False,
            )
            if r is None:
                continue
            m = r["measurement"]
            rows.append({
                "run_id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "scenario": scenario,
                "biosensor_type": "array",
                "noise_preset": "realistic",
                "kd": kd,
                "kd_scl": kd,
                "kd_ctx": kd_ctx,
                "kd_p1np": kd_p1np,
                "w_scl": w_scl,
                "w_ctx": w_ctx,
                "w_p1np": w_p1np,
                "sensitivity": sensitivity,
                "threshold": threshold,
                "response_time": float("nan"),  # array sensors don't use this
                "detection_rate": float(m["detection_rate"]),
                "false_negative_rate": float(m["false_negative_rate"]),
                "time_to_detection": float(m["time_to_detection"]),
                "snr_db": float(m.get("snr_db", 0.0)),
                "n_detections": int(m.get("n_detections", 0)),
                "sclerostin_mean": float(r.get("sclerostin_mean", float("nan"))),
                "sclerostin_std": float(r.get("sclerostin_std", float("nan"))),
                "ctx_mean": float(r.get("ctx_mean", float("nan"))),
                "p1np_mean": float(r.get("p1np_mean", float("nan"))),
                "metadata_file": "",
                "timeseries_file": "",
            })
    return rows


def retrain_surrogate(
    original_csv: Path,
    augmented_rows: list[dict],
    out_surrogate_dir: Path,
    version: str = None,
) -> SurrogateLoaderV3:
    """Merge original data + augmented rows, retrain surrogate, return new loader."""
    original_df = pd.read_csv(original_csv)
    aug_df = pd.DataFrame(augmented_rows)

    # Align columns: add any missing columns with NaN
    for col in original_df.columns:
        if col not in aug_df.columns:
            aug_df[col] = float("nan")
    aug_df = aug_df[original_df.columns]

    combined = pd.concat([original_df, aug_df], ignore_index=True)
    combined_csv = out_surrogate_dir / "master_index_augmented.csv"
    combined.to_csv(combined_csv, index=False)
    logger.info(f"  Combined dataset: {len(original_df)} original + {len(aug_df)} augmented = {len(combined)} rows")

    # Retrain
    builder = SurrogateBuilderV3()
    X_raw, df_prep = builder.load_and_prepare_data(out_surrogate_dir)
    X = builder.fit_scaler(X_raw)
    y_dr = df_prep["detection_rate"].values.astype(np.float32)
    y_fnr = df_prep["false_negative_rate"].values.astype(np.float32)
    y_ttd = df_prep["time_to_detection"].values.astype(np.float32)

    metrics = builder.train_all(X, y_dr, y_fnr, y_ttd)
    builder.save(out_surrogate_dir)

    logger.info(f"  Retrained surrogate:")
    logger.info(f"    DR AUC={metrics['detection_rate']['test_roc_auc']:.4f}  "
                f"FNR R²={metrics['fnr']['test_r2']:.4f}  "
                f"TTD R²={metrics['ttd']['test_r2']:.4f}")

    return SurrogateLoaderV3(results_dir=out_surrogate_dir)


def run_bo_round(
    objective: TherapeuticObjectiveV6,
    search_space: BiosensorSearchSpace,
    n_init: int,
    n_inner: int,
    seed: int,
) -> tuple[list[dict], np.ndarray]:
    """Run one round of BO. Returns (all_configs_sorted_by_score, sorted_scores)."""
    acq = ExpectedImprovement(xi=0.01)
    bo = GaussianProcessBO(objective, search_space, acq,
                           n_init=n_init, n_iter=n_inner, random_state=seed)
    results = bo.optimize()

    X_obs = results["X_observed"]
    y_obs = results["y_observed"]

    idx_sorted = np.argsort(y_obs)[::-1]
    configs_sorted = [search_space.vector_to_dict(X_obs[i]) for i in idx_sorted]
    scores_sorted = y_obs[idx_sorted]

    return configs_sorted, scores_sorted


def main():
    parser = argparse.ArgumentParser(description="Closed-loop BO with surrogate refinement")
    parser.add_argument("--n-rounds", type=int, default=5, help="Refinement rounds (default: 5)")
    parser.add_argument("--n-inner", type=int, default=50, help="BO iterations per round (default: 50)")
    parser.add_argument("--n-init", type=int, default=20, help="Initial LHS samples per round (default: 20)")
    parser.add_argument("--top-k", type=int, default=10, help="Configs validated with real sim per round (default: 10)")
    parser.add_argument("--n-trials", type=int, default=5, help="Simulator trials per (config × scenario) (default: 5)")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data_v19",
                        help="Training data directory (default: data_v18)")
    parser.add_argument("--surrogate-dir", type=Path, default=ROOT / "BO" / "bo_results",
                        help="Base surrogate directory (default: BO/bo_results)")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "BO" / "bo_results_closed_loop",
                        help="Output directory (default: BO/bo_results_closed_loop)")
    parser.add_argument("--early-stop", action="store_true",
                        help="Enable early stopping based on convergence criteria (default: False)")
    parser.add_argument("--min-improvement", type=float, default=0.01,
                        help="Min. improvement in mean real DR to continue (default: 0.01)")
    parser.add_argument("--max-bias-increase", type=float, default=0.10,
                        help="Max. allowed increase in abs(surrogate_bias) before stop (default: 0.10)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "results").mkdir(exist_ok=True)
    (args.out_dir / "saved_ml").mkdir(exist_ok=True)

    original_csv = args.data_dir / "master_index.csv"
    if not original_csv.exists():
        logger.error(f"Training data not found: {original_csv}")
        sys.exit(1)

    print("=" * 70)
    print("CLOSED-LOOP BAYESIAN OPTIMIZATION")
    print("=" * 70)
    print(f"  Rounds:         {args.n_rounds}")
    print(f"  BO iters/round: {args.n_init} init + {args.n_inner} iter = {args.n_init + args.n_inner} evals")
    print(f"  Real sim/round: {args.top_k} configs × 4 scenarios × {args.n_trials} trials = "
          f"{args.top_k * 4 * args.n_trials} real evaluations")
    print(f"  Total real sim: {args.n_rounds * args.top_k * 4 * args.n_trials} evaluations "
          f"on top of {sum(1 for _ in open(original_csv)) - 1} original rows")
    print()

    # Import simulator upfront to fail early
    try:
        from simulation.dataset.generator import DatasetGenerator
    except Exception as e:
        logger.error(f"Cannot import DatasetGenerator: {e}\n"
                     f"Run from project root: cd {ROOT}")
        sys.exit(1)

    search_space = BiosensorSearchSpace()
    physics_model = PhysicsForwardModel()
    augmented_rows: list[dict] = []
    round_summaries = []

    current_surrogate_dir = args.surrogate_dir

    for round_i in range(args.n_rounds):
        print(f"\n{'='*70}")
        print(f"ROUND {round_i} / {args.n_rounds - 1}")
        print(f"{'='*70}")

        # --- Load current surrogate ---
        loader = SurrogateLoaderV3(results_dir=current_surrogate_dir)
        objective = TherapeuticObjectiveV6(physics_model, loader)

        # --- Run BO ---
        print(f"[{round_i}.1] Running BO ({args.n_init} init + {args.n_inner} iter)...")
        seed = 42 + round_i * 100
        configs_sorted, scores_sorted = run_bo_round(
            objective, search_space, args.n_init, args.n_inner, seed
        )

        best_surr_score = float(scores_sorted[0])
        print(f"       Best surrogate score this round: {best_surr_score:.4f}")
        print(f"       Top-{min(5, args.top_k)} surrogate scores: "
              f"{', '.join(f'{s:.4f}' for s in scores_sorted[:5])}")

        # --- Validate top-K with real simulator ---
        top_k = min(args.top_k, len(configs_sorted))
        print(f"\n[{round_i}.2] Validating top-{top_k} with REAL simulator...")

        round_rows = []
        round_real_scores = []

        with tempfile.TemporaryDirectory() as tmpdir:
            gen = DatasetGenerator(
                antimony_model_path=ANT_MODEL,
                output_dir=tmpdir,
                seed=None,
            )
            for k_i, cfg in enumerate(configs_sorted[:top_k]):
                surr_score = float(scores_sorted[k_i])
                rows = simulate_config(cfg, args.n_trials, gen)
                if not rows:
                    print(f"  Config {k_i+1:2d}: FAILED")
                    continue

                round_rows.extend(rows)

                # Compute real DR from rows
                disease_rows = [r for r in rows if r["scenario"] in DISEASE_SCENARIOS]
                healthy_rows = [r for r in rows if r["scenario"] == "healthy"]
                real_dr = float(np.mean([r["detection_rate"] for r in disease_rows])) if disease_rows else 0.0
                real_fnr = float(np.mean([r["false_negative_rate"] for r in disease_rows])) if disease_rows else 1.0
                real_fp = float(np.mean([r["detection_rate"] for r in healthy_rows])) if healthy_rows else 0.0

                round_real_scores.append(real_dr)
                print(f"  Config {k_i+1:2d}/{top_k}: surr={surr_score:.4f}  "
                      f"real_DR={real_dr:.3f}  real_FNR={real_fnr:.3f}  real_FP={real_fp:.3f}")

        augmented_rows.extend(round_rows)

        mean_real_dr = float(np.mean(round_real_scores)) if round_real_scores else 0.0
        surrogate_bias = best_surr_score - mean_real_dr
        print(f"\n  Round {round_i} summary:")
        print(f"    Real DR mean (top-{top_k}): {mean_real_dr:.4f}")
        print(f"    Surrogate bias (score - real_dr): {surrogate_bias:+.4f}")
        print(f"    New real evaluations this round: {len(round_rows)}")
        print(f"    Total augmented rows so far: {len(augmented_rows)}")

        round_summaries.append({
            "round": round_i,
            "best_surrogate_score": best_surr_score,
            "mean_real_dr_top_k": mean_real_dr,
            "surrogate_bias": float(surrogate_bias),
            "new_rows_this_round": len(round_rows),
            "total_augmented_rows": len(augmented_rows),
        })

        # --- Early stopping check (if enabled) ---
        if args.early_stop and round_i > 0:
            prev_round = round_summaries[-2]
            prev_dr = prev_round["mean_real_dr_top_k"]
            prev_bias = abs(prev_round["surrogate_bias"])
            curr_bias = abs(surrogate_bias)

            dr_improvement = mean_real_dr - prev_dr
            bias_increase = curr_bias - prev_bias

            print(f"\n  Early-stop check:")
            print(f"    DR improvement: {dr_improvement:+.4f} (min required: {args.min_improvement:+.4f})")
            print(f"    Bias increase: {bias_increase:+.4f} (max allowed: {args.max_bias_increase:+.4f})")

            should_stop = False
            if dr_improvement < args.min_improvement:
                print(f"    [!] DR improvement below threshold")
                should_stop = True
            if bias_increase > args.max_bias_increase:
                print(f"    [!] Surrogate bias worsening beyond threshold")
                should_stop = True

            if should_stop:
                print(f"\n  => EARLY STOP: Convergence criteria met. Stopping at round {round_i}.")
                break

        # --- Retrain surrogate on augmented data ---
        if round_i < args.n_rounds - 1:
            print(f"\n[{round_i}.3] Retraining surrogate with augmented data...")
            new_surrogate_dir = args.out_dir

            # Copy original master_index.csv to output dir for builder to find it
            aug_master_csv = new_surrogate_dir / "master_index_augmented.csv"
            original_df = pd.read_csv(original_csv)
            aug_df = pd.DataFrame(augmented_rows)

            # Align columns: ensure aug_df has exactly same columns as original_df
            for col in original_df.columns:
                if col not in aug_df.columns:
                    aug_df[col] = float("nan")
            # Select only columns present in original, in original order
            aug_df = aug_df[original_df.columns]
            combined = pd.concat([original_df, aug_df], ignore_index=True)
            combined_csv = new_surrogate_dir / "master_index_augmented.csv"
            combined.to_csv(combined_csv, index=False)
            logger.info(f"  Combined dataset: {len(original_df)} original + {len(aug_df)} augmented = {len(combined)} rows")

            # Retrain
            builder = SurrogateBuilderV3()
            # Point builder at combined dataset
            builder_data_dir = new_surrogate_dir
            # Copy combined data as master_index.csv for builder to discover
            shutil.copy(combined_csv, builder_data_dir / "master_index.csv")

            try:
                X_raw, df_prep = builder.load_and_prepare_data(builder_data_dir)
                X = builder.fit_scaler(X_raw)
                y_dr = df_prep["detection_rate"].values.astype(np.float32)
                y_fnr = df_prep["false_negative_rate"].values.astype(np.float32)
                y_ttd = df_prep["time_to_detection"].values.astype(np.float32)
                metrics = builder.train_all(X, y_dr, y_fnr, y_ttd)
                builder.save(args.out_dir)
                print(f"  Surrogate round {round_i}: "
                      f"DR AUC={metrics['detection_rate']['test_roc_auc']:.4f}  "
                      f"FNR R²={metrics['fnr']['test_r2']:.4f}  "
                      f"TTD R²={metrics['ttd']['test_r2']:.4f}")
                current_surrogate_dir = args.out_dir
            except Exception as e:
                logger.warning(f"  Surrogate retrain failed ({e}), continuing with previous surrogate")

    # --- Final BO run on the best surrogate ---
    print(f"\n{'='*70}")
    print("FINAL ROUND — Best Refined Surrogate")
    print(f"{'='*70}")

    loader_final = SurrogateLoaderV3(results_dir=current_surrogate_dir)
    obj_final = TherapeuticObjectiveV6(physics_model, loader_final)
    print(f"  Using surrogate from {current_surrogate_dir}")
    print(f"  Running final BO ({args.n_init} init + {args.n_inner * 2} iter for thoroughness)...")

    configs_final, scores_final = run_bo_round(
        obj_final, search_space, args.n_init, args.n_inner * 2, seed=999
    )
    best_final_score = float(scores_final[0])
    best_final_cfg = configs_final[0]

    print(f"\n  Best final surrogate score: {best_final_score:.4f}")
    print(f"  Best config:")
    for k, v in best_final_cfg.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")

    # Quick real-simulator validation of the final best
    print("\n  Validating final best with real simulator (10 trials)...")
    with tempfile.TemporaryDirectory() as tmpdir:
        gen_final = DatasetGenerator(
            antimony_model_path=ANT_MODEL,
            output_dir=tmpdir,
            seed=None,
        )
        final_rows = simulate_config(best_final_cfg, n_trials=10, gen=gen_final)

    if final_rows:
        disease_rows = [r for r in final_rows if r["scenario"] in DISEASE_SCENARIOS]
        healthy_rows = [r for r in final_rows if r["scenario"] == "healthy"]
        final_real_dr = float(np.mean([r["detection_rate"] for r in disease_rows])) if disease_rows else 0.0
        final_real_fnr = float(np.mean([r["false_negative_rate"] for r in disease_rows])) if disease_rows else 1.0
        final_real_fp = float(np.mean([r["detection_rate"] for r in healthy_rows])) if healthy_rows else 0.0
        print(f"  Final validation:  DR={final_real_dr:.3f}  FNR={final_real_fnr:.3f}  FP={final_real_fp:.3f}")
    else:
        final_real_dr = None
        print("  WARNING: Final validation failed")

    # --- Save everything ---
    out_json = args.out_dir / "results" / "closed_loop_results.json"
    results_dict = {
        "timestamp": datetime.now().isoformat(),
        "n_rounds": args.n_rounds,
        "n_inner": args.n_inner,
        "top_k": args.top_k,
        "n_trials": args.n_trials,
        "total_augmented_rows": len(augmented_rows),
        "final_best_surrogate_score": best_final_score,
        "final_best_real_dr": final_real_dr,
        "round_summaries": round_summaries,
        "best_config": best_final_cfg,
    }
    with open(out_json, "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"\n{'='*70}")
    print("CLOSED-LOOP BO COMPLETE")
    print(f"{'='*70}")
    print(f"  Rounds: {args.n_rounds}")
    print(f"  Total augmented evaluations: {len(augmented_rows)}")
    print(f"  Final best surrogate score: {best_final_score:.4f}")
    if final_real_dr is not None:
        print(f"  Final best real DR: {final_real_dr:.3f}")
    print()
    print("Progress by round:")
    for rs in round_summaries:
        print(f"  Round {rs['round']}: surr={rs['best_surrogate_score']:.4f}  "
              f"real_DR={rs['mean_real_dr_top_k']:.3f}  "
              f"bias={rs['surrogate_bias']:+.4f}")

    print(f"\n  Results → {out_json}")
    print(f"  Refined surrogate → {args.out_dir}/saved_ml/")
    print()
    print("To run regular BO with the refined surrogate:")
    print(f"  python BO/bo_main.py --mode standard --surrogate-dir {args.out_dir}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
