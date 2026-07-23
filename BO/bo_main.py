#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GENEVO unified BO entry point.

Usage:
    python BO/bo_main.py [--mode MODE] [--n-runs N] [--n-init N] [--n-iter N]

Modes: standard (default), converged, benchmark, mobo, subtypes, topology,
       sim-validate, sobol, kd-scan, dist-shift, shap, validate, landscape,
       baseline, closed-loop, search-diag, robustness-regen
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent
_BO   = Path(__file__).parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_BO))


def _setup_logging(log_dir: Path, verbose: bool = False) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO

    console_fmt = logging.Formatter("%(message)s")
    file_fmt    = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(console_fmt)

    fh = logging.FileHandler(log_dir / "bo_optimization.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(file_fmt)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    if not root.handlers:
        root.addHandler(ch)
        root.addHandler(fh)

    return logging.getLogger("bo_main")


def _run_subprocess(cmd: list, check: bool = False) -> int:
    result = subprocess.run(cmd, text=True)
    if check and result.returncode != 0:
        print(f"[FAIL] Command exited with code {result.returncode}")
    return result.returncode


def run_standard(args) -> int:
    from BO.core.surrogate_loader import SurrogateLoaderV3
    from BO.core.build_surrogates import SurrogateBuilderV3
    from search_space.biosensor_space import BiosensorSearchSpace
    from evaluation.physics_forward_model import PhysicsForwardModel
    from evaluation.robustness_analyzer import RobustnessAnalyzer
    from evaluation.therapeutic_objective_v6 import TherapeuticObjectiveV6
    from acquisition.acquisition_functions import ExpectedImprovement
    from optimizer.gaussian_process_bo import GaussianProcessBO
    from optimizer.bo_pipeline import BOPipeline

    output_dir = args.output_dir

    if not args.data_dir.exists():
        print(f"[ERROR] Data directory not found: {args.data_dir}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logging(output_dir / "logs", verbose=args.verbose)

    logger.info("=" * 80)
    logger.info("GENEVO2 Bayesian Optimization  (mode: standard)")
    logger.info("=" * 80)
    logger.info(f"Started:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Surrogate:  BO/bo_results/saved_ml/ (rank-rho=0.8326 on data_v18)")
    logger.info(f"n_init:     {args.n_init}  |  n_iter: {args.n_iter}")
    logger.info(f"Output:     {output_dir.resolve()}")

    # Build or load surrogates
    saved_ml_dir  = args.surrogate_dir / "saved_ml"
    scaler_path   = saved_ml_dir / "scaler.pkl"
    surrogates_ok = scaler_path.exists()

    if args.retrain_surrogates or not surrogates_ok:
        logger.info("\n[1/5] Training surrogates from %s ...", args.data_dir)
        builder = SurrogateBuilderV3(logger)
        X_raw, df_results = builder.load_and_prepare_data(args.data_dir)
        X = builder.fit_scaler(X_raw)
        y_dr  = df_results["detection_rate"].values.astype(np.float32)
        y_fnr = df_results["false_negative_rate"].values.astype(np.float32)
        y_ttd = df_results["time_to_detection"].values.astype(np.float32)
        try:
            metrics = builder.train_all(X, y_dr, y_fnr, y_ttd)
            logger.info("  DR ROC-AUC: %.4f", metrics["detection_rate"]["test_roc_auc"])
            logger.info("  FNR R2:     %.4f", metrics["fnr"]["test_r2"])
            logger.info("  TTD R2:     %.4f", metrics["ttd"]["test_r2"])
            builder.save(args.surrogate_dir)
        except Exception as e:
            logger.error("Surrogate training failed: %s", e, exc_info=True)
            return 1
    else:
        logger.info("\n[1/5] Loading existing surrogates")

    try:
        surrogate_loader = SurrogateLoaderV3(args.surrogate_dir)
    except Exception as e:
        logger.error("Failed to load surrogates: %s", e)
        return 1

    # Initialize components
    logger.info("\n[2/5] Initializing BO components")
    search_space   = BiosensorSearchSpace()
    physics_model  = PhysicsForwardModel()
    objective_fn   = TherapeuticObjectiveV6(physics_model, surrogate_loader)
    robustness     = RobustnessAnalyzer(objective_fn)
    acquisition_fn = ExpectedImprovement(xi=0.01)
    logger.info("  Objective: V6 (relative threshold, sensitivity-independent)")
    logger.info(search_space.summary())

    # Run BO
    logger.info("\n[3/5] Running GP-EI Bayesian Optimization")
    optimizer = GaussianProcessBO(
        objective_fn=objective_fn,
        search_space=search_space,
        acquisition_fn=acquisition_fn,
        n_init=args.n_init,
        n_iter=args.n_iter,
        random_state=args.random_state,
    )
    pipeline = BOPipeline(
        optimizer=optimizer,
        objective_fn=objective_fn,
        search_space=search_space,
        robustness_analyzer=robustness,
        output_dir=output_dir,
    )
    result = pipeline.run()

    # Summary
    logger.info("\n[4/5] Results")
    best_cfg   = result["bo_result"]["config_best"]
    best_score = result["bo_result"]["y_best"]
    details    = result.get("details", {})
    logger.info("  Best composite score : %.4f", best_score)
    logger.info("  DR PMO-mild          : %.4f", details.get("dr_mild",  0.0))
    logger.info("  DR PMO               : %.4f", details.get("dr_pmo",   0.0))
    logger.info("  DR CKD               : %.4f", details.get("dr_ckd",   0.0))
    logger.info("  FP (healthy)         : %.4f", details.get("dr_healthy", 0.0))
    logger.info("  FNR mean             : %.4f", details.get("fnr_mean", 0.0))
    logger.info("\n  Best biosensor design:")
    for k, v in best_cfg.items():
        if isinstance(v, float):
            logger.info("    %-22s %.4f", k, v)
        else:
            logger.info("    %-22s %s", k, v)
    logger.info("\nResults: %s", output_dir / "results" / "best_config.json")
    logger.info("=" * 80)
    return 0


def run_mobo(args) -> int:
    from BO.core.surrogate_loader import SurrogateLoaderV3
    from mobo.mobo_pipeline import MOBOPipeline

    out_dir = args.output_dir / "mobo" if str(args.output_dir) == "BO/bo_results" else args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = _setup_logging(out_dir / "logs", verbose=args.verbose)
    logger.info("=" * 80)
    logger.info("GENEVO2 Multi-Objective BO  (mode: mobo)")
    logger.info("=" * 80)

    loader = SurrogateLoaderV3(args.surrogate_dir)

    # Warm start: seed MOBO with best known config to guarantee a feasible init point
    warm_configs = []
    best_cfg_path = args.surrogate_dir / "results" / "best_config.json"
    if best_cfg_path.exists():
        try:
            with open(best_cfg_path) as _f:
                _saved = json.load(_f)
            bd = _saved.get("biosensor_design", {})
            warm_configs.append({
                "biosensor_type": "array",
                "noise_preset":   "realistic",
                "target_scenario": "pmo",
                "response_time_s": 600.0,
                "kd_nm":       float(bd.get("kd_nm",       1.0)),
                "sensitivity": float(bd.get("sensitivity", 1.0)),
                "kd_ctx_nm":   float(bd.get("kd_ctx_nm",  1.0)),
                "kd_p1np_nm":  float(bd.get("kd_p1np_nm", 1.0)),
                "w_ctx":       float(bd.get("w_ctx",   0.1)),
                "w_p1np":      float(bd.get("w_p1np",  0.1)),
            })
            logger.info(f"MOBO warm start loaded from {best_cfg_path}")
        except Exception as _e:
            logger.warning(f"Could not load warm start config: {_e}")

    pipeline = MOBOPipeline(
        surrogate_loader=loader,
        n_init=args.n_init,
        n_iterations=args.n_iter,
        seed=args.random_state,
        output_dir=str(out_dir),
        warm_start_configs=warm_configs,
    )
    result = pipeline.run()
    pipeline.print_pareto_summary(result)
    logger.info("\n[OK] MOBO complete. %d Pareto solutions.", len(result.pareto_configs))
    return 0



def run_baseline(args) -> int:
    """Logistic regression on raw biomarker values. Answers: is 0.967 DR trivial?"""
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logging(out_dir / "logs", verbose=args.verbose)

    logger.info("=" * 80)
    logger.info("GENEVO2 Baseline Task Difficulty  (mode: baseline)")
    logger.info("  Question: What AUC does a logistic classifier get on SOST alone?")
    logger.info("  If AUC > 0.92 the detection task is trivially easy.")
    logger.info("=" * 80)

    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import roc_auc_score

    csv = args.data_dir / "master_index.csv"
    if not csv.exists():
        print(f"[ERROR] master_index.csv not found: {csv}")
        return 1

    df = pd.read_csv(csv)
    required = ["scenario", "sclerostin_mean"]
    if not all(c in df.columns for c in required):
        print(f"[ERROR] Required columns missing. Got: {list(df.columns)}")
        return 1

    # Binary: healthy vs disease
    df["label_bin"] = (df["scenario"] != "healthy").astype(int)
    # 4-class: healthy / pmo_mild / pmo / ckd_mbd
    le4 = LabelEncoder()
    df["label_4"] = le4.fit_transform(df["scenario"])

    feature_sets = {
        "SOST only":        ["sclerostin_mean"],
        "SOST + CTX":       [c for c in ["sclerostin_mean", "ctx_mean"] if c in df.columns],
        "SOST + P1NP":      [c for c in ["sclerostin_mean", "p1np_mean"] if c in df.columns],
        "SOST + CTX + P1NP": [c for c in ["sclerostin_mean", "ctx_mean", "p1np_mean"] if c in df.columns],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = LogisticRegression(max_iter=1000, C=1.0)

    results = {}
    print("\n--- Binary classification (healthy vs disease) ---")
    print(f"  {'Feature set':<25}  {'AUC':>7}  {'Acc':>7}")
    print(f"  {'-'*25}  {'-'*7}  {'-'*7}")

    for name, feats in feature_sets.items():
        if len(feats) < 1:
            continue
        X = df[feats].values
        y = df["label_bin"].values

        auc_scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
        acc_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        results[name] = {
            "binary_auc_mean": float(auc_scores.mean()),
            "binary_auc_std":  float(auc_scores.std()),
            "binary_acc_mean": float(acc_scores.mean()),
        }
        print(f"  {name:<25}  {auc_scores.mean():.4f}  {acc_scores.mean():.4f}")

    # Per-class analysis for SOST only
    if "ctx_mean" in df.columns and "p1np_mean" in df.columns:
        X_all = df[["sclerostin_mean", "ctx_mean", "p1np_mean"]].values
    else:
        X_all = df[["sclerostin_mean"]].values

    print("\n--- 4-class (healthy/pmo_mild/pmo/ckd_mbd) ---")
    print(f"  {'Feature set':<25}  {'Acc':>7}")
    print(f"  {'-'*25}  {'-'*7}")
    acc4 = cross_val_score(clf, X_all, df["label_4"].values, cv=cv, scoring="accuracy")
    print(f"  {'SOST+CTX+P1NP':<25}  {acc4.mean():.4f}")
    results["4class_acc"] = float(acc4.mean())

    # PMO-mild specifically (hardest class)
    df_disease = df[df["scenario"] != "healthy"].copy()
    df_disease["is_mild"] = (df_disease["scenario"] == "pmo_mild").astype(int)
    if "ctx_mean" in df.columns and "p1np_mean" in df.columns:
        X_mild = df_disease[["sclerostin_mean", "ctx_mean", "p1np_mean"]].values
    else:
        X_mild = df_disease[["sclerostin_mean"]].values
    y_mild = df_disease["is_mild"].values
    if y_mild.sum() > 10:
        auc_mild = cross_val_score(clf, X_mild, y_mild,
                                   cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                                   scoring="roc_auc")
        results["mild_auc_mean"] = float(auc_mild.mean())
        print(f"\n  PMO-mild vs other disease: AUC = {auc_mild.mean():.4f}")

    # Verdict
    sost_auc = results.get("SOST only", {}).get("binary_auc_mean", 0.0)
    all_auc  = results.get("SOST + CTX + P1NP", {}).get("binary_auc_mean", sost_auc)
    print("\n--- Verdict ---")
    if sost_auc > 0.92:
        verdict = "TRIVIALLY EASY (SOST alone AUC > 0.92). Biosensor optimization primarily adds safety/efficiency value, not detection capability."
    elif all_auc > 0.92:
        verdict = "MODERATELY EASY with all 3 biomarkers (AUC > 0.92). SOST alone is harder, showing sensor design adds value."
    else:
        verdict = "NON-TRIVIAL detection (AUC <= 0.92). Biosensor optimization genuinely improves performance."
    print(f"  {verdict}")
    results["verdict"] = verdict

    # Save
    out_path = out_dir / "baseline_auc_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Results saved: {out_path}")
    return 0



def run_landscape(args) -> int:
    """10k LHS samples scored by surrogate. Checks BO landscape and local optima."""
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logging(out_dir / "logs", verbose=args.verbose)

    logger.info("=" * 80)
    logger.info("GENEVO2 Global Landscape Audit  (mode: landscape)")
    logger.info("  Samples 10,000 configs via LHS and scores with v18 surrogate.")
    logger.info("  Answers: is the BO optimum a local optimum or global?")
    logger.info("=" * 80)

    from scipy.stats import qmc
    from BO.core.surrogate_loader import SurrogateLoaderV3
    from evaluation.physics_forward_model import PhysicsForwardModel
    from evaluation.therapeutic_objective_v6 import TherapeuticObjectiveV6
    from search_space.biosensor_space import BiosensorSearchSpace

    loader  = SurrogateLoaderV3(args.surrogate_dir)
    phys    = PhysicsForwardModel()
    v6      = TherapeuticObjectiveV6(phys, loader)
    space   = BiosensorSearchSpace()

    n_samples = args.n_lhs
    print(f"\n  Sampling {n_samples:,} configs via LHS ...")

    sampler = qmc.LatinHypercube(d=space.n_params, seed=42)
    X_unit  = sampler.random(n=n_samples)

    scores = []
    configs = []
    for i, x in enumerate(X_unit):
        cfg   = space.vector_to_dict(x)
        score = v6(cfg)
        scores.append(score)
        configs.append(cfg)
        if (i + 1) % 1000 == 0:
            print(f"    {i+1:6d}/{n_samples}  best_so_far={max(scores):.4f}", flush=True)

    scores = np.array(scores)
    top_pct = 0.01  # top 1%
    top_k   = max(10, int(n_samples * top_pct))
    top_idx = np.argsort(scores)[-top_k:][::-1]
    top_scores  = scores[top_idx]
    top_configs = [configs[i] for i in top_idx]

    # Score statistics
    print(f"\n  LHS landscape statistics ({n_samples:,} samples):")
    print(f"    Mean score : {scores.mean():.4f}")
    print(f"    Std score  : {scores.std():.4f}")
    print(f"    P95 score  : {np.percentile(scores, 95):.4f}")
    print(f"    Max score  : {scores.max():.4f}")
    print(f"    Top 1% (n={top_k}) mean : {top_scores.mean():.4f}")

    # Parameter distributions in top 1%
    print(f"\n  Top 1% parameter distributions:")
    for param in ["kd_nm", "sensitivity", "kd_ctx_nm", "kd_p1np_nm", "w_ctx", "w_p1np"]:
        vals = [c.get(param, 0.0) for c in top_configs if isinstance(c.get(param), float)]
        if vals:
            print(f"    {param:<16}  mean={np.mean(vals):.3f}  std={np.std(vals):.3f}"
                  f"  min={np.min(vals):.3f}  max={np.max(vals):.3f}")

    # Compare against BO best (if available)
    bo_best_path = args.surrogate_dir.parent / "bo_results" / "results" / "best_config.json"
    if bo_best_path.exists():
        with open(bo_best_path) as f:
            bo_cfg_raw = json.load(f)
        # Flatten biosensor_design + measurement_environment; rename "type" → "biosensor_type"
        bo_cfg = {**bo_cfg_raw.get("biosensor_design", bo_cfg_raw),
                  **bo_cfg_raw.get("measurement_environment", {})}
        if "type" in bo_cfg and "biosensor_type" not in bo_cfg:
            bo_cfg["biosensor_type"] = bo_cfg.pop("type")
        bo_score = v6(bo_cfg)
        pct_rank = 100.0 * (scores < bo_score).sum() / len(scores)
        print(f"\n  BO best score vs LHS landscape:")
        print(f"    BO score   : {bo_score:.4f}")
        print(f"    LHS max    : {scores.max():.4f}")
        print(f"    BO pct rank: {pct_rank:.1f}% (is BO in top {100-pct_rank:.1f}% of landscape?)")
        if pct_rank > 95:
            verdict = "BO found globally competitive region (top 5% of landscape)."
        elif pct_rank > 80:
            verdict = "BO is above average but not globally optimal — further exploration warranted."
        else:
            verdict = "BO may be stuck in local optimum. Consider restart with different acquisition."
        print(f"    Verdict: {verdict}")
    else:
        verdict = "No BO reference result found for comparison."
        print(f"\n  {verdict}")

    # Save results
    out = {
        "n_samples":       n_samples,
        "scores_summary": {
            "mean": float(scores.mean()),
            "std":  float(scores.std()),
            "p95":  float(np.percentile(scores, 95)),
            "max":  float(scores.max()),
        },
        "top1pct_mean_score": float(top_scores.mean()),
        "top_configs_top5": top_configs[:5],
        "top_scores_top5":  top_scores[:5].tolist(),
        "verdict": verdict,
    }
    out_path = out_dir / "landscape_audit.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[OK] Landscape audit saved: {out_path}")
    return 0



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GENEVO2 Unified BO Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode selector
    p.add_argument(
        "--mode",
        choices=["standard", "converged", "subtypes", "topology", "mobo",
                 "benchmark", "closed-loop", "baseline", "kd-scan", "landscape", "validate",
                 "sim-validate", "sobol", "search-diag", "dist-shift", "shap", "robustness-regen"],
        default="standard",
        help="Optimization mode (default: standard)",
    )

    # Shared paths
    p.add_argument("--data-dir",      type=Path, default=Path("data_v19"),
                   help="Training data directory (default: data_v19)")
    p.add_argument("--surrogate-dir", type=Path, default=Path("BO/bo_results"),
                   help="Surrogate models directory (default: BO/bo_results)")
    p.add_argument("--output-dir",    type=Path, default=None,
                   help="Output directory (default: mode-specific under BO/bo_results/)")

    # BO parameters
    p.add_argument("--n-init",      type=int, default=50,
                   help="LHS initial samples (default: 50)")
    p.add_argument("--n-iter",      type=int, default=150,
                   help="BO iterations (default: 150)")
    p.add_argument("--random-state", type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--retrain-surrogates", action="store_true",
                   help="Force surrogate retraining")

    # Multi-run parameters
    p.add_argument("--n-runs", type=int, default=5,
                   help="Number of independent runs (for converged/benchmark modes, default: 5)")

    # Landscape mode
    p.add_argument("--n-lhs", type=int, default=10000,
                   help="LHS samples for landscape audit (default: 10000)")

    # Subtypes
    p.add_argument("--subtypes", nargs="+",
                   choices=["young_pmo", "elderly_pmo", "ckd_controlled", "ckd_advanced"],
                   default=None,
                   help="Subtypes to run (default: all four)")

    p.add_argument("--verbose", "-v", action="store_true")

    return p.parse_args()


def _default_output_dir(mode: str) -> Path:
    base = Path("BO/bo_results")
    mapping = {
        "standard":         base,
        "converged":        base / "convergence",
        "subtypes":         base / "subtypes",
        "topology":         base / "topology",
        "mobo":             base / "mobo",
        "benchmark":        base / "benchmark",
        "closed-loop":      base / "closed_loop",
        "baseline":         base / "diagnostics",
        "kd-scan":          base / "diagnostics",
        "landscape":        base / "diagnostics",
        "validate":         base / "diagnostics",
        "sim-validate":     base / "diagnostics",
        "sobol":            base / "diagnostics",
        "search-diag":      base / "diagnostics",
        "dist-shift":       base / "diagnostics",
        "shap":             base / "diagnostics",
        "robustness-regen": base / "convergence",
    }
    return mapping.get(mode, base)


def main() -> int:
    args = parse_args()

    # Resolve output dir
    if args.output_dir is None:
        args.output_dir = _default_output_dir(args.mode)

    # ---- dispatch ----

    if args.mode == "standard":
        return run_standard(args)

    elif args.mode == "converged":
        cmd = [
            sys.executable, str(_BO / "bo_converged.py"),
            "--n-init",       str(args.n_init),
            "--n-iter",       str(args.n_iter),
            "--n-runs",       str(args.n_runs),
            "--surrogate-dir", str(args.surrogate_dir),
            "--output-dir",   str(args.output_dir),
        ]
        if args.verbose:
            cmd.append("--verbose")
        return _run_subprocess(cmd)

    elif args.mode == "subtypes":
        cmd = [
            sys.executable, str(_BO / "bo_patient_subtypes.py"),
            "--n-init",        str(args.n_init),
            "--n-iter",        str(args.n_iter),
            "--seed",          str(args.random_state),
            "--surrogate-dir", str(args.surrogate_dir),
            "--out-dir",       str(args.output_dir),
        ]
        if args.subtypes:
            cmd += ["--subtypes"] + args.subtypes
        if args.verbose:
            cmd.append("--verbose")
        return _run_subprocess(cmd)

    elif args.mode == "topology":
        cmd = [
            sys.executable, str(_BO / "bo_topology_search.py"),
            "--data-dir",      str(args.data_dir),
            "--surrogate-dir", str(args.surrogate_dir),
            "--output-dir",    str(args.output_dir),
            "--n-init",        str(args.n_init),
            "--n-iter",        str(args.n_iter),
            "--random-state",  str(args.random_state),
        ]
        if args.verbose:
            cmd.append("--verbose")
        return _run_subprocess(cmd)

    elif args.mode == "mobo":
        return run_mobo(args)

    elif args.mode == "benchmark":
        budget = args.n_init + args.n_iter
        cmd = [
            sys.executable, str(_BO / "benchmarks" / "multi_optimizer_benchmark.py"),
            "--surrogate-dir", str(args.surrogate_dir),
            "--runs",    str(args.n_runs),
            "--budget",  str(budget),
            "--n-init",  str(args.n_init),
            "--output",  str(args.output_dir / "benchmark_results.json"),
        ]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        return _run_subprocess(cmd)

    elif args.mode == "closed-loop":
        cmd = [
            sys.executable, str(_BO / "bo_closed_loop.py"),
            "--n-rounds",      "5",
            "--n-init",        str(args.n_init),
            "--n-inner",       str(args.n_iter),
            "--data-dir",      str(args.data_dir),
            "--surrogate-dir", str(args.surrogate_dir),
            "--out-dir",       str(args.output_dir),
        ]
        return _run_subprocess(cmd)

    elif args.mode == "baseline":
        return run_baseline(args)

    elif args.mode == "kd-scan":
        cmd = [
            sys.executable, str(_BO / "analysis" / "kd_ctx_scan.py"),
            "--n-trials", "20",
            "--out", str(args.output_dir / "kd_ctx_scan_results.json"),
        ]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        return _run_subprocess(cmd)

    elif args.mode == "landscape":
        return run_landscape(args)

    elif args.mode == "validate":
        cmd = [
            sys.executable, str(_BO / "analysis" / "rank_rho_v18.py"),
            "--out", str(args.output_dir / "rank_rho_results.json"),
        ]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        return _run_subprocess(cmd)

    elif args.mode == "sim-validate":
        cmd = [
            sys.executable, str(_BO / "analysis" / "validate_best_config.py"),
            "--convergence-report", str(Path("BO/bo_results/convergence/convergence_report.json")),
            "--seed-config", str(args.random_state if args.random_state != 42 else 888),
            "--n-trials", "20",
            "--out", str(args.output_dir / "best_config_validation.json"),
        ]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        return _run_subprocess(cmd)

    elif args.mode == "sobol":
        cmd = [
            sys.executable, str(_BO / "analysis" / "sobol_sensitivity.py"),
            "--surrogate-dir", str(args.surrogate_dir),
            "--N", str(getattr(args, "n_lhs", 512)),
            "--out", str(args.output_dir / "sobol_results.json"),
        ]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        return _run_subprocess(cmd)

    elif args.mode == "search-diag":
        cmd = [
            sys.executable, str(_BO / "analysis" / "search_space_diagnostics.py"),
            "--surrogate-dir", str(args.surrogate_dir),
            "--convergence-report", str(Path("BO/bo_results/convergence/convergence_report.json")),
            "--out", str(args.output_dir / "search_space_diagnostics.json"),
        ]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        return _run_subprocess(cmd)

    elif args.mode == "dist-shift":
        cmd = [
            sys.executable, str(_BO / "analysis" / "distribution_shift_test.py"),
            "--surrogate-dir", str(args.surrogate_dir),
            "--convergence-report", str(Path("BO/bo_results/convergence/convergence_report.json")),
            "--seed-config", str(args.random_state if args.random_state != 42 else 888),
            "--out", str(args.output_dir / "distribution_shift_results.json"),
        ]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        return _run_subprocess(cmd)

    elif args.mode == "shap":
        cmd = [
            sys.executable, str(_BO / "analysis" / "surrogate_shap.py"),
            "--data-dir",      str(args.data_dir),
            "--surrogate-dir", str(args.surrogate_dir),
            "--out", str(args.output_dir / "surrogate_interpretability.json"),
        ]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        return _run_subprocess(cmd)

    elif args.mode == "robustness-regen":
        cmd = [
            sys.executable, str(_BO / "analysis" / "regenerate_robustness.py"),
            "--convergence-report", str(Path("BO/bo_results/convergence/convergence_report.json")),
            "--n-trials", "10",
            "--out-patch", str(args.output_dir / "convergence_robustness_patch.json"),
        ]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        return _run_subprocess(cmd)

    else:
        print(f"[ERROR] Unknown mode: {args.mode}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
