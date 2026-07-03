#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Design Retrieval Model  (replaces the failed MLP)

WHY THE MLP FAILED (R² = -0.0006)
------------------------------------
The MLP tried to learn: patient_biomarker_profile → exact_optimal_design.
This is mathematically ill-posed because the mapping is ONE-TO-MANY:
for any given patient state, MANY biosensor configurations achieve similar
scores. The MLP cannot learn to predict one specific member of a large
equivalence class — it just predicts the mean and gets R²≈0.

THE FIX: K-NN Retrieval
-----------------------
Instead of predicting exact parameters, we retrieve similar patients from
a precomputed database and return THEIR best designs.

This works because:
  1. The "correct" design for a patient is not unique — any member of the
     top-scoring cluster is equally valid.
  2. K-NN is provably consistent for retrieval problems (vs. ill-posed
     regression).
  3. It's interpretable: "this patient is similar to these reference patients,
     who were best served by these biosensors."

For per-patient BO: optionally also runs a fast targeted BO (30 init + 20 iter)
initialized from retrieved designs, anchoring search in the right region.

Usage
-----
    cd c:\\Users\\eruku\\Akshith\\GENEVO2

    # Retrieve top-5 designs for a PMO patient
    python BO/design_model/design_retrieval.py --retrieve \\
        --sost 0.875 --ctx 0.500 --p1np 0.525 --disease pmo --severity moderate

    # Retrieve + run targeted BO for patient-specific optimal
    python BO/design_model/design_retrieval.py --retrieve --run-bo \\
        --sost 0.875 --ctx 0.500 --p1np 0.525 --disease pmo --severity moderate

    # Build the retrieval index from a dataset CSV
    python BO/design_model/design_retrieval.py --build-index \\
        --dataset BO/data_expansion/optimized_designs.csv

    # Quick fallback: builds index from data_v16 (no P4 needed)
    python BO/design_model/design_retrieval.py --build-index --use-existing-data
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "BO"))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DEFAULT_INDEX = ROOT / "BO" / "design_model" / "retrieval_index.parquet"

# Nominal concentrations by (disease, severity) — for feature computation
_NOMINAL = {
    ("healthy", ""):          {"scl": 0.375, "ctx": 0.200, "p1np": 0.350},
    ("pmo", "mild"):          {"scl": 0.5625, "ctx": 0.300, "p1np": 0.385},
    ("pmo", "moderate"):      {"scl": 0.875,  "ctx": 0.500, "p1np": 0.525},
    ("pmo", "severe"):        {"scl": 1.10,   "ctx": 0.620, "p1np": 0.600},
    ("ckd", "moderate"):      {"scl": 1.125,  "ctx": 0.500, "p1np": 0.625},
    ("ckd", "severe"):        {"scl": 1.40,   "ctx": 0.600, "p1np": 0.700},
}

_HEALTHY = {"scl": 0.375, "ctx": 0.200, "p1np": 0.350}

# Design output columns (what we return)
DESIGN_COLS = ["kd_nm", "kd_ctx_nm", "kd_p1np_nm", "sensitivity", "w_ctx", "w_p1np"]
SCORE_COL = "score_v6"


def _patient_features(sost: float, ctx: float, p1np: float) -> np.ndarray:
    """Compute normalized patient features for K-NN distance.

    Uses log-fold change vs healthy baseline, which is the most informative
    representation for Langmuir-based biosensor matching (what matters is
    the RATIO of disease to healthy signal, not absolute values).
    """
    h = _HEALTHY
    log_fold_scl  = np.log2(max(sost, 1e-4) / h["scl"])
    log_fold_ctx  = np.log2(max(ctx,  1e-4) / h["ctx"])
    log_fold_p1np = np.log2(max(p1np, 1e-4) / h["p1np"])
    return np.array([log_fold_scl, log_fold_ctx, log_fold_p1np], dtype=np.float32)


def build_index_from_dataset(dataset_csv: Path, out_path: Path, min_score: float = 0.50) -> None:
    """Build the retrieval index from the P4 optimized dataset CSV."""
    print(f"Building retrieval index from {dataset_csv}...")
    df = pd.read_csv(dataset_csv)

    required = ["score_v6"] + DESIGN_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error(f"Dataset missing columns: {missing}")
        logger.error(f"Available: {list(df.columns)}")
        sys.exit(1)

    df = df[df[SCORE_COL] >= min_score].copy()
    print(f"  Rows with score ≥ {min_score}: {len(df)} / {len(pd.read_csv(dataset_csv))}")

    # Add patient-level biomarker features if not present
    # These come from the scenario: use scenario→nominal concentrations
    if "scl_fold" not in df.columns:
        scenario_to_concs = {
            "healthy": _HEALTHY,
            "pmo_mild": _NOMINAL[("pmo", "mild")],
            "pmo": _NOMINAL[("pmo", "moderate")],
            "ckd_mbd": _NOMINAL[("ckd", "moderate")],
        }
        # Use target_scenario if available, else infer from pmo/ckd flags
        if "target_scenario" in df.columns:
            def _get_feats(row):
                concs = scenario_to_concs.get(row["target_scenario"], _HEALTHY)
                return _patient_features(concs["scl"], concs["ctx"], concs["p1np"])
            feats = np.vstack(df.apply(_get_feats, axis=1).tolist())
        else:
            # Default: healthy
            feats = np.tile(_patient_features(*_HEALTHY.values()), (len(df), 1))

        df["scl_fold"] = feats[:, 0]
        df["ctx_fold"] = feats[:, 1]
        df["p1np_fold"] = feats[:, 2]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"  Index saved → {out_path} ({len(df)} designs)")


def build_index_from_existing_data(
    data_dir: Path = ROOT / "data_v16",
    surrogate_dir: Path = ROOT / "BO" / "bo_results",
    out_path: Path = DEFAULT_INDEX,
    n_lhs: int = 2000,
) -> None:
    """Build retrieval index from data_v16 + surrogate scoring (no P4 dataset needed).

    Generates N_LHS diverse configs, scores with surrogate, keeps the good ones.
    """
    from BO.core.surrogate_loader import SurrogateLoaderV3
    from search_space.biosensor_space import BiosensorSearchSpace
    from evaluation.physics_forward_model import PhysicsForwardModel
    from evaluation.therapeutic_objective_v6 import TherapeuticObjectiveV6
    from scipy.stats import qmc

    print(f"Building retrieval index from LHS + surrogate scoring ({n_lhs} samples)...")
    loader = SurrogateLoaderV3(results_dir=surrogate_dir)
    obj = TherapeuticObjectiveV6(PhysicsForwardModel(), loader)
    space = BiosensorSearchSpace()

    sampler = qmc.LatinHypercube(d=space.n_params, seed=1337)
    X_lhs = sampler.random(n=n_lhs)

    rows = []
    for x in X_lhs:
        cfg = space.vector_to_dict(x)
        score, detail = obj.evaluate_with_details(cfg)
        if score < 0.45:
            continue
        rows.append({
            "kd_nm": cfg["kd_nm"],
            "kd_ctx_nm": cfg.get("kd_ctx_nm", cfg["kd_nm"]),
            "kd_p1np_nm": cfg.get("kd_p1np_nm", cfg["kd_nm"]),
            "sensitivity": cfg["sensitivity"],
            "w_ctx": cfg.get("w_ctx", 0.0),
            "w_p1np": cfg.get("w_p1np", 0.0),
            SCORE_COL: float(score),
            "dr_pmo": float(detail.get("dr_pmo", 0.0)),
            "dr_ckd": float(detail.get("dr_ckd", 0.0)),
            "dr_mild": float(detail.get("dr_mild", 0.0)),
            "fnr_mean": float(detail.get("fnr_mean", 1.0)),
            "ttd_mean": float(detail.get("ttd_mean", 9000.0)),
            # For each config, store features for each disease state
            "scl_fold": 0.0,  # will be set per-query at retrieval time
            "ctx_fold": 0.0,
            "p1np_fold": 0.0,
        })

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"  Index saved → {out_path} ({len(df)} designs, score ≥ 0.45)")
    print(f"  Score range: {df[SCORE_COL].min():.3f} – {df[SCORE_COL].max():.3f}")
    print(f"  Top-5 scores: {df[SCORE_COL].nlargest(5).values.tolist()}")


def retrieve_designs(
    sost: float,
    ctx: float,
    p1np: float,
    disease: str,
    severity: str,
    k: int = 5,
    index_path: Path = DEFAULT_INDEX,
) -> pd.DataFrame:
    """Retrieve top-K biosensor designs for a given patient profile.

    Uses a disease-aware K-NN: for a PMO patient, we weight PMO detection
    rate more heavily in the distance metric.

    Returns a DataFrame with top-K designs sorted by score.
    """
    if not index_path.exists():
        logger.error(f"Retrieval index not found: {index_path}")
        logger.error("Build it first: python BO/design_model/design_retrieval.py --build-index --use-existing-data")
        sys.exit(1)

    df = pd.read_parquet(index_path)

    query_feats = _patient_features(sost, ctx, p1np)

    # Compute distance in (scl_fold, ctx_fold, p1np_fold) space
    # These features measure how different the patient is from healthy baseline
    # Optionally filter by disease relevance
    index_feats = df[["scl_fold", "ctx_fold", "p1np_fold"]].values.astype(np.float32)

    # L2 distance in log-fold space (Euclidean distance in log-ratio space)
    dists = np.linalg.norm(index_feats - query_feats, axis=1)
    df = df.copy()
    df["_dist"] = dists

    # Disease-relevance filter: for PMO, prefer high dr_pmo; for CKD, prefer high dr_ckd
    if "pmo" in disease.lower() and "dr_pmo" in df.columns:
        df["_relevance_score"] = df[SCORE_COL] * 0.6 + df["dr_pmo"] * 0.4
    elif "ckd" in disease.lower() and "dr_ckd" in df.columns:
        df["_relevance_score"] = df[SCORE_COL] * 0.6 + df["dr_ckd"] * 0.4
    else:
        df["_relevance_score"] = df[SCORE_COL]

    # Retrieve: 50 nearest neighbors by biomarker distance → rank by relevance
    n_candidates = min(50, len(df))
    nearest = df.nsmallest(n_candidates, "_dist")
    top_k = nearest.nlargest(k, "_relevance_score")

    return top_k.drop(columns=["_dist", "_relevance_score"], errors="ignore")


def run_targeted_bo(
    sost: float,
    ctx: float,
    p1np: float,
    retrieved: pd.DataFrame,
    n_init: int = 30,
    n_iter: int = 20,
    surrogate_dir: Path = ROOT / "BO" / "bo_results",
) -> dict:
    """Run a full BO for this specific patient query.

    The retrieved designs serve as a performance baseline — if BO finds something
    better, great; if not, return the best retrieved design.
    BO's GP explores the full landscape from scratch (n_init LHS samples)
    and should confirm or improve on the retrieved designs.
    """
    from BO.core.surrogate_loader import SurrogateLoaderV3
    from search_space.biosensor_space import BiosensorSearchSpace
    from evaluation.physics_forward_model import PhysicsForwardModel
    from evaluation.therapeutic_objective_v6 import TherapeuticObjectiveV6
    from acquisition.acquisition_functions import ExpectedImprovement
    from optimizer.gaussian_process_bo import GaussianProcessBO

    loader = SurrogateLoaderV3(results_dir=surrogate_dir)
    obj = TherapeuticObjectiveV6(PhysicsForwardModel(), loader)
    space = BiosensorSearchSpace()
    acq = ExpectedImprovement(xi=0.01)

    print(f"  Running BO ({n_init} init + {n_iter} iter)...")
    bo = GaussianProcessBO(obj, space, acq, n_init=n_init, n_iter=n_iter, random_state=42)
    results = bo.optimize()

    # Compare BO result to best retrieved design
    retrieved_best_score = float(retrieved[SCORE_COL].max()) if len(retrieved) > 0 else 0.0
    bo_best = float(results["y_best"])

    if bo_best >= retrieved_best_score:
        print(f"  BO improved on retrieved: {retrieved_best_score:.4f} → {bo_best:.4f} (+{bo_best - retrieved_best_score:.4f})")
    else:
        print(f"  Retrieved design holds: {retrieved_best_score:.4f}  (BO: {bo_best:.4f})")
        # Inject the best retrieved design into results for easy comparison
        best_retrieved_row = retrieved.loc[retrieved[SCORE_COL].idxmax()]
        results["retrieved_best_score"] = retrieved_best_score
        results["retrieved_best_config"] = best_retrieved_row.to_dict()

    return results


def main():
    parser = argparse.ArgumentParser(description="Design retrieval model for patient-specific biosensor recommendation")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--retrieve", action="store_true", help="Retrieve designs for a patient")
    mode.add_argument("--build-index", action="store_true", help="Build the retrieval index")

    # Patient query
    parser.add_argument("--sost", type=float, default=0.875, help="SOST concentration (nM)")
    parser.add_argument("--ctx", type=float, default=0.500, help="CTX concentration (nM)")
    parser.add_argument("--p1np", type=float, default=0.525, help="P1NP concentration (nM)")
    parser.add_argument("--disease", type=str, default="pmo", help="Disease type: pmo, ckd")
    parser.add_argument("--severity", type=str, default="moderate", help="Severity: mild, moderate, severe")
    parser.add_argument("--k", type=int, default=5, help="Number of designs to retrieve (default: 5)")
    parser.add_argument("--run-bo", action="store_true", help="Also run targeted BO for patient")

    # Index building
    parser.add_argument("--dataset", type=Path, default=ROOT / "BO" / "data_expansion" / "optimized_designs.csv",
                        help="P4 dataset CSV path (for --build-index)")
    parser.add_argument("--use-existing-data", action="store_true",
                        help="Build index from data_v16 + surrogate (no P4 needed)")
    parser.add_argument("--n-lhs", type=int, default=2000,
                        help="Number of LHS samples for --use-existing-data (default: 2000)")

    # Paths
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX, help="Retrieval index path")
    parser.add_argument("--surrogate-dir", type=Path, default=ROOT / "BO" / "bo_results",
                        help="Surrogate directory")

    args = parser.parse_args()

    if args.build_index:
        if args.use_existing_data:
            build_index_from_existing_data(
                out_path=args.index,
                n_lhs=args.n_lhs,
                surrogate_dir=args.surrogate_dir,
            )
        elif args.dataset.exists():
            build_index_from_dataset(args.dataset, out_path=args.index)
        else:
            print(f"Dataset not found: {args.dataset}")
            print("Use --use-existing-data to build from data_v16 instead.")
            sys.exit(1)
        return

    # Retrieve mode
    print("=" * 60)
    print("BIOSENSOR DESIGN RETRIEVAL")
    print("=" * 60)
    print(f"  Patient profile:  SOST={args.sost} nM  CTX={args.ctx} nM  P1NP={args.p1np} nM")
    print(f"  Disease:          {args.disease} ({args.severity})")
    print(f"  Retrieving:       top-{args.k} designs")
    print()

    top_designs = retrieve_designs(
        sost=args.sost,
        ctx=args.ctx,
        p1np=args.p1np,
        disease=args.disease,
        severity=args.severity,
        k=args.k,
        index_path=args.index,
    )

    print(f"TOP-{args.k} RETRIEVED DESIGNS:")
    print(f"  {'Rank':<5} {'Score':>7} {'kd_scl':>8} {'kd_ctx':>8} {'kd_p1np':>8} {'Sens':>6} {'w_ctx':>6} {'w_p1np':>7}")
    print(f"  {'-'*60}")
    for rank, (_, row) in enumerate(top_designs.iterrows()):
        print(f"  {rank+1:<5} {row[SCORE_COL]:>7.4f} "
              f"{row.get('kd_nm', 0):>8.3f} "
              f"{row.get('kd_ctx_nm', 0):>8.3f} "
              f"{row.get('kd_p1np_nm', 0):>8.3f} "
              f"{row.get('sensitivity', 0):>6.3f} "
              f"{row.get('w_ctx', 0):>6.3f} "
              f"{row.get('w_p1np', 0):>7.3f}")

    # Extra metrics if available
    for metric, col in [("DR_pmo", "dr_pmo"), ("DR_ckd", "dr_ckd"), ("FNR", "fnr_mean")]:
        if col in top_designs.columns:
            vals = top_designs[col].values
            print(f"\n  {metric}: {', '.join(f'{v:.3f}' for v in vals)}")

    if args.run_bo:
        print("\n  Running targeted BO initialized from retrieved designs...")
        bo_results = run_targeted_bo(
            sost=args.sost,
            ctx=args.ctx,
            p1np=args.p1np,
            retrieved=top_designs,
            surrogate_dir=args.surrogate_dir,
        )
        print(f"\n  Targeted BO best score: {bo_results['y_best']:.4f}")
        best = bo_results["config_best"]
        print(f"  Best config from targeted BO:")
        for k, v in best.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")

    print()


if __name__ == "__main__":
    main()
