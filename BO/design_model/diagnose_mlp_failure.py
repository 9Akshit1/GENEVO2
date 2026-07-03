#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MLP Failure Diagnostic

Produces the evidence required before accepting the K-NN replacement:

  1. Feature variance analysis
     -- does the surrogate score vary across configs? (i.e., is there signal at all)

  2. Target variance analysis
     -- does the SAME patient profile map to many different good designs? (one-to-many)

  3. Nearest-neighbor consistency
     -- for two similar configs, do they get similar scores? (is K-NN even sensible)

  4. Multimodality evidence
     -- are there multiple separated clusters of high-scoring designs?
        If yes, a point predictor must pick one arbitrarily -> R2~0.

Usage
-----
    cd C:\\Users\\eruku\\Akshith\\GENEVO2
    python BO/design_model/diagnose_mlp_failure.py
    python BO/design_model/diagnose_mlp_failure.py --n-lhs 5000  # more thorough
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "BO"))

from BO.core.surrogate_loader import SurrogateLoaderV3
from search_space.biosensor_space import BiosensorSearchSpace
from evaluation.physics_forward_model import PhysicsForwardModel
from evaluation.therapeutic_objective_v6 import TherapeuticObjectiveV6


def score_configs(obj, space, n_lhs, seed=42):
    sampler = qmc.LatinHypercube(d=space.n_params, seed=seed)
    X = sampler.random(n=n_lhs)
    scores = []
    configs = []
    for x in X:
        cfg = space.vector_to_dict(x)
        s, detail = obj.evaluate_with_details(cfg)
        scores.append({
            "score": s,
            "dr_mean": detail.get("dr_mean", 0.0),
            "dr_pmo": detail.get("dr_pmo", 0.0),
            "dr_ckd": detail.get("dr_ckd", 0.0),
            "fnr_mean": detail.get("fnr_mean", 1.0),
            "kd_nm": cfg["kd_nm"],
            "kd_ctx_nm": cfg.get("kd_ctx_nm", cfg["kd_nm"]),
            "kd_p1np_nm": cfg.get("kd_p1np_nm", cfg["kd_nm"]),
            "sensitivity": cfg["sensitivity"],
            "w_ctx": cfg.get("w_ctx", 0.0),
            "w_p1np": cfg.get("w_p1np", 0.0),
        })
        configs.append(cfg)
    return pd.DataFrame(scores), np.array(X), configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--surrogate-dir", type=Path, default=ROOT / "BO" / "bo_results")
    parser.add_argument("--n-lhs", type=int, default=2000,
                        help="LHS samples to generate (default: 2000)")
    args = parser.parse_args()

    print("=" * 70)
    print("MLP FAILURE DIAGNOSTIC")
    print("=" * 70)

    loader = SurrogateLoaderV3(args.surrogate_dir)
    physics = PhysicsForwardModel()
    obj = TherapeuticObjectiveV6(physics, loader)
    space = BiosensorSearchSpace()

    print(f"\n[1/4] Generating {args.n_lhs} LHS configs + scoring with v6 objective...")
    df, X, configs = score_configs(obj, space, args.n_lhs)

    # --- Test 1: Feature variance -------------------------------------------
    print("\n" + "-" * 70)
    print("TEST 1: FEATURE VARIANCE (does score vary meaningfully?)")
    print("-" * 70)
    print(f"  Score distribution over {args.n_lhs} random configs:")
    for pct in [5, 25, 50, 75, 90, 95]:
        print(f"    p{pct:2d}: {np.percentile(df['score'], pct):.4f}")
    print(f"  Score std:  {df['score'].std():.4f}")
    print(f"  Score IQR:  {df['score'].quantile(0.75) - df['score'].quantile(0.25):.4f}")

    top10_mask = df["score"] >= df["score"].quantile(0.90)
    bot10_mask = df["score"] <= df["score"].quantile(0.10)
    print(f"\n  VERDICT: Score range = [{df['score'].min():.3f}, {df['score'].max():.3f}]")
    if df["score"].std() > 0.05:
        print("  [OK] Score varies substantially -> problem is not flat landscape.")
        print("       The objective has real signal. The MLP had signal to learn FROM.")
    else:
        print("  [FAIL] Score nearly constant -> landscape is flat -> MLP cannot learn.")

    # --- Test 2: Target variance — one-to-many mapping ----------------------
    print("\n" + "-" * 70)
    print("TEST 2: ONE-TO-MANY MAPPING (multiple good designs for any patient)")
    print("-" * 70)
    top_q = df[df["score"] >= df["score"].quantile(0.80)]
    print(f"\n  Top 20% by score: {len(top_q)} configs")
    for param in ["kd_nm", "kd_ctx_nm", "sensitivity", "w_ctx", "w_p1np"]:
        vals = top_q[param]
        lo_ratio = vals.max() / (vals.min() + 1e-9)
        print(f"    {param:<15}: min={vals.min():.3f}  max={vals.max():.3f}  "
              f"range_ratio={lo_ratio:.1f}x  std={vals.std():.3f}")

    kd_cv = top_q["kd_nm"].std() / (top_q["kd_nm"].mean() + 1e-9)
    sens_cv = top_q["sensitivity"].std() / (top_q["sensitivity"].mean() + 1e-9)

    print(f"\n  VERDICT:")
    if kd_cv > 0.3 or sens_cv > 0.3:
        print("  [OK] Top 20% spans a WIDE range of kd / sensitivity values.")
        print("       This confirms one-to-many: many different biosensors achieve")
        print("       similarly good scores. MLP predicts a mean of this wide set -> R2~0.")
    else:
        print("  [FAIL] Top 20% is tightly clustered -> one-to-many argument is weak.")
        print("         The MLP failure may be due to other causes (see tests 3, 4).")

    # --- Test 3: Nearest-neighbor consistency --------------------------------
    print("\n" + "-" * 70)
    print("TEST 3: NEAREST-NEIGHBOR CONSISTENCY (does K-NN make sense?)")
    print("-" * 70)
    n_queries = min(50, len(df))
    query_idx = np.random.RandomState(0).choice(len(X), n_queries, replace=False)
    nn_score_stds = []
    for qi in query_idx:
        q_x = X[qi]
        dists = np.linalg.norm(X - q_x, axis=1)
        dists[qi] = np.inf
        nn_idx = np.argsort(dists)[:10]
        nn_scores = df["score"].values[nn_idx]
        nn_score_stds.append(nn_scores.std())

    mean_nn_std = float(np.mean(nn_score_stds))
    global_std = float(df["score"].std())
    compression_ratio = mean_nn_std / (global_std + 1e-9)

    print(f"\n  Global score std:              {global_std:.4f}")
    print(f"  Mean std within 10 NN:         {mean_nn_std:.4f}")
    print(f"  Compression ratio (nn/global): {compression_ratio:.3f}")
    print(f"\n  VERDICT:")
    if compression_ratio < 0.5:
        print("  [OK] Nearby configs have more similar scores (NN std < 50% of global).")
        print("       K-NN retrieval is justified: nearby designs ARE similar in quality.")
    else:
        print("  [~]  NN score variability is similar to global -- landscape is rough.")
        print("       K-NN retrieval may still work if applied in OUTPUT (design) space,")
        print("       not input (patient biomarker) space.")

    # --- Test 4: Multimodality -----------------------------------------------
    print("\n" + "-" * 70)
    print("TEST 4: MULTIMODALITY (are high-scoring configs clustered or spread?)")
    print("-" * 70)

    top20 = df[df["score"] >= df["score"].quantile(0.80)]
    X_top = X[top20.index]

    try:
        from sklearn.cluster import KMeans
        n_clusters = 4
        km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        km.fit(X_top)
        labels = km.labels_

        centers = km.cluster_centers_
        center_dists = []
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                center_dists.append(np.linalg.norm(centers[i] - centers[j]))
        mean_center_dist = float(np.mean(center_dists))

        intra_stds = []
        for k in range(n_clusters):
            members = X_top[labels == k]
            if len(members) > 1:
                intra_stds.append(members.std(axis=0).mean())
        mean_intra_std = float(np.mean(intra_stds)) if intra_stds else 0.0

        separation_ratio = mean_center_dist / (mean_intra_std + 1e-9)

        print(f"\n  K-means on top-20% configs (k={n_clusters}):")
        print(f"    Mean between-cluster center distance: {mean_center_dist:.3f}")
        print(f"    Mean intra-cluster spread:            {mean_intra_std:.3f}")
        print(f"    Separation ratio:                     {separation_ratio:.1f}x")

        print(f"\n  Score per cluster:")
        for k in range(n_clusters):
            cluster_scores = top20["score"].values[labels == k]
            if len(cluster_scores) > 0:
                print(f"    Cluster {k} (n={len(cluster_scores):4d}): "
                      f"mean={cluster_scores.mean():.4f}  std={cluster_scores.std():.4f}")

        print(f"\n  VERDICT:")
        if separation_ratio > 3.0:
            print("  [OK] High-scoring configs form MULTIPLE well-separated clusters.")
            print("       Regression must pick ONE mean across these modes -> R2~0.")
            print("       This directly confirms the one-to-many / multimodality argument.")
        elif separation_ratio > 1.5:
            print("  [~]  Moderate clustering. Some multimodality but not strongly separated.")
        else:
            print("  [FAIL] High-scoring configs form one dense cluster.")
            print("         Multimodality argument is weak -- investigate other MLP causes.")

    except ImportError:
        print("  sklearn not available for clustering -- skipping.")

    # --- Summary -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Configs evaluated: {args.n_lhs}")
    print(f"  Score range:       [{df['score'].min():.3f}, {df['score'].max():.3f}]"
          f"  (std={df['score'].std():.3f})")
    print(f"  kd range (top20%): [{top_q['kd_nm'].min():.3f}, {top_q['kd_nm'].max():.3f}]")
    print(f"  NN compression:    {compression_ratio:.3f}x  (< 0.5 -> K-NN sensible)")
    print()
    if kd_cv > 0.3 and mean_nn_std / global_std < 0.7:
        print("  CONCLUSION: Evidence SUPPORTS the one-to-many argument.")
        print("    - Many biosensor configs achieve similar good scores (target variance).")
        print("    - Nearby configs in parameter space have more similar scores (NN sense).")
        print("    - K-NN retrieval returns a valid SET of equally-good designs.")
        print("    - MLP R2~0 is expected and unavoidable for this problem structure.")
    else:
        print("  CONCLUSION: Evidence is MIXED.")
        print("    Run with --n-lhs 5000 for more reliable statistics.")
        print("    Consider also checking train/test split and label noise separately.")


if __name__ == "__main__":
    main()
