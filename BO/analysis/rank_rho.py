#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Formal rank-rho (Spearman rank correlation) measurement for v18 surrogates.

Measures how well the surrogate RANKS configurations relative to their true
simulator-evaluated DR. A rank-rho of 1.0 means the surrogate's ranking is
identical to the simulator's; 0.0 means no correlation; 0.517 was the v4
baseline.

Method:
  - Load data_v18/master_index.csv (has true simulator DR values)
  - For each disease config (not healthy), predict DR with the v18 surrogate
  - Compute Spearman correlation between predicted rank and true rank
  - Do this per-scenario and overall

Usage:
    python BO/analysis/rank_rho_v18.py
    python BO/analysis/rank_rho_v18.py --surrogate-version v18 --n-sample 500
    python BO/analysis/rank_rho_v18.py --out BO/analysis/rank_rho_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from BO.core.surrogate_loader import SurrogateLoaderV3

logger = logging.getLogger(__name__)


def compute_rank_rho(
    data_dir: Path,
    surrogate_version: str = None,
    n_sample: int | None = None,
    seed: int = 42,
) -> dict:
    """
    Load master_index.csv, predict with surrogate, compute Spearman rank-rho.

    Returns a dict with overall and per-scenario results.
    """
    # Load data
    csv_path = data_dir / "master_index.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"master_index.csv not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Require columns
    required = [
        "scenario", "detection_rate",
        "kd_scl", "kd_ctx", "kd_p1np",
        "w_ctx", "w_p1np", "sensitivity",
        "noise_preset",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in master_index.csv: {missing}")

    # Drop healthy — surrogate trained for disease scenarios
    df_disease = df[df["scenario"] != "healthy"].copy().reset_index(drop=True)

    # Optional: subsample for speed
    if n_sample is not None and n_sample < len(df_disease):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(df_disease), size=n_sample, replace=False)
        df_disease = df_disease.iloc[idx].reset_index(drop=True)

    logger.info("Evaluating %d configs (disease scenarios only)", len(df_disease))

    # Load surrogate
    surrogate = SurrogateLoaderV3(results_dir=Path("BO/bo_results"))

    # Predict DR for each row
    predicted_dr = []
    for _, row in df_disease.iterrows():
        try:
            dr_pred, _, _ = surrogate.predict(
                kd_nm          = float(row["kd_scl"]),
                sensitivity    = float(row["sensitivity"]),
                response_time  = 600.0,
                biosensor_type = "array",
                noise_preset   = str(row.get("noise_preset", "realistic")),
                scenario       = str(row["scenario"]),
                kd_ctx  = float(row["kd_ctx"]),
                kd_p1np = float(row["kd_p1np"]),
                w_ctx   = float(row["w_ctx"]),
                w_p1np  = float(row["w_p1np"]),
            )
        except Exception as e:
            logger.debug("Prediction failed: %s", e)
            dr_pred = 0.5  # neutral fallback
        predicted_dr.append(dr_pred)

    df_disease = df_disease.copy()
    df_disease["dr_predicted"] = predicted_dr
    true_dr = df_disease["detection_rate"].values
    pred_dr = np.array(predicted_dr)

    # Overall rank-rho
    rho_all, p_all = stats.spearmanr(true_dr, pred_dr)

    # Per-scenario rank-rho
    scenario_results = {}
    for sc in df_disease["scenario"].unique():
        mask = df_disease["scenario"] == sc
        t = true_dr[mask.values]
        p = pred_dr[mask.values]
        if len(t) < 5:
            continue
        rho_sc, p_sc = stats.spearmanr(t, p)
        scenario_results[sc] = {
            "n": int(mask.sum()),
            "rank_rho": float(rho_sc),
            "p_value":  float(p_sc),
            "true_dr_mean":  float(t.mean()),
            "pred_dr_mean":  float(p.mean()),
        }

    # Calibration: bias (predicted - true) and RMSE
    bias = float((pred_dr - true_dr).mean())
    rmse = float(np.sqrt(((pred_dr - true_dr) ** 2).mean()))

    return {
        "n_evaluated":       len(df_disease),
        "overall": {
            "rank_rho": float(rho_all),
            "p_value":  float(p_all),
            "bias":     bias,
            "rmse":     rmse,
        },
        "by_scenario": scenario_results,
        "benchmark": {
            "v4_baseline_rank_rho":   0.517,
            "v5_cl_r3_estimate":      0.720,
            "threshold_acceptable":   0.600,
            "threshold_good":         0.700,
        },
    }


def print_report(result: dict) -> None:
    print("\n" + "=" * 64)
    print("RANK-RHO REPORT")
    print("=" * 64)
    ov = result["overall"]
    print(f"  Configurations evaluated : {result['n_evaluated']}")
    print(f"  Overall rank-rho         : {ov['rank_rho']:.4f}  (p={ov['p_value']:.2e})")
    print(f"  Prediction bias          : {ov['bias']:+.4f}  (predicted - true)")
    print(f"  Prediction RMSE          : {ov['rmse']:.4f}")

    bm = result["benchmark"]
    rho = ov["rank_rho"]
    print(f"\n  Benchmarks:")
    print(f"    v4 baseline : {bm['v4_baseline_rank_rho']:.3f}  {'<' if rho > bm['v4_baseline_rank_rho'] else '>'} current")
    print(f"    v5 estimate : {bm['v5_cl_r3_estimate']:.3f}  {'<' if rho > bm['v5_cl_r3_estimate'] else '>'} current")
    print(f"    Threshold (acceptable) : {bm['threshold_acceptable']:.3f}")
    print(f"    Threshold (good)       : {bm['threshold_good']:.3f}")

    if rho >= bm["threshold_good"]:
        verdict = "GOOD — surrogate suitable for BO guidance"
    elif rho >= bm["threshold_acceptable"]:
        verdict = "ACCEPTABLE — surrogate usable but improvement possible"
    else:
        verdict = "POOR — surrogate unreliable; retrain or collect more data"
    print(f"\n  Verdict: {verdict}")

    print("\n  Per-scenario rank-rho:")
    for sc, sc_r in result["by_scenario"].items():
        print(f"    {sc:<12s}: rho={sc_r['rank_rho']:.4f}  "
              f"n={sc_r['n']}  "
              f"true_mean={sc_r['true_dr_mean']:.3f}  "
              f"pred_mean={sc_r['pred_dr_mean']:.3f}")
    print("=" * 64)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rank-rho (Spearman) measurement for surrogates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir", dest="data_dir", default="data_v19",
        help="Dataset directory with master_index.csv (default: data_v19)",
    )
    parser.add_argument(
        "--n-sample", dest="n_sample", type=int, default=None,
        help="Subsample N configs (default: use all)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for subsampling (default: 42)",
    )
    parser.add_argument(
        "--out", default="BO/bo_results/diagnostics/rank_rho_results.json",
        help="Output JSON path (default: BO/bo_results/diagnostics/rank_rho_results.json)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error("Data dir not found: %s — run from project root", data_dir)
        return 1

    result = compute_rank_rho(
        data_dir=data_dir,
        n_sample=args.n_sample,
        seed=args.seed,
    )

    print_report(result)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Results saved: %s", out_path)

    return 0 if result["overall"]["rank_rho"] >= 0.6 else 1


if __name__ == "__main__":
    sys.exit(main())
