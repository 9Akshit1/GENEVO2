#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Priority 2: kd_ctx systematic scan with the real ODE simulator.

For each kd_ctx value in the scan range, all other parameters are fixed
at the current BO-optimal configuration. The simulator is run n_trials times
per kd_ctx (each with a different biological variability seed) to get a
reliable mean DR ± CI.

Answers: Is kd_ctx = 0.278 nM (BO result) actually optimal, or is the
Langmuir-theoretical 0.316 nM better? Is the anomalous 0.13 nM from
earlier sessions a surrogate artifact?

Usage:
    python BO/analysis/kd_ctx_scan.py
    python BO/analysis/kd_ctx_scan.py --config BO/bo_results/results/best_config.json
    python BO/analysis/kd_ctx_scan.py --n-trials 30 --out BO/analysis/kd_ctx_scan_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from simulation.dataset.generator import DatasetGenerator

logger = logging.getLogger(__name__)

# kd_ctx values to scan (nM)
# Includes: previous anomalous BO result (0.13), current BO result (0.278),
# Langmuir-theoretical optimum (0.316), and bracketing values.
KD_CTX_SCAN = [0.05, 0.10, 0.13, 0.20, 0.278, 0.316, 0.50, 1.00]

# Scenarios to evaluate (weighted average)
SCENARIOS = ["pmo_mild", "pmo", "ckd_mbd"]
SCENARIO_WEIGHTS = [0.28, 0.36, 0.36]   # match data_v18 disease distribution


def build_config_for_kd_ctx(best_config: dict, kd_ctx_nm: float) -> dict:
    """
    Build a biosensor config dict from best_config with kd_ctx replaced.

    Recomputes the threshold using the standard calibration formula so that
    the detection threshold stays consistent with the new kd_ctx value.
    """
    bd = best_config.get("biosensor_design", best_config)

    kd_scl  = float(bd.get("kd_nm",    bd.get("kd_scl",   5.81)))
    kd_p1np = float(bd.get("kd_p1np_nm", bd.get("kd_p1np", 0.953)))
    sensitivity = float(bd.get("sensitivity", 4.57))
    w_ctx   = float(bd.get("w_ctx",  0.155))
    w_p1np  = float(bd.get("w_p1np", 0.459))
    w_scl   = float(1.0 - w_ctx - w_p1np)

    # Healthy and PMO nominal concentrations for threshold calibration
    H = {"scl": 0.375, "ctx": 0.200, "p1np": 0.350}
    P = {"scl": 0.875, "ctx": 0.500, "p1np": 0.525}
    C = {"scl": 1.125, "ctx": 0.500, "p1np": 0.625}

    def occ(c, kd):
        return c / (kd + c + 1e-12)

    ref_scl  = occ(H["scl"],  kd_scl)
    ref_ctx  = occ(H["ctx"],  kd_ctx_nm)
    ref_p1np = occ(H["p1np"], kd_p1np)

    def composite(cd):
        n_scl  = occ(cd["scl"],  kd_scl)  / (ref_scl  + 1e-12)
        n_ctx  = occ(cd["ctx"],  kd_ctx_nm) / (ref_ctx + 1e-12)
        n_p1np = occ(cd["p1np"], kd_p1np) / (ref_p1np + 1e-12)
        return sensitivity * (w_scl * n_scl + w_ctx * n_ctx + w_p1np * n_p1np)

    sig_h = composite(H)
    sig_c = composite(C)

    # Threshold: healthy signal + 1.25 × (PMO/sensitivity - 1) gap
    sig_p_unit = composite({k: P[k] for k in P})
    hp_gap_ref = (sig_p_unit / sensitivity) - 1.0
    threshold = float(sig_h + 1.25 * hp_gap_ref)

    return {
        "circuit_type": "array",
        "kd_scl":        kd_scl,
        "kd_ctx":        kd_ctx_nm,
        "kd_p1np":       kd_p1np,
        "w_scl":         w_scl,
        "w_ctx":         w_ctx,
        "w_p1np":        w_p1np,
        "sensitivity":   sensitivity,
        "threshold":     threshold,
        "dynamic_range": (0.0, float(sig_c * 2.0)),
        "kd":            kd_scl,
    }


def run_scan(
    best_config: dict,
    kd_ctx_values: List[float],
    n_trials: int,
    model_path: str,
    seed: int,
) -> List[Dict]:
    """Run the kd_ctx scan. Returns list of per-value result dicts."""
    rng = np.random.RandomState(seed)
    # DatasetGenerator requires a real output_dir (creates metadata/ and timeseries/ subdirs).
    # Use a temp directory — the scan discards all output files.
    _tmp_dir = tempfile.mkdtemp(prefix="genevo2_kd_ctx_scan_")
    gen = DatasetGenerator(
        antimony_model_path=model_path,
        output_dir=_tmp_dir,
        seed=seed,
        sigma_measurement=0.0,
    )

    results = []

    for kd_ctx in kd_ctx_values:
        cfg = build_config_for_kd_ctx(best_config, kd_ctx)
        logger.info("kd_ctx = %.3f nM  (threshold=%.4f)", kd_ctx, cfg["threshold"])

        per_scenario: Dict[str, List[float]] = {sc: [] for sc in SCENARIOS}

        for trial in range(n_trials):
            trial_seed = int(rng.randint(0, 2**31))
            for scenario in SCENARIOS:
                record = gen.generate_single_simulation_instrumented(
                    scenario_name=scenario,
                    biosensor_config=cfg,
                    noise_preset="realistic",
                    duration=3600.0,
                    num_points=361,
                    apply_variability=True,
                    instrument=False,
                    rng_seed=trial_seed,
                )
                if record is not None:
                    dr = float(record.get("measurement", {}).get("detection_rate", 0.0))
                    per_scenario[scenario].append(dr)

        # Weighted average DR across scenarios
        dr_by_scenario = {}
        weighted_drs = []
        for sc, w in zip(SCENARIOS, SCENARIO_WEIGHTS):
            vals = per_scenario[sc]
            if vals:
                mean_dr = float(np.mean(vals))
                std_dr  = float(np.std(vals))
            else:
                mean_dr, std_dr = 0.0, 0.0
            dr_by_scenario[sc] = {"mean": mean_dr, "std": std_dr, "n": len(vals)}
            weighted_drs.extend([mean_dr] * max(1, len(vals)))

        # Weighted composite
        composite_dr = sum(
            dr_by_scenario[sc]["mean"] * w
            for sc, w in zip(SCENARIOS, SCENARIO_WEIGHTS)
        )

        results.append({
            "kd_ctx_nm":    kd_ctx,
            "threshold":    cfg["threshold"],
            "dr_composite": composite_dr,
            "by_scenario":  dr_by_scenario,
        })

        logger.info(
            "  DR composite=%.3f  pmo_mild=%.3f  pmo=%.3f  ckd=%.3f",
            composite_dr,
            dr_by_scenario["pmo_mild"]["mean"],
            dr_by_scenario["pmo"]["mean"],
            dr_by_scenario["ckd_mbd"]["mean"],
        )

    return results


def print_table(results: List[Dict]) -> None:
    bo_result  = 0.278
    langmuir   = 0.316
    anomalous  = 0.13

    print("\n" + "=" * 78)
    print("kd_ctx SCAN RESULTS")
    print("=" * 78)
    print(f"{'kd_ctx (nM)':<14} {'DR composite':<14} {'PMO-mild':<12} {'PMO':<10} {'CKD-MBD':<10}  {'Notes'}")
    print("-" * 78)
    for r in results:
        kd    = r["kd_ctx_nm"]
        note  = ""
        if abs(kd - bo_result) < 0.01:
            note = "<-- BO result"
        elif abs(kd - langmuir) < 0.01:
            note = "<-- Langmuir theory"
        elif abs(kd - anomalous) < 0.01:
            note = "<-- previous anomalous BO"
        sc = r["by_scenario"]
        print(
            f"{kd:<14.3f} {r['dr_composite']:<14.3f}"
            f" {sc['pmo_mild']['mean']:<12.3f}"
            f" {sc['pmo']['mean']:<10.3f}"
            f" {sc['ckd_mbd']['mean']:<10.3f}"
            f"  {note}"
        )
    print("=" * 78)

    best = max(results, key=lambda x: x["dr_composite"])
    print(f"\nPeak DR at kd_ctx = {best['kd_ctx_nm']:.3f} nM  (DR = {best['dr_composite']:.3f})")

    bo_r   = next((r for r in results if abs(r["kd_ctx_nm"] - bo_result)  < 0.01), None)
    lang_r = next((r for r in results if abs(r["kd_ctx_nm"] - langmuir)   < 0.01), None)
    anom_r = next((r for r in results if abs(r["kd_ctx_nm"] - anomalous)  < 0.01), None)

    print("\nKey comparisons:")
    if bo_r and lang_r:
        delta = lang_r["dr_composite"] - bo_r["dr_composite"]
        print(f"  Langmuir (0.316) vs BO result (0.278): {delta:+.3f} DR")
        if delta > 0.02:
            print("  => Surrogate pointed slightly sub-optimal; kd_ctx = 0.316 is better")
        elif delta < -0.02:
            print("  => BO result is better than Langmuir prediction; physics more complex")
        else:
            print("  => Essentially equivalent (within noise); BO result is near-optimal")
    if anom_r:
        delta_anom = best["dr_composite"] - anom_r["dr_composite"]
        print(f"  Anomalous 0.13 nM vs peak:  {delta_anom:+.3f} DR")
        if delta_anom > 0.02:
            print("  => Confirmed: 0.13 nM was a surrogate artifact (real DR worse)")
        else:
            print("  => Surprising: 0.13 nM performs comparably in real simulator")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="kd_ctx systematic scan with real ODE simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", default="BO/bo_results/results/best_config.json",
        help="Path to best_config.json",
    )
    parser.add_argument(
        "--n-trials", dest="n_trials", type=int, default=20,
        help="Simulator runs per kd_ctx value (default: 20)",
    )
    parser.add_argument(
        "--model", default="simulation/models/bone_environment.ant",
        help="ODE model path",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output JSON path (optional)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if not Path(args.model).exists():
        logger.error("ODE model not found: %s — run from project root", args.model)
        return 1
    if not Path(args.config).exists():
        logger.error("Config not found: %s", args.config)
        return 1

    with open(args.config) as f:
        best_config = json.load(f)

    logger.info("=" * 78)
    logger.info("kd_ctx SCAN  (n_trials=%d per value, %d values)", args.n_trials, len(KD_CTX_SCAN))
    logger.info("Fixed params from: %s", args.config)
    logger.info("=" * 78)

    results = run_scan(
        best_config=best_config,
        kd_ctx_values=KD_CTX_SCAN,
        n_trials=args.n_trials,
        model_path=args.model,
        seed=args.seed,
    )

    print_table(results)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved: %s", out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
