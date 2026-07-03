#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Patient-subtype-specific Bayesian Optimization (Blocker 3).

PROBLEM
-------
Blocker 3: All kd_ctx values give DR > 0.90 for established PMO/CKD — the V6
population-average objective is SATURATED for those scenarios. The real challenge
is to design biosensors optimized for clinically distinct patient SUBTYPES:

  young_pmo     : Age < 55, early PMO, BMD maximization most important,
                  false-positive tolerance higher (good compliance)
  elderly_pmo   : Age > 70, established PMO + comorbidities, safety-first
                  (false positives dangerous; co-meds amplify over-dosing risk)
  ckd_controlled: CKD Stage 3 (eGFR 30-60), kidney-safe dosing, CKD-BMD balance
  ckd_advanced  : CKD Stage 4-5 (eGFR < 30 or dialysis), strict safety ceiling

APPROACH
--------
Each subtype gets a different re-weighting of V6's component outputs:
  - V6 is NOT modified (frozen by CLAUDE.md)
  - A SubtypeObjective wrapper calls V6.evaluate_with_details() and
    re-combines the components with subtype-specific weights
  - Separate BO is run for each subtype

This gives 4 distinct optimal sensor configurations, one per patient cluster.

Usage:
    python BO/bo_patient_subtypes.py
    python BO/bo_patient_subtypes.py --subtypes young_pmo elderly_pmo
    python BO/bo_patient_subtypes.py --n-init 30 --n-iter 100
    python BO/bo_patient_subtypes.py --out-dir BO/bo_results_subtypes

Output:
    BO/bo_results_subtypes/{subtype}/best_config.json
    BO/bo_results_subtypes/subtype_comparison.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from BO.core.surrogate_loader import SurrogateLoaderV3
from evaluation.physics_forward_model import PhysicsForwardModel
from evaluation.therapeutic_objective_v6 import TherapeuticObjectiveV6
from search_space.biosensor_space import BiosensorSearchSpace
from optimizer.gaussian_process_bo import GaussianProcessBO
from acquisition.acquisition_functions import ExpectedImprovement

logger = logging.getLogger(__name__)

# ============================================================
# SUBTYPE DEFINITIONS
# ============================================================

SUBTYPE_CONFIGS: Dict[str, Dict] = {

    "young_pmo": {
        "description": "Age < 55, early / mild PMO, BMD maximization, higher FP tolerance",
        "clinical_context": (
            "Perimenopause onset (1-5 yr post-menopause). SOST only modestly elevated. "
            "Long bone-health horizon (30+ yr). Primary goal: maximize BMD gain; "
            "detect early-stage disease (pmo_mild critical). FP acceptable — "
            "younger patients have fewer co-morbidities and can handle monitoring."
        ),
        # Per-subtype score weights applied to V6 component outputs.
        # These replace the global V6 weights for the purpose of this subtype's BO.
        "score_weights": {
            "w_bmd_mild":     0.30,   # high: mild PMO BMD gain most important
            "w_bmd_pmo":      0.20,
            "w_bmd_ckd":      0.05,   # CKD not relevant for this subtype
            "w_dr_mild":      0.25,   # high: early detection is the bottleneck
            "w_dr_pmo":       0.08,
            "w_dr_ckd":       0.00,   # CKD scenarios excluded
            "w_fnr_penalty":  0.07,   # mild FNR penalty (missing early disease is bad)
            "w_fp_penalty":   0.05,   # low: FP tolerable in young patients
        },
        "penalize_ckd": False,        # do not require CKD DR coverage
    },

    "elderly_pmo": {
        "description": "Age > 70, established PMO, safety-first, co-medication interactions",
        "clinical_context": (
            "5-15 yr post-menopause. High fracture risk but also high polypharmacy risk. "
            "Over-dosing with romosozumab + bisphosphonate combination = ectopic calcification. "
            "Priority: minimize false positives. DR for established PMO already saturated — "
            "optimize for SAFETY and TTD (faster detection at lower drug load)."
        ),
        "score_weights": {
            "w_bmd_mild":     0.05,   # low: mild disease less relevant
            "w_bmd_pmo":      0.25,   # established PMO BMD gain important
            "w_bmd_ckd":      0.05,
            "w_dr_mild":      0.05,   # mild detection de-prioritized (safety > sensitivity)
            "w_dr_pmo":       0.15,
            "w_dr_ckd":       0.00,
            "w_fnr_penalty":  0.10,
            "w_fp_penalty":   0.35,   # HIGH: false positives lead to unnecessary dosing
        },
        "penalize_ckd": False,
    },

    "ckd_controlled": {
        "description": "CKD Stage 3 (eGFR 30-60), kidney-safe dosing emphasis",
        "clinical_context": (
            "CKD-MBD Stage 3: SOST ~3x elevated, phosphate imbalance. "
            "Romosozumab is NOT FDA-approved for CKD (cardiovascular safety concern), "
            "but is under investigation. Primary goal: maximize CKD detection rate "
            "while keeping dose within CKD-safe bounds (avoid Ca elevation). "
            "TTD fast detection important to minimize cumulative drug exposure."
        ),
        "score_weights": {
            "w_bmd_mild":     0.00,
            "w_bmd_pmo":      0.05,
            "w_bmd_ckd":      0.30,   # HIGH: CKD BMD gain is primary clinical goal
            "w_dr_mild":      0.00,
            "w_dr_pmo":       0.05,
            "w_dr_ckd":       0.30,   # HIGH: CKD detection must be reliable
            "w_fnr_penalty":  0.15,
            "w_fp_penalty":   0.15,   # moderate: CKD patients need careful monitoring
        },
        "penalize_ckd": True,         # enforce CKD DR >= 0.80 constraint
    },

    "ckd_advanced": {
        "description": "CKD Stage 4-5 / dialysis (eGFR < 30), strict safety ceiling",
        "clinical_context": (
            "CKD-MBD Stage 4-5 or dialysis. Very high SOST, severe mineral disorder. "
            "Drug dosing risk (hypercalcemia, cardiovascular events) is highest here. "
            "Design priority: SMALLEST effective dose that still detects CKD-MBD. "
            "FP rate must be near zero — unnecessary dosing in dialysis patients is "
            "life-threatening. Accept lower BMD gain in exchange for safety margin."
        ),
        "score_weights": {
            "w_bmd_mild":     0.00,
            "w_bmd_pmo":      0.00,
            "w_bmd_ckd":      0.15,   # modest: some BMD gain needed but not at cost of safety
            "w_dr_mild":      0.00,
            "w_dr_pmo":       0.00,
            "w_dr_ckd":       0.30,   # CKD detection required
            "w_fnr_penalty":  0.05,
            "w_fp_penalty":   0.50,   # VERY HIGH: false positives can kill in this population
        },
        "penalize_ckd": True,
    },
}


# ============================================================
# SUBTYPE OBJECTIVE WRAPPER
# ============================================================

class SubtypeObjective:
    """
    Wraps TherapeuticObjectiveV6 with subtype-specific re-weighting.

    V6 is called unchanged (frozen); its component outputs are re-combined
    using the subtype-specific weights from SUBTYPE_CONFIGS.

    Important: the V6 hard constraints (min DR >= 0.50 for disease scenarios,
    max FNR, max FP) are applied as before. The subtype weights only affect
    the composite score AFTER constraints are satisfied.
    """

    # Drug threshold (same as V6; needed for overdose scoring)
    _DRUG_THRESHOLD_FRAC = 1.08
    _D_SAFE              = 0.50
    _TTD_MAX             = 9000.0

    def __init__(self, v6_obj: TherapeuticObjectiveV6, subtype_cfg: dict):
        self.v6 = v6_obj
        self.cfg = subtype_cfg
        self.weights = subtype_cfg["score_weights"]
        self.penalize_ckd = subtype_cfg.get("penalize_ckd", False)

    def _normalize_weights(self) -> dict:
        """Normalize weights so they sum to 1.0."""
        w = self.weights
        total = sum(w.values())
        if total < 1e-9:
            total = 1.0
        return {k: v / total for k, v in w.items()}

    def __call__(self, config: dict) -> float:
        try:
            score, details = self.v6.evaluate_with_details(config)

            if "error" in details:
                return 0.0

            wn = self._normalize_weights()

            # Extract V6 component outputs
            bmd_mild = float(details.get("bmd_net_mild", 0.0))
            bmd_pmo  = float(details.get("bmd_net_pmo",  0.0))
            bmd_ckd  = float(details.get("bmd_net_ckd",  0.0))

            dr_mild = float(details.get("dr_mild",    0.0))
            dr_pmo  = float(details.get("dr_pmo",     0.0))
            dr_ckd  = float(details.get("dr_ckd",     0.0))

            fnr_mean = float(details.get("fnr_mean",   0.0))
            dr_healthy = float(details.get("dr_healthy", 0.0))

            # Subtype-weighted composite
            composite = (
                wn["w_bmd_mild"]   * max(bmd_mild, 0.0)
                + wn["w_bmd_pmo"]  * max(bmd_pmo,  0.0)
                + wn["w_bmd_ckd"]  * max(bmd_ckd,  0.0)
                + wn["w_dr_mild"]  * dr_mild
                + wn["w_dr_pmo"]   * dr_pmo
                + wn["w_dr_ckd"]   * dr_ckd
                - wn["w_fnr_penalty"] * fnr_mean
                - wn["w_fp_penalty"]  * dr_healthy
            )

            # CKD-specific constraint: penalize if CKD DR < 0.80
            if self.penalize_ckd and dr_ckd < 0.80:
                composite -= 0.30 * (0.80 - dr_ckd) / 0.80

            # Use V6's hard infeasibility penalty
            infeas = float(details.get("infeas_penalty", 0.0))
            composite -= infeas

            return float(np.clip(composite, -0.5, 1.0))

        except Exception as exc:
            logger.error("SubtypeObjective error: %s", exc)
            return 0.0


# ============================================================
# PER-SUBTYPE BO RUN
# ============================================================

def run_bo_for_subtype(
    subtype_name: str,
    subtype_cfg: dict,
    surrogate_dir: str,
    n_init: int,
    n_iter: int,
    seed: int,
    output_dir: Path,
    verbose: bool,
) -> dict:
    """Run one BO optimization for a single patient subtype."""

    logger.info("[%s] Starting BO (n_init=%d, n_iter=%d, seed=%d)",
                subtype_name, n_init, n_iter, seed)
    print(f"\n{'='*65}")
    print(f"SUBTYPE: {subtype_name}")
    print(f"  {subtype_cfg['description']}")
    print(f"{'='*65}")

    # Load surrogates
    loader = SurrogateLoaderV3(results_dir=surrogate_dir)
    phys = PhysicsForwardModel()
    v6 = TherapeuticObjectiveV6(phys, loader, apply_constraints=True)
    obj_fn = SubtypeObjective(v6, subtype_cfg)

    space = BiosensorSearchSpace()

    # GP optimizer
    acq = ExpectedImprovement(xi=0.01)
    gp = GaussianProcessBO(
        objective_fn=obj_fn,
        search_space=space,
        acquisition_fn=acq,
        n_init=n_init,
        n_iter=n_iter,
        random_state=seed,
    )

    bo_result = gp.optimize()

    best_x   = bo_result["x_best"]
    best_y   = float(bo_result.get("best_y") or bo_result.get("y_best") or 0.0)
    best_cfg = bo_result.get("config_best") or space.denormalize(best_x)

    # Get detailed V6 breakdown for the best config
    v6_score, details = v6.evaluate_with_details(best_cfg)
    subtype_score = obj_fn(best_cfg)

    result = {
        "subtype":         subtype_name,
        "description":     subtype_cfg["description"],
        "best_config":     best_cfg,
        "subtype_score":   subtype_score,
        "v6_score":        v6_score,
        "dr_pmo_mild":     details.get("dr_mild",    0.0),
        "dr_pmo":          details.get("dr_pmo",     0.0),
        "dr_ckd":          details.get("dr_ckd",     0.0),
        "dr_healthy_fp":   details.get("dr_healthy", 0.0),
        "fnr_mean":        details.get("fnr_mean",   0.0),
        "bmd_mild":        details.get("bmd_net_mild", 0.0),
        "bmd_pmo":         details.get("bmd_net_pmo",  0.0),
        "bmd_ckd":         details.get("bmd_net_ckd",  0.0),
        "therapeutic_mean": details.get("therapeutic_mean", 0.0),
        "score_weights":   subtype_cfg["score_weights"],
    }

    # Save per-subtype result
    sub_dir = output_dir / subtype_name
    sub_dir.mkdir(parents=True, exist_ok=True)
    with open(sub_dir / "best_config.json", "w") as f:
        json.dump(best_cfg, f, indent=2)
    with open(sub_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Result for {subtype_name}:")
    print(f"    Subtype score : {subtype_score:.4f}")
    print(f"    V6 score      : {v6_score:.4f}")
    print(f"    DR PMO-mild   : {result['dr_pmo_mild']:.3f}")
    print(f"    DR PMO        : {result['dr_pmo']:.3f}")
    print(f"    DR CKD        : {result['dr_ckd']:.3f}")
    print(f"    FP (healthy)  : {result['dr_healthy_fp']:.3f}")
    print(f"    FNR mean      : {result['fnr_mean']:.3f}")
    print(f"    BMD mild/pmo/ckd: {result['bmd_mild']:.3f}/{result['bmd_pmo']:.3f}/{result['bmd_ckd']:.3f}")
    print(f"    Best config:")
    for k, v in best_cfg.items():
        if isinstance(v, float):
            print(f"      {k:<22} {v:.4f}")
        else:
            print(f"      {k:<22} {v}")

    return result


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Patient-subtype-specific Bayesian Optimization (Blocker 3)",
    )
    parser.add_argument(
        "--subtypes", nargs="+",
        choices=list(SUBTYPE_CONFIGS.keys()),
        default=list(SUBTYPE_CONFIGS.keys()),
        help="Subtypes to run (default: all four)",
    )
    parser.add_argument("--n-init",       type=int, default=30,
                        help="LHS initial samples (default 30)")
    parser.add_argument("--n-iter",       type=int, default=100,
                        help="BO iterations (default 100)")
    parser.add_argument("--seed",         type=int, default=42,
                        help="Random seed (default 42)")
    parser.add_argument("--surrogate-dir", default="BO/bo_results",
                        help="Surrogate directory (default BO/bo_results)")
    parser.add_argument("--out-dir",      type=Path,
                        default=Path("BO/bo_results_subtypes"),
                        help="Output directory (default BO/bo_results_subtypes)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Suppress simulation noise
    for _noisy in ["simulation.simulator", "simulation.biosensor_engine",
                   "dataset.generator", "BO.core", "BO.optimizer"]:
        logging.getLogger(_noisy).setLevel(logging.WARNING)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("GENEVO2 PATIENT-SUBTYPE BAYESIAN OPTIMIZATION")
    print("=" * 65)
    print(f"Subtypes   : {args.subtypes}")
    print(f"n_init     : {args.n_init}  |  n_iter: {args.n_iter}")
    print(f"Seed       : {args.seed}")
    print(f"Surrogate  : {args.surrogate_dir}")
    print(f"Output     : {args.out_dir}")
    print()

    all_results = []
    for sname in args.subtypes:
        scfg = SUBTYPE_CONFIGS[sname]
        r = run_bo_for_subtype(
            subtype_name=sname,
            subtype_cfg=scfg,
            surrogate_dir=args.surrogate_dir,
            n_init=args.n_init,
            n_iter=args.n_iter,
            seed=args.seed,
            output_dir=args.out_dir,
            verbose=args.verbose,
        )
        all_results.append(r)

    # Comparison table
    print("\n" + "=" * 65)
    print("SUBTYPE COMPARISON SUMMARY")
    print("=" * 65)
    header = f"{'Subtype':<20} {'SubScore':>8} {'V6':>6} {'PMO-m':>6} {'PMO':>6} {'CKD':>6} {'FP':>6} {'FNR':>6}"
    print(header)
    print("-" * 65)
    for r in all_results:
        print(
            f"{r['subtype']:<20} {r['subtype_score']:>8.4f} {r['v6_score']:>6.3f}"
            f" {r['dr_pmo_mild']:>6.3f} {r['dr_pmo']:>6.3f} {r['dr_ckd']:>6.3f}"
            f" {r['dr_healthy_fp']:>6.3f} {r['fnr_mean']:>6.3f}"
        )
    print("=" * 65)

    # Key differences analysis
    print("\nKEY DIFFERENCES ACROSS SUBTYPES:")
    params = ["kd_nm", "kd_ctx_nm", "kd_p1np_nm", "sensitivity", "w_ctx", "w_p1np"]
    print(f"  {'Parameter':<22}", end="")
    for r in all_results:
        print(f" {r['subtype']:>14}", end="")
    print()
    print("  " + "-" * (22 + 15 * len(all_results)))
    for p in params:
        print(f"  {p:<22}", end="")
        for r in all_results:
            val = r["best_config"].get(p, 0.0)
            if isinstance(val, float):
                print(f" {val:>14.4f}", end="")
            else:
                print(f" {str(val):>14}", end="")
        print()

    # Save comparison
    comparison_path = args.out_dir / "subtype_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nComparison saved: {comparison_path}")

    # Clinical interpretation
    print("\nCLINICAL INTERPRETATION:")
    for r in all_results:
        sname = r["subtype"]
        fp = r["dr_healthy_fp"]
        dr_mild = r["dr_pmo_mild"]
        bmd_max = max(r["bmd_mild"], r["bmd_pmo"], r["bmd_ckd"])
        print(f"\n  {sname}:")
        print(f"    FP rate: {fp:.1%}  | PMO-mild DR: {dr_mild:.1%} | Max BMD score: {bmd_max:.3f}")
        scfg = SUBTYPE_CONFIGS[sname]
        print(f"    Context: {scfg['clinical_context'][:80]}...")

    return 0


if __name__ == "__main__":
    sys.exit(main())
