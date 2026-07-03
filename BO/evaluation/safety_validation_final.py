#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Safety validation for the final biosensor design.

Simulates N virtual patients across all disease scenarios, applies drug
interactions and patient-specific parameters, then checks every hard safety
constraint defined in safety_constraints.py.

Outputs:
  - Per-constraint violation rates (%)
  - BMD gain distribution (mean ± std per scenario)
  - Dose count distribution
  - Overall feasibility rate
  - JSON report for paper methods section

Usage:
    python BO/evaluation/safety_validation_final.py \\
        --config-json BO/bo_results/results/best_config.json \\
        --n-simulations 1000 \\
        --out BO/bo_results_final/safety_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from BO.evaluation.drug_interactions import (
    apply_drug_interactions,
    sample_patient_medications,
)
from BO.evaluation.pkpd_closed_loop import PKPDClosedLoop, PKPDResult
from BO.evaluation.safety_constraints import SafetyConstraints, SafetyResult

logger = logging.getLogger(__name__)

# Scenario distribution matching data_v18 (disease patients only)
_SCENARIO_DIST = {
    "pmo_mild": 0.28,
    "pmo":      0.28,
    "ckd_mbd":  0.44,
}

# Patient eGFR baselines (mL/min/1.73m²) by scenario
_EGFR_MEAN = {
    "pmo_mild": 80.0,
    "pmo":      72.0,
    "ckd_mbd":  32.0,   # CKD-MBD patients have severely reduced eGFR
}
_EGFR_STD = {
    "pmo_mild": 12.0,
    "pmo":      14.0,
    "ckd_mbd":  10.0,
}

# Serum calcium distribution (mg/dL) — slightly elevated baseline for CKD
_CA_MEAN = {"pmo_mild": 9.3, "pmo": 9.4, "ckd_mbd": 9.8}
_CA_STD  = {"pmo_mild": 0.3, "pmo": 0.4, "ckd_mbd": 0.5}


@dataclass
class PatientResult:
    scenario: str
    patient_id: int
    t_half_days: float
    medications: List[str]
    drug_efficacy_multiplier: float
    safety_flags: List[str]
    egfr_baseline: float
    egfr_simulated_decline_pct: float
    serum_calcium: float
    bmd_final_pct: float
    n_doses: int
    total_dose_fraction: float
    n_cycles: int
    cycle_days: int
    drug_normalized: bool
    safety: SafetyResult
    feasible: bool


@dataclass
class ScenarioStats:
    n_patients: int = 0
    feasible_pct: float = 0.0
    bmd_mean: float = 0.0
    bmd_std: float = 0.0
    dose_count_mean: float = 0.0
    dose_count_std: float = 0.0
    normalization_rate_pct: float = 0.0
    violation_rates: Dict[str, float] = field(default_factory=dict)


def sample_patient_params(
    rng: np.random.RandomState, scenario: str
) -> Dict:
    """Sample individual patient parameters for one virtual patient."""
    t_half = float(np.clip(rng.normal(180.0, 30.0), 90.0, 360.0))
    egfr_b = float(np.clip(rng.normal(_EGFR_MEAN[scenario], _EGFR_STD[scenario]), 10.0, 120.0))
    # eGFR decline: CKD patients worsen 1-3%/month during treatment
    if scenario == "ckd_mbd":
        decline_pct = float(np.clip(rng.normal(0.05, 0.02), 0.0, 0.20))
    else:
        decline_pct = float(np.clip(rng.normal(0.01, 0.01), 0.0, 0.08))
    serum_ca = float(np.clip(rng.normal(_CA_MEAN[scenario], _CA_STD[scenario]), 7.5, 13.0))
    medications = sample_patient_medications(rng=rng, scenario=scenario)
    return {
        "t_half_days": t_half,
        "egfr_baseline": egfr_b,
        "egfr_decline_pct": decline_pct,
        "serum_calcium": serum_ca,
        "medications": medications,
    }


def run_validation(
    config: dict,
    n_simulations: int = 1000,
    n_cycles: int = 12,
    cycle_days: int = 28,
    seed: int = 42,
) -> List[PatientResult]:
    """
    Simulate n_simulations virtual patients and check safety for each.

    Patients are drawn from _SCENARIO_DIST (disease scenarios only;
    healthy patients are excluded because they shouldn't receive treatment).
    """
    rng = np.random.RandomState(seed)
    sc = SafetyConstraints()
    sim = PKPDClosedLoop(config)

    scenarios = list(_SCENARIO_DIST.keys())
    probs     = list(_SCENARIO_DIST.values())

    results: List[PatientResult] = []

    for pid in range(n_simulations):
        scenario = rng.choice(scenarios, p=probs)
        params   = sample_patient_params(rng, scenario)

        # Run PK/PD simulation with patient-specific degradation
        try:
            pkpd: PKPDResult = sim.simulate(
                scenario=scenario,
                n_cycles=n_cycles,
                cycle_days=cycle_days,
                rng=np.random.RandomState(rng.randint(0, 2**31)),
                with_degradation=True,
                personalized_threshold=True,
                degradation_correction=True,
                patient_t_half_days=params["t_half_days"],
                apply_long_term_plateau=True,
            )
        except Exception as e:
            logger.warning("Patient %d simulation failed: %s", pid, e)
            continue

        # Drug interactions — modify effective BMD gain
        eff_mult, safety_flags = apply_drug_interactions(
            base_efficacy=1.0,
            medications=params["medications"],
        )

        # Adjusted BMD gain
        bmd_adjusted = pkpd.final_bmd_pct * eff_mult

        # Compute eGFR at end of treatment
        egfr_current = params["egfr_baseline"] * (1.0 - params["egfr_decline_pct"])

        # Serum calcium: thiazide or vitamin_d can bump it slightly
        serum_ca = params["serum_calcium"]
        if "thiazide" in params["medications"]:
            serum_ca += float(rng.normal(0.3, 0.1))

        # Build BMD trajectory in %
        bmd_traj_pct = [v / 0.775 * 100 for v in pkpd.bmd_trajectory]

        # Safety check
        safety = sc.check_all(
            cumulative_dose_fractional=pkpd.total_dose,
            n_months=n_cycles * cycle_days // 28,
            bmd_trajectory_pct=bmd_traj_pct,
            egfr_baseline=params["egfr_baseline"],
            egfr_current=egfr_current,
            serum_calcium_mg_dl=serum_ca,
        )

        results.append(PatientResult(
            scenario=scenario,
            patient_id=pid,
            t_half_days=params["t_half_days"],
            medications=params["medications"],
            drug_efficacy_multiplier=eff_mult,
            safety_flags=safety_flags,
            egfr_baseline=params["egfr_baseline"],
            egfr_simulated_decline_pct=params["egfr_decline_pct"] * 100.0,
            serum_calcium=serum_ca,
            bmd_final_pct=bmd_adjusted,
            n_doses=len(pkpd.doses),
            total_dose_fraction=pkpd.total_dose,
            n_cycles=n_cycles,
            cycle_days=cycle_days,
            drug_normalized=pkpd.drug_normalized,
            safety=safety,
            feasible=safety.feasible,
        ))

        if (pid + 1) % 100 == 0:
            n_done = pid + 1
            feas = sum(r.feasible for r in results)
            logger.info("  %d/%d complete | feasible: %d/%d (%.0f%%)",
                        n_done, n_simulations, feas, len(results),
                        100.0 * feas / len(results))

    return results


def aggregate_results(results: List[PatientResult]) -> Dict:
    """Compute summary statistics for the paper table."""
    n = len(results)
    if n == 0:
        return {"error": "no results"}

    # Overall
    feasible_all = [r for r in results if r.feasible]
    all_violations: Dict[str, int] = {}
    for r in results:
        for v in r.safety.violations:
            tag = v.split(":")[0]  # first word is the category
            all_violations[tag] = all_violations.get(tag, 0) + 1

    # Per-scenario
    by_scenario: Dict[str, List[PatientResult]] = {}
    for r in results:
        by_scenario.setdefault(r.scenario, []).append(r)

    scenario_stats: Dict[str, ScenarioStats] = {}
    for sc_name, sc_results in by_scenario.items():
        bmds  = [r.bmd_final_pct for r in sc_results]
        doses = [r.n_doses for r in sc_results]
        sc_viol: Dict[str, int] = {}
        for r in sc_results:
            for v in r.safety.violations:
                tag = v.split(":")[0]
                sc_viol[tag] = sc_viol.get(tag, 0) + 1

        scenario_stats[sc_name] = ScenarioStats(
            n_patients=len(sc_results),
            feasible_pct=100.0 * sum(r.feasible for r in sc_results) / len(sc_results),
            bmd_mean=float(np.mean(bmds)),
            bmd_std=float(np.std(bmds)),
            dose_count_mean=float(np.mean(doses)),
            dose_count_std=float(np.std(doses)),
            normalization_rate_pct=100.0 * sum(r.drug_normalized for r in sc_results) / len(sc_results),
            violation_rates={k: 100.0 * v / len(sc_results) for k, v in sc_viol.items()},
        )

    # Medication stats
    meds_all: Dict[str, int] = {}
    for r in results:
        for m in r.medications:
            meds_all[m] = meds_all.get(m, 0) + 1
    meds_pct = {m: 100.0 * c / n for m, c in meds_all.items()}

    # Drug efficacy multiplier distribution
    effs = [r.drug_efficacy_multiplier for r in results]

    return {
        "n_total": n,
        "n_feasible": len(feasible_all),
        "feasible_pct": 100.0 * len(feasible_all) / n,
        "infeasible_pct": 100.0 * (n - len(feasible_all)) / n,
        "violation_counts": all_violations,
        "violation_pct": {k: 100.0 * v / n for k, v in all_violations.items()},
        "bmd_overall": {
            "mean": float(np.mean([r.bmd_final_pct for r in results])),
            "std":  float(np.std([r.bmd_final_pct for r in results])),
            "p25":  float(np.percentile([r.bmd_final_pct for r in results], 25)),
            "p75":  float(np.percentile([r.bmd_final_pct for r in results], 75)),
        },
        "dose_count_overall": {
            "mean": float(np.mean([r.n_doses for r in results])),
            "std":  float(np.std([r.n_doses for r in results])),
        },
        "drug_interactions": {
            "medication_prevalence_pct": meds_pct,
            "efficacy_multiplier_mean": float(np.mean(effs)),
            "efficacy_multiplier_std":  float(np.std(effs)),
            "efficacy_multiplier_min":  float(np.min(effs)),
            "efficacy_multiplier_max":  float(np.max(effs)),
        },
        "normalization_rate_pct": 100.0 * sum(r.drug_normalized for r in results) / n,
        "by_scenario": {k: {
            "n": v.n_patients,
            "feasible_pct": v.feasible_pct,
            "bmd_mean_pct": v.bmd_mean,
            "bmd_std_pct":  v.bmd_std,
            "dose_count_mean": v.dose_count_mean,
            "normalization_rate_pct": v.normalization_rate_pct,
            "violation_rates": v.violation_rates,
        } for k, v in scenario_stats.items()},
    }


def print_report(summary: Dict) -> None:
    """Print formatted safety report to stdout."""
    print("\n" + "=" * 72)
    print("SAFETY VALIDATION REPORT")
    print("=" * 72)
    print(f"  Virtual patients simulated : {summary['n_total']}")
    print(f"  Feasible (no violations)   : {summary['n_feasible']}  ({summary['feasible_pct']:.1f}%)")
    print(f"  Infeasible (any violation) : {summary['n_total'] - summary['n_feasible']}  ({summary['infeasible_pct']:.1f}%)")

    print("\n  Violation breakdown:")
    if summary["violation_pct"]:
        for vtype, pct in sorted(summary["violation_pct"].items(), key=lambda x: -x[1]):
            print(f"    {vtype:<40s}: {pct:.1f}%")
    else:
        print("    [None]")

    print(f"\n  BMD gain (all scenarios)    : {summary['bmd_overall']['mean']:+.1f}% ± {summary['bmd_overall']['std']:.1f}%")
    print(f"  Doses per patient (mean)    : {summary['dose_count_overall']['mean']:.1f} ± {summary['dose_count_overall']['std']:.1f}")
    print(f"  Normalization rate          : {summary['normalization_rate_pct']:.1f}%")

    print("\n  Drug interaction summary:")
    di = summary["drug_interactions"]
    print(f"    Net efficacy multiplier   : {di['efficacy_multiplier_mean']:.2f} ± {di['efficacy_multiplier_std']:.2f}  (range {di['efficacy_multiplier_min']:.2f}–{di['efficacy_multiplier_max']:.2f})")
    print(f"    Medication prevalences (top 5):")
    top5 = sorted(di["medication_prevalence_pct"].items(), key=lambda x: -x[1])[:5]
    for med, pct in top5:
        print(f"      {med:<30s}: {pct:.1f}%")

    print("\n  Per-scenario breakdown:")
    for sc_name, sc in summary["by_scenario"].items():
        print(f"\n    [{sc_name.upper()}]  n={sc['n']}")
        print(f"      Feasible           : {sc['feasible_pct']:.1f}%")
        print(f"      BMD gain           : {sc['bmd_mean_pct']:+.1f}% ± {sc['bmd_std_pct']:.1f}%")
        print(f"      Doses given        : {sc['dose_count_mean']:.1f}")
        print(f"      Normalization rate : {sc['normalization_rate_pct']:.1f}%")
        if sc["violation_rates"]:
            print(f"      Violations         : {sc['violation_rates']}")

    print("\n" + "=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Safety validation: N virtual patients through PKPD + hard constraints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config-json", dest="config_json",
        default="BO/bo_results/results/best_config.json",
        help="Path to biosensor config JSON",
    )
    parser.add_argument(
        "--n-simulations", dest="n_simulations", type=int, default=1000,
        help="Number of virtual patients to simulate (default: 1000)",
    )
    parser.add_argument(
        "--n-cycles", dest="n_cycles", type=int, default=12,
        help="Monitoring cycles per patient (default: 12 = 12 months)",
    )
    parser.add_argument(
        "--cycle-days", dest="cycle_days", type=int, default=28,
        help="Days between monitoring cycles (default: 28)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output JSON path (optional)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Load config
    cfg_path = Path(args.config_json)
    if not cfg_path.exists():
        logger.error("Config not found: %s", cfg_path)
        return 1
    with open(cfg_path) as f:
        config = json.load(f)

    logger.info("=" * 72)
    logger.info("GENEVO2 Safety Validation — %d virtual patients", args.n_simulations)
    logger.info("  Config : %s", cfg_path)
    logger.info("  Cycles : %d × %d days", args.n_cycles, args.cycle_days)
    logger.info("  Seed   : %d", args.seed)
    logger.info("=" * 72)

    results = run_validation(
        config=config,
        n_simulations=args.n_simulations,
        n_cycles=args.n_cycles,
        cycle_days=args.cycle_days,
        seed=args.seed,
    )

    summary = aggregate_results(results)
    print_report(summary)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Report saved: %s", out_path)

    # Exit code: 0 if feasibility > 85%
    return 0 if summary.get("feasible_pct", 0) > 85.0 else 1


if __name__ == "__main__":
    sys.exit(main())
