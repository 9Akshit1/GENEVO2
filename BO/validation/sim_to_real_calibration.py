#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sim-to-Real Calibration Tool.

Compares GENEVO2 simulator outputs to published clinical literature values
across three dimensions:

  A. Biomarker fold-changes: simulator fold-change vs literature fold-change
     per scenario (pmo_mild, pmo, ckd_mbd)

  B. Patient variability: simulator CV (from lognormal sigma) vs published
     inter-individual CVs from Mirza 2010, Garnero 1994, Cejka 2011

  C. Detection rate plausibility: whether the achieved DR (0.96) is
     meaningful given the signal difficulty (biomarker separability AUC)

Outputs a calibration report with discrepancy flags [OK] / [WARN] / [FAIL].

Usage:
    python BO/validation/sim_to_real_calibration.py
    python BO/validation/sim_to_real_calibration.py --run-simulation --n-samples 1000
    python BO/validation/sim_to_real_calibration.py --data-csv BO/data/data_v10/master_index.csv
"""

import sys
import os
import json
import logging
import argparse
import numpy as np
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Literature reference values
# ---------------------------------------------------------------------------

# Fold-changes vs premenopausal healthy women
# Source, biomarker, scenario, mean, 95% CI or SD where available
LITERATURE_FOLD_CHANGES = {
    "pmo_mild": {
        "SOST": {
            "value": 1.5,
            "range": (1.2, 1.9),
            "source": "Mirza JCEM 2010 (1-3yr post-menopause: ~1.4-1.6x)",
        },
        "CTX": {
            "value": 1.5,
            "range": (1.2, 2.0),
            "source": "Garnero JBMR 1994 (early menopause: 1.2-2.0x)",
        },
        "P1NP": {
            "value": 1.1,
            "range": (0.9, 1.5),
            "source": "Seibel JBMR 2006 (early transition: 1.1-1.5x)",
        },
    },
    "pmo": {
        "SOST": {
            "value": 2.3,
            "range": (1.8, 2.9),
            "source": "Mirza JCEM 2010 (5+yr post-menopause: 2.4x, d=2.35)",
        },
        "CTX": {
            "value": 2.5,
            "range": (1.8, 3.5),
            "source": "Garnero JBMR 1994 (established PMO: 2.0-3.0x, CV=45-55%)",
        },
        "P1NP": {
            "value": 1.5,
            "range": (1.2, 1.8),
            "source": "Schafer JCEM 2011 (established PMO: 1.2-1.8x)",
        },
    },
    "ckd_mbd": {
        "SOST": {
            "value": 3.0,
            "range": (2.0, 4.5),
            "source": "Cejka Nephrol Dial Transplant 2011 (CKD Stage 3-5: 2.5-4.0x vs premenopause)",
        },
        "CTX": {
            "value": 2.5,
            "range": (1.5, 4.0),
            "source": "Hlaing 2011 (CKD Stage 3-4: 1.5-2.5x; Stage 5D: up to 8x)",
        },
        "P1NP": {
            "value": 1.8,
            "range": (1.3, 2.5),
            "source": "Kovesdy 2014 (CKD mixed stages: 1.3-2.5x)",
        },
    },
}

# ---------------------------------------------------------------------------
# External cohort data (real patient studies — independent confirmation)
# ---------------------------------------------------------------------------

# Wu et al. 2026, Frontiers Endocrinology, PMC12812736
# n=180 established osteoporosis + n=80 controls (Chinese cohort, modern immunoassays)
# NOTE: Controls appear to be younger/mixed-age (CTX=0.16 ng/mL << postmenopausal ref 0.30)
# Use postmenopausal reference values for fold-change denominator
WU_2026 = {
    "CTX_osteo_ngml":    (0.78, 0.13),    # osteoporosis mean, SD
    "CTX_control_ngml":  (0.16, 0.05),    # mixed-age control
    "P1NP_osteo_ngml":   (68.85, 6.79),
    "P1NP_control_ngml": (30.12, 2.43),
    "n_osteo": 180, "n_control": 80,
    "source": "Wu et al. 2026 Front Endocrinol PMC12812736",
}
# Postmenopausal healthy reference (IOF/ISCD consensus)
PMC_REFS = {
    "CTX_pmc_ngml": 0.30,    # healthy postmenopausal CTX upper ref (Wang 2021 JBMR meta-analysis)
    "P1NP_pmc_ngml": 45.0,   # healthy postmenopausal P1NP (NHBHA 2012 consensus)
}

# Ardawi et al. 2011 JBMR, n=1803 women (healthy pre- and post-menopausal Saudi women)
ARDAWI_2011 = {
    "SOST_premenopausal_ngml": (0.48, 0.15),
    "SOST_postmenopausal_ngml": (1.16, 0.38),
    "source": "Ardawi JBMR 2011 (doi:10.1002/jbmr.479)",
}

# Inter-individual CVs from literature (for sigma validation)
LITERATURE_CVS = {
    "SOST": {
        "healthy_CV_pct": 31.0,
        "source": "Mirza 2010 (0.48+/-0.15 ng/mL: CV=31%)",
    },
    "CTX": {
        "healthy_CV_pct": 47.0,
        "source": "Garnero 1994 (inter-individual CV 45-55%)",
    },
    "P1NP": {
        "healthy_CV_pct": 30.0,
        "source": "Vasikaran 2011 (inter-individual CV 25-35%)",
    },
    "SOST_PMO": {
        "PMO_CV_pct": 39.0,
        "source": "Mirza 2010 PMO cohort (estimated from IQR: CV~35-45%)",
    },
    "CTX_PMO": {
        "PMO_CV_pct": 50.0,
        "source": "EuBIVAS inter-individual: 45-55%",
    },
}

# Simulator configuration (from environment_configs.py)
# Nominal concentrations in sensor compartment (nM)
SIM_NOMINAL_CONCS = {
    "healthy":  {"SOST": 0.375, "CTX": 0.200, "P1NP": 0.350},
    "pmo_mild": {"SOST": 0.5625, "CTX": 0.300, "P1NP": 0.385},
    "pmo":      {"SOST": 0.875,  "CTX": 0.500, "P1NP": 0.525},
    "ckd_mbd":  {"SOST": 1.125,  "CTX": 0.500, "P1NP": 0.625},
}

# Sigma overrides (lognormal sigma in apply_variability)
# Synced with environment_configs.py healthy_sigma_overrides (v5.3 fix):
#   Sclerostin_bone: 0.30, CTX_bone: 0.45, P1NP_bone: 0.25 (Ueland 2009, Garnero 1994, Vasikaran 2011)
# CKD_MBD Sclerostin_bone: 0.30 (Cejka 2011, was 0.50)
SIM_SIGMA_OVERRIDES = {
    "healthy":  {"SOST": 0.30, "CTX": 0.45, "P1NP": 0.25},
    "pmo":      {"SOST": 0.35, "CTX": 0.45, "P1NP": 0.25},
    "ckd_mbd":  {"SOST": 0.30, "CTX": 0.45, "P1NP": 0.40},
    "pmo_mild": {"SOST": 0.20, "CTX": 0.25, "P1NP": 0.20},
}


# ---------------------------------------------------------------------------
# Calibration checks
# ---------------------------------------------------------------------------

def lognormal_cv(sigma: float) -> float:
    """Inter-individual CV% implied by lognormal sigma."""
    return 100.0 * (np.exp(sigma ** 2) - 1.0) ** 0.5


def check_fold_changes() -> List[Dict]:
    """
    Compare simulator nominal fold-changes to literature values.

    Returns:
        List of result dicts with status flags.
    """
    healthy_concs = SIM_NOMINAL_CONCS["healthy"]
    results = []

    for scenario in ["pmo_mild", "pmo", "ckd_mbd"]:
        disease_concs = SIM_NOMINAL_CONCS[scenario]
        for biomarker in ["SOST", "CTX", "P1NP"]:
            sim_fc  = disease_concs[biomarker] / healthy_concs[biomarker]
            lit     = LITERATURE_FOLD_CHANGES[scenario][biomarker]
            lit_val = lit["value"]
            lo, hi  = lit["range"]

            if lo <= sim_fc <= hi:
                status = "OK"
            elif abs(sim_fc - lit_val) / lit_val <= 0.20:
                status = "WARN"   # within 20% of literature mean but outside range
            else:
                status = "FAIL"

            discrepancy_pct = (sim_fc - lit_val) / lit_val * 100.0
            results.append({
                "scenario":         scenario,
                "biomarker":        biomarker,
                "sim_fc":           round(sim_fc, 3),
                "lit_fc":           round(lit_val, 3),
                "lit_range":        list(lit["range"]),
                "discrepancy_pct":  round(discrepancy_pct, 1),
                "status":           status,
                "source":           lit["source"],
            })

    return results


def check_variability() -> List[Dict]:
    """
    Compare simulator lognormal sigma -> implied CV% to published inter-individual CVs.
    """
    results = []

    for scenario, sigma_map in SIM_SIGMA_OVERRIDES.items():
        for biomarker, sigma in sigma_map.items():
            sim_cv = lognormal_cv(sigma)
            # Look up literature CV
            key = biomarker if scenario == "healthy" else f"{biomarker}_{scenario.upper()[:3]}"
            lit_entry = LITERATURE_CVS.get(key) or LITERATURE_CVS.get(biomarker)

            if lit_entry:
                if "healthy_CV_pct" in lit_entry:
                    lit_cv = lit_entry["healthy_CV_pct"]
                else:
                    lit_cv = lit_entry.get("PMO_CV_pct", None)
                if lit_cv:
                    discrepancy = abs(sim_cv - lit_cv) / lit_cv * 100.0
                    status = "OK" if discrepancy <= 25.0 else ("WARN" if discrepancy <= 50.0 else "FAIL")
                    results.append({
                        "scenario":    scenario,
                        "biomarker":   biomarker,
                        "sim_sigma":   round(sigma, 3),
                        "sim_cv_pct":  round(sim_cv, 1),
                        "lit_cv_pct":  round(lit_cv, 1),
                        "discrepancy_pct": round(discrepancy, 1),
                        "status":      status,
                        "source":      lit_entry.get("source", ""),
                    })

    return results


def check_detection_plausibility(
    achieved_dr: float = 0.960,
    pmo_mild_auc: float = 0.931,
    pmo_auc: float = 0.995,
    ckd_auc: float = 0.997,
) -> Dict:
    """
    Check whether achieved DR (0.96) is meaningful given baseline AUC.

    Key question: if logistic regression on biomarkers alone achieves AUC>0.93
    for PMO-mild, does 0.96 DR from BO represent genuine optimization gain?

    From AGENT_CONTEXT.md:
      - Baseline logistic regression AUC: PMO-mild=0.931, PMO=0.995, CKD=0.997
      - BO real DR: PMO-mild=0.90, PMO=1.00, CKD=0.98
    """
    # Naive upper bound on DR given AUC (Bamber 1975 relationship approximation)
    # DR_max ~ 2 * AUC * (1 - FP_threshold) / (1 + AUC)  [rough approximation]
    # More accurately: at optimal threshold with balanced accuracy
    baseline_dr_estimates = {
        "pmo_mild": min(1.0, pmo_mild_auc),    # AUC gives rough upper bound on DR
        "pmo":      min(1.0, pmo_auc),
        "ckd_mbd":  min(1.0, ckd_auc),
    }
    bo_drs = {
        "pmo_mild": 0.90,
        "pmo":      1.00,
        "ckd_mbd":  0.98,
    }

    analysis = {
        "achieved_dr_mean": achieved_dr,
        "baseline_auc_logistic": {
            "pmo_mild": pmo_mild_auc,
            "pmo":      pmo_auc,
            "ckd_mbd":  ckd_auc,
        },
        "bo_dr_per_scenario": bo_drs,
        "interpretation": [],
    }

    # PMO-mild is the critical case
    if pmo_mild_auc > 0.90:
        analysis["interpretation"].append(
            "WARN: PMO-mild AUC=0.931 is HIGH. Biomarker separability is already good. "
            "The detection task may not be challenging. "
            "BO improvement over logistic threshold is limited by inherent biomarker overlap."
        )
    else:
        analysis["interpretation"].append(
            "OK: PMO-mild AUC suggests real separation challenge for the biosensor."
        )

    if bo_drs["pmo_mild"] >= 0.85:
        analysis["interpretation"].append(
            "OK: PMO-mild DR=0.90 exceeds common clinical biosensor benchmarks (>80% sensitivity)."
        )

    # Marginal gain estimate
    # If logistic classifier gets AUC=0.93 with optimal threshold -> ~DR=0.93 at FP=0.07
    # BO optimized biosensor achieves DR=0.90 with FP=~0.05
    # This is roughly comparable — gain is FP reduction, not DR improvement
    analysis["interpretation"].append(
        "NOTE: The BO biosensor (DR=0.90, FP=0.05) vs logistic threshold (AUC=0.93, DR~0.93 at FP=0.07) "
        "shows BO gains mainly through FP reduction, not DR increase. "
        "This is clinically meaningful: lower false alarm rate reduces unnecessary drug delivery."
    )

    # Overall verdict
    analysis["verdict"] = (
        "The 0.96 mean DR represents genuine engineering achievement, but the HARD SCIENTIFIC CLAIM "
        "should be: 'Our optimized biosensor achieves [DR=0.90/1.00/0.98] at [FP=5%] for [PMO-mild/PMO/CKD], "
        "with kd optimized analytically from Langmuir theory.' Not simply '0.96 DR is impressive.'"
    )

    return analysis


def check_external_cohort() -> List[Dict]:
    """
    Validate simulator fold-changes against real patient cohort data.

    Uses Wu et al. 2026 (n=260) for CTX/P1NP and Ardawi 2011 (n=1803) for SOST.
    Compares model data_v16 fold-changes to independently-measured clinical values.
    Gate criterion: error < 15% = PASS, < 30% = WARN, else FAIL.
    """
    results = []

    # Model fold-changes from data_v16 (computed offline: pandas groupby on sclerostin_mean, ctx_mean, p1np_mean)
    MODEL_FC = {
        "pmo":     {"SOST": 2.368, "CTX": 2.409, "P1NP": 1.517},
        "ckd_mbd": {"SOST": 3.005, "CTX": 2.495, "P1NP": 1.883},
    }

    # SOST: Ardawi 2011 — postmenopausal/premenopausal = fold-change for PMO
    sost_pmo = ARDAWI_2011["SOST_postmenopausal_ngml"][0] / ARDAWI_2011["SOST_premenopausal_ngml"][0]
    # CKD SOST: ~2.5x from PLOS ONE 2017 (dialysis vs controls, n=49 ctrl + 100 HD)
    sost_ckd = 2.5

    # CTX/P1NP: Wu 2026 osteoporosis vs postmenopausal healthy reference (IOF/ISCD)
    ctx_pmo  = WU_2026["CTX_osteo_ngml"][0] / PMC_REFS["CTX_pmc_ngml"]
    p1np_pmo = WU_2026["P1NP_osteo_ngml"][0] / PMC_REFS["P1NP_pmc_ngml"]

    clinical_fc = {
        "pmo":     {"SOST": sost_pmo, "CTX": ctx_pmo, "P1NP": p1np_pmo},
        "ckd_mbd": {"SOST": sost_ckd, "CTX": None,    "P1NP": None},
    }
    sources = {
        "pmo":     {"SOST": ARDAWI_2011["source"],
                    "CTX":  WU_2026["source"] + " (vs IOF pmc ref 0.30 ng/mL)",
                    "P1NP": WU_2026["source"] + " (vs NHBHA pmc ref 45 ng/mL)"},
        "ckd_mbd": {"SOST": "PLOS ONE 2017 doi:10.1371/journal.pone.0176411 (dialysis 2.5x ctrl)"},
    }

    for scenario, bio_dict in MODEL_FC.items():
        for biomarker, sim_fc in bio_dict.items():
            clin_fc = clinical_fc[scenario].get(biomarker)
            if clin_fc is None:
                continue
            err_pct = abs(sim_fc - clin_fc) / clin_fc * 100.0
            if err_pct <= 15.0:
                status = "OK"
            elif err_pct <= 30.0:
                status = "WARN"
            else:
                status = "FAIL"
            results.append({
                "scenario":       scenario,
                "biomarker":      biomarker,
                "model_fc":       round(sim_fc, 3),
                "clinical_fc":    round(clin_fc, 3),
                "error_pct":      round(err_pct, 1),
                "status":         status,
                "source":         sources[scenario].get(biomarker, ""),
            })

    return results


def check_simulator_vs_csv(csv_path: str) -> Dict:
    """
    Load a simulation CSV and compare measured statistics to expected values.

    Checks:
      - Actual median fold-changes in data vs environment_configs.py nominal values
      - Actual CV% in data vs lognormal sigma expectation
    """
    try:
        import pandas as pd
    except ImportError:
        return {"error": "pandas required for CSV analysis"}

    df = pd.read_csv(csv_path)
    required = ["scenario", "detection_rate"]
    if not all(c in df.columns for c in required):
        return {"error": f"CSV missing required columns: {required}"}

    results = {}
    for scenario in ["healthy", "pmo_mild", "pmo", "ckd_mbd"]:
        mask = df["scenario"] == scenario
        n = mask.sum()
        if n < 10:
            continue
        subset = df[mask]
        dr_mean = float(subset["detection_rate"].mean())
        dr_std  = float(subset["detection_rate"].std())
        results[scenario] = {
            "n":       int(n),
            "dr_mean": round(dr_mean, 3),
            "dr_std":  round(dr_std, 3),
        }

        # Check biomarker columns if present
        for biomarker, col in [("SOST", "sclerostin_signal"), ("CTX", "ctx_signal"), ("P1NP", "p1np_signal")]:
            if col in df.columns:
                vals = subset[col].dropna()
                if len(vals) > 5:
                    sim_cv = float(np.std(vals, ddof=1) / np.mean(vals) * 100.0)
                    sim_fc = float(np.median(vals) / df[df["scenario"] == "healthy"][col].dropna().median())
                    results[scenario][f"{biomarker}_cv_pct"]  = round(sim_cv, 1)
                    results[scenario][f"{biomarker}_fc_sim"]  = round(sim_fc, 3)
                    results[scenario][f"{biomarker}_fc_expected"] = round(
                        SIM_NOMINAL_CONCS[scenario][biomarker] / SIM_NOMINAL_CONCS["healthy"][biomarker], 3
                    )

    return results


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------

def generate_calibration_report(
    data_csv: Optional[str] = None,
    out_path: Optional[str] = None,
) -> Dict:
    """
    Generate full sim-to-real calibration report.

    Returns:
        Dictionary with all check results.
    """
    report = {
        "sections": {
            "A_fold_changes":      [],
            "B_variability":       [],
            "C_detection":         {},
            "D_csv_stats":         {},
            "E_external_cohort":   [],
        },
        "summary": {
            "n_OK": 0, "n_WARN": 0, "n_FAIL": 0,
        }
    }

    # Section A: Fold-change comparison
    fc_results = check_fold_changes()
    report["sections"]["A_fold_changes"] = fc_results

    # Section B: Variability comparison
    var_results = check_variability()
    report["sections"]["B_variability"] = var_results

    # Section C: Detection plausibility
    det_result = check_detection_plausibility()
    report["sections"]["C_detection"] = det_result

    # Section D: CSV data stats (if provided)
    if data_csv:
        csv_result = check_simulator_vs_csv(data_csv)
        report["sections"]["D_csv_stats"] = csv_result

    # Section E: External cohort confirmation (real patient data)
    ext_results = check_external_cohort()
    report["sections"]["E_external_cohort"] = ext_results

    # Count status flags
    for r in fc_results + var_results + ext_results:
        status = r.get("status", "?")
        if status == "OK":
            report["summary"]["n_OK"] += 1
        elif status == "WARN":
            report["summary"]["n_WARN"] += 1
        elif status == "FAIL":
            report["summary"]["n_FAIL"] += 1

    if out_path:
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)

    return report


def print_report(report: Dict):
    """Print calibration report to console (ASCII only)."""
    print("\n" + "=" * 70)
    print("SIM-TO-REAL CALIBRATION REPORT")
    print("=" * 70)

    # Section A
    print("\n[A] FOLD-CHANGE CALIBRATION (simulator vs literature)")
    print("-" * 70)
    print(f"{'Scenario':<12} {'Biomarker':<8} {'Sim_FC':>7} {'Lit_FC':>7} {'Lit_Range':<14} {'Discrep%':>9} {'Status'}")
    print("-" * 70)
    for r in report["sections"]["A_fold_changes"]:
        lo, hi = r["lit_range"]
        print(
            f"{r['scenario']:<12} {r['biomarker']:<8} {r['sim_fc']:>7.3f} "
            f"{r['lit_fc']:>7.3f} [{lo:.1f},{hi:.1f}]{'':<5} "
            f"{r['discrepancy_pct']:>8.1f}% [{r['status']}]"
        )

    # Section B
    print("\n[B] VARIABILITY CALIBRATION (lognormal sigma vs literature CV%)")
    print("-" * 70)
    print(f"{'Scenario':<12} {'Biomarker':<8} {'Sigma':>6} {'SimCV%':>7} {'LitCV%':>7} {'Discrep%':>9} {'Status'}")
    print("-" * 70)
    for r in report["sections"]["B_variability"]:
        print(
            f"{r['scenario']:<12} {r['biomarker']:<8} {r['sim_sigma']:>6.2f} "
            f"{r['sim_cv_pct']:>7.1f} {r['lit_cv_pct']:>7.1f} "
            f"{r['discrepancy_pct']:>8.1f}% [{r['status']}]"
        )

    # Section C
    print("\n[C] DETECTION PLAUSIBILITY")
    print("-" * 70)
    det = report["sections"]["C_detection"]
    print(f"  Achieved DR mean: {det.get('achieved_dr_mean', 'N/A')}")
    print(f"  Baseline AUC (logistic regression on biomarkers):")
    for sc, auc in det.get("baseline_auc_logistic", {}).items():
        print(f"    {sc:<12}: AUC={auc:.3f}")
    print(f"\n  Interpretation:")
    for line in det.get("interpretation", []):
        print(f"    - {line}")
    print(f"\n  Verdict: {det.get('verdict', '')}")

    # Section E: External cohort
    print("\n[E] EXTERNAL COHORT CONFIRMATION (model data_v16 vs real patient studies)")
    print("-" * 70)
    print(f"{'Scenario':<12} {'Biomarker':<8} {'Model_FC':>9} {'Clinical_FC':>12} {'Error%':>8} {'Status'}")
    print("-" * 70)
    for r in report["sections"]["E_external_cohort"]:
        print(
            f"{r['scenario']:<12} {r['biomarker']:<8} {r['model_fc']:>9.3f} "
            f"{r['clinical_fc']:>12.3f} {r['error_pct']:>7.1f}% [{r['status']}]"
        )
        print(f"  Source: {r['source']}")

    # Summary
    s = report["summary"]
    print("\n" + "=" * 70)
    print(f"SUMMARY: [{s['n_OK']} OK] [{s['n_WARN']} WARN] [{s['n_FAIL']} FAIL]")
    if s["n_FAIL"] > 0:
        print("  FAIL: Simulator fold-changes or variability significantly diverges from literature.")
        print("        Review environment_configs.py sigma_overrides and scenario concentrations.")
    elif s["n_WARN"] > 0:
        print("  WARN: Minor discrepancies present. Review before publication.")
    else:
        print("  All checks passed. Simulator is consistent with literature constraints.")
    print("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Sim-to-real calibration audit")
    parser.add_argument("--data-csv", type=str, default=None, help="Path to master_index.csv for data stats")
    parser.add_argument("--out", type=str, default=None, help="Output path for JSON report")
    args = parser.parse_args()

    report = generate_calibration_report(
        data_csv=args.data_csv,
        out_path=args.out,
    )
    print_report(report)

    if args.out:
        print(f"\n[OK] Report saved to {args.out}")
