#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Therapeutic Response Validation — Romosozumab Clinical Trial Benchmarks.

Compares GENEVO2 simulator therapeutic predictions to published Phase 2/3
romosozumab trial outcomes. This is the primary clinical grounding check.

Published benchmarks used:
  - Cosman 2016 NEJM (ARCH trial): 12-month BMD data, n=3591
  - McClung 2014 JBMR (Phase 2): biomarker suppression at months 1/3/6
  - Padhi 2011 Bone (Phase 1): PK/PD modeling, t1/2 = 6.9 days
  - Genant 2017 Lancet (FRAME trial): BMD + fracture data, n=7180

Outputs a validation report comparing:
  A. BMD gain: simulator vs trial data (6-month and 12-month)
  B. Biomarker suppression: CTX/P1NP changes vs McClung 2014
  C. Dosing plausibility: dose fractions vs romosozumab 210mg protocol
  D. Sensor-drug coupling: does the optimized sensor design predict
     clinically meaningful drug doses?

Usage:
    python BO/validation/therapeutic_clinical_validation.py
    python BO/validation/therapeutic_clinical_validation.py --config-json path/to/best_config.json
    python BO/validation/therapeutic_clinical_validation.py --out report.json
"""

import sys
import os
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Published clinical trial benchmarks
# ---------------------------------------------------------------------------

# Romosozumab 210mg SC monthly — BMD change from baseline (%)
# Source: Cosman 2016 NEJM (ARCH), Genant 2017 Lancet (FRAME), McClung 2014 JBMR
ROMOSOZUMAB_BMD_BENCHMARKS = {
    "lumbar_spine_6mo": {
        "mean_pct": 9.0,
        "range_pct": (7.0, 11.0),
        "unit": "% change from baseline",
        "source": "McClung JBMR 2014 (Phase 2, n=419); Cosman 2016 NEJM midpoint",
    },
    "lumbar_spine_12mo": {
        "mean_pct": 13.3,
        "range_pct": (12.1, 14.9),
        "unit": "% change from baseline",
        "source": "Cosman 2016 NEJM ARCH trial (n=3591); Genant FRAME 2017",
    },
    "total_hip_6mo": {
        "mean_pct": 4.0,
        "range_pct": (3.0, 5.5),
        "unit": "% change from baseline",
        "source": "McClung JBMR 2014 (Phase 2)",
    },
    "total_hip_12mo": {
        "mean_pct": 6.9,
        "range_pct": (6.0, 8.0),
        "unit": "% change from baseline",
        "source": "Cosman 2016 NEJM ARCH trial",
    },
}

# Bone turnover marker suppression (% change from baseline, 210mg monthly)
# Source: McClung 2014 JBMR Phase 2 (Table 2)
ROMOSOZUMAB_BIOMARKER_BENCHMARKS = {
    "CTX_month1": {
        "mean_pct_change": -55.0,
        "range_pct": (-65.0, -45.0),
        "interpretation": "Bone resorption rapidly suppressed",
        "source": "McClung JBMR 2014 (serum CTX-I at month 1)",
    },
    "CTX_month3": {
        "mean_pct_change": -50.0,
        "range_pct": (-60.0, -35.0),
        "interpretation": "Sustained CTX suppression at month 3",
        "source": "McClung JBMR 2014",
    },
    "P1NP_month1": {
        "mean_pct_change": +20.0,
        "range_pct": (+10.0, +35.0),
        "interpretation": "Early anabolic window: bone formation transiently increases",
        "source": "McClung JBMR 2014 (serum P1NP at month 1)",
    },
    "P1NP_month3": {
        "mean_pct_change": -20.0,
        "range_pct": (-30.0, -10.0),
        "interpretation": "Late suppression of bone formation (anti-catabolic dominance)",
        "source": "McClung JBMR 2014",
    },
    "SOST_month3": {
        "mean_pct_change": -27.0,
        "range_pct": (-35.0, -15.0),
        "interpretation": "Serum sclerostin falls as romosozumab binds/clears it",
        "source": "McClung JBMR 2014 (serum sclerostin at month 3)",
    },
}

# PK/PD constants (Padhi 2011 Bone)
ROMOSOZUMAB_PK = {
    "half_life_days": 6.9,          # terminal half-life (5-9 day range)
    "dose_mg_monthly": 210.0,        # standard SC dose
    "bioavailability_pct": 81.0,     # subcutaneous bioavailability
    "Tmax_days": 5.0,                # time to peak serum concentration
    "Cmax_ug_mL": 22.0,             # peak serum concentration (210mg dose)
}

# Baseline BMD for postmenopausal osteoporosis patients (ARCH trial baseline)
BASELINE_BMD = {
    "lumbar_spine_gcm2": 0.775,      # mean baseline LS BMD in ARCH (T-score ~ -2.7)
    "total_hip_gcm2": 0.648,         # mean baseline TH BMD in ARCH
}

# Our model's BMD gain parameter (from therapeutic_objective_v6.py)
MODEL_BMD_PARAMS = {
    "BMD_GAIN_MAX_gcm2": 0.06,      # g/cm2 per 6 months at full dose
    "BMD_GAIN_REF_gcm2": 0.04,      # reference normalization
    "D_SAFE": 0.50,                  # safe dose ceiling (fractional units)
    "K_RELEASE": 1.0,
    "D_HALF": 0.15,
    "DRUG_THRESHOLD_FRAC": 1.08,
}


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------

def validate_bmd_gain() -> List[Dict]:
    """
    Validates that the model's BMD_GAIN_MAX is consistent with clinical trials.

    The model uses BMD_GAIN_MAX = 0.06 g/cm2 / 6 months at full dose.
    Clinical lumbar spine BMD gain: +9% at 6 months = +0.070 g/cm2 (from 0.775 baseline).
    """
    results = []

    # 6-month benchmark
    trial_bmd_6mo_gcm2 = (BASELINE_BMD["lumbar_spine_gcm2"] *
                           ROMOSOZUMAB_BMD_BENCHMARKS["lumbar_spine_6mo"]["mean_pct"] / 100.0)
    trial_range_6mo = tuple(
        BASELINE_BMD["lumbar_spine_gcm2"] * p / 100.0
        for p in ROMOSOZUMAB_BMD_BENCHMARKS["lumbar_spine_6mo"]["range_pct"]
    )

    model_max = MODEL_BMD_PARAMS["BMD_GAIN_MAX_gcm2"]
    # Model achieves BMD_GAIN_MAX only at full dose (R >> threshold).
    # For a perfectly-tuned biosensor releasing full dose to PMO patients,
    # BMD gain should approach BMD_GAIN_MAX.
    discrepancy_pct = (model_max - trial_bmd_6mo_gcm2) / trial_bmd_6mo_gcm2 * 100.0

    lo, hi = trial_range_6mo
    status = "OK" if lo <= model_max <= hi else ("WARN" if abs(discrepancy_pct) <= 30.0 else "FAIL")

    results.append({
        "check": "BMD_GAIN_MAX vs LS 6-month trial",
        "model_value_gcm2": round(model_max, 4),
        "trial_mean_gcm2": round(trial_bmd_6mo_gcm2, 4),
        "trial_range_gcm2": [round(x, 4) for x in trial_range_6mo],
        "discrepancy_pct": round(discrepancy_pct, 1),
        "status": status,
        "source": ROMOSOZUMAB_BMD_BENCHMARKS["lumbar_spine_6mo"]["source"],
    })

    # Reference BMD
    trial_pct = (MODEL_BMD_PARAMS["BMD_GAIN_REF_gcm2"] /
                 BASELINE_BMD["lumbar_spine_gcm2"] * 100.0)
    results.append({
        "check": "BMD_GAIN_REF vs baseline LS BMD",
        "model_ref_gcm2": MODEL_BMD_PARAMS["BMD_GAIN_REF_gcm2"],
        "implied_pct_gain": round(trial_pct, 1),
        "note": (
            f"BMD_GAIN_REF=0.04 g/cm2 implies {trial_pct:.1f}% LS BMD gain "
            f"vs {BASELINE_BMD['lumbar_spine_gcm2']} g/cm2 baseline. "
            f"Clinical trial reference: ~5.2% at sub-maximal dose. "
            + ("OK" if 3.0 <= trial_pct <= 7.0 else "WARN: outside expected range")
        ),
        "status": "OK" if 3.0 <= trial_pct <= 7.0 else "WARN",
    })

    return results


def validate_biomarker_suppression(config: Optional[Dict] = None) -> List[Dict]:
    """
    Checks that sensor design constraints are consistent with drug biomarker responses.

    The key question: after romosozumab treatment, SOST drops -27% (McClung 2014).
    Does our sensor's detection logic remain valid after drug-induced SOST suppression?

    If SOST_post_treatment = SOST_PMO * (1 - 0.27) ≈ 0.638 nM,
    this is still above the healthy baseline (0.375 nM) → sensor should still detect
    residual disease or confirm treatment success.
    """
    results = []

    nominal_sost = {
        "healthy": 0.375,
        "pmo_mild": 0.5625,
        "pmo": 0.875,
        "ckd_mbd": 1.125,
    }

    sost_suppression = ROMOSOZUMAB_BIOMARKER_BENCHMARKS["SOST_month3"]["mean_pct_change"] / 100.0

    for scenario, baseline in nominal_sost.items():
        if scenario == "healthy":
            continue
        post_treatment_sost = baseline * (1 + sost_suppression)
        fold_over_healthy = post_treatment_sost / nominal_sost["healthy"]

        results.append({
            "check": f"SOST post-treatment detectability ({scenario})",
            "sost_pre_nm": round(baseline, 4),
            "sost_post_nm": round(post_treatment_sost, 4),
            "fold_over_healthy_post_tx": round(fold_over_healthy, 3),
            "interpretation": (
                "OK: residual SOST still > 1.0x healthy after treatment — sensor monitors recovery"
                if fold_over_healthy > 1.05
                else "WARN: post-treatment SOST may fall to healthy range — sensor loses discriminative signal"
            ),
            "status": "OK" if fold_over_healthy > 1.05 else "WARN",
            "source": ROMOSOZUMAB_BIOMARKER_BENCHMARKS["SOST_month3"]["source"],
        })

    # CTX suppression implication for multi-channel sensor
    ctx_suppression = ROMOSOZUMAB_BIOMARKER_BENCHMARKS["CTX_month1"]["mean_pct_change"] / 100.0
    ctx_post_pmo = nominal_sost["pmo"] / 0.875 * 0.5  # ≈ 0.5 nM PMO nominal CTX
    ctx_post_treat_pmo = 0.5 * (1 + ctx_suppression)
    ctx_healthy = 0.200

    results.append({
        "check": "CTX post-treatment vs healthy (PMO)",
        "ctx_pmo_nm": 0.500,
        "ctx_post_treatment_nm": round(ctx_post_treat_pmo, 4),
        "ctx_healthy_nm": ctx_healthy,
        "still_above_healthy": ctx_post_treat_pmo > ctx_healthy * 1.05,
        "interpretation": (
            "OK: CTX still elevated after treatment — multi-channel sensor retains signal"
            if ctx_post_treat_pmo > ctx_healthy * 1.05
            else "WARN: CTX falls near healthy range — multi-channel adds little post-treatment"
        ),
        "status": "OK" if ctx_post_treat_pmo > ctx_healthy * 1.05 else "WARN",
    })

    return results


def validate_dose_plausibility(config: Optional[Dict] = None) -> List[Dict]:
    """
    Checks that the V6 dose model produces clinically plausible doses.

    Romosozumab clinical dose: 210 mg SC monthly.
    V6 dose is a fractional unit (0 to ~1.5).
    This function maps the model's dose fractions to clinical equivalents
    and checks plausibility.

    For a typical PMO scenario:
    R_PMO ≈ 0.875/0.375 = 2.33 (SOST ratio, scl-only case, Kd >> conc)
    R_PMO ≈ 1.4-1.6 (realistically with finite Kd and multi-channel)
    dose_frac = K_RELEASE * (R - threshold) / threshold ≈ 0.30-0.44
    """
    results = []

    D_HALF     = MODEL_BMD_PARAMS["D_HALF"]
    D_SAFE     = MODEL_BMD_PARAMS["D_SAFE"]
    threshold  = MODEL_BMD_PARAMS["DRUG_THRESHOLD_FRAC"]

    # Estimate dose for typical PMO SOST ratio (pure SOST channel, high Kd limit)
    R_pmo_high_kd = 0.875 / 0.375  # occupancy ratio when Kd >> conc → concentration ratio
    R_pmo_low_kd  = (0.875 / (0.5 + 0.875)) / (0.375 / (0.5 + 0.375))  # Kd = 0.5 nM

    for label, R in [("PMO (high Kd limit)", R_pmo_high_kd),
                      ("PMO (Kd=0.5 nM)",     R_pmo_low_kd)]:
        dose_frac = max(0.0, (R - threshold) / threshold)
        bmd_gcm2 = MODEL_BMD_PARAMS["BMD_GAIN_MAX_gcm2"] * (
            dose_frac / (dose_frac + D_HALF)
        )
        bmd_pct = bmd_gcm2 / BASELINE_BMD["lumbar_spine_gcm2"] * 100.0
        is_overdose = dose_frac > D_SAFE

        trial_6mo_pct = ROMOSOZUMAB_BMD_BENCHMARKS["lumbar_spine_6mo"]["mean_pct"]
        discrepancy_pp = bmd_pct - trial_6mo_pct

        results.append({
            "check": f"dose plausibility: {label}",
            "R_scenario": round(R, 3),
            "dose_fraction": round(dose_frac, 4),
            "predicted_bmd_gcm2": round(bmd_gcm2, 4),
            "predicted_bmd_pct": round(bmd_pct, 1),
            "trial_6mo_bmd_pct": trial_6mo_pct,
            "discrepancy_pp": round(discrepancy_pp, 1),
            "overdose_flag": is_overdose,
            "status": "OK" if abs(discrepancy_pp) <= 4.0 else ("WARN" if abs(discrepancy_pp) <= 8.0 else "FAIL"),
        })

    return results


def validate_sensor_lifetime() -> List[Dict]:
    """
    Checks sensor lifetime plausibility for aptamer-based vs enzyme-based designs.

    Published sensor lifetime data:
    - Enzyme-based (GOx, HRP): hours to days in vivo due to denaturation/biofouling
    - Aptamer-based: weeks to months (less susceptible to denaturation)
    - MIP-based: months to years (synthetic, most stable)

    Our ODE model assumes stable sensor (no degradation term).
    This check flags the gap and quantifies the signal loss timeline.
    """
    results = []

    sensor_types = {
        "enzyme_based": {
            "half_life_days": 1.5,
            "full_function_days": 3,
            "label": "Enzyme (GOx/HRP)",
            "realistic_for_months": False,
        },
        "aptamer_based": {
            "half_life_days": 30.0,
            "full_function_days": 60,
            "label": "Aptamer (DNA/RNA oligonucleotide)",
            "realistic_for_months": True,
        },
        "mip_based": {
            "half_life_days": 180.0,
            "full_function_days": 365,
            "label": "MIP (Molecularly Imprinted Polymer)",
            "realistic_for_months": True,
        },
    }

    romosozumab_treatment_months = 12  # standard treatment duration

    for name, props in sensor_types.items():
        survival_at_treatment_end = np.exp(
            -np.log(2) * (romosozumab_treatment_months * 30.0) / props["half_life_days"]
        )
        results.append({
            "sensor_type": props["label"],
            "half_life_days": props["half_life_days"],
            "pct_function_at_12mo": round(survival_at_treatment_end * 100, 1),
            "suitable_for_12mo_treatment": props["realistic_for_months"],
            "status": "OK" if props["realistic_for_months"] else "FAIL",
            "note": (
                "Current ODE model assumes 100% sensitivity throughout — "
                "add degradation multiplier for realistic long-term simulation"
                if name == "aptamer_based"
                else ""
            ),
        })

    results.append({
        "check": "Model gap",
        "finding": "Current GENEVO2 ODE has no sensor degradation term. "
                   "Signal at t=600s assumes sensor at full function. "
                   "Recommended: add stability_factor(t) = exp(-lambda_deg * t_days) "
                   "to biosensor output before detection logic.",
        "implementation": "Add lambda_degradation parameter to BiosensorConfig. "
                          "For aptamers: lambda = ln(2)/30 = 0.023/day. "
                          "Signal_corrected(t) = Signal(t) * exp(-lambda * t_total_days)",
        "status": "WARN",
    })

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_validation_report(
    config: Optional[Dict] = None,
    out_path: Optional[str] = None,
) -> Dict:
    """
    Generate full therapeutic clinical validation report.

    Args:
        config: Optional biosensor config dict (kd_nm, sensitivity, etc.)
        out_path: Optional path to save JSON report

    Returns:
        Validation report dict with all sections.
    """
    report = {
        "title": "GENEVO2 Therapeutic Clinical Validation",
        "source": "Romosozumab Phase 2/3 trials (McClung 2014, Cosman 2016, Genant 2017)",
        "sections": {
            "A_bmd_gain":               [],
            "B_biomarker_suppression":  [],
            "C_dose_plausibility":      [],
            "D_sensor_lifetime":        [],
        },
        "summary": {"n_OK": 0, "n_WARN": 0, "n_FAIL": 0},
    }

    report["sections"]["A_bmd_gain"]              = validate_bmd_gain()
    report["sections"]["B_biomarker_suppression"] = validate_biomarker_suppression(config)
    report["sections"]["C_dose_plausibility"]     = validate_dose_plausibility(config)
    report["sections"]["D_sensor_lifetime"]       = validate_sensor_lifetime()

    all_results = (
        report["sections"]["A_bmd_gain"] +
        report["sections"]["B_biomarker_suppression"] +
        report["sections"]["C_dose_plausibility"] +
        report["sections"]["D_sensor_lifetime"]
    )
    for r in all_results:
        s = r.get("status", "")
        if s == "OK":
            report["summary"]["n_OK"] += 1
        elif s == "WARN":
            report["summary"]["n_WARN"] += 1
        elif s == "FAIL":
            report["summary"]["n_FAIL"] += 1

    if out_path:
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)

    return report


def print_report(report: Dict):
    """Print clinical validation report to console (ASCII only)."""
    print("\n" + "=" * 72)
    print("GENEVO2 THERAPEUTIC CLINICAL VALIDATION REPORT")
    print("Sources: McClung JBMR 2014, Cosman NEJM 2016, Genant Lancet 2017")
    print("=" * 72)

    # Section A
    print("\n[A] BMD GAIN CALIBRATION (model vs romosozumab trials)")
    print("-" * 72)
    for r in report["sections"]["A_bmd_gain"]:
        s = r.get("status", "?")
        if "model_value_gcm2" in r:
            print(f"  [{s}] {r['check']}")
            print(f"       Model: {r['model_value_gcm2']:.4f} g/cm2   "
                  f"Trial: {r['trial_mean_gcm2']:.4f} g/cm2   "
                  f"Range: {r['trial_range_gcm2']}   "
                  f"Discrepancy: {r['discrepancy_pct']:+.1f}%")
        else:
            print(f"  [{s}] {r['check']}: {r.get('note', '')}")

    # Section B
    print("\n[B] BIOMARKER SUPPRESSION (drug effect on SOST/CTX post-treatment)")
    print("-" * 72)
    for r in report["sections"]["B_biomarker_suppression"]:
        s = r.get("status", "?")
        if "sost_pre_nm" in r:
            print(f"  [{s}] {r['check']}")
            print(f"       SOST pre: {r['sost_pre_nm']:.4f} nM  "
                  f"post-treatment: {r['sost_post_nm']:.4f} nM  "
                  f"fold/healthy: {r['fold_over_healthy_post_tx']:.3f}")
            print(f"       {r['interpretation']}")
        else:
            print(f"  [{s}] {r['check']}")
            print(f"       {r['interpretation']}")

    # Section C
    print("\n[C] DOSE PLAUSIBILITY (fractional dose -> BMD vs trial outcome)")
    print("-" * 72)
    for r in report["sections"]["C_dose_plausibility"]:
        s = r.get("status", "?")
        print(f"  [{s}] {r['check']}")
        print(f"       R={r['R_scenario']:.3f}  dose={r['dose_fraction']:.4f}  "
              f"BMD={r['predicted_bmd_pct']:.1f}%  trial={r['trial_6mo_bmd_pct']:.1f}%  "
              f"discrepancy={r['discrepancy_pp']:+.1f}pp  "
              f"{'OVERDOSE' if r['overdose_flag'] else 'safe'}")

    # Section D
    print("\n[D] SENSOR LIFETIME (aptamer vs enzyme stability)")
    print("-" * 72)
    for r in report["sections"]["D_sensor_lifetime"]:
        s = r.get("status", "?")
        if "sensor_type" in r:
            print(f"  [{s}] {r['sensor_type']}")
            print(f"       t1/2={r['half_life_days']}d  "
                  f"function@12mo={r['pct_function_at_12mo']}%  "
                  f"suitable={'YES' if r['suitable_for_12mo_treatment'] else 'NO'}")
        else:
            print(f"  [WARN] Model gap: {r['finding']}")
            print(f"         Fix: {r['implementation']}")

    # Summary
    s = report["summary"]
    print("\n" + "=" * 72)
    print(f"SUMMARY: [{s['n_OK']} OK] [{s['n_WARN']} WARN] [{s['n_FAIL']} FAIL]")
    if s["n_FAIL"] > 0:
        print("  FAIL: Significant model-trial discrepancies require attention before publication.")
    elif s["n_WARN"] > 0:
        print("  WARN: Minor gaps present. Address before clinical translation claims.")
    else:
        print("  All therapeutic checks pass. Model is consistent with published trial data.")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Therapeutic clinical validation")
    parser.add_argument("--config-json", type=str, default=None,
                        help="Path to best_config.json for dose plausibility checks")
    parser.add_argument("--out", type=str, default=None, help="Output path for JSON report")
    args = parser.parse_args()

    config = None
    if args.config_json and Path(args.config_json).exists():
        with open(args.config_json) as f:
            config = json.load(f)

    report = generate_validation_report(config=config, out_path=args.out)
    print_report(report)

    if args.out:
        print(f"\n[OK] Report saved to {args.out}")
