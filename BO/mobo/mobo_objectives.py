#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-objective formulation for biosensor MOBO.

Decomposes the V6 composite objective into three separate objectives
that a Pareto-based optimizer can trade off without arbitrary weighting:

  f1(config) = DR_mean        (maximize: mean disease detection rate)
  f2(config) = therapeutic    (maximize: net BMD gain from drug dosing)
  f3(config) = specificity    (maximize: 1 - FP_rate on healthy patients)

This is strictly more informative than V6's weighted sum, which hides trade-off
structure. The Pareto front exposes the full set of non-dominated designs for
a clinician to choose from.

Reference point for hypervolume computation:
  ref = [0.0, -0.5, 0.0]  (worst-case: no detection, harmful dosing, 100% FP)
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

# Nominal concentrations (sensor, nM) from environment_configs.py
_NOMINAL_CONCS = {
    "healthy":  {"scl": 0.375, "ctx": 0.200, "p1np": 0.350},
    "pmo_mild": {"scl": 0.5625, "ctx": 0.300, "p1np": 0.385},
    "pmo":      {"scl": 0.875,  "ctx": 0.500, "p1np": 0.525},
    "ckd_mbd":  {"scl": 1.125,  "ctx": 0.500, "p1np": 0.625},
}

# V6 therapeutic PK/PD constants (mirror TherapeuticObjectiveV6)
DRUG_THRESHOLD_FRAC = 1.08
K_RELEASE = 1.0
D_HALF = 0.15
D_SAFE = 0.50
ALPHA_OVERDOSE = 3.0
BMD_GAIN_MAX = 0.06
BMD_GAIN_REF = 0.04
TREAT_W_MILD = 0.55
TREAT_W_PMO  = 0.25
TREAT_W_CKD  = 0.20

# Hypervolume reference point for 3-objective space [DR_mean, therapeutic, specificity]
REFERENCE_POINT = np.array([0.0, -0.5, 0.0])

OBJECTIVE_NAMES = ["DR_mean", "therapeutic_mean", "specificity"]
N_OBJECTIVES = 3


def _occupancy(conc: float, kd: float) -> float:
    return float(conc / (kd + conc + 1e-12))


def _R_scenario(config: dict, scenario: str) -> float:
    """Sensitivity-independent occupancy ratio (disease / healthy)."""
    kd      = float(config.get("kd_nm", 1.0))
    kd_ctx  = float(config.get("kd_ctx_nm", 1.0))
    kd_p1np = float(config.get("kd_p1np_nm", 1.0))
    w_ctx   = float(config.get("w_ctx", 0.1))
    w_p1np  = float(config.get("w_p1np", 0.1))
    w_scl   = max(0.0, 1.0 - w_ctx - w_p1np)

    c_s = _NOMINAL_CONCS[scenario]
    c_h = _NOMINAL_CONCS["healthy"]

    occ_s = w_scl * _occupancy(c_s["scl"], kd) + w_ctx * _occupancy(c_s["ctx"], kd_ctx) + w_p1np * _occupancy(c_s["p1np"], kd_p1np)
    occ_h = w_scl * _occupancy(c_h["scl"], kd) + w_ctx * _occupancy(c_h["ctx"], kd_ctx) + w_p1np * _occupancy(c_h["p1np"], kd_p1np)
    return float(occ_s / max(occ_h, 1e-12))


def _drug_dose_bmd(config: dict, scenario: str) -> Tuple[float, float]:
    R = _R_scenario(config, scenario)
    frac = max(0.0, (R - DRUG_THRESHOLD_FRAC) / DRUG_THRESHOLD_FRAC)
    dose = K_RELEASE * frac
    overdose = max(0.0, dose / D_SAFE - 1.0) ** 2 * ALPHA_OVERDOSE
    if dose <= 0:
        bmd = 0.0
    else:
        bmd = BMD_GAIN_MAX * dose / (D_HALF + dose) / BMD_GAIN_REF - overdose
    return float(dose), float(bmd)


class MOBOObjectives:
    """
    Evaluates the three MOBO objectives for a biosensor configuration.

    All three objectives are to be MAXIMIZED.

    Usage:
        obj = MOBOObjectives(surrogate_loader)
        y = obj.evaluate(config)          # returns (3,) array [DR, therapeutic, specificity]
        is_feasible = obj.is_feasible(config, y)
    """

    # Hard constraint thresholds (per-scenario — PMO-mild is intrinsically harder)
    MIN_DR_PMO_MILD = 0.55   # PMO-mild: smaller biomarker contrast, inherently harder
    MIN_DR_PMO      = 0.80   # Established PMO: strong SOST signal
    MIN_DR_CKD      = 0.80   # CKD-MBD: strong SOST+CTX signal
    MAX_FP_RATE     = 0.10

    def __init__(self, surrogate_loader):
        self.loader = surrogate_loader

    def _predict(self, config: dict) -> dict:
        kd     = config.get("kd_nm", 1.0)
        sens   = config.get("sensitivity", 1.0)
        rt     = config.get("response_time_s", 500.0)
        btype  = config["biosensor_type"]
        noise  = config["noise_preset"]
        kd_ctx  = config.get("kd_ctx_nm", 0.0) if btype == "array" else 0.0
        kd_p1np = config.get("kd_p1np_nm", 0.0) if btype == "array" else 0.0
        w_ctx   = config.get("w_ctx", 0.0) if btype == "array" else 0.0
        w_p1np  = config.get("w_p1np", 0.0) if btype == "array" else 0.0

        results = {}
        for scenario in ("pmo_mild", "pmo", "ckd_mbd", "healthy"):
            dr, fnr, ttd = self.loader.predict(
                kd_nm=kd, sensitivity=sens, response_time=rt,
                biosensor_type=btype, noise_preset=noise, scenario=scenario,
                kd_ctx=kd_ctx, kd_p1np=kd_p1np, w_ctx=w_ctx, w_p1np=w_p1np,
            )
            results[scenario] = {"dr": dr, "fnr": fnr, "ttd": ttd}
        return results

    def evaluate(self, config: dict) -> np.ndarray:
        """
        Compute [f1, f2, f3] = [DR_mean, therapeutic_mean, specificity].

        Returns:
            (3,) float array, all in [-0.5, 1.0] range (approximately).
        """
        try:
            preds = self._predict(config)

            # f1: detection reliability (mean across disease scenarios)
            dr_vals = [preds[s]["dr"] for s in ("pmo_mild", "pmo", "ckd_mbd")]
            f1_dr = float(np.mean(dr_vals))

            # f2: therapeutic BMD gain (sensitivity-independent)
            if config.get("biosensor_type") == "array":
                _, bmd_mild = _drug_dose_bmd(config, "pmo_mild")
                _, bmd_pmo  = _drug_dose_bmd(config, "pmo")
                _, bmd_ckd  = _drug_dose_bmd(config, "ckd_mbd")
            else:
                bmd_mild = bmd_pmo = bmd_ckd = 0.5
            f2_therapeutic = float(np.clip(
                TREAT_W_MILD * bmd_mild + TREAT_W_PMO * bmd_pmo + TREAT_W_CKD * bmd_ckd,
                -0.5, 1.0
            ))

            # f3: specificity = 1 - FP_rate
            fp_rate = preds["healthy"]["dr"]
            f3_specificity = float(np.clip(1.0 - fp_rate, 0.0, 1.0))

            return np.array([f1_dr, f2_therapeutic, f3_specificity], dtype=float)

        except Exception as e:
            logger.error(f"MOBOObjectives.evaluate error: {e}")
            return np.array([0.0, -0.5, 0.0])

    def evaluate_with_details(self, config: dict) -> Tuple[np.ndarray, Dict]:
        """Evaluate objectives with full diagnostic breakdown."""
        try:
            preds = self._predict(config)
            y = self.evaluate(config)

            details = {
                "f1_dr_mean":        y[0],
                "f2_therapeutic":    y[1],
                "f3_specificity":    y[2],
                "dr_pmo_mild":       preds["pmo_mild"]["dr"],
                "dr_pmo":            preds["pmo"]["dr"],
                "dr_ckd_mbd":        preds["ckd_mbd"]["dr"],
                "dr_healthy":        preds["healthy"]["dr"],
                "fnr_mean":          float(np.mean([preds[s]["fnr"] for s in ("pmo_mild", "pmo", "ckd_mbd")])),
                "ttd_mean":          float(np.mean([preds[s]["ttd"] for s in ("pmo_mild", "pmo", "ckd_mbd")])),
            }
            if config.get("biosensor_type") == "array":
                for scenario, key in [("pmo_mild", "mild"), ("pmo", "pmo"), ("ckd_mbd", "ckd")]:
                    dose, bmd = _drug_dose_bmd(config, scenario)
                    details[f"dose_{key}"] = dose
                    details[f"bmd_{key}"] = bmd
                details["R_pmo_mild"] = _R_scenario(config, "pmo_mild")
                details["R_pmo"] = _R_scenario(config, "pmo")
                details["R_ckd"] = _R_scenario(config, "ckd_mbd")

            return y, details

        except Exception as e:
            logger.error(f"MOBOObjectives.evaluate_with_details error: {e}")
            return np.array([0.0, -0.5, 0.0]), {"error": str(e)}

    def is_feasible(self, config: dict, y: Optional[np.ndarray] = None) -> bool:
        """Check if config meets hard DR constraints per scenario.

        FP rate is NOT constrained here because the surrogate systematically
        overestimates FP for the healthy scenario (predicts ~18% vs real ~5%).
        FP is instead optimised as the specificity objective (f3 = 1-FP), which
        the Pareto front exposes naturally without requiring a hard threshold on
        a biased surrogate prediction.
        """
        if y is None:
            y = self.evaluate(config)
        preds = self._predict(config)
        dr_pmo  = preds["pmo"]["dr"]
        dr_ckd  = preds["ckd_mbd"]["dr"]
        dr_mild = preds["pmo_mild"]["dr"]
        return bool(
            dr_mild >= self.MIN_DR_PMO_MILD
            and dr_pmo  >= self.MIN_DR_PMO
            and dr_ckd  >= self.MIN_DR_CKD
        )
