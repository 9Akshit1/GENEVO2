#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hard safety constraints for GENEVO2 therapeutic objective.

These are non-negotiable clinical limits. Configs that violate them are
infeasible — they should be excluded from optimization, not just penalized.

Constraint categories (from document):
  1. Dosing ceiling:   cumulative dose ≤ 0.5 mg/month × n_months
  2. BMD gain rate:    ≤ 3% per 6 months (rapid gain → brittle bone)
  3. Kidney function:  eGFR must not decline > 15% from baseline
  4. Hypercalcemia:    serum calcium must remain ≤ 10.5 mg/dL
  5. False positive:   FP rate ≤ 15% (unnecessary dosing burden)

Usage:
    from BO.evaluation.safety_constraints import SafetyConstraints, check_simulation_safety

    sc = SafetyConstraints()
    result = sc.check(patient_state)
    if not result.feasible:
        print(result.violations)

    # Or check a PKPDResult from pkpd_closed_loop.py
    verdict = check_simulation_safety(pkpd_result, patient_egfr_baseline=65.0)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hard constraint thresholds
# ---------------------------------------------------------------------------
# Dosing: romosozumab is approved at 210 mg SC monthly (FDA label 2019).
# One monthly dose = 1.0 fractional unit in the PKPD model.
# Over-dosing means > 1 dose per 28-day cycle on average.
MAX_DOSE_PER_CYCLE_FRACTION  = 1.0    # max 1 full dose per monitoring cycle
# BMD gain rate: romosozumab ARCH trial shows +6.7%/6mo at target dosing.
# A rate >15%/6mo would indicate implausible model behavior or extreme dosing.
# The old 3%/6mo limit was wrong — romosozumab IS designed for rapid BMD gain.
MAX_BMD_GAIN_RATE_6MO        = 0.15   # 15% per 6 months (safety ceiling, ~2× ARCH trial)
MAX_EGFR_DECLINE_FRACTION    = 0.15   # 15% decline from baseline is harmful
MAX_SERUM_CALCIUM_MG_DL      = 10.5  # hypercalcemia threshold (mg/dL)
MAX_FP_RATE                  = 0.15   # false positive dosing rate ceiling
MIN_DR                       = 0.50   # minimum detection rate (hard floor)

# Keep backward-compatible alias
MAX_DOSE_PER_MONTH_MG = MAX_DOSE_PER_CYCLE_FRACTION


@dataclass
class SafetyResult:
    """Outcome of a safety constraint check."""
    feasible: bool
    violations: List[str] = field(default_factory=list)
    warnings: List[str]   = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        status = "FEASIBLE" if self.feasible else "INFEASIBLE"
        lines = [f"Safety check: {status}"]
        if self.violations:
            lines.append("  Violations:")
            for v in self.violations:
                lines.append(f"    - {v}")
        if self.warnings:
            lines.append("  Warnings:")
            for w in self.warnings:
                lines.append(f"    ~ {w}")
        for k, v in self.metrics.items():
            lines.append(f"  {k}: {v:.4f}")
        return "\n".join(lines)


class SafetyConstraints:
    """
    Evaluates hard safety constraints for a biosensor-driven treatment plan.

    Parameters can be overridden at construction for patient-specific limits.
    """

    def __init__(
        self,
        max_dose_per_month: float = MAX_DOSE_PER_CYCLE_FRACTION,
        max_bmd_gain_rate_6mo: float = MAX_BMD_GAIN_RATE_6MO,
        max_egfr_decline: float = MAX_EGFR_DECLINE_FRACTION,
        max_serum_calcium: float = MAX_SERUM_CALCIUM_MG_DL,
        max_fp_rate: float = MAX_FP_RATE,
        min_dr: float = MIN_DR,
    ):
        self.max_dose_per_month  = max_dose_per_month
        self.max_bmd_gain_rate   = max_bmd_gain_rate_6mo
        self.max_egfr_decline    = max_egfr_decline
        self.max_serum_calcium   = max_serum_calcium
        self.max_fp_rate         = max_fp_rate
        self.min_dr              = min_dr

    def check_dosing(
        self, cumulative_dose_fractional: float, n_months: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Check that average dosing does not exceed one full dose per cycle.

        cumulative_dose_fractional: sum of dose_fraction values from PKPDResult
            (1.0 = one standard 210 mg romosozumab dose per cycle)
        n_months: number of treatment months elapsed (≈ number of 28-day cycles)

        Violation: average dose fraction per cycle > MAX_DOSE_PER_CYCLE_FRACTION
        Clinical context: romosozumab is approved at 210 mg SC once monthly
        (Cosman 2016 NEJM; FDA label Apr 2019). Exceeding 1 dose/cycle implies
        the model is triggering additional doses within a single 28-day window.
        """
        if n_months <= 0:
            return True, None
        dose_per_cycle = cumulative_dose_fractional / n_months
        ok = dose_per_cycle <= self.max_dose_per_month
        msg = (
            None if ok else
            f"Dosing exceeded: {dose_per_cycle:.2f} fractional doses/cycle > "
            f"{self.max_dose_per_month:.1f} limit (1 = one 210mg dose/month)"
        )
        return ok, msg

    def check_bmd_rate(
        self, bmd_trajectory_pct: List[float], cycle_days: int = 28
    ) -> Tuple[bool, Optional[str]]:
        """
        Check BMD gain rate does not exceed 3% per 6 months.

        bmd_trajectory_pct: BMD gain (%) at each monitoring cycle
        """
        if len(bmd_trajectory_pct) < 7:
            return True, None  # Not enough data for 6-month window

        cycles_per_6mo = int(round(182 / cycle_days))
        max_rate = 0.0
        for i in range(cycles_per_6mo, len(bmd_trajectory_pct)):
            rate_6mo = (bmd_trajectory_pct[i] - bmd_trajectory_pct[i - cycles_per_6mo]) / 100.0
            if rate_6mo > max_rate:
                max_rate = rate_6mo

        ok = max_rate <= self.max_bmd_gain_rate
        msg = (
            None if ok else
            f"BMD gain rate {max_rate*100:.1f}%/6mo exceeds {self.max_bmd_gain_rate*100:.1f}% limit"
        )
        return ok, msg

    def check_egfr(
        self,
        egfr_baseline: float,
        egfr_current: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check eGFR has not declined more than 15% from baseline.

        egfr_baseline: mL/min/1.73m² at enrollment
        egfr_current:  mL/min/1.73m² at evaluation point
        """
        if egfr_baseline <= 0:
            return True, None
        decline_frac = (egfr_baseline - egfr_current) / egfr_baseline
        ok = decline_frac <= self.max_egfr_decline
        msg = (
            None if ok else
            f"eGFR declined {decline_frac*100:.1f}% (from {egfr_baseline:.0f} to "
            f"{egfr_current:.0f}), exceeds {self.max_egfr_decline*100:.0f}% limit"
        )
        return ok, msg

    def check_serum_calcium(
        self, serum_calcium_mg_dl: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Check serum calcium does not exceed hypercalcemia threshold.

        serum_calcium_mg_dl: current measured or estimated serum Ca (mg/dL).
        Normal range: 8.5-10.2 mg/dL; hypercalcemia: >10.5 mg/dL.
        """
        ok = serum_calcium_mg_dl <= self.max_serum_calcium
        msg = (
            None if ok else
            f"Hypercalcemia: Ca {serum_calcium_mg_dl:.1f} mg/dL > {self.max_serum_calcium:.1f} limit"
        )
        return ok, msg

    def check_detection_performance(
        self,
        dr: float,
        fp_rate: float,
    ) -> Tuple[bool, List[str]]:
        """
        Check detection rate floor and false-positive ceiling.

        dr:      Detection rate (0-1) across disease scenarios
        fp_rate: False-positive rate (0-1) on healthy patients
        """
        violations = []
        if dr < self.min_dr:
            violations.append(
                f"DR {dr:.3f} below minimum {self.min_dr:.2f} — too many missed detections"
            )
        if fp_rate > self.max_fp_rate:
            violations.append(
                f"FP rate {fp_rate:.3f} above maximum {self.max_fp_rate:.2f} — over-dosing risk"
            )
        return len(violations) == 0, violations

    def check_all(
        self,
        *,
        cumulative_dose_fractional: float = 0.0,
        n_months: int = 0,
        bmd_trajectory_pct: Optional[List[float]] = None,
        egfr_baseline: Optional[float] = None,
        egfr_current: Optional[float] = None,
        serum_calcium_mg_dl: Optional[float] = None,
        dr: Optional[float] = None,
        fp_rate: Optional[float] = None,
    ) -> SafetyResult:
        """
        Run all applicable safety checks and return a SafetyResult.

        Pass None for any metric that is unavailable — that check is skipped.
        """
        violations: List[str] = []
        warnings:   List[str] = []
        metrics: Dict[str, float] = {}

        # 1. Dosing
        ok, msg = self.check_dosing(cumulative_dose_fractional, n_months)
        metrics["dose_per_month_fraction"] = (
            cumulative_dose_fractional / max(n_months, 1)
        )
        if not ok and msg:
            violations.append(msg)

        # 2. BMD gain rate
        if bmd_trajectory_pct is not None and len(bmd_trajectory_pct) >= 7:
            ok, msg = self.check_bmd_rate(bmd_trajectory_pct)
            if not ok and msg:
                violations.append(msg)

        # 3. eGFR
        if egfr_baseline is not None and egfr_current is not None:
            ok, msg = self.check_egfr(egfr_baseline, egfr_current)
            metrics["egfr_decline_pct"] = max(0, (egfr_baseline - egfr_current) / egfr_baseline * 100)
            if not ok and msg:
                violations.append(msg)
            elif egfr_current < 45:
                warnings.append(f"eGFR {egfr_current:.0f} mL/min — CKD Stage 3b, monitor closely")

        # 4. Serum calcium
        if serum_calcium_mg_dl is not None:
            ok, msg = self.check_serum_calcium(serum_calcium_mg_dl)
            metrics["serum_calcium"] = serum_calcium_mg_dl
            if not ok and msg:
                violations.append(msg)
            elif serum_calcium_mg_dl > 10.0:
                warnings.append(f"Serum Ca {serum_calcium_mg_dl:.1f} mg/dL — approaching hypercalcemia limit")

        # 5. Detection performance
        if dr is not None and fp_rate is not None:
            ok, det_violations = self.check_detection_performance(dr, fp_rate)
            metrics["dr"] = dr
            metrics["fp_rate"] = fp_rate
            violations.extend(det_violations)

        return SafetyResult(
            feasible=len(violations) == 0,
            violations=violations,
            warnings=warnings,
            metrics=metrics,
        )


def check_simulation_safety(
    pkpd_result,
    patient_egfr_baseline: Optional[float] = None,
    patient_egfr_current: Optional[float] = None,
    serum_calcium_mg_dl: Optional[float] = None,
    dr: Optional[float] = None,
    fp_rate: Optional[float] = None,
    constraints: Optional[SafetyConstraints] = None,
) -> SafetyResult:
    """
    Convenience wrapper: check safety of a PKPDResult object.

    Parameters
    ----------
    pkpd_result : PKPDResult
        Output of PKPDClosedLoop.simulate().
    patient_egfr_baseline : float, optional
        eGFR at enrollment (mL/min/1.73m²). Pass for CKD patients.
    patient_egfr_current : float, optional
        eGFR at last measurement. Estimated from dialysis flag if not provided.
    serum_calcium_mg_dl : float, optional
        Latest serum calcium measurement. Estimated from nominal if not provided.
    dr, fp_rate : float, optional
        Detection rate and false-positive rate from surrogate evaluation.
    constraints : SafetyConstraints, optional
        Custom constraint instance. Defaults to population-level limits.

    Returns
    -------
    SafetyResult
    """
    if constraints is None:
        constraints = SafetyConstraints()

    n_months = pkpd_result.n_cycles * pkpd_result.cycle_days // 28

    return constraints.check_all(
        cumulative_dose_fractional=pkpd_result.total_dose,
        n_months=n_months,
        bmd_trajectory_pct=pkpd_result.bmd_trajectory,
        egfr_baseline=patient_egfr_baseline,
        egfr_current=patient_egfr_current,
        serum_calcium_mg_dl=serum_calcium_mg_dl,
        dr=dr,
        fp_rate=fp_rate,
    )


def is_feasible(config: dict, patient_state: Optional[dict] = None) -> bool:
    """
    Quick feasibility check for BO surrogate filtering.

    Designed to be called inside the BO acquisition loop to discard configs
    that violate hard constraints before GP prediction.

    Parameters
    ----------
    config : dict
        Biosensor config from BO search space.
    patient_state : dict, optional
        {cumulative_dose_fractional, n_months, egfr_baseline, egfr_current,
         serum_calcium_mg_dl, bmd_trajectory_pct, dr, fp_rate}

    Returns
    -------
    bool : True if no violations detected.
    """
    if patient_state is None:
        patient_state = {}

    sc = SafetyConstraints()
    result = sc.check_all(
        cumulative_dose_fractional=float(patient_state.get("cumulative_dose_fractional", 0.0)),
        n_months=int(patient_state.get("n_months", 0)),
        bmd_trajectory_pct=patient_state.get("bmd_trajectory_pct"),
        egfr_baseline=patient_state.get("egfr_baseline"),
        egfr_current=patient_state.get("egfr_current"),
        serum_calcium_mg_dl=patient_state.get("serum_calcium_mg_dl"),
        dr=patient_state.get("dr"),
        fp_rate=patient_state.get("fp_rate"),
    )
    if not result.feasible:
        logger.debug("Config infeasible: %s", result.violations)
    return result.feasible
