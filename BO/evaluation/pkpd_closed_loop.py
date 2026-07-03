#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PK/PD Closed-Loop Biosensor Simulation.

Models a multi-dose, feedback-controlled therapeutic cycle using the
GENEVO2 biosensor as the sensing front-end.

WHAT THIS IS (closed-loop biosensor):
  t=0:      Sensor measures SOST/CTX/P1NP → composite signal R > threshold
            → Drug dose released (romosozumab equivalent, fractional units)
  t=7d:     Drug partially clears (t1/2 = 6.9 days, Padhi 2011 Bone)
            SOST suppressed by ~27% (McClung 2014 JBMR)
  t=7d:     Sensor re-measures → if R still > threshold → next dose
  t=14d, 21d, ...: Repeat until biomarker normalizes or max_cycles reached

WHAT THIS IS NOT:
  - Reinforcement learning (no learned policy — simple threshold rule)
  - Real patient simulation (no ODE dynamics — uses steady-state biomarker levels)

KEY EQUATIONS:
  Drug concentration (multi-dose Bateman):
    C_drug(t) = sum_i[ D_i * exp(-k_el * (t - t_i)) ]  for t > t_i
    k_el = ln(2) / t_half  =  ln(2) / 6.9 days

  Biomarker suppression (empirical, McClung 2014):
    SOST_eff(t) = SOST_disease * (1 - f_supp_sost * C_drug(t) / Cmax)
    CTX_eff(t)  = CTX_disease  * (1 - f_supp_ctx  * C_drug(t) / Cmax)
    P1NP_eff(t) = P1NP_disease * (1 - f_supp_p1np * C_drug(t) / Cmax)

    f_supp_sost = 0.27  (SOST -27% at steady state)
    f_supp_ctx  = 0.55  (CTX  -55% at month 1 peak)
    f_supp_p1np = 0.20  (P1NP -20% at month 3 trough)

  Sensor composite signal (relative to healthy):
    R(t) = w_scl * (theta_scl(t)/theta_scl_h) + w_ctx * ... + w_p1np * ...
    Drug releases if R(t) > threshold (1.08)

  BMD gain (cumulative over multi-dose cycle):
    BMD(t) = sum over doses: dose_i * BMD_gain_per_dose * (1 - e^{-t_recovery/T_bmd})

Usage:
    from BO.evaluation.pkpd_closed_loop import PKPDClosedLoop, run_closed_loop_sim
    result = run_closed_loop_sim(config, scenario="pmo", n_cycles=12)
    print(result.summary())
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PK constants (Padhi 2011 Bone, romosozumab 210mg SC)
# ---------------------------------------------------------------------------
T_HALF_DAYS       = 6.9     # terminal half-life (days)
K_EL              = np.log(2) / T_HALF_DAYS  # elimination rate constant (/day)
C_MAX_NORMALIZED  = 1.0     # normalized peak serum concentration after one dose

# ---------------------------------------------------------------------------
# Biomarker suppression fractions at Cmax (McClung 2014 JBMR, 210mg monthly)
# ---------------------------------------------------------------------------
F_SUPP_SOST  = 0.27   # SOST -27% at month 3
F_SUPP_CTX   = 0.55   # CTX  -55% at month 1
F_SUPP_P1NP  = 0.20   # P1NP -20% at month 3 (transient +20% before suppression)

# ---------------------------------------------------------------------------
# BMD gain model (calibrated to Cosman 2016 NEJM)
# ---------------------------------------------------------------------------
BMD_MAX_PER_DOSE  = 0.06 / 4.0   # g/cm2 per weekly-equivalent dose cycle
                                   # 0.06 g/cm2 over 6mo = 0.01 g/cm2 per 2-week cycle
BMD_T_RECOVERY    = 30.0          # BMD gain time constant (days)
BMD_BASELINE_GCMS = 0.775         # ARCH trial mean baseline LS BMD (g/cm2)

# ---------------------------------------------------------------------------
# Nominal biomarker concentrations at sensor (nM, from environment_configs.py)
# ---------------------------------------------------------------------------
_NOMINAL_CONCS = {
    "healthy":  {"scl": 0.375, "ctx": 0.200, "p1np": 0.350},
    "pmo_mild": {"scl": 0.5625, "ctx": 0.300, "p1np": 0.385},
    "pmo":      {"scl": 0.875,  "ctx": 0.500, "p1np": 0.525},
    "ckd_mbd":  {"scl": 1.125,  "ctx": 0.500, "p1np": 0.625},
}

# Detection threshold (from V6 objective)
DRUG_THRESHOLD_FRAC = 1.08

# Sensor degradation (matching biosensor_engine.py)
# Modified/locked DNA aptamers in physiological buffer, t½=180 days.
# Ref: Delcanale et al. ACS Chem Biol 2021.
SENSOR_HALF_LIFE_DAYS: float = 180.0
_K_DEG_SENSOR: float = np.log(2) / SENSOR_HALF_LIFE_DAYS

# ---------------------------------------------------------------------------
# Long-term BMD dynamics (24-month model)
# ---------------------------------------------------------------------------
# Romosozumab data (Cosman 2016 NEJM) shows diminishing returns after 6 months.
# The plateau factor reduces effective BMD gain per dose as cumulative treatment
# months increase:
#   - Month 0-6:  factor = 1.0  (peak efficacy)
#   - Month 6-12: factor decays to 0.6 (tolerance develops, rebound partly offsets)
#   - Month 12+:  factor stabilizes at ~0.4 (long-term adapted state)
# Formula: plateau(t_months) = 0.4 + 0.6 * exp(-t_months / TAU_PLATEAU)
# TAU_PLATEAU=6 months → at 6mo: 0.4+0.6*exp(-1)=0.62; at 12mo: 0.4+0.6*exp(-2)=0.48
BMD_PLATEAU_FLOOR   = 0.40   # minimum long-term efficacy fraction
BMD_PLATEAU_PEAK    = 0.60   # additional acute efficacy (decays over time)
BMD_PLATEAU_TAU_MO  = 6.0    # time constant (months) for plateau development


@dataclass
class DoseEvent:
    """A single drug dose event."""
    day: float
    dose_fraction: float   # fractional dose (0 to 1.5; 1.0 = standard full dose)
    triggered_by_R: float  # sensor composite ratio that triggered this dose


@dataclass
class PKPDResult:
    """Result of a closed-loop multi-dose simulation."""
    scenario: str
    n_cycles: int
    cycle_days: int
    doses: List[DoseEvent]
    biomarker_trajectories: Dict[str, List[float]]  # {biomarker: [value at each cycle]}
    R_trajectory: List[float]                        # true composite signal R at each cycle
    R_measured_trajectory: List[float]               # degraded measured signal (what sensor sees)
    bmd_trajectory: List[float]                      # cumulative BMD (g/cm2)
    degradation_trajectory: List[float]              # sensor degradation factor per cycle [0,1]
    total_dose: float
    final_bmd_pct: float
    final_R: float
    drug_normalized: bool   # True if R < threshold by end of treatment
    overdose_cycles: int
    missed_doses_due_to_degradation: int             # cycles where true R > threshold but sensor missed
    personalized_threshold: float                    # threshold used (may differ from DRUG_THRESHOLD_FRAC)
    R_corrected_trajectory: List[float]              # degradation-corrected R (R_measured / deg_factor)
    rate_detection_triggers: int                     # cycles triggered by dR/dt alone (not abs threshold)
    degradation_correction: bool                     # True if threshold-decay compensation was applied

    def summary(self) -> str:
        final_deg = self.degradation_trajectory[-1] if self.degradation_trajectory else 1.0
        deg_corr_str = "ON (threshold tracks sensor)" if self.degradation_correction else "OFF"
        lines = [
            f"Closed-Loop Simulation: {self.scenario} | {self.n_cycles} cycles x {self.cycle_days}d",
            f"  Degradation correction:{deg_corr_str}",
            f"  Doses given:           {len(self.doses)} / {self.n_cycles}",
            f"  Missed (degradation):  {self.missed_doses_due_to_degradation}",
            f"  Rate-detection fires:  {self.rate_detection_triggers}",
            f"  Total dose:            {self.total_dose:.3f} (fractional)",
            f"  Final BMD gain:        {self.final_bmd_pct:+.1f}% ({self.final_bmd_pct * BMD_BASELINE_GCMS / 100:.4f} g/cm2)",
            f"  Final R (corrected):   {self.R_corrected_trajectory[-1] if self.R_corrected_trajectory else self.final_R:.3f} (threshold={self.personalized_threshold:.3f})",
            f"  Final sensor factor:   {final_deg:.2f} ({(1-final_deg)*100:.0f}% signal lost)",
            f"  Normalized:            {'YES' if self.drug_normalized else 'NO'}",
            f"  Overdose cycles:       {self.overdose_cycles}",
        ]
        return "\n".join(lines)


class PKPDClosedLoop:
    """
    Simulates a multi-cycle closed-loop therapeutic protocol.

    Each cycle:
      1. Compute biomarker levels accounting for current drug concentration
      2. Compute sensor composite ratio R
      3. If R > threshold: release dose_fraction, record event
      4. Advance time by cycle_days
      5. Update drug concentration (exponential decay + new bolus)
      6. Accumulate BMD gain

    Parameters
    ----------
    config : dict
        Biosensor config with kd_nm, kd_ctx_nm, kd_p1np_nm, w_ctx, w_p1np, sensitivity
    """

    def __init__(self, config: Dict):
        self.kd      = float(config.get("kd_nm", 1.0))
        self.kd_ctx  = float(config.get("kd_ctx_nm", 1.0))
        self.kd_p1np = float(config.get("kd_p1np_nm", 1.0))
        self.w_ctx   = float(config.get("w_ctx", 0.1))
        self.w_p1np  = float(config.get("w_p1np", 0.1))
        self.w_scl   = max(0.0, 1.0 - self.w_ctx - self.w_p1np)

        # Drug dose release model (V6 objective parameters)
        self.K_RELEASE     = 1.0
        self.D_HALF        = 0.15
        self.D_SAFE        = 0.50
        self.ALPHA_OD      = 3.0

    # ------------------------------------------------------------------
    # Langmuir occupancy helpers
    # ------------------------------------------------------------------

    def _occupancy(self, conc: float, kd: float) -> float:
        return float(conc / (kd + conc))

    def _composite_R(self, scl_nm: float, ctx_nm: float, p1np_nm: float) -> float:
        """Normalized composite ratio (healthy = 1.0 by construction)."""
        h = _NOMINAL_CONCS["healthy"]
        th_scl_h  = self._occupancy(h["scl"],  self.kd)
        th_ctx_h  = self._occupancy(h["ctx"],  self.kd_ctx)
        th_p1np_h = self._occupancy(h["p1np"], self.kd_p1np)

        th_scl  = self._occupancy(scl_nm,  self.kd)
        th_ctx  = self._occupancy(ctx_nm,  self.kd_ctx)
        th_p1np = self._occupancy(p1np_nm, self.kd_p1np)

        eps = 1e-12
        R = (
            self.w_scl   * th_scl   / max(th_scl_h,  eps) +
            self.w_ctx   * th_ctx   / max(th_ctx_h,  eps) +
            self.w_p1np  * th_p1np  / max(th_p1np_h, eps)
        )
        return float(R)

    def _dose_from_R(self, R: float) -> Tuple[float, float]:
        """
        Compute dose fraction and net BMD gain for a single release event.

        Returns (dose_fraction, bmd_gross_per_dose_gcm2).
        """
        dose = self.K_RELEASE * max(0.0, (R - DRUG_THRESHOLD_FRAC) / DRUG_THRESHOLD_FRAC)
        bmd_gross = BMD_MAX_PER_DOSE * (dose / (dose + self.D_HALF))

        # Overdose penalty
        if dose > self.D_SAFE:
            overdose = (dose / self.D_SAFE - 1.0) ** 2 * self.ALPHA_OD
            bmd_gross = max(0.0, bmd_gross - overdose * BMD_MAX_PER_DOSE * 0.2)

        return dose, bmd_gross

    # ------------------------------------------------------------------
    # Multi-dose simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        scenario: str,
        n_cycles: int = 12,
        cycle_days: int = 28,  # monthly monitoring
        noise_sigma: float = 0.0,
        rng: Optional[np.random.RandomState] = None,
        with_degradation: bool = True,
        personalized_threshold: bool = True,
        degradation_correction: bool = True,
        use_rate_detection: bool = False,
        rate_threshold: float = 0.05,
        patient_t_half_days: Optional[float] = None,
        apply_long_term_plateau: bool = True,
    ) -> PKPDResult:
        """
        Run a closed-loop multi-dose simulation.

        Parameters
        ----------
        scenario : str
            Disease scenario: "pmo_mild", "pmo", "ckd_mbd"
        n_cycles : int
            Number of monitoring cycles (default 12 = 12 months at monthly)
        cycle_days : int
            Days between monitoring/dosing events (default 28 = monthly)
        noise_sigma : float
            Lognormal sigma for measurement noise (default 0 = deterministic)
        rng : RandomState, optional
            Random state for noise (default: new random state)
        """
        if rng is None:
            rng = np.random.RandomState()

        if scenario not in _NOMINAL_CONCS:
            raise ValueError(f"Unknown scenario: {scenario}. Choose from {list(_NOMINAL_CONCS.keys())}")

        disease_concs = _NOMINAL_CONCS[scenario]
        scl0  = disease_concs["scl"]
        ctx0  = disease_concs["ctx"]
        p1np0 = disease_concs["p1np"]

        # Patient-specific threshold calibration.
        # At implant time (day 0), the sensor measures the patient's true biomarker
        # state and sets the drug-release threshold to 90% of that initial reading.
        # This accounts for patient-to-patient baseline variation (e.g., CKD patients
        # start at R≈3.0, PMO at R≈1.5) rather than using a one-size-fits-all 1.08.
        # If personalized_threshold=False, use the global V6 default (1.08).
        R_baseline = self._composite_R(scl0, ctx0, p1np0)
        if personalized_threshold:
            # Threshold = 90% of the patient's measured baseline R.
            # Minimum floor = DRUG_THRESHOLD_FRAC (don't release drug for healthy patients).
            active_threshold = max(DRUG_THRESHOLD_FRAC, R_baseline * 0.90)
        else:
            active_threshold = DRUG_THRESHOLD_FRAC

        # State variables
        C_drug    = 0.0     # current normalized drug concentration
        bmd_total = 0.0     # cumulative BMD gain (g/cm2)
        doses: List[DoseEvent] = []
        R_traj      = []
        R_meas_traj = []
        bmd_traj    = []
        scl_traj    = []
        ctx_traj    = []
        p1np_traj   = []
        deg_traj    = []
        overdose_cycles = 0
        missed_due_to_degradation = 0
        R_corr_traj = []
        rate_det_count = 0
        prev_R_corrected: Optional[float] = None

        for cycle in range(n_cycles):
            t_day = cycle * cycle_days

            # Sensor degradation factor at this point in time.
            # Uses patient-specific half-life when provided (Normal(180,30) per patient).
            if with_degradation:
                k_deg = (np.log(2) / patient_t_half_days
                         if patient_t_half_days is not None else _K_DEG_SENSOR)
                deg_factor = float(np.exp(-k_deg * t_day))
            else:
                deg_factor = 1.0

            # Long-term efficacy plateau (24-month model).
            # BMD gain per dose decays as tolerance develops after month 6.
            if apply_long_term_plateau:
                t_months = t_day / 30.0
                plateau_factor = (BMD_PLATEAU_FLOOR
                                  + BMD_PLATEAU_PEAK * np.exp(-t_months / BMD_PLATEAU_TAU_MO))
            else:
                plateau_factor = 1.0

            # Effective biomarker levels under current drug concentration
            supp_sost  = F_SUPP_SOST  * C_drug
            supp_ctx   = F_SUPP_CTX   * C_drug
            supp_p1np  = F_SUPP_P1NP  * C_drug

            scl_eff  = scl0  * (1.0 - supp_sost)
            ctx_eff  = ctx0  * (1.0 - supp_ctx)
            p1np_eff = p1np0 * (1.0 - supp_p1np)

            # Measurement noise (optional)
            if noise_sigma > 0:
                scl_eff  *= float(rng.lognormal(0.0, noise_sigma))
                ctx_eff  *= float(rng.lognormal(0.0, noise_sigma))
                p1np_eff *= float(rng.lognormal(0.0, noise_sigma))

            # True composite ratio (what the disease actually produces)
            R_true = self._composite_R(scl_eff, ctx_eff, p1np_eff)
            # Measured ratio (attenuated by sensor degradation)
            R_measured = R_true * deg_factor

            # Degradation-corrected signal: device knows its own degradation curve
            # (pre-programmed with t½ at manufacture) and compensates accordingly.
            # R_corrected ≈ R_true; noise is amplified by 1/deg_factor (trade-off).
            safe_deg = max(deg_factor, 0.05)  # prevent runaway at >94% signal loss
            R_corrected = R_measured / safe_deg if (with_degradation and degradation_correction) else R_measured

            # Effective detection threshold: decays with sensor so detection sensitivity
            # is preserved over the device lifetime (equivalent to R_corrected > active_threshold).
            if degradation_correction and with_degradation:
                effective_threshold = active_threshold * safe_deg
            else:
                effective_threshold = active_threshold

            # Rate-of-change detection: catch rising disease even below absolute threshold.
            dR = R_corrected - prev_R_corrected if prev_R_corrected is not None else 0.0
            prev_R_corrected = R_corrected
            rate_triggered = (
                use_rate_detection
                and prev_R_corrected is not None
                and dR > rate_threshold
                and R_measured <= effective_threshold  # only fires when abs threshold missed
            )

            R_traj.append(R_true)
            R_meas_traj.append(R_measured)
            R_corr_traj.append(R_corrected)
            deg_traj.append(deg_factor)
            scl_traj.append(scl_eff)
            ctx_traj.append(ctx_eff)
            p1np_traj.append(p1np_eff)
            bmd_traj.append(bmd_total)

            # Drug release — use R_corrected for dose scaling (reflects true disease severity)
            if R_measured > effective_threshold or rate_triggered:
                if rate_triggered and R_measured <= effective_threshold:
                    rate_det_count += 1
                dose_frac, bmd_gain = self._dose_from_R(R_corrected)
                if dose_frac > self.D_SAFE:
                    overdose_cycles += 1
                bmd_total += bmd_gain * plateau_factor
                C_drug += dose_frac
                doses.append(DoseEvent(day=t_day, dose_fraction=dose_frac, triggered_by_R=R_corrected))
            elif R_true > active_threshold and not rate_triggered:
                # True disease active but sensor (even corrected) missed detection
                missed_due_to_degradation += 1

            # Drug elimination over next cycle
            C_drug *= np.exp(-K_EL * cycle_days)
            C_drug = min(C_drug, 3.0)

        # Final state — use corrected R for normalization check
        final_R   = R_traj[-1] if R_traj else 1.0
        final_R_corr = R_corr_traj[-1] if R_corr_traj else final_R
        final_bmd_pct = bmd_total / BMD_BASELINE_GCMS * 100.0
        normalized = final_R_corr < active_threshold

        return PKPDResult(
            scenario=scenario,
            n_cycles=n_cycles,
            cycle_days=cycle_days,
            doses=doses,
            biomarker_trajectories={
                "SOST": scl_traj, "CTX": ctx_traj, "P1NP": p1np_traj,
            },
            R_trajectory=R_traj,
            R_measured_trajectory=R_meas_traj,
            R_corrected_trajectory=R_corr_traj,
            bmd_trajectory=bmd_traj,
            degradation_trajectory=deg_traj,
            total_dose=sum(d.dose_fraction for d in doses),
            final_bmd_pct=final_bmd_pct,
            final_R=final_R,
            drug_normalized=normalized,
            overdose_cycles=overdose_cycles,
            missed_doses_due_to_degradation=missed_due_to_degradation,
            personalized_threshold=active_threshold,
            rate_detection_triggers=rate_det_count,
            degradation_correction=degradation_correction,
        )


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def run_closed_loop_sim(
    config: Dict,
    scenario: str = "pmo",
    n_cycles: int = 12,
    cycle_days: int = 28,
    noise_sigma: float = 0.15,
    n_patients: int = 100,
    seed: int = 42,
    with_degradation: bool = True,
    personalized_threshold: bool = True,
    degradation_correction: bool = True,
    use_rate_detection: bool = False,
) -> Dict:
    """
    Run closed-loop simulation on n_patients virtual patients.

    Each patient gets independent measurement noise (lognormal sigma).
    with_degradation: apply aptamer half-life signal decay (t½=180d).
    personalized_threshold: calibrate trigger threshold to patient baseline R.
    degradation_correction: pre-program threshold to decay with sensor (fixes missed-dose bug).
    use_rate_detection: also trigger dose if dR/dt > rate_threshold.

    Returns:
        dict with population statistics and trajectory arrays.
    """
    sim = PKPDClosedLoop(config)
    rng = np.random.RandomState(seed)

    results_per_patient = []
    for _ in range(n_patients):
        patient_seed = rng.randint(0, 100000)
        r = sim.simulate(
            scenario=scenario,
            n_cycles=n_cycles,
            cycle_days=cycle_days,
            noise_sigma=noise_sigma,
            rng=np.random.RandomState(patient_seed),
            with_degradation=with_degradation,
            personalized_threshold=personalized_threshold,
            degradation_correction=degradation_correction,
            use_rate_detection=use_rate_detection,
        )
        results_per_patient.append(r)

    bmd_finals  = [r.final_bmd_pct  for r in results_per_patient]
    doses_given = [len(r.doses)      for r in results_per_patient]
    normalized  = [r.drug_normalized for r in results_per_patient]
    total_doses = [r.total_dose      for r in results_per_patient]
    missed_deg  = [r.missed_doses_due_to_degradation for r in results_per_patient]
    rate_fires  = [r.rate_detection_triggers for r in results_per_patient]

    # Final degradation factor (same for all patients — deterministic by t_day)
    final_deg = results_per_patient[0].degradation_trajectory[-1] if results_per_patient else 1.0

    return {
        "scenario":              scenario,
        "n_patients":            n_patients,
        "n_cycles":              n_cycles,
        "cycle_days":            cycle_days,
        "with_degradation":      with_degradation,
        "degradation_correction": degradation_correction,
        "personalized_threshold": personalized_threshold,
        "sensor_half_life_days": SENSOR_HALF_LIFE_DAYS,
        "final_degradation_factor": float(final_deg),
        "bmd_final_pct":    {
            "mean":  float(np.mean(bmd_finals)),
            "std":   float(np.std(bmd_finals)),
            "p25":   float(np.percentile(bmd_finals, 25)),
            "p75":   float(np.percentile(bmd_finals, 75)),
        },
        "doses_per_patient": {
            "mean":  float(np.mean(doses_given)),
            "std":   float(np.std(doses_given)),
            "max":   int(np.max(doses_given)),
            "min":   int(np.min(doses_given)),
        },
        "normalization_rate_pct": float(np.mean(normalized) * 100.0),
        "total_dose": {
            "mean":  float(np.mean(total_doses)),
            "std":   float(np.std(total_doses)),
        },
        "overdose_pct": float(
            np.mean([r.overdose_cycles > 0 for r in results_per_patient]) * 100.0
        ),
        "missed_doses_due_to_degradation": {
            "mean": float(np.mean(missed_deg)),
            "total": int(np.sum(missed_deg)),
        },
        "rate_detection_triggers": {
            "mean": float(np.mean(rate_fires)),
            "total": int(np.sum(rate_fires)),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_config() -> Dict:
    """Best config from BO results (BO/bo_results/results/best_config.json)."""
    return {
        "kd_nm":      5.81,
        "sensitivity": 4.57,
        "response_time_s": 600.0,
        "kd_ctx_nm":  0.278,
        "kd_p1np_nm": 0.953,
        "w_ctx":      0.155,
        "w_p1np":     0.459,
        "biosensor_type": "array",
        "noise_preset": "realistic",
    }


if __name__ == "__main__":
    import argparse
    import json
    import sys

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="PK/PD closed-loop biosensor simulation")
    parser.add_argument("--config-json", type=str, default=None,
                        help="Path to biosensor config JSON (default: built-in best config)")
    parser.add_argument("--scenario", type=str, default="pmo",
                        choices=["pmo_mild", "pmo", "ckd_mbd"],
                        help="Disease scenario to simulate (default: pmo)")
    parser.add_argument("--n-cycles", type=int, default=12,
                        help="Number of monitoring cycles (default: 12)")
    parser.add_argument("--cycle-days", type=int, default=28,
                        help="Days between doses (default: 28 = monthly)")
    parser.add_argument("--n-patients", type=int, default=200,
                        help="Virtual patients for population statistics (default: 200)")
    parser.add_argument("--noise", type=float, default=0.15,
                        help="Measurement noise sigma (default: 0.15)")
    parser.add_argument("--all-scenarios", action="store_true",
                        help="Run all disease scenarios")
    parser.add_argument("--no-degradation-correction", action="store_true",
                        help="Disable threshold-decay compensation (shows raw degradation impact)")
    parser.add_argument("--rate-detection", action="store_true",
                        help="Enable supplemental dR/dt detection")
    parser.add_argument("--out", type=str, default=None,
                        help="Output JSON path")
    args = parser.parse_args()

    config = _default_config()
    if args.config_json:
        with open(args.config_json) as f:
            raw = json.load(f)
        # Handle nested best_config.json format
        if "biosensor_design" in raw:
            d = raw["biosensor_design"]
            config.update({
                "kd_nm": d.get("kd_nm", config["kd_nm"]),
                "sensitivity": d.get("sensitivity", config["sensitivity"]),
                "kd_ctx_nm": d.get("kd_ctx_nm", config["kd_ctx_nm"]),
                "kd_p1np_nm": d.get("kd_p1np_nm", config["kd_p1np_nm"]),
                "w_ctx": d.get("w_ctx", config["w_ctx"]),
                "w_p1np": d.get("w_p1np", config["w_p1np"]),
            })
        else:
            config.update(raw)

    scenarios = ["pmo_mild", "pmo", "ckd_mbd"] if args.all_scenarios else [args.scenario]

    deg_correction = not args.no_degradation_correction

    print("\n" + "=" * 72)
    print("GENEVO2 PK/PD CLOSED-LOOP BIOSENSOR SIMULATION")
    print(f"Config: kd={config['kd_nm']:.3f} nm  w_ctx={config['w_ctx']:.3f}  w_p1np={config['w_p1np']:.3f}")
    print(f"Protocol: {args.n_cycles} cycles x {args.cycle_days}d  |  n_patients={args.n_patients}")
    print(f"Degradation correction: {'ON' if deg_correction else 'OFF'}  |  Rate detection: {'ON' if args.rate_detection else 'OFF'}")
    print("=" * 72)

    all_results = {}
    for sc in scenarios:
        res = run_closed_loop_sim(
            config=config,
            scenario=sc,
            n_cycles=args.n_cycles,
            cycle_days=args.cycle_days,
            noise_sigma=args.noise,
            n_patients=args.n_patients,
            seed=42,
            degradation_correction=deg_correction,
            use_rate_detection=args.rate_detection,
        )
        all_results[sc] = res

        print(f"\n[{sc.upper()}]")
        print(f"  BMD gain:              {res['bmd_final_pct']['mean']:+.1f}% +/- {res['bmd_final_pct']['std']:.1f}%")
        print(f"  Doses given:           {res['doses_per_patient']['mean']:.1f} +/- {res['doses_per_patient']['std']:.1f}  (range {res['doses_per_patient']['min']}-{res['doses_per_patient']['max']})")
        print(f"  Missed (degradation):  {res['missed_doses_due_to_degradation']['mean']:.1f} avg / patient")
        print(f"  Rate-detect triggers:  {res['rate_detection_triggers']['mean']:.1f} avg / patient")
        print(f"  Sensor at end:         {res['final_degradation_factor']*100:.0f}% original sensitivity")
        print(f"  Normalization:         {res['normalization_rate_pct']:.0f}% of patients")
        print(f"  Total dose:            {res['total_dose']['mean']:.3f} fractional")
        print(f"  Overdose events:       {res['overdose_pct']:.0f}% of patients")

    # Compare to trial benchmark
    trial_bmd_12mo_pct = 13.3  # Cosman 2016 NEJM LS BMD at 12 months
    if "pmo" in all_results:
        sim_bmd = all_results["pmo"]["bmd_final_pct"]["mean"]
        print(f"\nBenchmark check: sim BMD ({sim_bmd:+.1f}%) vs ARCH trial 12mo ({trial_bmd_12mo_pct:+.1f}%)")
        if abs(sim_bmd - trial_bmd_12mo_pct) < 5.0:
            print("  [OK] Within 5pp of clinical trial outcome")
        else:
            print(f"  [WARN] {abs(sim_bmd - trial_bmd_12mo_pct):.1f}pp discrepancy — model may need recalibration")

    print("\n" + "=" * 72)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"[OK] Results saved to {args.out}")
