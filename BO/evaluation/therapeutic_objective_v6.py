#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Therapeutic Objective V6 -- Decoupled kd/sensitivity optimization.

WHY V6 IS DIFFERENT FROM V5
============================

V5 uses dose = DR * K_RELEASE * fractional_margin.
The DR factor (from the surrogate) is strongly sensitivity-dependent.
This reintroduces sensitivity into the therapeutic term, producing
r=0.994 correlation with v3 even after mild-centric reweighting.

V6 FIXES: Two structural changes that break the sensitivity-therapeutic coupling.

1. RELATIVE THRESHOLD (pure kd/weight discrimination)
   The drug-release trigger is set proportional to the healthy signal:
       R_scenario = composite_scenario / composite_healthy
                  = (w_scl*ratio_scl + w_ctx*ratio_ctx + w_p1np*ratio_p1np)
   Since composite_healthy = sensitivity * 1.0 (by construction), dividing by it
   cancels sensitivity completely.  The drug dose becomes:

       dose = K_RELEASE * max(0, (R - DRUG_THRESHOLD_FRAC) / DRUG_THRESHOLD_FRAC)

   where DRUG_THRESHOLD_FRAC = 1.08 (drug releases when disease signal
   is at least 8% above healthy baseline).  This is INDEPENDENT of sensitivity.

2. OVERDOSE PENALTY (non-monotone kd landscape)
   Very low kd values push R_PMO/R_CKD high, delivering excess drug.
   A quadratic penalty activates when dose > D_SAFE:

       overdose = max(0, dose/D_SAFE - 1)^2 * ALPHA_OVERDOSE

   Net BMD = bmd_gross - overdose

   Effect on landscape:
   - kd too high → R_mild near threshold → little drug → poor BMD
   - kd optimal  → R_mild ≈ 1.3, R_PMO ≈ 1.5 → good BMD, no overdose
   - kd too low  → R_PMO ≈ 2.0 → overdose penalty → reduced net BMD for PMO/CKD

   This creates a true kd sweet spot, breaking the "always lower kd is better"
   monotone that plagued v3/v4/v5.

EXPECTED LANDSCAPE PROPERTIES:
   - Therapeutic axis: pure kd/weight optimization, sensitivity-flat
   - Detection axis: surrogate-based, sensitivity-dominant
   - Pearson r(v6, v3) target: 0.70-0.85 (down from 0.994 in v5)

NOMINAL SENSOR CONCENTRATIONS (from environment_configs.py)
-------------------------------------------------------------
    Analyte    Healthy   PMO-mild   PMO     CKD-MBD
    --------   -------   --------   -----   -------
    SOST       0.375     0.5625     0.875   1.125   nM
    CTX        0.200     0.300      0.500   0.500   nM
    P1NP       0.350     0.385      0.525   0.625   nM

Objective weights:
  Therapeutic BMD (kd-optimal)  : 0.40
  Detection reliability (DR)    : 0.25
  FNR miss rate                 : 0.15
  FP toxicity penalty           : 0.10
  TTD speed                     : 0.05
  Resource/toxicity             : 0.05
  Total                         : 1.00
"""

import numpy as np
from typing import Dict, Tuple
import logging
from scipy.special import erfc

logger = logging.getLogger(__name__)

_NOMINAL_CONCS = {
    "healthy": {"scl": 0.375, "ctx": 0.200, "p1np": 0.350},
    "pmo_mild": {"scl": 0.5625, "ctx": 0.300, "p1np": 0.385},
    "pmo":      {"scl": 0.875,  "ctx": 0.500, "p1np": 0.525},
    "ckd_mbd":  {"scl": 1.125,  "ctx": 0.500, "p1np": 0.625},
}


class TherapeuticObjectiveV6:
    """
    V6 objective: decoupled kd/sensitivity optimization via relative threshold.

    API-compatible with ObjectiveFunctionV3, V4, and TherapeuticObjectiveV5.
    """

    # --- Objective weights (sum to 1.0) ---
    WEIGHT_THERAPEUTIC = 0.40
    WEIGHT_DR          = 0.25
    WEIGHT_FNR         = 0.15
    WEIGHT_FP          = 0.10
    WEIGHT_TTD         = 0.05
    WEIGHT_TOXICITY    = 0.05

    # --- Hard constraint thresholds (same as v3/v4/v5) ---
    MIN_DR     = 0.50
    MAX_FNR    = 0.60
    MAX_FP_DR  = 0.15
    TTD_MAX    = 9000.0

    CATASTROPHIC_PENALTY     = 0.0
    MIN_DISCRIMINATION_RATIO = 1.15
    _HEALTHY_CONC = 0.375
    _PMO_CONC     = 0.875

    # --- V6 PKPD parameters ---
    # Drug-release relative threshold: drug is released when the scenario
    # occupancy-ratio R > DRUG_THRESHOLD_FRAC (8% above healthy baseline).
    DRUG_THRESHOLD_FRAC = 1.08

    # Hill equation calibration:
    #   D_HALF=0.15 → half-max BMD at R_mild ≈ 1.24 (10% margin above mild threshold)
    K_RELEASE   = 1.0
    D_HALF      = 0.15

    # Overdose penalty: quadratic above D_SAFE.
    # Calibrated so typical R_PMO=1.5 (dose≈0.39) is just above safe limit,
    # creating a gentle penalty that grows as kd→0 (R_PMO→2.0).
    D_SAFE         = 0.50   # safe dose ceiling (fractional margin units)
    ALPHA_OVERDOSE = 3.0    # quadratic penalty strength

    # BMD reference (romosozumab 6-month data, Cosman 2016 NEJM)
    BMD_GAIN_MAX = 0.06     # g/cm2 / 6-months at full dose
    BMD_GAIN_REF = 0.04     # reference BMD gain for normalization

    # Therapeutic sub-weights (mild-centric; must sum to 1.0)
    TREAT_W_MILD = 0.55
    TREAT_W_PMO  = 0.25
    TREAT_W_CKD  = 0.20

    DISEASE_SCENARIOS = ["ckd_mbd", "pmo", "pmo_mild"]

    def __init__(
        self,
        physics_model,
        surrogate_loader_v3,
        apply_constraints: bool = True,
        **kwargs,
    ):
        self.surrogate_loader = surrogate_loader_v3
        self.apply_constraints = apply_constraints
        logger.info(
            "TherapeuticObjectiveV6 (decoupled kd/sensitivity): "
            f"drug_threshold_frac={self.DRUG_THRESHOLD_FRAC}, "
            f"D_SAFE={self.D_SAFE}, alpha_od={self.ALPHA_OVERDOSE}"
        )

    # ------------------------------------------------------------------
    # Analytical composite (Langmuir occupancy, normalized to healthy)
    # ------------------------------------------------------------------

    def _occupancy(self, conc: float, kd: float) -> float:
        return float(conc / (kd + conc))

    def _composite_signal(self, config: dict, scenario: str) -> float:
        """sensitivity * (weighted sum of normalized occupancies)."""
        kd      = float(config.get("kd_nm", 1.0))
        kd_ctx  = float(config.get("kd_ctx_nm", 1.0))
        kd_p1np = float(config.get("kd_p1np_nm", 1.0))
        w_ctx   = float(config.get("w_ctx", 0.1))
        w_p1np  = float(config.get("w_p1np", 0.1))
        w_scl   = max(0.0, 1.0 - w_ctx - w_p1np)
        sens    = float(config.get("sensitivity", 1.0))

        concs_s = _NOMINAL_CONCS[scenario]
        concs_h = _NOMINAL_CONCS["healthy"]

        theta_scl   = self._occupancy(concs_s["scl"],  kd)
        theta_ctx   = self._occupancy(concs_s["ctx"],  kd_ctx)
        theta_p1np  = self._occupancy(concs_s["p1np"], kd_p1np)
        theta_scl_h  = self._occupancy(concs_h["scl"],  kd)
        theta_ctx_h  = self._occupancy(concs_h["ctx"],  kd_ctx)
        theta_p1np_h = self._occupancy(concs_h["p1np"], kd_p1np)

        eps = 1e-12
        norm_scl   = theta_scl   / max(theta_scl_h,   eps)
        norm_ctx   = theta_ctx   / max(theta_ctx_h,   eps)
        norm_p1np  = theta_p1np  / max(theta_p1np_h,  eps)

        return float(sens * (w_scl * norm_scl + w_ctx * norm_ctx + w_p1np * norm_p1np))

    # ------------------------------------------------------------------
    # Analytical healthy FP rate
    # ------------------------------------------------------------------

    # Biological variability parameters (literature-calibrated, environment_configs.py v5.4)
    _BIO_SIGMA_BONE = {'scl': 0.30, 'ctx': 0.45, 'p1np': 0.25}
    _BIO_SIGMA_RATE = 0.15   # k_prod and k_deg sigma each
    _BIO_CORR = np.array([
        [1.00, 0.50, 0.35],
        [0.50, 1.00, 0.45],
        [0.35, 0.45, 1.00],
    ])
    _THRESHOLD_MARGIN_FACTOR = 1.25  # matches generate_random_array_config()

    def _analytical_healthy_fp_rate(self, config: dict) -> float:
        """
        Analytically compute healthy FP rate from biosensor parameters.

        The real simulator draws each patient's biomarkers from correlated
        log-normal distributions (σ_SOST=0.30, σ_CTX=0.45, σ_P1NP=0.25,
        r_SOST-CTX=0.50, r_CTX-P1NP=0.45), with k_prod/k_deg each at σ=0.15
        adding in quadrature to effective biomarker σ. The Langmuir occupancy
        ratio R_i = θ(C_i,Kd_i)/θ(H_i,Kd_i) is approximately log-normal with
        effective σ_R_i = (1−θ_H_i) × σ_eff_i (first-order Taylor expansion).

        The composite signal R = Σ w_i R_i is approximately normal (CLT) with
        mean=1.0 and variance = w^T (CORR ⊙ σ_r σ_r^T) w.

        FP rate = P(R > threshold/sensitivity) ≈ P(Z > z_fp).

        This formula is surrogate-independent and consistent with the ODE
        simulator's biological variability model.
        """
        kd      = float(config.get('kd_nm',     1.0))
        kd_ctx  = float(config.get('kd_ctx_nm', 1.0))
        kd_p1np = float(config.get('kd_p1np_nm', 1.0))
        w_ctx   = float(config.get('w_ctx',  0.0))
        w_p1np  = float(config.get('w_p1np', 0.0))
        w_scl   = max(0.0, 1.0 - w_ctx - w_p1np)
        H       = _NOMINAL_CONCS['healthy']

        # Langmuir occupancy at healthy nominal → elasticity η = 1 − θ
        theta_h = np.array([
            H['scl']  / (kd      + H['scl']  + 1e-12),
            H['ctx']  / (kd_ctx  + H['ctx']  + 1e-12),
            H['p1np'] / (kd_p1np + H['p1np'] + 1e-12),
        ])
        eta = 1.0 - theta_h   # d(ln R_i)/d(ln C_i)

        # Effective sigma for each ODE steady-state (IC variability + rate variability)
        sigma_rate_eff = np.sqrt(2.0) * self._BIO_SIGMA_RATE   # ~0.2121
        eff_sig = np.array([
            np.sqrt(self._BIO_SIGMA_BONE['scl'] **2 + sigma_rate_eff**2),
            np.sqrt(self._BIO_SIGMA_BONE['ctx'] **2 + sigma_rate_eff**2),
            np.sqrt(self._BIO_SIGMA_BONE['p1np']**2 + sigma_rate_eff**2),
        ])

        # Log-normal sigma of normalized occupancy ratio per channel
        sig_r = eta * eff_sig
        ws    = np.array([w_scl, w_ctx, w_p1np])

        # Composite variance: w^T (CORR ⊙ outer(sig_r)) w
        cov_mat = self._BIO_CORR * np.outer(sig_r, sig_r)
        sigma_comp = float(np.sqrt(max(float(ws @ cov_mat @ ws), 1e-9)))

        # Threshold in normalized composite space:
        #   threshold = sensitivity + MARGIN * (R_pmo - 1.0)
        #   composite_healthy ≈ sensitivity × R_h   (R_h ~ 1 ± sigma_comp)
        #   FP: R_h > 1 + MARGIN * (R_pmo - 1.0) / sensitivity
        R_pmo  = self._R_scenario(config, 'pmo')
        sens   = float(config.get('sensitivity', 1.0))
        threshold_margin = self._THRESHOLD_MARGIN_FACTOR * (R_pmo - 1.0) / max(sens, 1e-9)
        z_fp   = threshold_margin / max(sigma_comp, 1e-9)
        fp_rate = float(0.5 * erfc(z_fp / np.sqrt(2.0)))
        return float(np.clip(fp_rate, 0.0, 1.0))

    def _R_scenario(self, config: dict, scenario: str) -> float:
        """
        Normalized composite occupancy ratio: scenario / healthy.

        This equals the weighted sum of Langmuir occupancy ratios:
            R = w_scl * (θ_scl_s/θ_scl_h) + w_ctx * (θ_ctx_s/θ_ctx_h) + w_p1np * (...)

        By construction R_healthy = 1.0.
        Sensitivity cancels out completely (composite_s / composite_h = R, no sensitivity).
        """
        sig_h = self._composite_signal(config, "healthy")
        sig_s = self._composite_signal(config, scenario)
        if sig_h < 1e-9:
            return 1.0
        return float(sig_s / sig_h)

    # ------------------------------------------------------------------
    # Drug dose model (V6: relative threshold, no DR multiplier)
    # ------------------------------------------------------------------

    def _drug_dose_and_overdose(self, config: dict, scenario: str) -> Tuple[float, float]:
        """
        Compute drug dose and overdose penalty for a given scenario.

        dose = K_RELEASE * max(0, (R - DRUG_THRESHOLD_FRAC) / DRUG_THRESHOLD_FRAC)

        Dose is INDEPENDENT of sensitivity (R cancels sensitivity out).
        Overdose penalty activates quadratically when dose > D_SAFE.

        Returns: (raw_dose, overdose_penalty)
        """
        R = self._R_scenario(config, scenario)
        frac_margin = max(0.0, (R - self.DRUG_THRESHOLD_FRAC) / self.DRUG_THRESHOLD_FRAC)
        raw_dose = float(self.K_RELEASE * frac_margin)
        overdose = float(max(0.0, raw_dose / self.D_SAFE - 1.0) ** 2 * self.ALPHA_OVERDOSE)
        return raw_dose, overdose

    def _bmd_net(self, raw_dose: float, overdose: float) -> float:
        """
        Net BMD gain = Hill(dose) - overdose_penalty.

        Normalized by BMD_GAIN_REF so 1.0 = reference therapeutic outcome.
        """
        if raw_dose <= 0:
            return 0.0
        bmd_gross = self.BMD_GAIN_MAX * raw_dose / (self.D_HALF + raw_dose)
        bmd_normalized = min(bmd_gross / self.BMD_GAIN_REF, 2.0)
        return float(bmd_normalized - overdose)

    # ------------------------------------------------------------------
    # Surrogate predictions
    # ------------------------------------------------------------------

    def _predict_all(self, config: dict) -> dict:
        kd_nm         = config.get("kd_nm", 1.0)
        sensitivity   = config.get("sensitivity", 1.0)
        response_time = config.get("response_time_s", 500.0)
        btype         = config["biosensor_type"]
        noise         = config["noise_preset"]
        kd_ctx        = config.get("kd_ctx_nm", 0.0) if btype == "array" else 0.0
        kd_p1np       = config.get("kd_p1np_nm", 0.0) if btype == "array" else 0.0
        w_ctx         = config.get("w_ctx", 0.0) if btype == "array" else 0.0
        w_p1np        = config.get("w_p1np", 0.0) if btype == "array" else 0.0

        def _p(scenario):
            return self.surrogate_loader.predict(
                kd_nm=kd_nm, sensitivity=sensitivity, response_time=response_time,
                biosensor_type=btype, noise_preset=noise, scenario=scenario,
                kd_ctx=kd_ctx, kd_p1np=kd_p1np, w_ctx=w_ctx, w_p1np=w_p1np,
            )

        return {
            "ckd":     _p("ckd_mbd"),
            "pmo":     _p("pmo"),
            "mild":    _p("pmo_mild"),
            "healthy": _p("healthy"),
        }

    # ------------------------------------------------------------------
    # __call__
    # ------------------------------------------------------------------

    def __call__(self, config: Dict) -> float:
        try:
            kd_nm          = config.get("kd_nm", 1.0)
            biosensor_type = config["biosensor_type"]

            if self.apply_constraints and biosensor_type == "direct_binding":
                theta_H = self._HEALTHY_CONC / (kd_nm + self._HEALTHY_CONC)
                theta_P = self._PMO_CONC     / (kd_nm + self._PMO_CONC)
                if theta_H > 1e-9 and theta_P / theta_H < self.MIN_DISCRIMINATION_RATIO:
                    return self.CATASTROPHIC_PENALTY

            preds = self._predict_all(config)
            dr_ckd,  fnr_ckd,  ttd_ckd  = preds["ckd"]
            dr_pmo,  fnr_pmo,  ttd_pmo  = preds["pmo"]
            dr_mild, fnr_mild, ttd_mild = preds["mild"]
            # Use analytical FP rate (not surrogate-predicted): the surrogate
            # underestimates healthy FP by ~10× because it can't integrate the
            # full biological variability distribution. The analytical formula
            # uses the same log-normal parameters as the ODE simulator.
            dr_healthy = self._analytical_healthy_fp_rate(config)

            # V6: relative-threshold drug dosing (sensitivity-independent)
            if biosensor_type == "array":
                dose_ckd, od_ckd  = self._drug_dose_and_overdose(config, "ckd_mbd")
                dose_pmo, od_pmo  = self._drug_dose_and_overdose(config, "pmo")
                dose_mild, od_mild = self._drug_dose_and_overdose(config, "pmo_mild")
            else:
                # non-array fallback: use sensitivity as a proxy R
                R_proxy = 1.5
                dose_pmo = dose_ckd = self.K_RELEASE * (R_proxy - self.DRUG_THRESHOLD_FRAC) / self.DRUG_THRESHOLD_FRAC
                dose_mild = dose_pmo * 0.5
                od_pmo = od_ckd = od_mild = 0.0

            bmd_ckd  = self._bmd_net(dose_ckd,  od_ckd)
            bmd_pmo  = self._bmd_net(dose_pmo,  od_pmo)
            bmd_mild = self._bmd_net(dose_mild, od_mild)

            therapeutic_mean = (
                self.TREAT_W_MILD * bmd_mild
                + self.TREAT_W_PMO  * bmd_pmo
                + self.TREAT_W_CKD  * bmd_ckd
            )
            therapeutic_mean = max(therapeutic_mean, -1.0)

            # Surrogate-based detection terms
            dr_mean    = (dr_ckd + dr_pmo + dr_mild) / 3.0
            dr_min     = min(dr_ckd, dr_pmo, dr_mild)
            fnr_mean   = (fnr_ckd + fnr_pmo + fnr_mild) / 3.0
            ttd_mean   = (ttd_ckd + ttd_pmo + ttd_mild) / 3.0
            dr_disease = 0.5 * dr_mean + 0.5 * dr_min

            # Hard constraint penalties
            infeas_penalty = 0.0
            if self.apply_constraints:
                if dr_ckd < self.MIN_DR:
                    infeas_penalty += 0.30 * (self.MIN_DR - dr_ckd) / self.MIN_DR
                if dr_pmo < self.MIN_DR:
                    infeas_penalty += 0.30 * (self.MIN_DR - dr_pmo) / self.MIN_DR
                _fnr_max = max(fnr_ckd, fnr_pmo)
                if _fnr_max > self.MAX_FNR:
                    infeas_penalty += 0.25 * (_fnr_max - self.MAX_FNR) / (1.0 - self.MAX_FNR)
                if dr_healthy > self.MAX_FP_DR:
                    infeas_penalty += 0.30 * (dr_healthy - self.MAX_FP_DR) / (1.0 - self.MAX_FP_DR)

            # Soft penalties (reduced mild-DR penalty vs v5 to avoid sensitivity bias)
            soft_penalty = 0.0
            if dr_mean < 0.75:
                soft_penalty += 0.15 * max(0.0, (0.75 - dr_mean) / 0.75)
            if dr_min < 0.60:
                soft_penalty += 0.10 * max(0.0, (0.60 - dr_min) / 0.60)
            if fnr_mean > 0.25:
                soft_penalty += 0.10 * max(0.0, (fnr_mean - 0.25) / 0.75)
            if dr_healthy > 0.05:
                soft_penalty += 0.20 * max(0.0, (dr_healthy - 0.05) / 0.10)
            # Note: dr_mild soft penalty removed vs v5 (was 0.40 weight, created
            # excessive sensitivity bias; therapeutic term now handles mild efficacy)

            objective = (
                self.WEIGHT_THERAPEUTIC * therapeutic_mean
                + self.WEIGHT_DR        * dr_disease
                + self.WEIGHT_FNR       * (1.0 - fnr_mean)
                + self.WEIGHT_TTD       * (1.0 - ttd_mean / self.TTD_MAX)
                - self.WEIGHT_FP        * dr_healthy
                - self.WEIGHT_TOXICITY  * dr_healthy
            ) - soft_penalty - infeas_penalty

            return float(np.clip(objective, -0.5, 1.0))

        except Exception as e:
            logger.error(f"TherapeuticObjectiveV6 error: {e}")
            return 0.0

    def evaluate_with_details(self, config: Dict) -> Tuple[float, Dict]:
        """Detailed evaluation with per-component breakdown."""
        try:
            biosensor_type = config["biosensor_type"]
            kd_nm = config.get("kd_nm", 1.0)

            if self.apply_constraints and biosensor_type == "direct_binding":
                theta_H = self._HEALTHY_CONC / (kd_nm + self._HEALTHY_CONC)
                theta_P = self._PMO_CONC     / (kd_nm + self._PMO_CONC)
                if theta_H > 1e-9 and theta_P / theta_H < self.MIN_DISCRIMINATION_RATIO:
                    return self.CATASTROPHIC_PENALTY, {"error": "Langmuir guard triggered"}

            preds = self._predict_all(config)
            dr_ckd, fnr_ckd, ttd_ckd     = preds["ckd"]
            dr_pmo, fnr_pmo, ttd_pmo     = preds["pmo"]
            dr_mild, fnr_mild, ttd_mild  = preds["mild"]
            dr_healthy = self._analytical_healthy_fp_rate(config)

            R_ckd  = self._R_scenario(config, "ckd_mbd")
            R_pmo  = self._R_scenario(config, "pmo")
            R_mild = self._R_scenario(config, "pmo_mild")
            R_h    = self._R_scenario(config, "healthy")

            if biosensor_type == "array":
                dose_ckd, od_ckd  = self._drug_dose_and_overdose(config, "ckd_mbd")
                dose_pmo, od_pmo  = self._drug_dose_and_overdose(config, "pmo")
                dose_mild, od_mild = self._drug_dose_and_overdose(config, "pmo_mild")
            else:
                R_proxy = 1.5
                dose_pmo = dose_ckd = self.K_RELEASE * (R_proxy - self.DRUG_THRESHOLD_FRAC) / self.DRUG_THRESHOLD_FRAC
                dose_mild = dose_pmo * 0.5
                od_pmo = od_ckd = od_mild = 0.0

            bmd_ckd  = self._bmd_net(dose_ckd,  od_ckd)
            bmd_pmo  = self._bmd_net(dose_pmo,  od_pmo)
            bmd_mild = self._bmd_net(dose_mild, od_mild)
            therapeutic_mean = (
                self.TREAT_W_MILD * bmd_mild
                + self.TREAT_W_PMO  * bmd_pmo
                + self.TREAT_W_CKD  * bmd_ckd
            )
            therapeutic_mean = max(therapeutic_mean, -1.0)

            dr_mean    = (dr_ckd + dr_pmo + dr_mild) / 3.0
            dr_min     = min(dr_ckd, dr_pmo, dr_mild)
            fnr_mean   = (fnr_ckd + fnr_pmo + fnr_mild) / 3.0
            ttd_mean   = (ttd_ckd + ttd_pmo + ttd_mild) / 3.0
            dr_disease = 0.5 * dr_mean + 0.5 * dr_min

            infeas_penalty = 0.0
            if self.apply_constraints:
                if dr_ckd < self.MIN_DR:
                    infeas_penalty += 0.30 * (self.MIN_DR - dr_ckd) / self.MIN_DR
                if dr_pmo < self.MIN_DR:
                    infeas_penalty += 0.30 * (self.MIN_DR - dr_pmo) / self.MIN_DR
                if max(fnr_ckd, fnr_pmo) > self.MAX_FNR:
                    infeas_penalty += 0.25 * (max(fnr_ckd, fnr_pmo) - self.MAX_FNR) / (1.0 - self.MAX_FNR)
                if dr_healthy > self.MAX_FP_DR:
                    infeas_penalty += 0.30 * (dr_healthy - self.MAX_FP_DR) / (1.0 - self.MAX_FP_DR)

            soft_penalty = 0.0
            if dr_mean < 0.75:
                soft_penalty += 0.15 * max(0.0, (0.75 - dr_mean) / 0.75)
            if dr_min < 0.60:
                soft_penalty += 0.10 * max(0.0, (0.60 - dr_min) / 0.60)
            if fnr_mean > 0.25:
                soft_penalty += 0.10 * max(0.0, (fnr_mean - 0.25) / 0.75)
            if dr_healthy > 0.05:
                soft_penalty += 0.20 * max(0.0, (dr_healthy - 0.05) / 0.10)

            objective = (
                self.WEIGHT_THERAPEUTIC * therapeutic_mean
                + self.WEIGHT_DR        * dr_disease
                + self.WEIGHT_FNR       * (1.0 - fnr_mean)
                + self.WEIGHT_TTD       * (1.0 - ttd_mean / self.TTD_MAX)
                - self.WEIGHT_FP        * dr_healthy
                - self.WEIGHT_TOXICITY  * dr_healthy
            ) - soft_penalty - infeas_penalty
            objective = float(np.clip(objective, -0.5, 1.0))

            return objective, {
                "dr_mean":           dr_mean,
                "dr_ckd":            dr_ckd,
                "dr_pmo":            dr_pmo,
                "dr_mild":           dr_mild,
                "dr_min":            dr_min,
                "fnr_mean":          fnr_mean,
                "ttd_mean":          ttd_mean,
                "dr_healthy":        dr_healthy,
                "R_healthy":         R_h,
                "R_pmo_mild":        R_mild,
                "R_pmo":             R_pmo,
                "R_ckd":             R_ckd,
                "dose_mild":         dose_mild,
                "dose_pmo":          dose_pmo,
                "dose_ckd":          dose_ckd,
                "overdose_mild":     od_mild,
                "overdose_pmo":      od_pmo,
                "overdose_ckd":      od_ckd,
                "bmd_net_mild":      bmd_mild,
                "bmd_net_pmo":       bmd_pmo,
                "bmd_net_ckd":       bmd_ckd,
                "therapeutic_mean":  therapeutic_mean,
                "infeas_penalty":    infeas_penalty,
                "soft_penalty":      soft_penalty,
            }

        except Exception as e:
            logger.error(f"TherapeuticObjectiveV6.evaluate_with_details error: {e}")
            return 0.0, {"error": str(e)}
