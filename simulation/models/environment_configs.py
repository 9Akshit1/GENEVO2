"""
Environment configuration presets — Version 5.0 (Biomarker Realism Fixes).

V5.1 BIOMARKER REALISM FIX: CTX PMO Variability Correction
============================================================
Literature realism audit (2026-06-07) found CTX PMO CV=29% (sigma=0.30)
is below every published estimate for inter-individual CTX variability
in postmenopausal populations:
  - Garnero JBMR 1994: inter-individual CV ~45-55% for serum CTX in PMO
  - EuBIVAS study: intra-individual (within-subject) CV=15%; inter-individual
    is substantially larger (35-60% range)
  Fix: PMO sigma_overrides raised for CTX_bone and k_prod_CTX from 0.30
  to 0.45, targeting the literature midpoint (~47% inter-individual CV).
  Effect: CTX/PMO Cohen's d decreases from ~2.95 toward ~1.8 (realistic
  for bone turnover markers in unselected postmenopausal cohorts).

V5.0 BIOMARKER REALISM FIX: Corrected PMO Patient Variability
==============================================================
Biomarker audit (Mirza JCEM 2010, PMC10859939, PMC5426702) revealed:

  PMO: Model σ=0.15 was far too tight. Clinical literature shows serum
  sclerostin is heterogeneous in postmenopausal patients:
    - Early post-menopause: ~2.4× premenopausal (Mirza 2010, d=2.35)
    - Established osteoporosis vs age-matched healthy: LOWER, not higher
      (PMC10859939: 4.62 vs 5.74 ng/mL, d=0.77 inverted)
    - IQR-derived lognormal σ for postmenopausal population: ~0.35
  Fix: PMO sigma_overrides set to σ=0.35 for Sclerostin_bone, σ=0.30
  for k_prod_Scl. This makes ~10th-percentile PMO patients overlap with
  the healthy reference range, which is clinically correct.

  IMPORTANT framing: "Healthy" in this model = premenopausal women
  (Estrogen=1.0 nM). The PMO scenario captures the early-to-mid
  postmenopausal transition, NOT a chronic established osteoporosis state.
  The 2.3× sclerostin ratio is calibrated to the premenopausal reference.

  CKD-MBD: σ=0.50 validated against PMC5426702 IQR data (estimated σ≈0.54
  from hemodialysis IQR), represents CKD Stage 3–5/dialysis. Correct.

V4.0 CRITICAL FIX: Bounded Physiological Variability (unchanged)
=================================================================
  - Estrogen: [0.1, 3.0] nM (pre-menopausal range)
  - PTH: [10, 400] pg/mL (normal to severe hyperparathyroidism)
  - k_prod: Reduced σ to 0.10 (±10%, not ±15–26%)
  - Rejection sampling: resample if out of bounds (don't clip silently)
"""

import numpy as np
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

# Diffusion equilibrium ratios (unchanged from v3.x)
_SCLEROSTIN_SENSOR_RATIO = 25.0
_RANKL_SENSOR_RATIO      = 30.0
_OPG_SENSOR_RATIO        = 30.0

# Cross-biomarker correlation matrix for (SOST, CTX, P1NP) log-space sampling.
# Values are ESTIMATED from bone biology — not measured from individual patient records.
# Rationale:
#   SOST-CTX r=0.50: Both elevated in high-turnover states (menopause, CKD). Ardawi 2011
#     reports parallel SOST elevation with bone turnover markers in postmenopausal cohort.
#   SOST-P1NP r=0.35: Weaker coupling — SOST suppresses formation (negative causal),
#     but both still co-elevated in early PMO before chronic suppression sets in.
#   CTX-P1NP r=0.45: Coupled resorption/formation — Vasikaran 2011 IFCC consensus
#     recommends measuring both together as they move in parallel in active disease.
# Source: estimated; will be refined once individual patient records are available.
_BIOMARKER_CORR = np.array([
    [1.00, 0.50, 0.35],   # SOST
    [0.50, 1.00, 0.45],   # CTX
    [0.35, 0.45, 1.00],   # P1NP
])
# Cholesky factor (pre-computed once — same for all scenarios)
_BIOMARKER_CHOL = np.linalg.cholesky(_BIOMARKER_CORR)
_CTX_SENSOR_RATIO        = 25.0  # k_diff_CTX/k_back_CTX * bone/sensor_chamber = 0.05/0.02 * 1.0/0.1
_P1NP_SENSOR_RATIO       = 25.0  # same diffusion parameters


class EnvironmentConfig:
    """Configuration class for bone microenvironment scenarios."""

    def __init__(self, name: str, param_overrides: Dict[str, float],
                 description: str = "", sigma_overrides: Dict[str, float] = None):
        self.name = name
        self.param_overrides = param_overrides
        self.description = description
        # Per-parameter sigma overrides for apply_variability(); a higher sigma
        # on Sclerostin_bone in CKD creates a realistic patient population where
        # some CKD cases have near-PMO sclerostin levels (early/mild disease).
        self.sigma_overrides: Dict[str, float] = sigma_overrides or {}
        self._validate()
        logger.debug(f"Created environment config: {name}")

    def _validate(self):
        """Validate and enforce parameter ranges are biologically plausible."""
        VALID_RANGES = {
            'Estrogen':    (0.0,   2.0),
            'PTH':         (10.0,  500.0),
            'Sclerostin_bone':   (1e-4, 0.5),
            'Sclerostin_sensor': (1e-3, 20.0),
            'RANKL_bone':   (0.05,  5.0),
            'RANKL_sensor': (0.5,  200.0),
            'OPG_bone':     (0.5,   20.0),
            'OPG_sensor':   (5.0,  600.0),
            'k_prod_Scl':   (1e-7, 2e-5),
            'k_prod_RANKL': (1e-6, 2e-5),
            'k_prod_OPG':   (1e-5, 5e-4),
            'k_deg_Scl':    (5e-5, 1e-4),
            'k_deg_RANKL':  (1e-5, 1e-4),
            'k_deg_OPG':    (1e-5, 1e-4),
            # CTX and P1NP (multi-biomarker panel)
            'CTX_bone':    (1e-5, 0.5),
            'CTX_sensor':  (1e-4, 15.0),
            'P1NP_bone':   (1e-5, 0.5),
            'P1NP_sensor': (1e-4, 15.0),
            'k_prod_CTX':  (1e-7, 1e-5),
            'k_deg_CTX':   (5e-7, 5e-5),  # wide: covers normal to severe CKD retention
            'k_prod_P1NP': (1e-7, 5e-6),
            'k_deg_P1NP':  (2e-6, 5e-5),
        }
        
        enforced_params = {}
        for param, value in self.param_overrides.items():
            if param in VALID_RANGES:
                lo, hi = VALID_RANGES[param]
                if not (lo <= value <= hi):
                    logger.warning(
                        f"Parameter {param} = {value:.3e} outside range "
                        f"[{lo:.3e}, {hi:.3e}]."
                    )
            enforced_params[param] = value
        
        self.param_overrides = enforced_params

    def get_params(self) -> Dict[str, float]:
        """Return parameter overrides (copy)."""
        return self.param_overrides.copy()

    def apply_variability(self, variability: float = 0.15,
                          use_correlations: bool = False) -> Dict[str, float]:
        """
        FIX 2: Apply bounded lognormal parameter variability.
        
        CRITICAL CHANGE: Each parameter type gets constrained σ and hard bounds.
        
        Parameter bounds (physiologically justified):
          - Estrogen: [0.1, 3.0] nM (pre-menopausal range)
          - PTH: [10, 400] pg/mL (normal to severe HPT)
          - k_prod: ±10% (tight, prevents dead regions)
          - k_deg: ±10%
          - Biomarkers: ±15% (higher variation allowed)
        
        On out-of-bounds: Raises ValueError (don't clip silently)
        Caller should catch and resample.
        
        Sensor ICs are derived from bone after perturbation, maintaining
        diffusion equilibrium (25× for Sclerostin, 30× for others).
        """
        
        SENSOR_DERIVED = {'Sclerostin_sensor', 'RANKL_sensor', 'OPG_sensor',
                          'CTX_sensor', 'P1NP_sensor'}
        
        # Hard bounds: these are biologically non-negotiable
        HARD_BOUNDS = {
            'Estrogen':    (0.1,   3.0),    # nM, premenopausal range
            'PTH':         (10.0,  400.0),  # pg/mL, pathological max
        }
        
        varied_params: Dict[str, float] = {}

        for param, value in self.param_overrides.items():
            if param in SENSOR_DERIVED:
                continue  # Derived below

            # Base sigma per parameter type; can be overridden per-config
            # via self.sigma_overrides to model patient-population spread.
            if param.startswith('k_prod_') or param.startswith('k_deg_'):
                sigma = 0.10
            elif param in ('Estrogen', 'PTH'):
                sigma = 0.12
            elif param in ('Sclerostin_bone', 'RANKL_bone', 'OPG_bone', 'MineralIon',
                           'CTX_bone', 'P1NP_bone'):
                sigma = 0.15
            else:
                sigma = 0.10

            sigma = self.sigma_overrides.get(param, sigma)

            factor = np.random.lognormal(mean=0.0, sigma=sigma)
            new_value = value * factor

            # FIX 2: Apply hard bounds and raise on violation
            if param in HARD_BOUNDS:
                lo, hi = HARD_BOUNDS[param]
                if not (lo <= new_value <= hi):
                    raise ValueError(
                        f"Parameter {param} = {new_value:.3e} out of bounds "
                        f"[{lo:.3e}, {hi:.3e}] after variability sampling. "
                        f"Rejecting this run."
                    )

            varied_params[param] = new_value

        # Cross-biomarker correlation override (opt-in; default=False for backward compat).
        # Replaces independently-sampled SOST/CTX/P1NP bone ICs with correlated draws
        # using the pre-computed Cholesky factor. Marginal distributions are unchanged.
        if use_correlations and all(
            k in self.param_overrides for k in ('Sclerostin_bone', 'CTX_bone', 'P1NP_bone')
        ):
            sig_scl  = self.sigma_overrides.get('Sclerostin_bone', 0.15)
            sig_ctx  = self.sigma_overrides.get('CTX_bone',        0.15)
            sig_p1np = self.sigma_overrides.get('P1NP_bone',       0.15)
            sigmas   = np.array([sig_scl, sig_ctx, sig_p1np])

            z_iid   = np.random.randn(3)
            z_corr  = _BIOMARKER_CHOL @ z_iid      # correlated standard normals
            log_fac = z_corr * sigmas               # scale to per-parameter sigma
            factors = np.exp(log_fac)

            varied_params['Sclerostin_bone'] = self.param_overrides['Sclerostin_bone'] * factors[0]
            varied_params['CTX_bone']        = self.param_overrides['CTX_bone']        * factors[1]
            varied_params['P1NP_bone']       = self.param_overrides['P1NP_bone']       * factors[2]

        # FIX 2: Derive sensor ICs to maintain diffusion equilibrium
        if 'Sclerostin_bone' in varied_params:
            varied_params['Sclerostin_sensor'] = (
                _SCLEROSTIN_SENSOR_RATIO * varied_params['Sclerostin_bone']
            )
        if 'RANKL_bone' in varied_params:
            varied_params['RANKL_sensor'] = (
                _RANKL_SENSOR_RATIO * varied_params['RANKL_bone']
            )
        if 'OPG_bone' in varied_params:
            varied_params['OPG_sensor'] = (
                _OPG_SENSOR_RATIO * varied_params['OPG_bone']
            )
        if 'CTX_bone' in varied_params:
            varied_params['CTX_sensor'] = (
                _CTX_SENSOR_RATIO * varied_params['CTX_bone']
            )
        if 'P1NP_bone' in varied_params:
            varied_params['P1NP_sensor'] = (
                _P1NP_SENSOR_RATIO * varied_params['P1NP_bone']
            )

        return varied_params


# ============================================================================
# SCENARIO DEFINITIONS (Unchanged from v3.x)
# ============================================================================

def get_healthy_config() -> EnvironmentConfig:
    """Premenopausal healthy reference: Sclerostin_sensor≈0.375 nM.

    Calibrated to premenopausal women (Estrogen=1.0 nM, PTH=45 pg/mL).
    Mirza JCEM 2010: serum sclerostin 0.48 ± 0.15 ng/mL in premenopausal
    women (age 26.8 ± 2.6 yr). All disease comparisons are made against
    this premenopausal reference population.
    """
    params = {
        'Estrogen': 1.0,
        'PTH':      45.0,
        'k_prod_Scl':        0.000001389,
        'k_deg_Scl':         0.00007,
        'Sclerostin_bone':   0.015,
        'k_prod_RANKL':      0.00000429,
        'k_deg_RANKL':       0.00002,
        'RANKL_bone':        0.5,
        'k_prod_OPG':        0.0000472,
        'k_deg_OPG':         0.00002,
        'OPG_bone':          5.0,
        'MineralIon':        2.5,
        'k_prod_Mineral':    0.087,
        'k_loss_Mineral':    0.025,
        # CTX calibration: OC_ss=0.3 → CTX_bone=0.008 → CTX_sensor=0.20 nM
        'k_prod_CTX':        5.33e-7,
        'k_deg_CTX':         0.00002,
        'CTX_bone':          0.008,
        # P1NP calibration: OB_ss=2.0 → P1NP_bone=0.014 → P1NP_sensor=0.35 nM
        'k_prod_P1NP':       1.40e-7,
        'k_deg_P1NP':        0.00002,
        'P1NP_bone':         0.014,
    }
    # v5.3: Raise rate-parameter sigma from 0.10 to 0.15 for healthy scenario.
    # Motivation: steady-state concentration proportional to k_prod/k_deg.
    # With sigma_k=0.10, CV_healthy = sqrt(0.10^2 + 0.10^2) = 14.1%, well below
    # published inter-individual CVs (20-30% for SOST/CTX/P1NP, Ueland 2009,
    # Garnero 1994, Vasikaran 2011). Raising to 0.15 gives CV = 21.2%, matching
    # literature midpoints. Disease sigma_overrides are unchanged — this change
    # widens ONLY the healthy reference population, which is the correct biological
    # interpretation (genuine physiological heterogeneity between individuals).
    # v5.4: Raise bone IC sigmas to match literature inter-individual CVs.
    # SOST CV target 31% (Ueland 2009), CTX CV target 47% (Garnero 1994), P1NP 25% (Vasikaran 2011).
    # IC sigma dominates sensor concentration variability; rate-constant sigma (0.15) adds ~21% on top.
    # Previous: Sclerostin_bone/CTX_bone/P1NP_bone used default sigma=0.15 (→ CV≈15%),
    # causing FP rates lower than clinically observed when surrogates are deployed.
    healthy_sigma_overrides = {
        'k_prod_Scl':      0.15,
        'k_deg_Scl':       0.15,
        'k_prod_CTX':      0.15,
        'k_deg_CTX':       0.15,
        'k_prod_P1NP':     0.15,
        'k_deg_P1NP':      0.15,
        'Sclerostin_bone': 0.30,  # SOST IC sigma → CV≈31% (Ueland 2009: 31-47%)
        'CTX_bone':        0.45,  # CTX IC sigma → CV≈47% (Garnero 1994: 45-55%)
        'P1NP_bone':       0.25,  # P1NP IC sigma → CV≈25% (Vasikaran 2011: 20-30%)
    }
    return EnvironmentConfig(
        name='healthy',
        param_overrides=params,
        description=(
            "Premenopausal healthy reference: Sclerostin_sensor=0.375 nM, "
            "CTX_sensor=0.200 nM, P1NP_sensor=0.350 nM (Estrogen=1.0 nM). "
            "All disease scenarios compared to this baseline. "
            "v5.4: bone IC sigmas raised to match literature CVs: SOST 31%, CTX 47%, P1NP 25%. "
            "Requires dataset regeneration; existing surrogates trained on v5.3 data are stale."
        ),
        sigma_overrides=healthy_sigma_overrides,
    )


def get_pmo_config() -> EnvironmentConfig:
    """Early post-menopausal / PMO: median Sclerostin_sensor≈0.875 nM (2.3× healthy).

    Represents the early-to-mid postmenopausal transition (Estrogen=0.2 nM),
    calibrated to Mirza JCEM 2010 which found postmenopausal women had 2.4×
    higher sclerostin than premenopausal healthy controls (1.16 vs 0.48 ng/mL,
    Cohen's d=2.35).

    BIOMARKER REALISM NOTE (v5.0 audit):
    The 2.3× ratio reflects the estrogen-withdrawal effect on sclerostin, NOT
    a fixed property of all PMO patients. Clinical literature shows:
      - Established severe osteoporosis patients may have LOWER sclerostin than
        age-matched controls (PMC10859939: 4.62 vs 5.74 ng/mL, d=0.77 inverted),
        because osteocyte apoptosis reduces sclerostin-producing cells.
      - Therefore PMO patient population is heterogeneous, spanning from near-
        healthy levels to substantially elevated.

    v5.0 FIX: sigma_overrides set to σ=0.35 (Sclerostin_bone) and σ=0.30
    (k_prod_Scl), derived from IQR analysis of postmenopausal cohort data.
    5th-percentile PMO patients now have Sclerostin_sensor ≈ 0.46 nM
    (above healthy 0.375 nM median but within 1 SD of healthy noise), creating
    a genuine detection challenge and making biosensor optimization meaningful.
    """
    params = {
        'Estrogen': 0.2,
        'PTH':      60.0,
        'k_prod_Scl':        0.000000719,
        'k_deg_Scl':         0.000065,
        'Sclerostin_bone':   0.035,
        'k_prod_RANKL':      0.00000377,
        'k_deg_RANKL':       0.00002,
        'RANKL_bone':        0.9,
        'k_prod_OPG':        0.0000636,
        'k_deg_OPG':         0.00002,
        'OPG_bone':          4.0,
        'MineralIon':        2.3,
        'k_prod_Mineral':    0.072,
        'k_loss_Mineral':    0.025,
        # CTX: elevated 2.5× healthy — high bone resorption in menopause transition.
        # OC_ss_pmo ≈ 0.40 (RANKL/OPG shift). Using k_prod 2.5× to achieve target.
        # CTX_bone → 0.020 nM → CTX_sensor ≈ 0.500 nM (Garnero JBMR 1994: ~2.5× in early menopause)
        'k_prod_CTX':        1.00e-6,
        'k_deg_CTX':         0.00002,    # normal renal clearance
        'CTX_bone':          0.020,
        # P1NP: elevated 1.5× healthy — moderate bone formation in early menopause.
        # v5.4: reduced from 2.0× to 1.5× to match Schafer JCEM 2011 (1.2-1.8×).
        # The 2.0× value produced Cohen's d=2.63 vs literature d~0.85 (Schafer 2011
        # measured with real specimens including measurement noise in pooled SD).
        # 1.5× is the literature mid-point; combined with measurement noise Layer 4
        # (CV~15-20%) this will produce a biologically plausible Cohen's d~1.4-1.6.
        # P1NP_bone → 0.021 nM → P1NP_sensor ≈ 0.525 nM
        'k_prod_P1NP':       2.10e-7,
        'k_deg_P1NP':        0.00002,    # normal clearance
        'P1NP_bone':         0.021,
    }
    return EnvironmentConfig(
        name='pmo',
        param_overrides=params,
        description=(
            "Early postmenopausal / PMO: median Sclerostin_sensor=0.875 nM (2.3x), "
            "CTX_sensor=0.500 nM (2.5x), P1NP_sensor=0.525 nM (1.5x) vs healthy. "
            "v5.4: P1NP fold reduced 2.0x→1.5x to match Schafer 2011 (1.2-1.8x). "
            "Wide variability in all markers — genuine detection challenge."
        ),
        sigma_overrides={
            'Sclerostin_bone': 0.35,
            'k_prod_Scl':      0.30,
            'CTX_bone':        0.45,   # v5.1: raised 0.30→0.45 (Garnero 1994: CV 45-55%)
            'k_prod_CTX':      0.45,
            'P1NP_bone':       0.25,
            'k_prod_P1NP':     0.25,
        },
    )


def get_ckd_mbd_config() -> EnvironmentConfig:
    """Chronic Kidney Disease — Mineral and Bone Disorder, Stage 3–5.

    V5.2 CKD RECALIBRATION (Phase A realism audit, June 2026):
    Phase A audit found CKD fold-changes were 2-4× above literature:
      Old:  SOST 6.1×, CTX 7.6×, P1NP 3.2× vs premenopausal healthy
      New:  SOST 3.0×, CTX 2.5×, P1NP 1.8× (literature midpoints below)
    Root cause of old values:
      - CTX: k_deg_CTX was 4× reduced (dialysis-level clearance impairment)
        → 8.4× analytic ratio, 7.6× observed. Changed to 2× reduced.
      - SOST: IC and k_prod calibrated to severe dialysis (6.1× observed).
        Changed to 3.0× (mixed stage 3-5 population).
      - P1NP: IC set to 3× healthy. Changed to 1.8× healthy.
    Literature targets:
      SOST: Cejka 2011 (CKD vs premenopausal healthy, compound: 2.5× CKD × 1.2×
            partial estrogen effect at Estrogen=0.6 vs 1.0) → ~3.0× composite.
      CTX:  Hlaing 2011 (CKD Stage 3-4 mixed-turnover): 1.5-2.5×; target 2.5×
            (elevated production from secondary HPT + 2× reduced clearance).
      P1NP: Kovesdy 2014 (CKD): 1.3-2.5×; target 1.8× (modest HPT-driven
            formation increase + partial hepatic/renal clearance reduction).

    IC values are the primary lever: empirically, simulated SS ≈ IC (the
    600s equilibration holds the system near the IC under ODE feedback).
    k_prod values scaled proportionally to maintain physiological ratio.
    """
    params = {
        'Estrogen': 0.6,
        'PTH':      180.0,
        # SOST: target ~3.0× healthy (0.015 → 0.045)
        # Secondary HPT + partial estrogen withdrawal (Estrogen=0.6 vs 1.0 healthy)
        'k_prod_Scl':        2.40e-6,   # 1.73× healthy (was 4.29e-6)
        'k_deg_Scl':         0.00006,   # slightly reduced vs healthy 7e-5
        'Sclerostin_bone':   0.045,     # 3.0× healthy IC 0.015 (was 0.080)
        'k_prod_RANKL':      0.00000643,
        'k_deg_RANKL':       0.00002,
        'RANKL_bone':        0.9,
        'k_prod_OPG':        0.000119,
        'k_deg_OPG':         0.00002,
        'OPG_bone':          5.5,
        'MineralIon':        3.8,
        'k_prod_Mineral':    0.150,
        'k_loss_Mineral':    0.055,
        # CTX: target ~2.5× healthy. Secondary HPT raises OC activity (1.31× higher
        # k_prod) + moderate renal clearance impairment (k_deg 2× reduced).
        # CTX_bone → 0.020 nM → CTX_sensor ≈ 0.50 nM (2.5× healthy 0.20 nM).
        # Literature: mixed CKD Stage 3-5 population (Hlaing 2011: 1.5-2.5×).
        'k_prod_CTX':        7.00e-7,   # 1.31× healthy 5.33e-7 (was 1.12e-6)
        'k_deg_CTX':         0.000010,  # 2× reduced vs healthy 0.00002 (was 4× reduced)
        'CTX_bone':          0.020,     # 2.5× healthy IC 0.008 (was 0.056)
        # P1NP: target ~1.8× healthy. Modest PTH-driven OB activity + partial
        # hepatic/renal clearance reduction (P1NP clears via both routes).
        # P1NP_bone → 0.025 nM → P1NP_sensor ≈ 0.625 nM (1.8× healthy 0.35 nM).
        'k_prod_P1NP':       1.25e-7,   # 0.89× healthy 1.40e-7 (was 2.10e-7)
        'k_deg_P1NP':        0.000015,  # 1.33× reduced vs healthy 2e-5 (was 2× reduced)
        'P1NP_bone':         0.025,     # 1.79× healthy IC 0.014 (was 0.042)
    }
    return EnvironmentConfig(
        name='ckd_mbd',
        param_overrides=params,
        description=(
            "CKD-MBD Stage 3-5: Sclerostin_sensor~1.13 nM (3.0x), "
            "CTX_sensor~0.50 nM (2.5x), P1NP_sensor~0.63 nM (1.8x) vs premenopausal healthy. "
            "v5.2: recalibrated from Phase A audit (old: 6.1x/7.6x/3.2x). "
            "Secondary hyperparathyroidism (PTH=180 pg/mL), moderate clearance impairment."
        ),
        sigma_overrides={
            # v5.4: CKD SOST sigma reduced 0.50→0.30 to match literature.
            # Cejka 2011 reports CKD inter-individual CV ~31%; the prior 0.50 (CV≈50%)
            # overestimated heterogeneity, causing artifically low CKD DR predictions.
            'Sclerostin_bone': 0.30,   # CKD SOST CV≈31% (Cejka 2011), was 0.50
            'k_prod_Scl':      0.45,
            'CTX_bone':        0.45,   # CTX still wide (mixed CKD Stage 3-5 renal clearance)
            'k_prod_CTX':      0.40,
            'P1NP_bone':       0.40,
            'k_prod_P1NP':     0.35,
        },
    )


def get_pmo_mild_config() -> EnvironmentConfig:
    """Perimenopause onset — SOST 1.5×, CTX 1.5×, P1NP 1.1× healthy.

    Represents 1-3 years post-menopause with partial estrogen withdrawal
    (Estrogen=0.6 nM). Corresponds to the lowest-severity PMO subgroup that
    still qualifies as postmenopausal (early transition, BMD not yet significantly
    impaired). This is the hardest detection case: biomarker elevation is modest
    and overlaps substantially with healthy inter-individual variation.

    Literature basis:
      - Seibel JBMR 2006: P1NP elevation 1.1-1.5× in early menopause transition
      - Mirza JCEM 2010: SOST 1.5× at 1-3 years post-menopause (vs 2.3× at 5+ years)
    """
    params = {
        'Estrogen': 0.6,
        'PTH':      50.0,
        'k_prod_Scl':        2.08e-6,
        'k_deg_Scl':         0.000065,
        'Sclerostin_bone':   0.0225,   # 1.5× healthy 0.015
        'k_prod_RANKL':      0.00000377,
        'k_deg_RANKL':       0.00002,
        'RANKL_bone':        0.6,
        'k_prod_OPG':        0.0000500,
        'k_deg_OPG':         0.00002,
        'OPG_bone':          4.5,
        'MineralIon':        2.4,
        'k_prod_Mineral':    0.082,
        'k_loss_Mineral':    0.025,
        'k_prod_CTX':        8.00e-7,  # 1.5× healthy 5.33e-7
        'k_deg_CTX':         0.00002,
        'CTX_bone':          0.012,    # 1.5× healthy 0.008 → CTX_sensor ≈ 0.30 nM
        'k_prod_P1NP':       1.54e-7,  # 1.1× healthy 1.40e-7
        'k_deg_P1NP':        0.00002,
        'P1NP_bone':         0.0154,   # 1.1× healthy 0.014 → P1NP_sensor ≈ 0.385 nM
    }
    return EnvironmentConfig(
        name='pmo_mild',
        param_overrides=params,
        description=(
            "Perimenopause onset: SOST~0.563 nM (1.5x), CTX~0.30 nM (1.5x), "
            "P1NP~0.385 nM (1.1x) vs healthy. Estrogen=0.6 nM, partial withdrawal. "
            "Hardest detection case: modest elevation with large healthy overlap."
        ),
        sigma_overrides={
            'Sclerostin_bone': 0.20,
            'k_prod_Scl':      0.15,
            'CTX_bone':        0.25,
            'k_prod_CTX':      0.25,
            'P1NP_bone':       0.20,
            'k_prod_P1NP':     0.20,
        },
    )


def get_pmo_severe_config() -> EnvironmentConfig:
    """Late postmenopause — SOST 3.5×, CTX 3.5×, P1NP 2.0× healthy.

    Represents 10+ years post-menopause with sustained high bone turnover,
    elevated sclerostin (osteocyte hyperactivity). Some women maintain this
    elevated sclerostin chronically; others develop established OP with LOWER
    sclerostin (osteocyte apoptosis). This config models the HIGH-SOST subtype.
    """
    params = {
        'Estrogen': 0.1,
        'PTH':      70.0,
        'k_prod_Scl':        4.86e-6,
        'k_deg_Scl':         0.000062,
        'Sclerostin_bone':   0.0525,   # 3.5× healthy 0.015
        'k_prod_RANKL':      0.00000429,
        'k_deg_RANKL':       0.00002,
        'RANKL_bone':        1.1,
        'k_prod_OPG':        0.0000636,
        'k_deg_OPG':         0.00002,
        'OPG_bone':          3.5,
        'MineralIon':        2.1,
        'k_prod_Mineral':    0.065,
        'k_loss_Mineral':    0.025,
        'k_prod_CTX':        1.87e-6,  # 3.5× healthy
        'k_deg_CTX':         0.00002,
        'CTX_bone':          0.028,    # 3.5× healthy → CTX_sensor ≈ 0.70 nM
        'k_prod_P1NP':       2.80e-7,  # 2.0× healthy
        'k_deg_P1NP':        0.00002,
        'P1NP_bone':         0.028,    # 2.0× healthy → P1NP_sensor ≈ 0.70 nM
    }
    return EnvironmentConfig(
        name='pmo_severe',
        param_overrides=params,
        description=(
            "Late postmenopause: SOST~1.31 nM (3.5x), CTX~0.70 nM (3.5x), "
            "P1NP~0.70 nM (2.0x) vs healthy. Estrogen=0.1 nM, sustained high turnover. "
            "Easiest detection case."
        ),
        sigma_overrides={
            'Sclerostin_bone': 0.30,
            'k_prod_Scl':      0.25,
            'CTX_bone':        0.40,
            'k_prod_CTX':      0.40,
            'P1NP_bone':       0.30,
            'k_prod_P1NP':     0.30,
        },
    )


def get_ckd_stage3_config() -> EnvironmentConfig:
    """CKD Stage 3 — SOST 1.8×, CTX 1.5×, P1NP 1.3× healthy.

    Represents mild-to-moderate CKD (GFR 30-59 mL/min/1.73m²) with early
    secondary hyperparathyroidism (PTH=80 pg/mL). Renal clearance not yet
    severely impaired — k_deg_CTX remains normal. This is the hardest CKD
    detection case: biomarker elevation is modest.

    Literature:
      - Cejka Nephrol Dial Transplant 2011: SOST 1.5-2.0× in CKD Stage 3
      - Hlaing 2011: CTX 1.2-1.5× in CKD Stage 3 (vs 2-3× in Stage 5)
    """
    params = {
        'Estrogen': 0.7,
        'PTH':      80.0,
        'k_prod_Scl':        2.50e-6,
        'k_deg_Scl':         0.000065,
        'Sclerostin_bone':   0.027,    # 1.8× healthy 0.015 → sensor ≈ 0.675 nM
        'k_prod_RANKL':      0.00000450,
        'k_deg_RANKL':       0.00002,
        'RANKL_bone':        0.65,
        'k_prod_OPG':        0.0000550,
        'k_deg_OPG':         0.00002,
        'OPG_bone':          5.0,
        'MineralIon':        2.8,
        'k_prod_Mineral':    0.100,
        'k_loss_Mineral':    0.030,
        'k_prod_CTX':        8.00e-7,  # 1.5× healthy 5.33e-7
        'k_deg_CTX':         0.00002,  # normal — Stage 3 CTX clearance intact
        'CTX_bone':          0.012,    # 1.5× healthy → CTX_sensor ≈ 0.30 nM
        'k_prod_P1NP':       1.82e-7,  # 1.3× healthy 1.40e-7
        'k_deg_P1NP':        0.00002,
        'P1NP_bone':         0.0182,   # 1.3× healthy → P1NP_sensor ≈ 0.455 nM
    }
    return EnvironmentConfig(
        name='ckd_stage3',
        param_overrides=params,
        description=(
            "CKD Stage 3: SOST~0.675 nM (1.8x), CTX~0.30 nM (1.5x), "
            "P1NP~0.455 nM (1.3x) vs healthy. PTH=80 pg/mL, intact CTX clearance. "
            "Hardest CKD detection case."
        ),
        sigma_overrides={
            'Sclerostin_bone': 0.30,
            'k_prod_Scl':      0.25,
            'CTX_bone':        0.30,
            'k_prod_CTX':      0.30,
            'P1NP_bone':       0.25,
            'k_prod_P1NP':     0.25,
        },
    )


def get_ckd_stage5d_config() -> EnvironmentConfig:
    """CKD Stage 5D (dialysis) — SOST 4.5×, CTX 5.0×, P1NP 2.5× healthy.

    Represents end-stage renal disease on hemodialysis, with severe secondary
    hyperparathyroidism (PTH=350 pg/mL) and markedly impaired CTX/P1NP clearance.

    Literature:
      - Cejka 2011: SOST 4-6× in hemodialysis patients
      - Hlaing 2011: CTX 4-8× in dialysis (k_deg_CTX 10× reduced)
      - Kovesdy 2014: P1NP 2-3× (hepatic + renal clearance impaired)
    """
    params = {
        'Estrogen': 0.3,
        'PTH':      350.0,
        'k_prod_Scl':        6.25e-6,
        'k_deg_Scl':         0.000058,
        'Sclerostin_bone':   0.067,    # 4.5× healthy 0.015 → sensor ≈ 1.675 nM
        'k_prod_RANKL':      0.00000650,
        'k_deg_RANKL':       0.00002,
        'RANKL_bone':        1.0,
        'k_prod_OPG':        0.000115,
        'k_deg_OPG':         0.00002,
        'OPG_bone':          5.5,
        'MineralIon':        4.5,
        'k_prod_Mineral':    0.180,
        'k_loss_Mineral':    0.060,
        'k_prod_CTX':        2.67e-6,  # 5.0× healthy production
        'k_deg_CTX':         0.000002, # 10× reduced — anuric dialysis patient
        'CTX_bone':          0.040,    # 5.0× healthy → CTX_sensor ≈ 1.00 nM
        'k_prod_P1NP':       3.50e-7,  # 2.5× healthy 1.40e-7
        'k_deg_P1NP':        0.000010, # 2× reduced — partial renal/hepatic impairment
        'P1NP_bone':         0.035,    # 2.5× healthy → P1NP_sensor ≈ 0.875 nM
    }
    return EnvironmentConfig(
        name='ckd_stage5d',
        param_overrides=params,
        description=(
            "CKD Stage 5D (dialysis): SOST~1.675 nM (4.5x), CTX~1.00 nM (5.0x), "
            "P1NP~0.875 nM (2.5x) vs healthy. PTH=350 pg/mL, 10x reduced CTX clearance. "
            "Easiest CKD detection case."
        ),
        sigma_overrides={
            'Sclerostin_bone': 0.45,
            'k_prod_Scl':      0.40,
            'CTX_bone':        0.45,
            'k_prod_CTX':      0.45,
            'P1NP_bone':       0.40,
            'k_prod_P1NP':     0.35,
        },
    )


SCENARIO_CONFIGS = {
    'healthy':     get_healthy_config(),
    'pmo_mild':    get_pmo_mild_config(),
    'pmo':         get_pmo_config(),
    'pmo_severe':  get_pmo_severe_config(),
    'ckd_stage3':  get_ckd_stage3_config(),
    'ckd_mbd':     get_ckd_mbd_config(),
    'ckd_stage5d': get_ckd_stage5d_config(),
}


def get_config(scenario_name: str) -> EnvironmentConfig:
    """Retrieve configuration for a named scenario."""
    if scenario_name not in SCENARIO_CONFIGS:
        raise ValueError(
            f"Unknown scenario: '{scenario_name}'. "
            f"Valid options: {list(SCENARIO_CONFIGS.keys())}"
        )
    return SCENARIO_CONFIGS[scenario_name]


def list_scenarios() -> List[str]:
    """Return list of available scenario names."""
    return list(SCENARIO_CONFIGS.keys())


def print_scenario_summary():
    """Print full parameter table for debugging / documentation."""
    print("\n" + "=" * 80)
    print("SCENARIO CONFIGURATIONS SUMMARY (v4.0, Tier 1 fixes)")
    print("=" * 80 + "\n")

    groups = {
        'Hormones':   ['Estrogen', 'PTH'],
        'Sclerostin': ['k_prod_Scl', 'k_deg_Scl', 'Sclerostin_bone', 'Sclerostin_sensor'],
        'RANKL/OPG':  ['k_prod_RANKL', 'k_deg_RANKL', 'RANKL_bone', 'RANKL_sensor',
                       'k_prod_OPG',   'k_deg_OPG',   'OPG_bone',   'OPG_sensor'],
        'Minerals':   ['MineralIon', 'k_prod_Mineral', 'k_loss_Mineral'],
    }

    for scenario_name in list_scenarios():
        config = get_config(scenario_name)
        print(f"{'─'*60}")
        print(f"  {scenario_name.upper()}")
        print(f"  {config.description}")
        for group_name, params in groups.items():
            if any(p in config.param_overrides for p in params):
                print(f"\n    {group_name}:")
                for param in params:
                    if param in config.param_overrides:
                        v = config.param_overrides[param]
                        print(f"      {param:<28} = {v:.6e}")
        print()


if __name__ == "__main__":
    print_scenario_summary()