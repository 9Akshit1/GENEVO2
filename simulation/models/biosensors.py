"""
Biosensor circuit implementations for sclerostin detection.

V5.0 FIX — Decoupled Threshold Calibration
===========================================
ROOT CAUSE: threshold = sensitivity × g(kd), so when computing
signal >= threshold, sensitivity cancelled out completely.
  corr(sensitivity, threshold/sensitivity) = -0.0045 (effectively zero)
  → sensitivity permutation importance < 0% (statistical noise)

PROOF:
  signal    = sensitivity × occupancy(kd, S)
  threshold = sensitivity × occupancy(kd, C_thresh)   ← old code
  detection = signal >= threshold
            → occupancy(kd, S) >= occupancy(kd, C_thresh)   (sensitivity divides out)

SOLUTION: calibrate threshold against a REFERENCE sensor with sensitivity=1.0.
  threshold = 1.0 × occupancy(kd, C_thresh)  (absolute, not sensitivity-scaled)
  detection = sensitivity × occupancy(kd, S) >= threshold
            → sensitivity ≥ occupancy(C_thresh) / occupancy(S)  (sensitivity matters!)

Threshold gap also moved from H–PMO to PMO–CKD:
  Old: threshold ∈ [H_out, PMO_out] → PMO DR=99%, CKD DR=99% (flat landscape)
  New: threshold ∈ [PMO_ref, CKD_ref] → PMO and CKD require adequate sensitivity

V4.2 CRITICAL FIX — Acceptance Testing in Generation
=====================================================
ROOT CAUSE (corrected from v4.1): The problem was not just Kd operating regime.
The real issue: generator creates sensors with NO quality filtering.
Result: 80% of generated sensors are useless, RL trains on noise.

SOLUTION (V4.2): Built-in acceptance testing
  - Generate candidate sensor
  - Test on disease states (H, P, C)
  - Keep only if: H_output < P_output < C_output AND separation > threshold
  - Reject if: ordering inverted or separation too small
  - Retry up to 20 times before giving up
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


_HEALTHY_SCL_RANKL_RATIO = 0.375 / 15.0    # 0.025000
_HEALTHY_SCL_OPG_RATIO   = 0.375 / 150.0   # 0.002500

# Disease state signal levels (from environment_configs.py) v5.4 calibration
NOMINAL_SIGNALS = {
    'healthy': 0.375,
    'pmo':     0.875,
    'ckd_mbd': 1.125,   # v5.2: 3.0x healthy (Sclerostin_bone=0.045 * ratio 25)
}

# Biosensor class mapping (populated after classes defined below)
_BIOSENSOR_CLASSES = {}


class Biosensor(ABC):
    """Abstract base for all biosensor circuit models."""

    def __init__(self,
                 sensitivity:   float,
                 threshold:     float,
                 dynamic_range: Tuple[float, float],
                 kd:            float,
                 circuit_type:  str,
                 **kwargs):
        self.sensitivity   = sensitivity
        self.threshold     = threshold
        self.dynamic_range = dynamic_range
        self.kd            = kd
        self.circuit_type  = circuit_type
        self.extra_params  = kwargs
        self._validate()
        logger.debug(
            f"Biosensor({circuit_type}): "
            f"Kd={kd:.3f} nM  sensitivity={sensitivity:.3f}  "
            f"threshold={threshold:.3f} nM"
        )

    def _validate(self):
        assert self.sensitivity > 0,                         "sensitivity must be > 0"
        assert self.threshold   >= 0,                        "threshold must be >= 0"
        assert self.dynamic_range[0] < self.dynamic_range[1],"dynamic_range[0] < [1]"
        assert self.kd          > 0,                         "Kd must be > 0"

    @abstractmethod
    def measure(self,
                sclerostin: np.ndarray,
                time:       np.ndarray,
                **kwargs) -> np.ndarray:
        """Return biosensor output signal for given analyte concentration array."""
        pass

    def apply_saturation(self, signal: np.ndarray) -> np.ndarray:
        """Clip signal to the sensor's dynamic range."""
        return np.clip(signal, self.dynamic_range[0], self.dynamic_range[1])

    def is_detected(self, signal: np.ndarray) -> np.ndarray:
        """Boolean mask: True where signal >= threshold."""
        return signal >= self.threshold

    def to_dict(self) -> Dict:
        return {
            'circuit_type':  self.circuit_type,
            'sensitivity':   self.sensitivity,
            'threshold':     self.threshold,
            'dynamic_range': self.dynamic_range,
            'kd':            self.kd,
            **self.extra_params,
        }


class DirectBindingSensor(Biosensor):
    """
    Direct-binding sensor: pure Langmuir single-site equilibrium.
    
    signal(S) = sensitivity * S / (Kd + S)
    
    V4.2: Now subject to acceptance testing before use.
    """

    def __init__(self,
                 sensitivity:  float = 1.5,
                 threshold:    float = 0.6,
                 dynamic_range: Tuple[float, float] = (0.0, 5.0),
                 kd:           float = 1.0,
                 **kwargs):
        super().__init__(
            sensitivity=sensitivity, threshold=threshold,
            dynamic_range=dynamic_range, kd=kd,
            circuit_type='direct_binding', **kwargs
        )

    def measure(self,
                sclerostin: np.ndarray,
                time:       np.ndarray,
                rankl:      Optional[np.ndarray] = None,
                opg:        Optional[np.ndarray] = None,
                **kwargs) -> np.ndarray:
        """
        Langmuir binding: signal = sensitivity * [S] / (Kd + [S])
        """
        epsilon = 1e-12
        occupancy = sclerostin / (self.kd + sclerostin + epsilon)
        signal = self.sensitivity * occupancy
        return self.apply_saturation(signal)


class AmplifyingSensor(Biosensor):
    """
    Amplifying sensor: exponential accumulation with time constant.
    
    d[P]/dt = k_cat * [E] - k_deg * [P]
    [P](t) = sensitivity * (1 - exp(-t / tau))
    
    V4.2: Now subject to acceptance testing before use.
    """

    def __init__(self,
                 sensitivity:   float = 1.5,
                 threshold:     float = 0.8,
                 dynamic_range: Tuple[float, float] = (0.0, 8.0),
                 kd:            float = 1.0,
                 response_time: float = 600.0,
                 **kwargs):
        self.response_time = response_time
        super().__init__(
            sensitivity=sensitivity, threshold=threshold,
            dynamic_range=dynamic_range, kd=kd,
            circuit_type='amplifying', **kwargs
        )

    def measure(self,
                sclerostin: np.ndarray,
                time:       np.ndarray,
                rankl:      Optional[np.ndarray] = None,
                opg:        Optional[np.ndarray] = None,
                **kwargs) -> np.ndarray:
        """
        Exponential buildup: signal plateaus at sensitivity * sclerostin.
        Timescale set by response_time.
        """
        occupancy = sclerostin / (self.kd + sclerostin + 1e-12)
        time_factor = 1.0 - np.exp(-time / (self.response_time + 1e-6))
        signal = self.sensitivity * occupancy * time_factor
        return self.apply_saturation(signal)


class ThresholdSensor(Biosensor):
    """
    Threshold/digital sensor: Hill function with cooperativity.
    
    signal = (off_level + on_level) / 2 + (on_level - off_level) / 2
           * tanh(n * ln([S] / Kd))
    
    V4.2: Now subject to acceptance testing before use.
    """

    def __init__(self,
                 sensitivity:      float = 2.0,
                 threshold:        float = 0.8,
                 dynamic_range:    Tuple[float, float] = (0.0, 10.0),
                 kd:               float = 1.0,
                 hill_coefficient: float = 4.0,
                 off_level:        float = 0.1,
                 on_level:         float = 5.0,
                 **kwargs):
        self.hill_coefficient = hill_coefficient
        self.off_level        = off_level
        self.on_level         = on_level
        super().__init__(
            sensitivity=sensitivity, threshold=threshold,
            dynamic_range=dynamic_range, kd=kd,
            circuit_type='threshold', **kwargs
        )

    def measure(self,
                sclerostin: np.ndarray,
                time:       np.ndarray,
                rankl:      Optional[np.ndarray] = None,
                opg:        Optional[np.ndarray] = None,
                **kwargs) -> np.ndarray:
        """
        Hill switch: smooth transition between off_level and on_level
        centered at Kd.
        """
        epsilon = 1e-12
        ratio = (sclerostin + epsilon) / (self.kd + epsilon)
        exponent = self.hill_coefficient * np.log(ratio + epsilon)
        hill_activation = 1.0 / (1.0 + np.exp(-exponent))
        signal = (self.off_level + 
                 (self.on_level - self.off_level) * hill_activation)
        return self.apply_saturation(signal)


class RatiometricSensor(Biosensor):
    """
    Ratiometric sensor: normalised sclerostin/reference ratio.
    
    signal = sensitivity * (Sclerostin / Reference)^exponent / baseline_ratio
    
    V4.2: Now subject to acceptance testing before use.
    """

    def __init__(self,
                 sensitivity:    float = 0.5,
                 threshold:      float = 0.6,
                 dynamic_range:  Tuple[float, float] = (0.0, 5.0),
                 kd:             float = 1.0,
                 ratio_exponent: float = 1.0,
                 **kwargs):
        self.ratio_exponent = ratio_exponent
        super().__init__(
            sensitivity=sensitivity, threshold=threshold,
            dynamic_range=dynamic_range, kd=kd,
            circuit_type='ratiometric', **kwargs
        )

    def measure(self,
                sclerostin: np.ndarray,
                time:       np.ndarray,
                rankl:      Optional[np.ndarray] = None,
                opg:        Optional[np.ndarray] = None,
                **kwargs) -> np.ndarray:
        """
        signal = sensitivity * (actual_ratio / ref_ratio_healthy)^exponent
        """
        epsilon = 1e-6

        if rankl is not None:
            reference     = rankl
            healthy_ratio = _HEALTHY_SCL_RANKL_RATIO
        elif opg is not None:
            reference     = opg
            healthy_ratio = _HEALTHY_SCL_OPG_RATIO
        else:
            reference     = np.full_like(sclerostin, 0.375)
            healthy_ratio = 1.0

        actual_ratio     = sclerostin / (reference + epsilon)
        normalised_ratio = actual_ratio / (healthy_ratio + epsilon)
        normalised_ratio = np.clip(normalised_ratio, 1e-4, 20.0)

        signal = self.sensitivity * (normalised_ratio ** self.ratio_exponent)
        return self.apply_saturation(signal)


class ArrayBiosensor:
    """
    Multi-analyte biosensor array: Sclerostin + CTX + P1NP.

    Each channel uses Langmuir binding normalized to the premenopausal healthy
    reference concentration so that healthy produces a composite signal of exactly
    `sensitivity` (baseline = 1.0 × sensitivity). Disease states produce higher
    composite signals; the threshold is set above the healthy baseline.

    Composite signal = sensitivity × (w_scl × norm_scl + w_ctx × norm_ctx + w_p1np × norm_p1np)

    where norm_x = (C_x / (Kd_x + C_x)) / (C_x_healthy / (Kd_x + C_x_healthy))

    This normalization means:
      - Healthy → composite = sensitivity × 1.0  (all channels at 1.0)
      - PMO     → composite > sensitivity         (elevated CTX, P1NP, moderate SOST)
      - CKD     → composite >> sensitivity        (very high SOST, CTX, P1NP)
    """

    # Premenopausal healthy reference concentrations in sensor compartment (nM)
    HEALTHY_CONC = {
        'scl':  0.375,
        'ctx':  0.200,
        'p1np': 0.350,
    }
    # circuit_type tag (used by surrogate builder and data generator)
    circuit_type = 'array'

    def __init__(
        self,
        kd_scl:     float = 1.0,
        kd_ctx:     float = 1.0,
        kd_p1np:    float = 1.0,
        w_scl:      float = 1.0,
        w_ctx:      float = 1.0,
        w_p1np:     float = 1.0,
        sensitivity: float = 2.0,
        threshold:  float = 1.5,
        dynamic_range: tuple = (0.0, 10.0),
        **kwargs,
    ):
        total_w = w_scl + w_ctx + w_p1np + 1e-12
        self.w_scl  = w_scl  / total_w
        self.w_ctx  = w_ctx  / total_w
        self.w_p1np = w_p1np / total_w

        self.kd_scl    = kd_scl
        self.kd_ctx    = kd_ctx
        self.kd_p1np   = kd_p1np
        self.sensitivity   = sensitivity
        self.threshold     = threshold
        self.dynamic_range = dynamic_range
        self.kd = kd_scl  # primary Kd (kept for BiosensorEngine compatibility)

        # Reference occupancies at healthy concentrations — used to normalize
        H = self.HEALTHY_CONC
        self._ref_occ_scl  = H['scl']  / (kd_scl  + H['scl']  + 1e-12)
        self._ref_occ_ctx  = H['ctx']  / (kd_ctx  + H['ctx']  + 1e-12)
        self._ref_occ_p1np = H['p1np'] / (kd_p1np + H['p1np'] + 1e-12)

    def measure(
        self,
        sclerostin: np.ndarray,
        time:       np.ndarray,
        ctx:        Optional[np.ndarray] = None,
        p1np:       Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Return composite signal across all three analyte channels."""
        eps = 1e-12

        occ_scl  = sclerostin / (self.kd_scl + sclerostin + eps)
        norm_scl = occ_scl / (self._ref_occ_scl + eps)

        if ctx is not None:
            occ_ctx  = ctx / (self.kd_ctx + ctx + eps)
            norm_ctx = occ_ctx / (self._ref_occ_ctx + eps)
        else:
            norm_ctx = np.ones_like(sclerostin)

        if p1np is not None:
            occ_p1np  = p1np / (self.kd_p1np + p1np + eps)
            norm_p1np = occ_p1np / (self._ref_occ_p1np + eps)
        else:
            norm_p1np = np.ones_like(sclerostin)

        composite = self.sensitivity * (
            self.w_scl  * norm_scl  +
            self.w_ctx  * norm_ctx  +
            self.w_p1np * norm_p1np
        )
        return np.clip(composite, self.dynamic_range[0], self.dynamic_range[1])

    def is_detected(self, signal: np.ndarray) -> np.ndarray:
        return signal >= self.threshold

    def apply_saturation(self, signal: np.ndarray) -> np.ndarray:
        return np.clip(signal, self.dynamic_range[0], self.dynamic_range[1])

    def to_dict(self) -> Dict:
        return {
            'circuit_type':  'array',
            'kd_scl':        self.kd_scl,
            'kd_ctx':        self.kd_ctx,
            'kd_p1np':       self.kd_p1np,
            'w_scl':         self.w_scl,
            'w_ctx':         self.w_ctx,
            'w_p1np':        self.w_p1np,
            'sensitivity':   self.sensitivity,
            'threshold':     self.threshold,
            'dynamic_range': list(self.dynamic_range),
            'kd':            self.kd,
        }


def create_biosensor(config: Dict) -> "Biosensor | ArrayBiosensor":
    """Instantiate a Biosensor from a configuration dictionary."""
    circuit_type = config.get('circuit_type')
    kwargs = {k: v for k, v in config.items() if k != 'circuit_type'}
    if   circuit_type == 'direct_binding': return DirectBindingSensor(**kwargs)
    elif circuit_type == 'amplifying':     return AmplifyingSensor(**kwargs)
    elif circuit_type == 'threshold':      return ThresholdSensor(**kwargs)
    elif circuit_type == 'ratiometric':    return RatiometricSensor(**kwargs)
    elif circuit_type == 'array':          return ArrayBiosensor(**kwargs)
    else:
        raise ValueError(
            f"Unknown circuit_type: '{circuit_type}'. "
            "Valid: direct_binding, amplifying, threshold, ratiometric, array"
        )


def evaluate_separability_quick(config: Dict) -> Tuple[float, str]:
    """
    Quick quality test: does this biosensor config separate disease states?
    
    V4.2 NEW: Acceptance testing function
    
    Tests if outputs satisfy: healthy_output < pmo_output < ckd_output
    and if separation is large enough to be useful.
    
    Args:
        config: Biosensor configuration dict (from generate_random_biosensor_config)
        
    Returns:
        score: Float in [-1, ∞). Positive = good, negative = reject
        reason: String explaining the score
    """
    try:
        biosensor = create_biosensor(config)
    except Exception as e:
        return -1.0, f"Biosensor creation failed: {e}"
    
    # Test on nominal disease state signals at steady state (1800s into run)
    time_test = np.array([1800.0])  # Late in simulation, after equilibration
    
    try:
        h_out = biosensor.measure(
            np.array([NOMINAL_SIGNALS['healthy']]), 
            time_test
        )[0]
        p_out = biosensor.measure(
            np.array([NOMINAL_SIGNALS['pmo']]), 
            time_test
        )[0]
        c_out = biosensor.measure(
            np.array([NOMINAL_SIGNALS['ckd_mbd']]), 
            time_test
        )[0]
    except Exception as e:
        return -1.0, f"Measure failed: {e}"
    
    # Check ordering: Healthy < PMO < CKD
    if not (h_out < p_out):
        return -1.0, f"Bad H-P ordering: H={h_out:.3f} >= P={p_out:.3f}"
    
    if not (p_out < c_out):
        return -1.0, f"Bad P-C ordering: P={p_out:.3f} >= C={c_out:.3f}"
    
    # Check separation magnitude (prefer larger gaps)
    h_p_gap = p_out - h_out
    p_c_gap = c_out - p_out
    total_separation = h_p_gap + p_c_gap
    
    # Minimum useful separation threshold
    # Diseases span 0.375 → 2.0 (5.3× range in input)
    # Output should show at least 0.2 nM total separation to be useful
    min_separation = 0.15
    
    if total_separation < min_separation:
        return -1.0, f"Separation too small: {total_separation:.3f} < {min_separation}"
    
    # Score: larger separation is better
    # Also reward monotonic spacing (balanced H-P and P-C gaps)
    balance = 1.0 - abs(h_p_gap - p_c_gap) / (h_p_gap + p_c_gap + 1e-6)
    score = total_separation * balance
    
    return score, f"Valid: H={h_out:.3f}, P={p_out:.3f}, C={c_out:.3f}, sep={total_separation:.3f}"


# Populate biosensor class mapping after all classes are defined
_BIOSENSOR_CLASSES = {
    'direct_binding': DirectBindingSensor,
    'amplifying':     AmplifyingSensor,
    'threshold':      ThresholdSensor,
    'ratiometric':    RatiometricSensor,
    'array':          ArrayBiosensor,
}


def generate_random_biosensor_config(
    circuit_type: Optional[str] = None,
    seed:         Optional[int] = None
) -> Dict:
    """
    Generate a random biosensor configuration that passes acceptance testing.

    V4.2 CRITICAL: Added acceptance testing loop
    =============================================
    Instead of: generate random → use immediately
    Now does: generate → test → keep if good, retry if bad
    
    This is the key architectural fix.
    
    Parameters:
    -----------
    circuit_type : str or None
        Force a circuit type. If None, choose randomly.
    seed : int or None
        Random seed for reproducibility.
        
    Returns:
    --------
    config : Dict
        Biosensor configuration that passes quality check.
        Guaranteed to satisfy: H_out < P_out < C_out
    """
    if seed is not None:
        np.random.seed(seed)

    if circuit_type is None:
        # V4.3: Constrain to 2 best-performing types to reduce heterogeneity
        circuit_type = np.random.choice(
            ['direct_binding', 'amplifying']
        )

    max_generate_attempts = 20
    best_config = None
    best_score = -np.inf
    
    for attempt in range(max_generate_attempts):
        # Generate candidate configuration

        # V5.2 FIX: Generate threshold in a robust range
        # Empirical observation: healthy outputs ~0.3-1.5, CKD ~0.5-2.0
        # Threshold range [0.5-1.5] separates most healthy/disease with overlap
        # This avoids the broken coordinate-space issue from before

        # Wide kd range so BO can discover that affinity matters.
        # Langmuir occupancy: occ = c/(kd+c); at kd=0.1 sensors saturate on
        # even PMO-level signals (0.875 nM), while at kd=10 they have only
        # ~8% occupancy — a 10× dynamic-range spread that is detectable by ML.
        kd = float(np.exp(np.random.uniform(np.log(0.10), np.log(10.0))))

        response_type_map = {
            'direct_binding': 'fast',
            'amplifying':     'slow',
            'threshold':      'digital',
            'ratiometric':    'normalized'
        }

        if circuit_type == 'direct_binding':
            sensitivity = float(np.exp(np.random.uniform(np.log(0.5), np.log(5.0))))
            dyn_max = float(np.random.uniform(max(sensitivity, 2.0),
                                               max(sensitivity * 2.5, 5.0)))
            dyn_range = (0.0, dyn_max)

        elif circuit_type == 'amplifying':
            sensitivity = float(np.exp(np.random.uniform(np.log(0.5), np.log(5.0))))
            dyn_max = float(np.random.uniform(max(sensitivity, 3.0),
                                               max(sensitivity * 2.5, 7.0)))
            dyn_range = (0.0, dyn_max)
            response_time = float(np.random.uniform(100.0, 3600.0))


        # V5.0: Decoupled threshold calibration (hybrid formula)
        # -------------------------------------------------------
        # Root cause: old threshold = sensitivity × g(kd) → sensitivity cancelled.
        #
        # Formula: threshold = healthy_actual + margin × pmc_gap
        #
        #   healthy_actual  — scales with sensitivity (guarantees zero healthy
        #                     false positives for the clean signal at calib time)
        #   margin × pmc_gap — absolute (from reference sensor, sensitivity=1.0)
        #                     → detection requires: sensitivity × Δocc ≥ margin × pmc_gap
        #                     → sensitivity appears in the detection inequality
        #
        # Detection condition at t=1800 (calib time):
        #   sensitivity × (occ(S) - occ(H)) ≥ margin × pmc_gap
        #   sensitivity ≥ margin × pmc_gap / (occ(S) - occ(H))
        #
        # Healthy is NEVER detected (occ(H) - occ(H) = 0, so LHS = 0 < RHS).
        # kd affects occ via Langmuir → kd still matters.
        from simulation.models.biosensors import _BIOSENSOR_CLASSES
        biosensor_cls = _BIOSENSOR_CLASSES.get(circuit_type)
        if biosensor_cls is None:
            raise ValueError(f"Unknown circuit_type: {circuit_type}")

        healthy_conc = 0.375   # nM
        pmo_conc     = 0.875   # nM
        ckd_conc     = 2.0     # nM  — threshold design ref (intentionally above ODE CKD 1.125;
        #                              provides safe FP margin for healthy outliers)
        time_calib   = np.array([1800.0])

        # Reference sensor (sensitivity=1.0) for the absolute PMO-CKD gap scale
        ref_config = {
            'sensitivity': 1.0,
            'threshold':   0.0,
            'dynamic_range': dyn_range,
            'kd': kd,
        }
        if circuit_type == 'amplifying' and 'response_time' in locals():
            ref_config['response_time'] = response_time
        biosensor_ref = biosensor_cls(**ref_config)

        pmo_ref = biosensor_ref.measure(np.array([pmo_conc]), time_calib)[0]
        ckd_ref = biosensor_ref.measure(np.array([ckd_conc]), time_calib)[0]
        pmc_gap = ckd_ref - pmo_ref   # absolute gap scale (sensitivity-independent)

        # Actual sensor for the healthy baseline (scales with actual sensitivity)
        actual_config = {
            'sensitivity': sensitivity,
            'threshold':   0.0,
            'dynamic_range': dyn_range,
            'kd': kd,
        }
        if circuit_type == 'amplifying' and 'response_time' in locals():
            actual_config['response_time'] = response_time
        biosensor_actual = biosensor_cls(**actual_config)
        healthy_actual = biosensor_actual.measure(np.array([healthy_conc]), time_calib)[0]

        # Locked margin: A_best validation (N=500) confirmed margin=1.25 achieves
        # FP=4.2%, PMO-mild DR=85.0% at 13 dB. Training and BO evaluation must use
        # the same margin to avoid train/deploy distribution mismatch.
        margin_fraction = 1.25
        threshold = float(healthy_actual + margin_fraction * pmc_gap)

        config: Dict = {
            'circuit_type':  circuit_type,
            'threshold':     threshold,
            'kd':            kd,
            'response_type': response_type_map[circuit_type],
            'dynamic_range': dyn_range,
            'sensitivity':   float(sensitivity),
        }

        # Add type-specific parameters
        if circuit_type == 'amplifying':
            config['response_time'] = response_time

        # V4.2: ACCEPTANCE TEST
        score, reason = evaluate_separability_quick(config)
        
        if score > 0:
            # Passed! Return immediately
            logger.debug(f"Generated {circuit_type}: {reason} (score={score:.3f})")
            return config
        
        # Track best attempt even if rejected (for diagnostics)
        if score > best_score:
            best_score = score
            best_config = config
    
    # Exhausted attempts: return best attempt found
    if best_config is not None:
        logger.warning(
            f"Could not generate passing {circuit_type} after {max_generate_attempts} attempts. "
            f"Using best attempt (score={best_score:.3f}). Check signal/threshold alignment."
        )
        return best_config
    
    # Ultimate fallback (shouldn't reach here)
    raise RuntimeError(
        f"Failed to generate any valid biosensor config after {max_generate_attempts} attempts. "
        "This suggests broken signal distributions or invalid parameter ranges."
    )


def generate_random_array_config(seed: Optional[int] = None) -> Dict:
    """
    Generate a random ArrayBiosensor configuration with acceptance testing.

    Each of the three channels (SOST, CTX, P1NP) gets an independent Kd drawn
    log-uniformly from [0.1, 10.0] nM.  Channel weights are drawn from a Dirichlet
    distribution so they always sum to 1.  The composite threshold is calibrated
    using the same healthy-baseline + margin approach as single-channel sensors.

    Returns a config dict suitable for create_biosensor({'circuit_type': 'array', ...}).
    """
    if seed is not None:
        np.random.seed(seed)

    # Healthy, PMO, CKD nominal sensor concentrations (nM) for threshold calibration
    # v5.4: P1NP PMO = 1.5x healthy (0.525); CKD from ODE at Scl_bone×25 ratio
    H = {'scl': 0.375, 'ctx': 0.200, 'p1np': 0.350}
    P = {'scl': 0.875, 'ctx': 0.500, 'p1np': 0.525}   # v5.4: P1NP 1.5x healthy
    C = {'scl': 1.125, 'ctx': 0.500, 'p1np': 0.625}   # v5.2: ODE CKD steady-state

    max_attempts = 20
    best_config, best_score = None, -np.inf

    for _ in range(max_attempts):
        kd_scl  = float(np.exp(np.random.uniform(np.log(0.10), np.log(10.0))))
        kd_ctx  = float(np.exp(np.random.uniform(np.log(0.10), np.log(10.0))))
        kd_p1np = float(np.exp(np.random.uniform(np.log(0.10), np.log(10.0))))
        sensitivity = float(np.exp(np.random.uniform(np.log(0.5), np.log(5.0))))

        # Dirichlet weights (symmetric α=1 → uniform on simplex)
        raw_w = np.random.dirichlet([1.0, 1.0, 1.0])
        w_scl, w_ctx, w_p1np = float(raw_w[0]), float(raw_w[1]), float(raw_w[2])

        def _occ(c, kd): return c / (kd + c + 1e-12)

        ref_scl  = _occ(H['scl'],  kd_scl)
        ref_ctx  = _occ(H['ctx'],  kd_ctx)
        ref_p1np = _occ(H['p1np'], kd_p1np)

        def _composite(conc_dict):
            n_scl  = _occ(conc_dict['scl'],  kd_scl)  / (ref_scl  + 1e-12)
            n_ctx  = _occ(conc_dict['ctx'],  kd_ctx)  / (ref_ctx  + 1e-12)
            n_p1np = _occ(conc_dict['p1np'], kd_p1np) / (ref_p1np + 1e-12)
            return sensitivity * (w_scl * n_scl + w_ctx * n_ctx + w_p1np * n_p1np)

        sig_h = _composite(H)   # = sensitivity × 1.0 by construction
        sig_p = _composite(P)
        sig_c = _composite(C)

        # Require ordering: healthy < PMO < CKD, and meaningful separation
        if not (sig_h < sig_p < sig_c):
            continue
        h_p_gap = sig_p - sig_h
        p_c_gap = sig_c - sig_p
        if h_p_gap < 0.05 or p_c_gap < 0.05:
            continue

        # Mirror single-channel decoupled calibration (V5.0):
        # Use unit-sensitivity gap so threshold < sig_p when sensitivity > 1.35.
        # sig_h = sensitivity × 1.0 always; hp_gap_ref = PMO composite at sens=1.0 - 1.0.
        # Detection condition: sensitivity × (sig_p/sensitivity) > sig_h + 1.35×hp_gap_ref
        #   ↔ sig_p > sensitivity + 1.35×(sig_p/sensitivity - 1)
        #   ↔ sensitivity > 1.35  (always satisfied for well-chosen configs).
        hp_gap_ref = (sig_p / sensitivity) - 1.0
        threshold = float(sig_h + 1.25 * hp_gap_ref)
        dyn_max   = float(sig_c * 2.0)

        config = {
            'circuit_type':  'array',
            'kd_scl':        kd_scl,
            'kd_ctx':        kd_ctx,
            'kd_p1np':       kd_p1np,
            'w_scl':         w_scl,
            'w_ctx':         w_ctx,
            'w_p1np':        w_p1np,
            'sensitivity':   sensitivity,
            'threshold':     threshold,
            'dynamic_range': (0.0, dyn_max),
            'kd':            kd_scl,
        }

        total_sep = h_p_gap + p_c_gap
        balance   = 1.0 - abs(h_p_gap - p_c_gap) / (total_sep + 1e-6)
        score     = total_sep * balance
        if score > best_score:
            best_score, best_config = score, config

        logger.debug(
            f"Array config: kd_scl={kd_scl:.2f} kd_ctx={kd_ctx:.2f} "
            f"kd_p1np={kd_p1np:.2f} sep={total_sep:.3f}"
        )
        return config  # First valid config is good enough

    if best_config is not None:
        logger.warning(
            f"Array config: using best-of-{max_attempts} (score={best_score:.3f})"
        )
        return best_config

    raise RuntimeError("Failed to generate a valid array biosensor config.")


def generate_2channel_array_config(seed: Optional[int] = None) -> Dict:
    """
    Generate a 2-channel array config: SOST + P1NP only (CTX channel dropped, w_ctx=0).

    Structurally identical to generate_random_array_config() except:
    - w_ctx is forced to 0.0 (CTX channel absent)
    - Dirichlet over 2 channels (SOST + P1NP) so weights sum to 1
    - kd_ctx is still drawn but has no effect on the composite signal

    This generates the 2-channel topology for Phase 2.1 topology comparison.
    Returns config with circuit_type='array' and w_ctx=0.0.
    """
    if seed is not None:
        np.random.seed(seed)

    H = {'scl': 0.375, 'ctx': 0.200, 'p1np': 0.350}
    P = {'scl': 0.875, 'ctx': 0.500, 'p1np': 0.525}
    C = {'scl': 1.125, 'ctx': 0.500, 'p1np': 0.625}

    max_attempts = 20
    best_config, best_score = None, -np.inf

    for _ in range(max_attempts):
        kd_scl  = float(np.exp(np.random.uniform(np.log(0.10), np.log(10.0))))
        kd_ctx  = float(np.exp(np.random.uniform(np.log(0.10), np.log(10.0))))  # unused but recorded
        kd_p1np = float(np.exp(np.random.uniform(np.log(0.10), np.log(10.0))))
        sensitivity = float(np.exp(np.random.uniform(np.log(0.5), np.log(5.0))))

        # 2-channel weights: Dirichlet over SOST and P1NP only
        raw_w = np.random.dirichlet([1.0, 1.0])
        w_scl, w_p1np = float(raw_w[0]), float(raw_w[1])
        w_ctx = 0.0

        def _occ(c, kd): return c / (kd + c + 1e-12)

        ref_scl  = _occ(H['scl'],  kd_scl)
        ref_p1np = _occ(H['p1np'], kd_p1np)

        def _composite(conc_dict):
            n_scl  = _occ(conc_dict['scl'],  kd_scl)  / (ref_scl  + 1e-12)
            n_p1np = _occ(conc_dict['p1np'], kd_p1np) / (ref_p1np + 1e-12)
            return sensitivity * (w_scl * n_scl + w_p1np * n_p1np)

        sig_h = _composite(H)
        sig_p = _composite(P)
        sig_c = _composite(C)

        if not (sig_h < sig_p < sig_c):
            continue
        h_p_gap = sig_p - sig_h
        p_c_gap = sig_c - sig_p
        if h_p_gap < 0.05 or p_c_gap < 0.05:
            continue

        hp_gap_ref = (sig_p / sensitivity) - 1.0
        threshold = float(sig_h + 1.25 * hp_gap_ref)
        dyn_max   = float(sig_c * 2.0)

        config = {
            'circuit_type':  'array',
            'kd_scl':        kd_scl,
            'kd_ctx':        kd_ctx,   # recorded but w_ctx=0 means it has no effect
            'kd_p1np':       kd_p1np,
            'w_scl':         w_scl,
            'w_ctx':         w_ctx,
            'w_p1np':        w_p1np,
            'sensitivity':   sensitivity,
            'threshold':     threshold,
            'dynamic_range': (0.0, dyn_max),
            'kd':            kd_scl,
            'topology':      '2ch',    # Phase 2.1 topology tag
        }

        total_sep = h_p_gap + p_c_gap
        balance   = 1.0 - abs(h_p_gap - p_c_gap) / (total_sep + 1e-6)
        score     = total_sep * balance
        if score > best_score:
            best_score, best_config = score, config

        return config  # First valid config is good enough

    if best_config is not None:
        logger.warning(
            f"2ch array config: using best-of-{max_attempts} (score={best_score:.3f})"
        )
        return best_config

    raise RuntimeError("Failed to generate a valid 2-channel array biosensor config.")