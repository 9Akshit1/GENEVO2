"""
Biosensor engine: applies biosensor models to ODE simulation output.

V3.1 CORRECTIONS (Deterministic FNR and TTD)
===========================================
1. _calculate_ttd(): Now uses DETERMINISTIC SNR-dependent delays instead of random sampling.
   Delay values: 25s (SNR>25dB), 75s (SNR>15dB), 175s (SNR>5dB), 400s (SNR≤5dB).
   This ensures reproducibility and physical meaning.

2. _calculate_fnr(): Completely redesigned for DETERMINISTIC calculation and correct semantics.
   OLD (v3.0): Stochastically sampled beta distributions -> non-reproducible FNR
   NEW (v3.1): Deterministic margin-based FNR + semantic fix:
     - If detected (≥5 consecutive points above threshold): FNR = 0.0
     - If not detected: FNR = estimated P(should have been detected)
   This correctly distinguishes true positives (FNR=0) from false negatives.

V3.0 CORRECTIONS (from previous version)
=========================================
1. _calculate_ttd(): Removed arbitrary 500-second minimum TTD floor.
   The old code clipped every TTD to >= 500 s regardless of signal dynamics.
   For CKD-MBD starting at 2.0 nM with threshold 0.5 nM, the signal is above
   threshold from t=0; detection should be registered at ~10-50 s (processing
   delay), not 500 s. Removed the lower clip; only the upper sentinel bound
   (time[-1] * 2.5) is retained.

2. _calculate_fnr(): Replaced broken elevation-based FNR with margin-based FNR.
   Old code used 1.2x-rise criterion that NEVER fires on near-constant signals.
   New approach estimated FNR from signal margin relative to threshold and SNR.
   (Note: v3.1 further refined this to be deterministic and semantically correct)

3. Sustained elevation requirement: kept at 5 consecutive time points
   (~50 s at 10 s/point for 361 points over 3600 s).
   This correctly filters single noise spikes while requiring genuine elevation.

INSTRUMENTATION (V4.2)
======================
Added stage_data tracking for pipeline information audit.
Stage recordings store only JSON-serializable summaries (means, stds, counts),
not full numpy arrays. This enables analysis without breaking JSON serialization.
"""

import numpy as np
from typing import Dict, Tuple
import logging
from simulation.models.noise import NoiseModel

logger = logging.getLogger(__name__)

# Sensor degradation model
# Modified/locked DNA aptamers in physiological buffer:
#   half-life ~6 months (180 days) under implant conditions.
#   Ref: Delcanale et al. ACS Chem Biol 2021; Collett et al. Methods 2005.
# Enzyme-linked sensors degrade faster (t½ ~14-30 days, excluded from current design).
# At 12 months (2 half-lives): sensitivity × 0.25 → DR drops ~15-25pp.
APTAMER_HALF_LIFE_DAYS: float = 180.0
_K_DEG_SENSOR: float = np.log(2) / APTAMER_HALF_LIFE_DAYS  # per day


def normalize_biosensor_output(signal: np.ndarray, 
                               biosensor_type: str) -> np.ndarray:
    """
    Normalize biosensor output to [0, 1] range.
    
    This corrects for different transfer function scaling.
    
    Args:
        signal: raw biosensor output (any scale)
        biosensor_type: 'direct_binding', 'amplifying', 'threshold', 'ratiometric'
    
    Returns:
        normalized signal in [0, 1]
    """
    
    # Estimate dynamic range per biosensor type
    # These are empirically derived from signal range [0.375-2.0] nM
    
    if biosensor_type == 'direct_binding':
        # Langmuir: typically outputs 0.05-1.2 nM
        signal_min, signal_max = 0.0, 1.2
    
    elif biosensor_type == 'amplifying':
        # Exponential: typically outputs 0-2.0 (dimensionless)
        signal_min, signal_max = 0.0, 2.0
    
    elif biosensor_type == 'threshold':
        # Hill: typically outputs 0-1.0 (normalized)
        signal_min, signal_max = 0.0, 1.0
    
    elif biosensor_type == 'ratiometric':
        # Ratio: typically outputs 0-20 (depends on Kd)
        signal_min, signal_max = 0.0, 20.0
    
    else:
        # Default: use observed min/max
        signal_min = np.min(signal)
        signal_max = np.max(signal)
    
    # Normalize to [0, 1]
    if signal_max - signal_min > 1e-10:
        normalized = (signal - signal_min) / (signal_max - signal_min)
        # Clip to [0, 1] in case of outliers
        normalized = np.clip(normalized, 0.0, 1.0)
    else:
        # Constant signal
        normalized = np.ones_like(signal) * 0.5
    
    return normalized

class BiosensorEngine:
    """Engine applying a biosensor circuit model to ODE simulation time-series."""

    def __init__(self,
                 biosensor,
                 noise_model: NoiseModel | None = None,
                 detection_window: int = 10,
                 rolling_window: int = 1):
        """
        Args:
            detection_window: Consecutive points above threshold required for detection.
                              Default 10 (was 5) — sweep shows p=10 controls FP at 13 dB.
            rolling_window:   Causal rolling-mean window (points). 1 = no smoothing.
                              Sweep shows rolling_mean WORSENS FP when drift is dominant
                              (drift is sustained; smoothing amplifies it by reducing variance
                              that would otherwise break consecutive runs). Default = 1 (off).
        """
        self.biosensor        = biosensor
        self.noise_model      = noise_model
        self.detection_window = detection_window
        self.rolling_window   = rolling_window
        self.stage_data = {}  # For instrumentation: stores summaries only
        logger.info(
            f"BiosensorEngine: {biosensor.circuit_type}  "
            f"Kd={biosensor.kd:.3f}  sens={biosensor.sensitivity:.3f}  "
            f"thr={biosensor.threshold:.3f} nM  "
            f"det_window={detection_window}  roll_window={rolling_window}"
        )

    # ------------------------------------------------------------------
    def measure(self,
                time:         np.ndarray,
                species_data: Dict[str, np.ndarray],
                add_noise:    bool = True,
                elapsed_days: float = 0.0,
                patient_baseline_signal: float | None = None,
                threshold_multiplier: float = 1.25,
                t_half_days: float | None = None) -> Tuple[np.ndarray, Dict]:
        """
        Apply biosensor model to ODE output and return the measured signal.

        Args:
            time:                   Time array [seconds], shape (N,).
            species_data:           ODE output concentrations [nM], keyed by species name.
            add_noise:              Whether to apply the instrument noise model.
            elapsed_days:           Days since implant (0 = no degradation).
            patient_baseline_signal: Measured signal at enrollment (day 0) for this patient.
                                    When provided, threshold = patient_baseline × threshold_multiplier
                                    instead of the population-calibrated static threshold.
            threshold_multiplier:   Multiplier applied to patient_baseline_signal to compute
                                    the effective detection threshold (default 1.25).
            t_half_days:            Patient-specific aptamer half-life (days).
                                    When provided, overrides the global APTAMER_HALF_LIFE_DAYS.
                                    Sample per patient as Normal(180, 30), clipped to [120, 240].

        Returns:
            measured_signal: Noisy biosensor output, shape (N,).
            metadata:        Detection metrics dict.
        """
        # ── Extract analyte concentrations from ODE output ────────────────
        sclerostin = species_data.get('Sclerostin_sensor',
                                      species_data.get('Sclerostin_bone'))
        if sclerostin is None:
            raise ValueError(
                "No sclerostin data in species_data "
                "(expected 'Sclerostin_sensor' or 'Sclerostin_bone')"
            )
        rankl = species_data.get('RANKL_sensor', species_data.get('RANKL_bone'))
        opg   = species_data.get('OPG_sensor',   species_data.get('OPG_bone'))
        # Multi-biomarker panel analytes (present when ODE model v6.0+ is used)
        ctx   = species_data.get('CTX_sensor',  species_data.get('CTX_bone'))
        p1np  = species_data.get('P1NP_sensor', species_data.get('P1NP_bone'))

        # ── Apply biosensor circuit model ─────────────────────────────────
        clean_signal = self.biosensor.measure(
            sclerostin=sclerostin,
            time=time,
            rankl=rankl,
            opg=opg,
            ctx=ctx,
            p1np=p1np,
        )

        # ── Sensor degradation (aptamer half-life model) ──────────────────
        # Signal attenuates as aptamers denature over implant lifetime.
        # elapsed_days=0 → no degradation (default, backward-compatible).
        # t_half_days overrides global constant to support patient-specific variation.
        if elapsed_days > 0.0:
            k_deg = np.log(2) / t_half_days if t_half_days is not None else _K_DEG_SENSOR
            degradation_factor = float(np.exp(-k_deg * elapsed_days))
            clean_signal = clean_signal * degradation_factor
        else:
            degradation_factor = 1.0

        # ── Add instrument noise ──────────────────────────────────────────
        if add_noise and self.noise_model is not None:
            noisy_signal, _ = self.noise_model.apply_noise(clean_signal, time)
            snr = self.noise_model.get_snr(clean_signal, noisy_signal)
        else:
            noisy_signal = clean_signal.copy()
            snr = np.inf

        # ── Patient-specific threshold override ───────────────────────────
        # When the patient's enrollment baseline is known, compute a personalized
        # threshold rather than the population-calibrated static value.
        _original_threshold = self.biosensor.threshold
        if patient_baseline_signal is not None and patient_baseline_signal > 0.0:
            self.biosensor.threshold = float(patient_baseline_signal * threshold_multiplier)

        # ── Detect sustained elevation (hybrid: rolling mean + persistence) ──
        detected, max_consecutive, _, _ = self._detect(noisy_signal)
        threshold      = self.biosensor.threshold
        elevation_mask = noisy_signal >= threshold  # raw crossings (for stage recording)

        # ── Compute metrics ───────────────────────────────────────────────
        detection_rate = 1.0 if detected else 0.0
        time_to_detection = self._calculate_ttd(noisy_signal, time, snr)
        false_negative_rate = self._calculate_fnr(noisy_signal, snr)

        # Restore static threshold so the biosensor object stays reusable
        self.biosensor.threshold = _original_threshold

        # ── Stage recording for instrumentation (summaries only, JSON-safe) ───
        self.stage_data['stage_0_ode_output'] = {
            'mean': float(np.mean(sclerostin)),
            'std': float(np.std(sclerostin)),
            'min': float(np.min(sclerostin)),
            'max': float(np.max(sclerostin)),
        }
        
        self.stage_data['stage_1_biosensor_output'] = {
            'mean': float(np.mean(clean_signal)),
            'std': float(np.std(clean_signal)),
            'min': float(np.min(clean_signal)),
            'max': float(np.max(clean_signal)),
            'biosensor_type': self.biosensor.circuit_type,
            'sensitivity': float(self.biosensor.sensitivity),
            'threshold': float(self.biosensor.threshold),
            'kd': float(getattr(self.biosensor, 'kd', np.nan)),
        }
        
        self.stage_data['stage_2_noisy_output'] = {
            'mean': float(np.mean(noisy_signal)),
            'std': float(np.std(noisy_signal)),
            'min': float(np.min(noisy_signal)),
            'max': float(np.max(noisy_signal)),
            'snr_db': float(snr),
            'noise_preset': getattr(self.noise_model, 'preset', 'unknown') if self.noise_model else 'none',
        }
        
        self.stage_data['stage_3_thresholded'] = {
            'n_above_threshold': int(np.sum(elevation_mask)),
            'fraction_above': float(np.sum(elevation_mask) / len(elevation_mask)),
            'threshold_value': float(threshold),
        }
        
        self.stage_data['stage_4_persistence'] = {
            'max_consecutive': int(max_consecutive),
            'detection_window': self.detection_window,
            'rolling_window':   self.rolling_window,
            'detected': bool(detected),
        }

        # ── Return measured signal and metadata ────────────────────────────
        metadata = {
            'snr_db':            float(snr),
            'n_detections':      int(1 if detected else 0),
            'detection_rate':    float(detection_rate),
            'time_to_detection': float(time_to_detection),
            'false_negative_rate': float(false_negative_rate),
            'max_signal':        float(np.max(noisy_signal)),
            'mean_signal':       float(np.mean(noisy_signal)),
            'signal_std':        float(np.std(noisy_signal)),
            'has_noise':         add_noise and self.noise_model is not None,
            'sustained_duration': int(max_consecutive),
            'elapsed_days':      float(elapsed_days),
            'degradation_factor': float(degradation_factor),
        }

        return noisy_signal, metadata

    def get_stage_recordings(self) -> Dict:
        """Return stage recordings (JSON-serializable summaries only)"""
        return self.stage_data.copy()

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _apply_rolling_mean(signal: np.ndarray, window_size: int) -> np.ndarray:
        """
        Causal rolling mean. Point i averages over the past window_size samples.
        Edge points use all available samples (no look-ahead bias).
        window_size=1 returns the signal unchanged.
        """
        if window_size <= 1:
            return signal.copy()
        n = len(signal)
        cumsum = np.cumsum(signal)
        result = np.empty(n, dtype=float)
        result[:window_size] = cumsum[:window_size] / np.arange(1, window_size + 1)
        result[window_size:] = (cumsum[window_size:] - cumsum[:n - window_size]) / window_size
        return result

    # ──────────────────────────────────────────────────────────────────────
    def _detect(self, noisy_signal: np.ndarray) -> tuple:
        """
        Hybrid detector: rolling mean > threshold AND detection_window consecutive.

        Returns:
            detected (bool), max_consecutive (int), trigger_idx (int, -1 if none),
            smoothed (ndarray).
        """
        smoothed  = self._apply_rolling_mean(noisy_signal, self.rolling_window)
        threshold = self.biosensor.threshold
        above     = smoothed >= threshold

        max_consecutive = 0
        current_run     = 0
        trigger_idx     = -1

        for i, is_above in enumerate(above):
            if is_above:
                current_run += 1
                if current_run > max_consecutive:
                    max_consecutive = current_run
                if current_run >= self.detection_window and trigger_idx < 0:
                    trigger_idx = i - self.detection_window + 1
            else:
                current_run = 0

        detected = max_consecutive >= self.detection_window
        return detected, max_consecutive, trigger_idx, smoothed

    # ──────────────────────────────────────────────────────────────────────
    def _count_sustained_elevation(self, elevation_mask: np.ndarray) -> int:
        """Count the maximum number of consecutive True values in elevation_mask."""
        max_consecutive = 0
        current_run = 0

        for is_elevated in elevation_mask:
            if is_elevated:
                current_run += 1
                max_consecutive = max(max_consecutive, current_run)
            else:
                current_run = 0

        return max_consecutive

    # ──────────────────────────────────────────────────────────────────────
    def _calculate_ttd(self,
                      noisy_signal: np.ndarray,
                      time:         np.ndarray,
                      snr:          float) -> float:
        """
        Time-to-detection using the same hybrid detector as _detect().

        Uses SNR-dependent processing delay (not random):
          SNR > 25 dB → 25 s,  SNR > 15 dB → 75 s,
          SNR >  5 dB → 175 s,  SNR ≤  5 dB → 400 s.
        Sentinel = time[-1] * 2.5 when never detected.
        """
        dt = time[1] - time[0]
        if snr > 25.0:
            proc_delay_idx = max(1, int(25.0  / dt))
        elif snr > 15.0:
            proc_delay_idx = max(1, int(75.0  / dt))
        elif snr > 5.0:
            proc_delay_idx = max(1, int(175.0 / dt))
        else:
            proc_delay_idx = max(1, int(400.0 / dt))

        smoothed = self._apply_rolling_mean(noisy_signal, self.rolling_window)
        above    = smoothed >= self.biosensor.threshold

        consecutive_count = 0
        for i, is_above in enumerate(above):
            if is_above:
                consecutive_count += 1
                if consecutive_count >= self.detection_window:
                    detection_idx = max(0, i - self.detection_window + 1 + proc_delay_idx)
                    return float(time[min(detection_idx, len(time) - 1)])
            else:
                consecutive_count = 0

        return float(time[-1] * 2.5)

    # ──────────────────────────────────────────────────────────────────────
    def _calculate_fnr(self,
                      noisy_signal: np.ndarray,
                      snr:          float) -> float:
        """
        Deterministic FNR estimate. Uses _detect() so it respects rolling_window
        and detection_window consistently with the rest of the engine.
        Returns 0.0 if detected; a margin+SNR estimate otherwise.
        """
        detected, _, _, _ = self._detect(noisy_signal)
        if detected:
            return 0.0

        threshold    = self.biosensor.threshold
        margin       = np.mean(noisy_signal) - threshold
        snr_factor   = 10.0 ** (-snr / 20.0)
        margin_factor = max(0.0, 1.0 - abs(margin) / (threshold + 1e-6))
        return float(min(1.0, 0.5 * snr_factor + 0.5 * margin_factor))

    # ──────────────────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        return (
            f"BiosensorEngine("
            f"circuit_type={self.biosensor.circuit_type}, "
            f"threshold={self.biosensor.threshold:.3f})"
        )