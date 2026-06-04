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
from models.noise import NoiseModel

logger = logging.getLogger(__name__)


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
                 noise_model: NoiseModel | None = None):
        self.biosensor   = biosensor
        self.noise_model = noise_model
        self.stage_data = {}  # For instrumentation: stores summaries only
        logger.info(
            f"BiosensorEngine: {biosensor.circuit_type}  "
            f"Kd={biosensor.kd:.3f}  sens={biosensor.sensitivity:.3f}  "
            f"thr={biosensor.threshold:.3f} nM"
        )

    # ------------------------------------------------------------------
    def measure(self,
                time:         np.ndarray,
                species_data: Dict[str, np.ndarray],
                add_noise:    bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Apply biosensor model to ODE output and return the measured signal.

        Args:
            time:         Time array [seconds], shape (N,).
            species_data: ODE output concentrations [nM], keyed by species name.
            add_noise:    Whether to apply the instrument noise model.

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

        # ── Apply biosensor circuit model ─────────────────────────────────
        clean_signal = self.biosensor.measure(
            sclerostin=sclerostin,
            time=time,
            rankl=rankl,
            opg=opg,
        )

        # ── Add instrument noise ──────────────────────────────────────────
        if add_noise and self.noise_model is not None:
            noisy_signal, noise_components = self.noise_model.apply_noise(
                clean_signal, time
            )
            snr = self.noise_model.get_snr(clean_signal, noisy_signal)
        else:
            noisy_signal = clean_signal.copy()
            snr = np.inf

        # ── Detect sustained elevation ────────────────────────────────────
        threshold        = self.biosensor.threshold
        elevation_mask   = noisy_signal >= threshold
        max_consecutive  = self._count_sustained_elevation(elevation_mask)
        detected         = max_consecutive >= 5

        # ── Compute metrics ───────────────────────────────────────────────
        detection_rate = 1.0 if detected else 0.0
        time_to_detection = self._calculate_ttd(noisy_signal, sclerostin, time, snr)
        false_negative_rate = self._calculate_fnr(noisy_signal, sclerostin, time, snr)

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
            'persistence_window': 5,
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
        }

        return noisy_signal, metadata

    def get_stage_recordings(self) -> Dict:
        """Return stage recordings (JSON-serializable summaries only)"""
        return self.stage_data.copy()

    # ──────────────────────────────────────────────────────────────────────
    def _count_sustained_elevation(self, elevation_mask: np.ndarray) -> int:
        """
        Count the maximum number of consecutive points at or above threshold.
        """
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
                      noisy_signal:  np.ndarray,
                      true_signal:   np.ndarray,
                      time:          np.ndarray,
                      snr:           float) -> float:
        """
        Time to detection: How long until signal sustains >= 5 consecutive
        points above threshold?

        DETERMINISTIC behavior (v3.1):
          - Uses SNR-dependent processing delay (not random sampling)
          - Delay values: 25s (SNR>25dB), 75s (SNR>15dB), 175s (SNR>5dB),
            400s (SNR≤5dB)
          - Returns time index when 5 consecutive points sustain above threshold
          - Sentinel value: time[-1] * 2.5 if never detected

        Args:
            noisy_signal: Biosensor output (after noise)
            true_signal:  True analyte concentration (for margin calc)
            time:         Time array
            snr:          Signal-to-noise ratio (dB)

        Returns:
            Time (seconds) to when signal sustains detection.
            Sentinel value (time[-1] * 2.5) if never detected.
        """
        threshold = self.biosensor.threshold
        elevation_mask = noisy_signal >= threshold

        # Compute SNR-dependent processing delay (deterministic, not random)
        if snr > 25.0:
            processing_delay_idx = max(1, int(25.0 / (time[1] - time[0])))
        elif snr > 15.0:
            processing_delay_idx = max(1, int(75.0 / (time[1] - time[0])))
        elif snr > 5.0:
            processing_delay_idx = max(1, int(175.0 / (time[1] - time[0])))
        else:
            processing_delay_idx = max(1, int(400.0 / (time[1] - time[0])))

        # Find first index where 5 consecutive points are >= threshold
        consecutive_count = 0
        for i, is_elevated in enumerate(elevation_mask):
            if is_elevated:
                consecutive_count += 1
                if consecutive_count >= 5:
                    # Detection time is when the 5-point window starts,
                    # plus processing delay
                    detection_idx = max(0, i - 4 + processing_delay_idx)
                    if detection_idx < len(time):
                        return float(time[detection_idx])
                    else:
                        return float(time[-1])
            else:
                consecutive_count = 0

        # Never detected: return sentinel (time[-1] * 2.5)
        return float(time[-1] * 2.5)

    # ──────────────────────────────────────────────────────────────────────
    def _calculate_fnr(self,
                      noisy_signal: np.ndarray,
                      true_signal:  np.ndarray,
                      time:         np.ndarray,
                      snr:          float) -> float:
        """
        False negative rate (FNR): Probability that detection should have
        occurred but didn't.

        DETERMINISTIC behavior (v3.1):
          - If detected (≥5 consecutive points above threshold): FNR = 0.0
          - If not detected: FNR = estimated P(should have been detected)
            based on signal margin and SNR

        Args:
            noisy_signal: Biosensor output (after noise)
            true_signal:  True analyte concentration
            time:         Time array
            snr:          Signal-to-noise ratio (dB)

        Returns:
            FNR in [0, 1], where 0 means detection was successful,
            1 means detection failed completely.
        """
        threshold = self.biosensor.threshold
        elevation_mask = noisy_signal >= threshold

        # Check if detection succeeded
        if self._count_sustained_elevation(elevation_mask) >= 5:
            return 0.0  # Detected successfully, no false negative

        # Not detected: estimate FNR based on signal margin and SNR
        margin = np.mean(noisy_signal) - threshold

        # SNR-dependent FNR curve
        snr_factor = 10.0 ** (-snr / 20.0)  # Convert dB to linear
        margin_factor = max(0.0, 1.0 - abs(margin) / (threshold + 1e-6))

        fnr = min(1.0, 0.5 * snr_factor + 0.5 * margin_factor)

        return float(fnr)

    # ──────────────────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        return (
            f"BiosensorEngine("
            f"circuit_type={self.biosensor.circuit_type}, "
            f"threshold={self.biosensor.threshold:.3f})"
        )