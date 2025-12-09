"""
Noise models for biosensor measurements.

Implements multiple noise sources commonly encountered in biosensing:
1. Additive Gaussian noise (thermal, electronic)
2. Multiplicative noise (proportional to signal)
3. Baseline drift (temperature, aging)
4. Shot noise (Poisson)
5. Outliers (sporadic interference)

References:
- Hassibi et al. (2004). Biological shot-noise and quantum-limited SNR in 
  biosensors. J Appl Phys. doi:10.1063/1.1755429
- Ramanathan & Roy (2004). Fast, accurate and computationally efficient method
  for prediction of biosensor performance. Biosens Bioelectron. 
  doi:10.1016/j.bios.2003.11.021
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class NoiseModel:
    """
    Comprehensive noise model for biosensor measurements.
    """
    
    def __init__(self, 
                 additive_sigma: float = 0.01,
                 multiplicative_sigma: float = 0.05,
                 drift_rate: float = 0.0001,
                 outlier_probability: float = 0.001,
                 outlier_magnitude: float = 5.0,
                 shot_noise_enabled: bool = False,
                 seed: Optional[int] = None):
        """
        Initialize noise model.
        
        Args:
            additive_sigma: Standard deviation of additive Gaussian noise (nM)
            multiplicative_sigma: Fractional noise (e.g., 0.05 = 5%)
            drift_rate: Linear drift rate per time unit
            outlier_probability: Probability of outlier per measurement
            outlier_magnitude: Outlier magnitude (multiples of sigma)
            shot_noise_enabled: Enable Poisson shot noise
            seed: Random seed for reproducibility
        """
        self.additive_sigma = additive_sigma
        self.multiplicative_sigma = multiplicative_sigma
        self.drift_rate = drift_rate
        self.outlier_probability = outlier_probability
        self.outlier_magnitude = outlier_magnitude
        self.shot_noise_enabled = shot_noise_enabled
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        self._validate()
        logger.debug(f"Initialized NoiseModel with additive_sigma={additive_sigma}")
    
    def _validate(self):
        """Validate noise parameters."""
        assert self.additive_sigma >= 0, "Additive sigma must be non-negative"
        assert self.multiplicative_sigma >= 0, "Multiplicative sigma must be non-negative"
        assert 0 <= self.outlier_probability <= 1, "Outlier probability must be in [0,1]"
    
    def apply_noise(self, 
                    signal: np.ndarray, 
                    time: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Apply comprehensive noise to clean signal.
        
        Args:
            signal: Clean signal array (N,)
            time: Time array (N,)
        
        Returns:
            noisy_signal: Signal with all noise sources applied
            noise_components: Dictionary of individual noise contributions
        """
        n = len(signal)
        
        # Component 1: Additive Gaussian noise
        # Represents thermal noise, electronic noise
        additive_noise = np.random.normal(0, self.additive_sigma, size=n)
        
        # Component 2: Multiplicative (proportional) noise
        # Represents signal-dependent noise (e.g., shot noise approximation)
        multiplicative_noise = signal * np.random.normal(0, self.multiplicative_sigma, size=n)
        
        # Component 3: Baseline drift
        # Linear drift over time (temperature, sensor aging)
        drift = self.drift_rate * time
        
        # Component 4: Outliers (sporadic)
        # Random spikes from interference
        outliers = np.zeros(n)
        outlier_mask = np.random.rand(n) < self.outlier_probability
        outliers[outlier_mask] = np.random.choice(
            [-1, 1], 
            size=np.sum(outlier_mask)
        ) * self.outlier_magnitude * self.additive_sigma
        
        # Component 5: Shot noise (optional, Poisson)
        shot_noise = np.zeros(n)
        if self.shot_noise_enabled and np.all(signal >= 0):
            # Poisson noise: variance = mean
            # Scale signal to photon counts (assuming 1 nM ~ 1000 photons, adjustable)
            photon_scale = 1000.0
            photon_counts = np.maximum(signal * photon_scale, 0)
            poisson_counts = np.random.poisson(photon_counts)
            shot_noise = (poisson_counts - photon_counts) / photon_scale
        
        # Combine all noise sources
        total_noise = (additive_noise + multiplicative_noise + 
                      drift + outliers + shot_noise)
        noisy_signal = signal + total_noise
        
        # Enforce non-negativity constraint (physical concentrations)
        noisy_signal = np.maximum(noisy_signal, 0.0)
        
        noise_components = {
            'additive': additive_noise,
            'multiplicative': multiplicative_noise,
            'drift': drift,
            'outliers': outliers,
            'shot': shot_noise,
            'total': total_noise
        }
        
        return noisy_signal, noise_components
    
    def get_snr(self, signal: np.ndarray, noisy_signal: np.ndarray) -> float:
        """
        Calculate signal-to-noise ratio.
        
        Args:
            signal: Clean signal
            noisy_signal: Noisy signal
        
        Returns:
            SNR in dB
        """
        signal_power = np.mean(signal ** 2)
        noise = noisy_signal - signal
        noise_power = np.mean(noise ** 2)
        
        if noise_power == 0:
            return np.inf
        
        snr_linear = signal_power / noise_power
        snr_db = 10 * np.log10(snr_linear)
        
        return snr_db
    
    def to_dict(self) -> Dict[str, float]:
        """Export noise parameters as dictionary."""
        return {
            'additive_sigma': self.additive_sigma,
            'multiplicative_sigma': self.multiplicative_sigma,
            'drift_rate': self.drift_rate,
            'outlier_probability': self.outlier_probability,
            'outlier_magnitude': self.outlier_magnitude,
            'shot_noise_enabled': self.shot_noise_enabled,
            'seed': self.seed
        }


# ============================================================================
# NOISE PRESETS
# ============================================================================

def get_low_noise() -> NoiseModel:
    """Low noise conditions (laboratory, optimal)."""
    return NoiseModel(
        additive_sigma=0.01,
        multiplicative_sigma=0.02,
        drift_rate=0.00005,
        outlier_probability=0.0005,
        outlier_magnitude=3.0
    )


def get_medium_noise() -> NoiseModel:
    """Medium noise conditions (typical clinical)."""
    return NoiseModel(
        additive_sigma=0.05,
        multiplicative_sigma=0.08,
        drift_rate=0.0002,
        outlier_probability=0.002,
        outlier_magnitude=5.0
    )


def get_high_noise() -> NoiseModel:
    """High noise conditions (challenging, in vivo)."""
    return NoiseModel(
        additive_sigma=0.15,
        multiplicative_sigma=0.15,
        drift_rate=0.0005,
        outlier_probability=0.005,
        outlier_magnitude=8.0,
        shot_noise_enabled=True
    )


NOISE_PRESETS = {
    'low': get_low_noise(),
    'medium': get_medium_noise(),
    'high': get_high_noise()
}


def get_noise_model(preset: str, **kwargs) -> NoiseModel:
    """
    Get noise model by preset name with optional parameter overrides.
    
    Args:
        preset: One of 'low', 'medium', 'high'
        **kwargs: Parameter overrides
    
    Returns:
        NoiseModel instance
    """
    if preset not in NOISE_PRESETS:
        raise ValueError(f"Unknown noise preset: {preset}. "
                        f"Must be one of {list(NOISE_PRESETS.keys())}")
    
    base_noise = NOISE_PRESETS[preset]
    params = base_noise.to_dict()
    params.update(kwargs)
    
    return NoiseModel(**params)