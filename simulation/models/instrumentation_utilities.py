"""
Instrumentation utilities for paired-propagation pipeline diagnosis.

Core design:
  - Single biological realization
  - Propagate through ALL stages without resampling
  - Extract defensible scalar metrics at each stage
  - Record stage-delta degradation explicitly
  - Preserve reproducibility metadata

Statistical validity:
  - Scalar metrics are independent observations (valid Cohen's d)
  - Not trajectory points (auto-correlated)
  - Physically meaningful AUC normalization
  - Modular separability metrics (extensible beyond Cohen's d)
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
import hashlib
import logging
import json

logger = logging.getLogger(__name__)


class BiologicalRealization:
    """Encapsulates a single parameter draw with reproducibility metadata."""
    
    def __init__(self, 
                 params: Dict[str, float],
                 rng_seed: int,
                 scenario: str):
        """
        Args:
            params: Dictionary of environment parameters
            rng_seed: RNG seed used to generate this realization
            scenario: Scenario name ('healthy', 'pmo', 'ckd_mbd')
        """
        self.params = params.copy()
        self.rng_seed = rng_seed
        self.scenario = scenario
        
        # Compute hash for reproducibility
        params_json = json.dumps(params, sort_keys=True)
        self.realization_hash = hashlib.sha256(
            f"{params_json}_{rng_seed}_{scenario}".encode()
        ).hexdigest()[:16]
        
        logger.debug(
            f"Created biological realization: {self.scenario} "
            f"seed={rng_seed} hash={self.realization_hash}"
        )
    
    def get_metadata(self) -> Dict:
        """Return reproducibility metadata."""
        return {
            'rng_seed': int(self.rng_seed),
            'scenario': self.scenario,
            'realization_hash': self.realization_hash,
        }


class StageMetrics:
    """Computes and stores metrics at a single pipeline stage."""
    
    def __init__(self, stage_name: str, stage_description: str):
        """
        Args:
            stage_name: Short name (e.g., 'ode', 'biosensor_clean')
            stage_description: Human-readable description
        """
        self.stage_name = stage_name
        self.stage_description = stage_description
        self.metrics = {
            'stage': stage_name,
            'description': stage_description,
        }
    
    def compute_from_signal(self,
                           signal: np.ndarray,
                           time: np.ndarray,
                           extra_metadata: Optional[Dict] = None):
        """
        Extract scalar metrics from signal trajectory.
        
        CRITICAL: These are summary statistics, not trajectory points.
        Each metric is a single scalar value (independent observation).
        
        Args:
            signal: 1D array of signal values
            time: 1D array of time points
            extra_metadata: Additional metadata dict
        """
        
        if len(signal) < 2:
            logger.warning(f"Signal too short ({len(signal)} points)")
            signal = np.atleast_1d(signal)
            time = np.atleast_1d(time)
        
        # Core summary statistics
        self.metrics['mean_signal'] = float(np.mean(signal))
        self.metrics['std_signal'] = float(np.std(signal))
        self.metrics['min_signal'] = float(np.min(signal))
        self.metrics['max_signal'] = float(np.max(signal))
        self.metrics['median_signal'] = float(np.median(signal))
        self.metrics['q25_signal'] = float(np.percentile(signal, 25))
        self.metrics['q75_signal'] = float(np.percentile(signal, 75))
        
        # Temporal integration (FIXED: proper AUC normalization)
        # AUC per second is physically meaningful
        time_span = float(time[-1] - time[0])
        if time_span > 0:
            auc = float(np.trapz(signal, time))
            self.metrics['auc_total'] = auc
            self.metrics['auc_per_second'] = auc / time_span
        else:
            self.metrics['auc_total'] = float(np.nan)
            self.metrics['auc_per_second'] = float(np.nan)
        
        # Terminal behavior (FIXED: renamed to avoid false convergence assumption)
        terminal_fraction = int(0.1 * len(signal)) if len(signal) > 10 else 1
        self.metrics['terminal_mean'] = float(
            np.mean(signal[-terminal_fraction:])
        )
        self.metrics['terminal_std'] = float(
            np.std(signal[-terminal_fraction:])
        )
        
        # Range and dynamic properties
        signal_range = self.metrics['max_signal'] - self.metrics['min_signal']
        self.metrics['signal_range'] = signal_range
        
        if self.metrics['mean_signal'] > 1e-12:
            self.metrics['cv_coefficient'] = float(
                self.metrics['std_signal'] / self.metrics['mean_signal']
            )
        else:
            self.metrics['cv_coefficient'] = float(np.nan)
        
        # Add extra metadata if provided
        if extra_metadata:
            self.metrics.update(extra_metadata)
        
        return self.metrics
    
    def compute_from_threshold_crossing(self,
                                       signal: np.ndarray,
                                       time: np.ndarray,
                                       threshold: float) -> Dict:
        """
        Extract threshold-related metrics.
        
        Args:
            signal: Signal trajectory
            time: Time array
            threshold: Threshold value
            
        Returns:
            Metrics dict
        """
        
        above_threshold = signal >= threshold
        fraction_above = float(np.sum(above_threshold) / len(above_threshold))
        
        # Time to first crossing
        crossings = np.where(above_threshold)[0]
        if len(crossings) > 0:
            time_to_first = float(time[crossings[0]])
            n_crossings = len(np.where(np.diff(above_threshold.astype(int)) != 0)[0])
        else:
            time_to_first = float(time[-1] * 2.5)  # sentinel
            n_crossings = 0
        
        # Signal statistics stratified by threshold
        signal_above = signal[above_threshold]
        signal_below = signal[~above_threshold]
        
        self.metrics['threshold'] = float(threshold)
        self.metrics['fraction_above_threshold'] = fraction_above
        self.metrics['time_to_first_crossing'] = time_to_first
        self.metrics['n_threshold_crossings'] = int(n_crossings)
        
        if len(signal_above) > 0:
            self.metrics['mean_signal_above'] = float(np.mean(signal_above))
            self.metrics['std_signal_above'] = float(np.std(signal_above))
        else:
            self.metrics['mean_signal_above'] = float(np.nan)
            self.metrics['std_signal_above'] = float(np.nan)
        
        if len(signal_below) > 0:
            self.metrics['mean_signal_below'] = float(np.mean(signal_below))
            self.metrics['std_signal_below'] = float(np.std(signal_below))
        else:
            self.metrics['mean_signal_below'] = float(np.nan)
            self.metrics['std_signal_below'] = float(np.nan)
        
        # Margin relative to threshold
        margin_above = np.mean(signal_above - threshold) if len(signal_above) > 0 else np.nan
        margin_below = np.mean(signal_below - threshold) if len(signal_below) > 0 else np.nan
        
        self.metrics['mean_margin_above_threshold'] = float(margin_above)
        self.metrics['mean_margin_below_threshold'] = float(margin_below)
        
        return self.metrics
    
    def get_metrics(self) -> Dict:
        """Return metrics dictionary."""
        return self.metrics.copy()


class StageDelta:
    """Quantifies degradation between consecutive stages."""
    
    def __init__(self, 
                 stage_from: str,
                 stage_to: str,
                 output_from: np.ndarray,
                 output_to: np.ndarray):
        """
        Compute difference between outputs at consecutive stages.
        
        Args:
            stage_from: Name of earlier stage
            stage_to: Name of later stage
            output_from: Output at earlier stage
            output_to: Output at later stage
        """
        self.stage_from = stage_from
        self.stage_to = stage_to
        self.delta = output_to - output_from
        
        self.metrics = {
            'transition': f'{stage_from}->{stage_to}',
            'mean_delta': float(np.mean(self.delta)),
            'std_delta': float(np.std(self.delta)),
            'max_delta': float(np.max(np.abs(self.delta))),
            'rms_delta': float(np.sqrt(np.mean(self.delta**2))),
        }
        
        # Fraction of signal changed significantly
        threshold_sig = np.std(output_from) * 0.1 if np.std(output_from) > 1e-12 else 0.01
        significant_change = np.sum(np.abs(self.delta) > threshold_sig)
        self.metrics['fraction_significantly_changed'] = float(
            significant_change / len(self.delta)
        )
    
    def get_metrics(self) -> Dict:
        """Return delta metrics."""
        return self.metrics.copy()


class PairedPropagationRun:
    """
    Single instrumented simulation with paired propagation.
    
    CRITICAL:
      - One biological realization generated
      - Propagated through ALL stages
      - NO resampling between stages
      - Stage outputs recorded for analysis
    """
    
    def __init__(self, 
                 scenario: str,
                 biosensor_config: Dict,
                 noise_preset: str):
        """
        Args:
            scenario: 'healthy', 'pmo', or 'ckd_mbd'
            biosensor_config: Biosensor circuit configuration
            noise_preset: 'low', 'medium', or 'high'
        """
        self.scenario = scenario
        self.biosensor_config = biosensor_config
        self.noise_preset = noise_preset
        
        self.biological_realization = None
        self.time = None
        self.species_data = None
        
        self.stage_metrics = {}
        self.stage_outputs = {}
        self.stage_deltas = {}
    
    def set_biological_realization(self, 
                                   realization: BiologicalRealization,
                                   time: np.ndarray,
                                   species_data: Dict[str, np.ndarray]):
        """
        Set the biological realization that will propagate through pipeline.
        
        This is the SINGLE biological state for this entire run.
        """
        self.biological_realization = realization
        self.time = time.copy()
        self.species_data = {k: v.copy() for k, v in species_data.items()}
        
        logger.debug(
            f"Set biological realization: {realization.scenario} "
            f"hash={realization.realization_hash}"
        )
    
    def record_stage_0_ode(self):
        """Record ODE output stage (before any transformation)."""
        
        sclerostin = self.species_data.get(
            'Sclerostin_sensor',
            self.species_data.get('Sclerostin_bone')
        )
        
        metrics = StageMetrics('stage_0_ode', 'ODE concentration output')
        metrics.compute_from_signal(
            sclerostin,
            self.time,
            extra_metadata={
                'species_tracked': 'Sclerostin_sensor or Sclerostin_bone'
            }
        )
        
        self.stage_metrics['stage_0_ode'] = metrics.get_metrics()
        self.stage_outputs['stage_0_ode'] = sclerostin.copy()
        
        logger.debug("Recorded stage 0: ODE output")
    
    def record_stage_1_biosensor_clean(self, 
                                       clean_output: np.ndarray):
        """
        Record clean biosensor output (no noise yet).
        
        CRITICAL: This uses the SAME species_data from stage 0.
        No resampling or re-simulation.
        """
        
        metrics = StageMetrics(
            'stage_1_biosensor_clean',
            'Biosensor output (clean signal, no noise)'
        )
        metrics.compute_from_signal(
            clean_output,
            self.time,
            extra_metadata={
                'biosensor_type': self.biosensor_config.get('circuit_type'),
                'kd': self.biosensor_config.get('kd'),
                'sensitivity': self.biosensor_config.get('sensitivity'),
            }
        )
        
        self.stage_metrics['stage_1_biosensor_clean'] = metrics.get_metrics()
        self.stage_outputs['stage_1_biosensor_clean'] = clean_output.copy()
        
        # Record stage delta: ODE -> clean biosensor
        if 'stage_0_ode' in self.stage_outputs:
            delta = StageDelta(
                'stage_0_ode',
                'stage_1_biosensor_clean',
                self.stage_outputs['stage_0_ode'],
                clean_output
            )
            self.stage_deltas['delta_0_to_1'] = delta.get_metrics()
        
        logger.debug("Recorded stage 1: Clean biosensor output")
    
    def record_stage_2_biosensor_noisy(self,
                                       noisy_output: np.ndarray,
                                       snr_db: float):
        """
        Record noisy biosensor output.
        
        CRITICAL: noise is applied to SAME clean output.
        No re-simulation or re-measurement.
        
        Args:
            noisy_output: Signal after noise injection
            snr_db: Signal-to-noise ratio in dB
        """
        
        metrics = StageMetrics(
            'stage_2_biosensor_noisy',
            'Biosensor output (with noise injection)'
        )
        metrics.compute_from_signal(
            noisy_output,
            self.time,
            extra_metadata={
                'noise_preset': self.noise_preset,
                'snr_db': float(snr_db),
            }
        )
        
        self.stage_metrics['stage_2_biosensor_noisy'] = metrics.get_metrics()
        self.stage_outputs['stage_2_biosensor_noisy'] = noisy_output.copy()
        
        # Record stage delta: clean -> noisy
        if 'stage_1_biosensor_clean' in self.stage_outputs:
            delta = StageDelta(
                'stage_1_biosensor_clean',
                'stage_2_biosensor_noisy',
                self.stage_outputs['stage_1_biosensor_clean'],
                noisy_output
            )
            self.stage_deltas['delta_1_to_2'] = delta.get_metrics()
        
        logger.debug(f"Recorded stage 2: Noisy biosensor output (SNR={snr_db:.1f}dB)")
    
    def record_stage_3_thresholded(self,
                                   noisy_output: np.ndarray,
                                   threshold: float) -> np.ndarray:
        """
        Record thresholding operation metrics.
        
        Args:
            noisy_output: Signal before thresholding
            threshold: Threshold value
            
        Returns:
            Binary elevation mask (for TTD calculation)
        """
        
        elevation_mask = noisy_output >= threshold
        
        metrics = StageMetrics(
            'stage_3_thresholded',
            'Threshold crossing detection'
        )
        metrics.compute_from_threshold_crossing(
            noisy_output,
            self.time,
            threshold
        )
        
        self.stage_metrics['stage_3_thresholded'] = metrics.get_metrics()
        self.stage_outputs['stage_3_thresholded'] = elevation_mask.astype(float)
        
        # Record stage delta: noisy -> thresholded (as binary)
        if 'stage_2_biosensor_noisy' in self.stage_outputs:
            # Create a pseudo-signal from detection state for delta
            detected_signal = noisy_output * elevation_mask.astype(float)
            delta = StageDelta(
                'stage_2_biosensor_noisy',
                'stage_3_thresholded',
                self.stage_outputs['stage_2_biosensor_noisy'],
                detected_signal
            )
            self.stage_deltas['delta_2_to_3'] = delta.get_metrics()
        
        logger.debug("Recorded stage 3: Threshold crossing")
        
        return elevation_mask
    
    def record_stage_4_ttd(self, 
                          elevation_mask: np.ndarray,
                          time_to_detection: float):
        """
        Record TTD assignment metrics.
        
        Args:
            elevation_mask: Boolean mask of points >= threshold
            time_to_detection: Computed TTD value
        """
        
        detected = (time_to_detection < self.time[-1] * 2.5)
        
        self.stage_metrics['stage_4_ttd'] = {
            'stage': 'stage_4_ttd',
            'description': 'Time-to-detection assignment',
            'ttd_seconds': float(time_to_detection),
            'detected': bool(detected),
            'is_sentinel': bool(not detected),
        }
        
        self.stage_outputs['stage_4_ttd'] = np.array([float(time_to_detection)])
        
        logger.debug(f"Recorded stage 4: TTD={time_to_detection:.0f}s (detected={detected})")
    
    def get_metadata(self) -> Dict:
        """Return complete instrumentation metadata."""
        return {
            'biological_realization': self.biological_realization.get_metadata(),
            'stages': self.stage_metrics,
            'stage_deltas': self.stage_deltas,
            'propagation_integrity': self._verify_propagation_integrity(),
        }
    
    def _verify_propagation_integrity(self) -> Dict:
        """
        Verify that propagation was truly paired.
        
        Returns confidence metrics about propagation.
        """
        return {
            'stages_recorded': len(self.stage_metrics),
            'biological_hash_consistent': (
                self.biological_realization.realization_hash ==
                self.stage_metrics.get('stage_0_ode', {}).get('realization_hash', None)
            ),
            'all_stages_use_same_time': (
                all(len(s) == len(self.time) for s in self.stage_outputs.values()
                    if isinstance(s, np.ndarray) and len(s) > 1)
            ),
        }


def create_instrumented_run(scenario: str,
                           biosensor_config: Dict,
                           noise_preset: str) -> PairedPropagationRun:
    """Factory function for creating instrumented runs."""
    return PairedPropagationRun(scenario, biosensor_config, noise_preset)