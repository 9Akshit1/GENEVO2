"""
Biosensor measurement engine that interfaces simulation with biosensor models.
"""

import numpy as np
from typing import Dict, Optional, Tuple
import logging
from models.biosensors import Biosensor, create_biosensor
from models.noise import NoiseModel

logger = logging.getLogger(__name__)


class BiosensorEngine:
    """
    Engine for applying biosensor models to simulation data.
    """
    
    def __init__(self, 
                 biosensor: Biosensor,
                 noise_model: Optional[NoiseModel] = None):
        """
        Initialize biosensor engine.
        
        Args:
            biosensor: Biosensor model instance
            noise_model: Optional noise model to apply
        """
        self.biosensor = biosensor
        self.noise_model = noise_model
        
        logger.info(f"Initialized BiosensorEngine with {biosensor.circuit_type} sensor")
    
    def measure(self, 
                time: np.ndarray,
                species_data: Dict[str, np.ndarray],
                add_noise: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Apply biosensor to simulation data and generate measurement signal.
        
        Args:
            time: Time array (seconds)
            species_data: Dictionary of species concentrations
            add_noise: Whether to add measurement noise
        
        Returns:
            measured_signal: Biosensor output with noise
            metadata: Dictionary with measurement info
        """
        # Extract sclerostin from sensor chamber
        sclerostin = species_data.get('Sclerostin_sensor', 
                                     species_data.get('Sclerostin_bone'))
        
        if sclerostin is None:
            raise ValueError("No sclerostin data found in species_data")
        
        # Get additional analytes if needed (for ratiometric sensors)
        rankl = species_data.get('RANKL_sensor', 
                                species_data.get('RANKL_bone'))
        opg = species_data.get('OPG_sensor', 
                              species_data.get('OPG_bone'))
        
        # Apply biosensor model
        clean_signal = self.biosensor.measure(
            sclerostin=sclerostin,
            time=time,
            rankl=rankl,
            opg=opg
        )
        
        # Add noise if requested
        if add_noise and self.noise_model is not None:
            noisy_signal, noise_components = self.noise_model.apply_noise(
                clean_signal, time
            )
            snr = self.noise_model.get_snr(clean_signal, noisy_signal)
        else:
            noisy_signal = clean_signal
            noise_components = {}
            snr = np.inf
        
        # Compute detection metrics
        detection_events = self.biosensor.is_detected(noisy_signal)
        n_detections = np.sum(detection_events)
        detection_rate = n_detections / len(detection_events)
        
        # Time to first detection
        if n_detections > 0:
            first_detection_idx = np.where(detection_events)[0][0]
            time_to_detection = time[first_detection_idx]
        else:
            time_to_detection = np.nan
        
        metadata = {
            'snr_db': snr,
            'n_detections': int(n_detections),
            'detection_rate': float(detection_rate),
            'time_to_detection': float(time_to_detection),
            'max_signal': float(np.max(noisy_signal)),
            'mean_signal': float(np.mean(noisy_signal)),
            'signal_std': float(np.std(noisy_signal)),
            'has_noise': add_noise and self.noise_model is not None
        }
        
        logger.debug(f"Measurement complete: SNR={snr:.1f}dB, detections={n_detections}")
        
        return noisy_signal, metadata