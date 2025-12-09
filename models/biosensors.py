"""
Biosensor circuit implementations for sclerostin detection.

Implements multiple biosensor architectures:
1. DirectBindingSensor - Simple receptor-ligand binding
2. AmplifyingSensor - Enzymatic cascade amplification
3. ThresholdSensor - Digital (on/off) response
4. RatiometricSensor - Dual-analyte ratiometric detection

All sensors model:
- Binding kinetics (mass-action, Kd)
- Signal transduction
- Response curves (linear, Hill, Michaelis-Menten)
- Circuit-specific dynamics

References:
- Sapsford et al. (2006). Biosensor detection systems. Biosens Bioelectron.
  doi:10.1016/j.bios.2005.09.008
- Thévenot et al. (2001). Electrochemical biosensors: recommended definitions.
  Biosens Bioelectron. doi:10.1016/S0956-5663(01)00115-4
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
from scipy.integrate import odeint
import logging

logger = logging.getLogger(__name__)


class Biosensor(ABC):
    """
    Abstract base class for biosensor models.
    """
    
    def __init__(self, 
                 sensitivity: float,
                 threshold: float,
                 dynamic_range: Tuple[float, float],
                 kd: float,
                 response_type: str,
                 circuit_type: str,
                 **kwargs):
        """
        Initialize biosensor base class.
        
        Args:
            sensitivity: Sensor sensitivity constant
            threshold: Detection threshold (nM)
            dynamic_range: (min, max) detection range (nM)
            kd: Dissociation constant for binding (nM)
            response_type: 'linear', 'hill', 'michaelis_menten'
            circuit_type: Biosensor circuit architecture name
            **kwargs: Additional sensor-specific parameters
        """
        self.sensitivity = sensitivity
        self.threshold = threshold
        self.dynamic_range = dynamic_range
        self.kd = kd
        self.response_type = response_type
        self.circuit_type = circuit_type
        self.extra_params = kwargs
        
        self._validate()
        logger.debug(f"Initialized {circuit_type} biosensor with Kd={kd} nM")
    
    def _validate(self):
        """Validate biosensor parameters."""
        assert self.sensitivity > 0, "Sensitivity must be positive"
        assert self.threshold >= 0, "Threshold must be non-negative"
        assert self.dynamic_range[0] < self.dynamic_range[1], \
            "Dynamic range min must be < max"
        assert self.kd > 0, "Kd must be positive"
        assert self.response_type in ['linear', 'hill', 'michaelis_menten'], \
            f"Unknown response type: {self.response_type}"
    
    @abstractmethod
    def measure(self, 
                sclerostin: np.ndarray, 
                time: np.ndarray,
                **kwargs) -> np.ndarray:
        """
        Compute biosensor measurement signal from analyte concentration.
        
        Args:
            sclerostin: Sclerostin concentration array (nM)
            time: Time array
            **kwargs: Additional analyte concentrations if needed
        
        Returns:
            Measurement signal array
        """
        pass
    
    def apply_saturation(self, signal: np.ndarray) -> np.ndarray:
        """Apply dynamic range saturation."""
        return np.clip(signal, self.dynamic_range[0], self.dynamic_range[1])
    
    def get_response_curve(self, concentration: np.ndarray) -> np.ndarray:
        """
        Compute response based on response_type.
        
        Args:
            concentration: Analyte concentration array (nM)
        
        Returns:
            Response signal (arbitrary units, before sensitivity scaling)
        """
        if self.response_type == 'linear':
            return concentration
        
        elif self.response_type == 'hill':
            # Hill equation: S = S_max * C^n / (Kd^n + C^n)
            n = self.extra_params.get('hill_coefficient', 2.0)
            return (concentration ** n) / (self.kd ** n + concentration ** n)
        
        elif self.response_type == 'michaelis_menten':
            # Michaelis-Menten: S = S_max * C / (Km + C)
            # Use Kd as Km
            return concentration / (self.kd + concentration)
        
        else:
            raise ValueError(f"Unknown response type: {self.response_type}")
    
    def is_detected(self, signal: np.ndarray) -> np.ndarray:
        """Return boolean array indicating detection events."""
        return signal >= self.threshold
    
    def to_dict(self) -> Dict:
        """Export biosensor configuration as dictionary."""
        return {
            'circuit_type': self.circuit_type,
            'sensitivity': self.sensitivity,
            'threshold': self.threshold,
            'dynamic_range': self.dynamic_range,
            'kd': self.kd,
            'response_type': self.response_type,
            **self.extra_params
        }


class DirectBindingSensor(Biosensor):
    """
    Direct binding biosensor.
    
    Mechanism:
    - Receptor (R) binds sclerostin (S) to form complex (RS)
    - Signal proportional to [RS]
    - No amplification
    
    Kinetics:
    R + S <--> RS
    kon, koff rates
    Kd = koff / kon
    
    References:
    - Pollard (2010). A guide to simple and informative binding assays.
      Mol Biol Cell. doi:10.1091/mbc.E10-08-0683
    """
    
    def __init__(self, 
                 sensitivity: float = 1.0,
                 threshold: float = 0.1,
                 dynamic_range: Tuple[float, float] = (0.0, 10.0),
                 kd: float = 1.0,
                 receptor_concentration: float = 1.0,  # Reduced for low nM analyte range
                 kon: float = 1e5,
                 koff: float = 1e-1,
                 response_type: str = 'linear',
                 **kwargs):
        """
        Initialize direct binding sensor.
        
        Args:
            sensitivity: Signal per unit [RS] complex
            threshold: Detection threshold
            dynamic_range: Sensor saturation limits
            kd: Dissociation constant (nM)
            receptor_concentration: Total receptor concentration (nM)
            kon: Association rate constant (1/(nM*s))
            koff: Dissociation rate constant (1/s)
            response_type: Response curve type
        """
        super().__init__(
            sensitivity=sensitivity,
            threshold=threshold,
            dynamic_range=dynamic_range,
            kd=kd,
            response_type=response_type,
            circuit_type='direct_binding',
            receptor_concentration=receptor_concentration,
            kon=kon,
            koff=koff,
            **kwargs
        )
        
        self.receptor_concentration = receptor_concentration
        self.kon = kon
        self.koff = koff
        
        # Validate Kd consistency
        calculated_kd = koff / kon
        if not np.isclose(calculated_kd, kd, rtol=0.1):
            logger.warning(f"Provided Kd ({kd}) differs from koff/kon ({calculated_kd})")
    
    def measure(self, 
                sclerostin: np.ndarray, 
                time: np.ndarray,
                **kwargs) -> np.ndarray:
        """
        Measure sclerostin via direct binding.
        
        Assumes quasi-equilibrium (fast binding kinetics):
        [RS] = [R_total] * [S] / (Kd + [S])
        
        Args:
            sclerostin: Sclerostin concentration (nM)
            time: Time array (used for potential time-dependent effects)
        
        Returns:
            Sensor signal
        """
        # Equilibrium binding (Langmuir isotherm)
        # [RS] = [R_total] * [S] / (Kd + [S])
        bound_complex = (self.receptor_concentration * sclerostin / 
                        (self.kd + sclerostin))
        
        # Apply response curve
        response = self.get_response_curve(bound_complex)
        
        # Scale by sensitivity
        signal = self.sensitivity * response
        
        # Apply saturation
        signal = self.apply_saturation(signal)
        
        return signal


class AmplifyingSensor(Biosensor):
    """
    Enzymatic amplification biosensor.
    
    Mechanism:
    - R + S <--> RS (binding)
    - RS activates enzyme E
    - E catalyzes reporter production (amplification)
    - Signal proportional to [Reporter]
    
    Amplification gain: single binding event → many reporter molecules
    
    References:
    - Ronkainen et al. (2010). Electrochemical biosensors. Chem Soc Rev.
      doi:10.1039/b714449k
    """
    
    def __init__(self, 
                 sensitivity: float = 1.0,
                 threshold: float = 0.1,
                 dynamic_range: Tuple[float, float] = (0.0, 100.0),
                 kd: float = 1.0,
                 receptor_concentration: float = 1.0,    # Match analyte scale
                 enzyme_concentration: float = 10.0,     # Proportionally reduced
                 amplification_gain: float = 500.0,      # Increased for sensitivity
                 response_time: float = 60.0,
                 response_type: str = 'michaelis_menten',
                 **kwargs):
        """
        Initialize amplifying sensor.
        
        Args:
            amplification_gain: Number of reporter molecules per binding event
            enzyme_concentration: Enzyme concentration (nM)
            response_time: Characteristic response time (seconds)
        """
        super().__init__(
            sensitivity=sensitivity,
            threshold=threshold,
            dynamic_range=dynamic_range,
            kd=kd,
            response_type=response_type,
            circuit_type='amplifying',
            receptor_concentration=receptor_concentration,
            enzyme_concentration=enzyme_concentration,
            amplification_gain=amplification_gain,
            response_time=response_time,
            **kwargs
        )
        
        self.receptor_concentration = receptor_concentration
        self.enzyme_concentration = enzyme_concentration
        self.amplification_gain = amplification_gain
        self.response_time = response_time
    
    def measure(self, 
                sclerostin: np.ndarray, 
                time: np.ndarray,
                **kwargs) -> np.ndarray:
        """
        Measure sclerostin via amplifying cascade.
        
        Simplified model:
        - Instantaneous binding equilibrium
        - First-order enzyme activation by [RS]
        - Catalytic reporter production
        
        Args:
            sclerostin: Sclerostin concentration (nM)
            time: Time array (seconds)
        
        Returns:
            Amplified sensor signal
        """
        # Binding equilibrium
        bound_complex = (self.receptor_concentration * sclerostin / 
                        (self.kd + sclerostin))
        
        # Enzyme activation (simplified): E_active ∝ [RS]
        enzyme_active_fraction = bound_complex / (self.kd + bound_complex)
        enzyme_active = self.enzyme_concentration * enzyme_active_fraction
        
        # Reporter production (simplified first-order accumulation)
        # Reporter(t) = Gain * E_active * (1 - exp(-t/tau))
        # For discrete time points, assume steady-state approximation
        tau = self.response_time
        time_factor = 1.0 - np.exp(-time / tau)
        
        reporter = self.amplification_gain * enzyme_active * time_factor
        
        # Apply response curve
        response = self.get_response_curve(reporter)
        
        # Scale by sensitivity
        signal = self.sensitivity * response
        
        # Apply saturation
        signal = self.apply_saturation(signal)
        
        return signal


class ThresholdSensor(Biosensor):
    """
    Digital threshold sensor (on/off response).
    
    Mechanism:
    - Binary output: OFF if C < threshold, ON if C >= threshold
    - Mimics genetic circuit toggle switches
    - Sharp transition via cooperative binding (high Hill coefficient)
    
    References:
    - Gardner et al. (2000). Construction of a genetic toggle switch. Nature.
      doi:10.1038/35002131
    """
    
    def __init__(self, 
                 sensitivity: float = 10.0,
                 threshold: float = 2.0,
                 dynamic_range: Tuple[float, float] = (0.0, 10.0),
                 kd: float = 2.0,
                 hill_coefficient: float = 4.0,
                 off_level: float = 0.1,
                 on_level: float = 10.0,
                 response_type: str = 'hill',
                 **kwargs):
        """
        Initialize threshold sensor.
        
        Args:
            hill_coefficient: Cooperativity (higher = sharper transition)
            off_level: Signal output when OFF
            on_level: Signal output when ON
        """
        super().__init__(
            sensitivity=sensitivity,
            threshold=threshold,
            dynamic_range=dynamic_range,
            kd=kd,
            response_type=response_type,
            circuit_type='threshold',
            hill_coefficient=hill_coefficient,
            off_level=off_level,
            on_level=on_level,
            **kwargs
        )
        
        self.hill_coefficient = hill_coefficient
        self.off_level = off_level
        self.on_level = on_level
    
    def measure(self, 
                sclerostin: np.ndarray, 
                time: np.ndarray,
                **kwargs) -> np.ndarray:
        """
        Measure sclerostin with digital threshold response.
        
        Args:
            sclerostin: Sclerostin concentration (nM)
            time: Time array
        
        Returns:
            Digital sensor signal (bimodal distribution)
        """
        # Hill function with high cooperativity
        activation = (sclerostin ** self.hill_coefficient) / \
                    (self.kd ** self.hill_coefficient + sclerostin ** self.hill_coefficient)
        
        # Map to OFF/ON levels
        signal = self.off_level + (self.on_level - self.off_level) * activation
        
        # Apply saturation
        signal = self.apply_saturation(signal)
        
        return signal


class RatiometricSensor(Biosensor):
    """
    Ratiometric dual-analyte sensor.
    
    Mechanism:
    - Measures ratio of two analytes (e.g., sclerostin / OPG)
    - Provides disease discrimination (PMO: high RANKL/OPG)
    - Self-calibrating (ratio cancels common-mode drift)
    
    References:
    - Grynkiewicz et al. (1985). A new generation of Ca2+ indicators. J Biol Chem.
      PMID: 3838314
    - Boyce & Xing (2008). RANKL/OPG ratio determines bone remodeling fate
    """
    
    def __init__(self, 
                 sensitivity: float = 1.0,
                 threshold: float = 1.5,
                 dynamic_range: Tuple[float, float] = (0.0, 5.0),
                 kd: float = 1.0,
                 reference_kd: float = 1.0,
                 ratio_exponent: float = 1.0,
                 response_type: str = 'linear',
                 **kwargs):
        """
        Initialize ratiometric sensor.
        
        Args:
            reference_kd: Kd for reference analyte binding
            ratio_exponent: Power for ratio calculation
        """
        super().__init__(
            sensitivity=sensitivity,
            threshold=threshold,
            dynamic_range=dynamic_range,
            kd=kd,
            response_type=response_type,
            circuit_type='ratiometric',
            reference_kd=reference_kd,
            ratio_exponent=ratio_exponent,
            **kwargs
        )
        
        self.reference_kd = reference_kd
        self.ratio_exponent = ratio_exponent
    
    def measure(self, 
                sclerostin: np.ndarray, 
                time: np.ndarray,
                rankl: Optional[np.ndarray] = None,
                opg: Optional[np.ndarray] = None,
                **kwargs) -> np.ndarray:
        """
        Measure sclerostin ratiometrically with RANKL or OPG.
        
        Args:
            sclerostin: Sclerostin concentration (nM)
            time: Time array
            rankl: RANKL concentration (pM), if used as denominator
            opg: OPG concentration (pM), if used as denominator
        
        Returns:
            Ratiometric sensor signal
        """
        # Choose reference analyte (prefer RANKL if both provided)
        if rankl is not None:
            reference = rankl / 1000.0  # Convert pM to nM for scaling
            ref_kd = self.reference_kd
        elif opg is not None:
            reference = opg / 1000.0
            ref_kd = self.reference_kd
        else:
            # Fallback: use constant reference
            reference = np.ones_like(sclerostin)
            ref_kd = 1.0
        
        # Compute ratio (with saturation avoidance)
        epsilon = 1e-6  # Prevent division by zero
        ratio = sclerostin / (reference + epsilon)
        
        # Apply exponent
        ratio_signal = ratio ** self.ratio_exponent
        
        # Apply response curve
        response = self.get_response_curve(ratio_signal)
        
        # Scale by sensitivity
        signal = self.sensitivity * response
        
        # Apply saturation
        signal = self.apply_saturation(signal)
        
        return signal


# ============================================================================
# BIOSENSOR FACTORY
# ============================================================================

def create_biosensor(config: Dict) -> Biosensor:
    """
    Factory function to create biosensor from configuration dictionary.
    
    Args:
        config: Dictionary with 'circuit_type' key and sensor parameters
    
    Returns:
        Biosensor instance
    
    Raises:
        ValueError: If circuit_type is unknown
    """
    circuit_type = config.get('circuit_type')
    
    if circuit_type == 'direct_binding':
        return DirectBindingSensor(**{k: v for k, v in config.items() 
                                     if k != 'circuit_type'})
    elif circuit_type == 'amplifying':
        return AmplifyingSensor(**{k: v for k, v in config.items() 
                                  if k != 'circuit_type'})
    elif circuit_type == 'threshold':
        return ThresholdSensor(**{k: v for k, v in config.items() 
                                 if k != 'circuit_type'})
    elif circuit_type == 'ratiometric':
        return RatiometricSensor(**{k: v for k, v in config.items() 
                                   if k != 'circuit_type'})
    else:
        raise ValueError(f"Unknown circuit type: {circuit_type}")


def generate_random_biosensor_config(circuit_type: Optional[str] = None,
                                     seed: Optional[int] = None) -> Dict:
    """
    Generate random biosensor configuration for dataset diversity.
    
    Args:
        circuit_type: Specific circuit type, or None for random choice
        seed: Random seed
    
    Returns:
        Configuration dictionary
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Choose random circuit type if not specified
    if circuit_type is None:
        circuit_type = np.random.choice([
            'direct_binding', 'amplifying', 'threshold', 'ratiometric'
        ])
    
    # Sample parameters from reasonable ranges (log-uniform for wide ranges)
    config = {
        'circuit_type': circuit_type,
        'sensitivity': np.random.lognormal(mean=3.0, sigma=0.8),  # Even higher sensitivity
        'threshold': np.random.uniform(0.005, 0.05),  # Lower threshold
        'dynamic_range': (0.0, np.random.uniform(0.5, 10.0)),  # Tighter range
        'kd': np.random.lognormal(mean=np.log(0.03), sigma=0.8),  # Lower Kd for better binding
        'response_type': np.random.choice(['linear', 'hill', 'michaelis_menten'])
    }
    
    # Circuit-specific parameters
    if circuit_type == 'direct_binding':
        # Calculate the value of kon first
        new_kon = np.random.lognormal(mean=np.log(1e6), sigma=1.0)

        # Now use new_kon to define both 'kon' and 'koff' in one update call
        config.update({
            'receptor_concentration': np.random.uniform(0.1, 5.0),  # Lower for nM range
            'kon': new_kon,
            'koff': config['kd'] * new_kon 
        })
    
    elif circuit_type == 'amplifying':
        config.update({
            'receptor_concentration': np.random.uniform(0.1, 5.0),    # Lower
            'enzyme_concentration': np.random.uniform(2.0, 20.0),     # Lower
            'amplification_gain': np.random.uniform(100.0, 5000.0),   # Higher gain
            'response_time': np.random.uniform(10.0, 300.0)
        })
    
    elif circuit_type == 'threshold':
        config.update({
            'hill_coefficient': np.random.uniform(2.0, 6.0),
            'off_level': np.random.uniform(0.01, 0.5),
            'on_level': np.random.uniform(5.0, 20.0)
        })
    
    elif circuit_type == 'ratiometric':
        config.update({
            'reference_kd': np.random.lognormal(mean=0.0, sigma=1.5),
            'ratio_exponent': np.random.uniform(0.5, 2.0)
        })
    
    return config