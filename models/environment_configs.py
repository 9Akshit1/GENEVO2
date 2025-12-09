"""
Environment configuration presets for different clinical scenarios.

This module defines parameter overrides for three scenarios:
1. Healthy baseline
2. Post-Menopausal Osteoporosis (PMO)
3. Chronic Kidney Disease-Mineral Bone Disorder (CKD-MBD)

All parameter values are based on published literature.
"""

import numpy as np
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class EnvironmentConfig:
    """
    Configuration class for bone microenvironment scenarios.
    """
    
    def __init__(self, name: str, param_overrides: Dict[str, float], 
                 description: str = ""):
        """
        Initialize environment configuration.
        
        Args:
            name: Scenario name (healthy, pmo, ckd_mbd)
            param_overrides: Dictionary of parameter names to override values
            description: Human-readable description of the scenario
        """
        self.name = name
        self.param_overrides = param_overrides
        self.description = description
        
        # Validate parameters
        self._validate()
        
        logger.debug(f"Created environment config: {name}")
    
    def _validate(self):
        """Validate parameter ranges are biologically plausible."""
        # Check for negative values that should be positive
        for param, value in self.param_overrides.items():
            if param.startswith('k_') and value < 0:
                raise ValueError(f"Rate constant {param} cannot be negative: {value}")
            
            # Check estrogen range
            if param == 'Estrogen' and not (0.0 <= value <= 10.0):
                logger.warning(f"Estrogen value {value} nM outside typical range [0-10]")
            
            # Check PTH range
            if param == 'PTH' and not (0.0 <= value <= 500.0):
                logger.warning(f"PTH value {value} pM outside typical range [0-500]")
    
    def get_params(self) -> Dict[str, float]:
        """Return parameter overrides."""
        return self.param_overrides.copy()
    
    def apply_variability(self, variability: float = 0.15) -> Dict[str, float]:
        """
        Apply random variability to parameters for dataset diversity.
        
        Args:
            variability: Fractional variability (0.15 = ±15%)
        
        Returns:
            Dictionary with varied parameters
        """
        varied_params = {}
        for param, value in self.param_overrides.items():
            # Multiplicative lognormal noise
            factor = np.random.lognormal(mean=0.0, sigma=variability)
            varied_params[param] = value * factor
        
        return varied_params


# ============================================================================
# SCENARIO DEFINITIONS
# ============================================================================

def get_healthy_config() -> EnvironmentConfig:
    """
    Healthy baseline configuration.
    
    References:
    - Estrogen: 0.3-1.5 nM in healthy adults (Mödder et al., 2011)
    - PTH: 10-65 pM normal range (Souberbielle et al., 2010)
    - RANKL/OPG: Balanced ratio ~1:2 (Boyce & Xing, 2008)
    """
    params = {
        'Estrogen': 1.0,              # Normal estrogen [nM]
        'PTH': 50.0,                  # Normal PTH [pM]
        'k_prod_Scl': 0.0000035,      # Baseline sclerostin production
        'k_prod_RANKL': 0.00001,     # Balanced RANKL
        'k_prod_OPG': 0.0001,        # Balanced OPG
        'Sclerostin_bone': 0.05,     # ~50 pmol/L steady-state
        'RANKL_bone': 0.5,           # ~0.5 pM
        'OPG_bone': 5.0,             # ~5 pM, ratio 1:10 (healthy)
    }
    
    return EnvironmentConfig(
        name='healthy',
        param_overrides=params,
        description="Healthy bone homeostasis with normal hormone levels"
    )


def get_pmo_config() -> EnvironmentConfig:
    """
    Post-Menopausal Osteoporosis (PMO) configuration.
    
    Key features:
    - Low estrogen (postmenopausal state)
    - Elevated RANKL (increased bone resorption)
    - Decreased OPG (reduced osteoclast inhibition)
    - Slightly elevated sclerostin
    
    References:
    - Eastell et al. (2016). Postmenopausal osteoporosis. Nat Rev Dis Primers.
      doi:10.1038/nrdp.2016.69
    - Eghbali-Fatourechi et al. (2003). Role of RANK ligand in mediating 
      increased bone resorption in early postmenopausal women. JCEM.
      doi:10.1210/jc.2002-021215
    - Mödder et al. (2011). Regulation of circulating sclerostin levels by 
      sex steroids in women and men. J Bone Miner Res. doi:10.1002/jbmr.128
    """
    params = {
        'Estrogen': 0.2,              # Low postmenopausal estrogen
        'PTH': 55.0,                  # Slightly elevated PTH
        'k_prod_Scl': 0.00000525,     # +50% sclerostin (Mödder 2011)
        'k_prod_RANKL': 0.000025,    # 2.5x RANKL (Eghbali-Fatourechi 2003)
        'k_prod_OPG': 0.000053,      # -47% OPG (Hofbauer 1999)
        'Sclerostin_bone': 0.075,    # ~75 pmol/L (+50% vs healthy)
        'RANKL_bone': 1.2,           # ~1.2 pM (elevated)
        'OPG_bone': 2.5,             # ~2.5 pM (reduced)
                                      # RANKL:OPG ratio becomes ~1:2 (unhealthy)
    }
    
    return EnvironmentConfig(
        name='pmo',
        param_overrides=params,
        description="Post-menopausal osteoporosis with estrogen deficiency"
    )


def get_ckd_mbd_config() -> EnvironmentConfig:
    """
    Chronic Kidney Disease-Mineral and Bone Disorder (CKD-MBD) configuration.
    
    Key features:
    - Highly dysregulated (elevated) PTH
    - Markedly elevated sclerostin
    - Abnormal mineral metabolism
    - Disrupted bone remodeling
    
    References:
    - Quarles (2012). Role of FGF23 in vitamin D and phosphate metabolism. 
      Pediatr Nephrol. doi:10.1007/s00467-011-1838-5
    - Cejka et al. (2011). Sclerostin serum levels correlate positively with 
      bone mineral density and microarchitecture in haemodialysis patients. 
      Nephrol Dial Transplant. doi:10.1093/ndt/gfr270
    - Brandenburg et al. (2010). Relationship between sclerostin and cardiovascular 
      calcification in hemodialysis patients. Kidney Int. doi:10.1038/ki.2010.219
    - Moe et al. (2006). Definition, evaluation, and classification of renal 
      osteodystrophy. Kidney Int. doi:10.1038/sj.ki.5000414
    """
    params = {
        'Estrogen': 0.8,              # Near-normal
        'PTH': 250.0,                 # 5x elevated (severe CKD)
        'k_prod_Scl': 0.0000105,      # 3x sclerostin (Cejka 2011)
        'k_deg_Scl': 0.00005,         # Reduced clearance in CKD (50% slower)
        'k_prod_RANKL': 0.000015,    # Moderately elevated
        'k_prod_OPG': 0.00009,       # Slightly reduced
        'Sclerostin_bone': 0.15,     # ~150 pmol/L (3x healthy)
        'RANKL_bone': 0.8,           # Moderately elevated
        'OPG_bone': 4.5,             # Slightly reduced
        'MineralIon': 3.2,           # Elevated Ca/PO4
        'k_prod_Mineral': 0.15,      # Increased dysregulation
    }
    
    return EnvironmentConfig(
        name='ckd_mbd',
        param_overrides=params,
        description="CKD-MBD with severe PTH elevation and sclerostin dysregulation"
    )


# ============================================================================
# CONFIGURATION REGISTRY
# ============================================================================

SCENARIO_CONFIGS = {
    'healthy': get_healthy_config(),
    'pmo': get_pmo_config(),
    'ckd_mbd': get_ckd_mbd_config(),
}


def get_config(scenario_name: str) -> EnvironmentConfig:
    """
    Retrieve configuration for a scenario.
    
    Args:
        scenario_name: One of 'healthy', 'pmo', 'ckd_mbd'
    
    Returns:
        EnvironmentConfig object
    
    Raises:
        ValueError: If scenario name is not recognized
    """
    if scenario_name not in SCENARIO_CONFIGS:
        raise ValueError(
            f"Unknown scenario: {scenario_name}. "
            f"Must be one of {list(SCENARIO_CONFIGS.keys())}"
        )
    
    return SCENARIO_CONFIGS[scenario_name]


def list_scenarios() -> List[str]:
    """Return list of available scenario names."""
    return list(SCENARIO_CONFIGS.keys())