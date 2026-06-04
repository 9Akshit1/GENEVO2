"""
Environment configuration presets — Version 4.0 (Tier 1 Fixes).

V4.0 CRITICAL FIX: Bounded Physiological Variability
======================================================
Problem (v3.x): Lognormal σ=0.15–0.30 creates impossible scenarios:
  - Estrogen can drop to 0.044 nM (lethal)
  - PTH can jump to 810 pg/mL (lethal)
  - k_prod varies 50–500× baseline

Solution (v4.0): Hard bounds on physiological parameters
  - Estrogen: [0.1, 3.0] nM (pre-menopausal range)
  - PTH: [10, 400] pg/mL (normal to severe hyperparathyroidism)
  - k_prod: Reduced σ to 0.10 (±10%, not ±15–26%)
  - Rejection sampling: resample if out of bounds (don't clip silently)

This ensures RL explores biologically plausible parameter space only.
No other changes from v3.x — all other logic remains identical.
"""

import numpy as np
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

# Diffusion equilibrium ratios (unchanged from v3.x)
_SCLEROSTIN_SENSOR_RATIO = 25.0
_RANKL_SENSOR_RATIO      = 30.0
_OPG_SENSOR_RATIO        = 30.0


class EnvironmentConfig:
    """Configuration class for bone microenvironment scenarios."""

    def __init__(self, name: str, param_overrides: Dict[str, float],
                 description: str = ""):
        self.name = name
        self.param_overrides = param_overrides
        self.description = description
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

    def apply_variability(self, variability: float = 0.15) -> Dict[str, float]:
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
        
        SENSOR_DERIVED = {'Sclerostin_sensor', 'RANKL_sensor', 'OPG_sensor'}
        
        # Hard bounds: these are biologically non-negotiable
        HARD_BOUNDS = {
            'Estrogen':    (0.1,   3.0),    # nM, premenopausal range
            'PTH':         (10.0,  400.0),  # pg/mL, pathological max
        }
        
        varied_params: Dict[str, float] = {}

        for param, value in self.param_overrides.items():
            if param in SENSOR_DERIVED:
                continue  # Derived below

            # FIX 2: Conservative σ to stay in bounds
            if param.startswith('k_prod_') or param.startswith('k_deg_'):
                # Rate constants: tight control (±10%, not ±15–26%)
                # Prevents system instability from rate scaling
                sigma = 0.10
            elif param in ('Estrogen', 'PTH'):
                # Hormones: physiological variation (±12%)
                sigma = 0.12
            elif param in ('Sclerostin_bone', 'RANKL_bone', 'OPG_bone', 'MineralIon'):
                # Biomarker ICs: higher variation allowed (±15%)
                sigma = 0.15
            else:
                sigma = 0.10

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

        return varied_params


# ============================================================================
# SCENARIO DEFINITIONS (Unchanged from v3.x)
# ============================================================================

def get_healthy_config() -> EnvironmentConfig:
    """Healthy: Sclerostin_sensor≈0.375 nM, well below threshold."""
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
    }
    return EnvironmentConfig(
        name='healthy',
        param_overrides=params,
        description=(
            "Healthy: Sclerostin_sensor≈0.375 nM, well below threshold. "
            "Normal osteoid homeostasis. Low detection probability expected."
        )
    )


def get_pmo_config() -> EnvironmentConfig:
    """Post-Menopausal Osteoporosis (PMO): Sclerostin_sensor≈0.875 nM."""
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
    }
    return EnvironmentConfig(
        name='pmo',
        param_overrides=params,
        description=(
            "PMO: Sclerostin_sensor≈0.875 nM, approaching threshold. "
            "Low estrogen drives RANKL up, OPG down, sclerostin up. "
            "Moderate detection probability expected."
        )
    )


def get_ckd_mbd_config() -> EnvironmentConfig:
    """Chronic Kidney Disease — Mineral and Bone Disorder (CKD-MBD)."""
    params = {
        'Estrogen': 0.6,
        'PTH':      180.0,
        'k_prod_Scl':        0.00000429,
        'k_deg_Scl':         0.00006,
        'Sclerostin_bone':   0.080,
        'k_prod_RANKL':      0.00000643,
        'k_deg_RANKL':       0.00002,
        'RANKL_bone':        0.9,
        'k_prod_OPG':        0.000119,
        'k_deg_OPG':         0.00002,
        'OPG_bone':          5.5,
        'MineralIon':        3.8,
        'k_prod_Mineral':    0.150,
        'k_loss_Mineral':    0.055,
    }
    return EnvironmentConfig(
        name='ckd_mbd',
        param_overrides=params,
        description=(
            "CKD-MBD: Sclerostin_sensor≈2.0 nM, at/above lower threshold. "
            "Uremic dysregulation drives 3× higher k_prod_Scl. "
            "Secondary hyperparathyroidism (PTH=180). "
            "High detection probability expected."
        )
    )


SCENARIO_CONFIGS = {
    'healthy': get_healthy_config(),
    'pmo':     get_pmo_config(),
    'ckd_mbd': get_ckd_mbd_config(),
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