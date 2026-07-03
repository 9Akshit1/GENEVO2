#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Biosensor parameter search space for Bayesian Optimization.

Defines scientifically-constrained parameter bounds for biosensor design.
All ranges justified by biochemical literature and simulation data.
"""

import numpy as np
from typing import Tuple, Dict, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ParameterBounds:
    """Bounds for a single parameter."""

    name: str
    param_type: str  # 'continuous', 'categorical', 'integer'
    lower: float = None
    upper: float = None
    values: List[str] = None
    scale: str = "linear"  # or 'log'


class BiosensorSearchSpace:
    """
    Nine-dimensional search space for biosensor design optimization.

    response_time_s is fixed at 600 s (not a search parameter — surrogate PI=0
    because all array biosensors in training data used a constant response time).

    Shared parameters (all types):
    1. biosensor_type (categorical): {array}
    2. kd_nm (continuous, log): [0.1, 10.0] nM  (SOST channel Kd)
    3. sensitivity (continuous, log): [0.5, 5.0]
    4. noise_preset (categorical): {realistic}
    5. target_scenario (categorical): {pmo, ckd_mbd}

    Array-specific parameters (zeroed out for non-array types in the objective):
    6. kd_ctx_nm  (continuous, log): [0.1, 10.0] nM  (CTX channel Kd)
    7. kd_p1np_nm (continuous, log): [0.1, 10.0] nM  (P1NP channel Kd)
    8. w_ctx  (continuous, linear): [0.001, 0.49]  (CTX weight, pre-normalisation)
    9. w_p1np (continuous, linear): [0.001, 0.49]  (P1NP weight, pre-normalisation)
    """

    # Fixed non-search parameter: all array biosensors operate with this
    # response window (clinically defined maximum implant response window).
    RESPONSE_TIME_S: float = 600.0

    # Sclerostin concentrations at sensor (biochemical basis) v5.2 calibration
    SCLEROSTIN_CONC_NM = {
        "healthy": 0.375,
        "pmo": 0.875,
        "ckd_mbd": 1.125,   # ODE CKD: Sclerostin_bone=0.045 * ratio 25
    }

    # Noise preset: deployment conditions only (13.6 dB, matches models/noise.py "realistic")
    NOISE_FRACTIONS = {
        "realistic": {"additive": 0.18, "multiplicative": 0.10},
    }

    def __init__(self):
        """Initialize the search space with scientifically justified bounds."""
        self.parameters = self._define_parameters()
        self.n_params = len(self.parameters)

    def _define_parameters(self) -> Dict[str, ParameterBounds]:
        """
        Define parameter bounds with scientific justification.

        Returns:
            Dictionary mapping parameter names to ParameterBounds objects
        """
        return {
            "biosensor_type": ParameterBounds(
                name="biosensor_type",
                param_type="categorical",
                values=["array"],
            ),
            "kd_nm": ParameterBounds(
                name="kd_nm",
                param_type="continuous",
                lower=0.1,
                upper=10.0,
                scale="log",
            ),
            "sensitivity": ParameterBounds(
                name="sensitivity",
                param_type="continuous",
                lower=0.5,
                upper=5.0,
                scale="log",
            ),
            "noise_preset": ParameterBounds(
                name="noise_preset",
                param_type="categorical",
                values=["realistic"],
            ),
            "target_scenario": ParameterBounds(
                name="target_scenario",
                param_type="categorical",
                values=["pmo", "ckd_mbd"],  # Removed "healthy" - not a disease scenario, always has DR≈0.0
            ),
            # Array-specific parameters (zeroed for non-array types in objective function)
            "kd_ctx_nm": ParameterBounds(
                name="kd_ctx_nm",
                param_type="continuous",
                lower=0.1,
                upper=10.0,
                scale="log",
            ),
            "kd_p1np_nm": ParameterBounds(
                name="kd_p1np_nm",
                param_type="continuous",
                lower=0.1,
                upper=10.0,
                scale="log",
            ),
            "w_ctx": ParameterBounds(
                name="w_ctx",
                param_type="continuous",
                lower=0.001,  # Phase 2.1: near-zero allowed → implicit channel dropping
                upper=0.49,   # w_ctx + w_p1np ≤ 0.98 guaranteed; w_scl ≥ 0.02
                scale="linear",
            ),
            "w_p1np": ParameterBounds(
                name="w_p1np",
                param_type="continuous",
                lower=0.001,  # Phase 2.1: near-zero allowed → implicit channel dropping
                upper=0.49,
                scale="linear",
            ),
        }

    # ── Phase 2.1 topology helpers ───────────────────────────────────────────

    def enforce_topology(self, config: Dict, topology: str) -> Dict:
        """
        Apply topology constraint to a config dict.

        topology='2ch' → w_ctx forced to 0, remaining weight renormalized to SOST+P1NP.
        topology='3ch' → config unchanged.
        """
        if topology == "2ch":
            config = dict(config)
            w_p1np = max(float(config.get("w_p1np", 0.1)), 0.001)
            # SOST gets remaining weight (w_scl = 1 − w_p1np via normalization in ArrayBiosensor)
            config["w_ctx"] = 0.0
            config["w_p1np"] = min(w_p1np, 0.999)  # ensure SOST still contributes
        return config

    def sample_random_for_topology(
        self, topology: str, rng: np.random.RandomState = None
    ) -> Dict:
        """Sample a random config constrained to the given topology."""
        config = self.sample_random(rng)
        return self.enforce_topology(config, topology)

    def get_bounds(self) -> Dict[str, Tuple]:
        """
        Get bounds for all continuous parameters.

        Returns:
            Dictionary {param_name: (lower, upper)} for continuous params
        """
        bounds = {}
        for name, param in self.parameters.items():
            if param.param_type == "continuous":
                bounds[name] = (param.lower, param.upper)
        return bounds

    def get_categorical_params(self) -> Dict[str, List[str]]:
        """
        Get categorical parameter values.

        Returns:
            Dictionary {param_name: list_of_values}
        """
        categorical = {}
        for name, param in self.parameters.items():
            if param.param_type == "categorical":
                categorical[name] = param.values
        return categorical

    def normalize_continuous(self, param_name: str, value: float) -> float:
        """
        Normalize a continuous parameter to [0, 1].

        Args:
            param_name: Name of the parameter
            value: Raw parameter value

        Returns:
            Normalized value in [0, 1]
        """
        param = self.parameters[param_name]
        if param.param_type != "continuous":
            raise ValueError(f"{param_name} is not continuous")

        if param.scale == "log":
            # Log-uniform scaling
            log_lower = np.log10(param.lower)
            log_upper = np.log10(param.upper)
            log_value = np.log10(value)
            return (log_value - log_lower) / (log_upper - log_lower)
        else:
            # Linear scaling
            return (value - param.lower) / (param.upper - param.lower)

    def denormalize_continuous(self, param_name: str, norm_value: float) -> float:
        """
        Denormalize a continuous parameter from [0, 1] to original scale.

        Args:
            param_name: Name of the parameter
            norm_value: Normalized value in [0, 1]

        Returns:
            Raw parameter value
        """
        param = self.parameters[param_name]
        if param.param_type != "continuous":
            raise ValueError(f"{param_name} is not continuous")

        if param.scale == "log":
            # Log-uniform inverse
            log_lower = np.log10(param.lower)
            log_upper = np.log10(param.upper)
            log_value = log_lower + norm_value * (log_upper - log_lower)
            return 10.0 ** log_value
        else:
            # Linear inverse
            return param.lower + norm_value * (param.upper - param.lower)

    def encode_categorical(self, param_name: str, value: str) -> int:
        """
        Encode a categorical value to integer.

        Args:
            param_name: Name of the parameter
            value: Categorical value

        Returns:
            Integer encoding [0, num_values-1]
        """
        param = self.parameters[param_name]
        if param.param_type != "categorical":
            raise ValueError(f"{param_name} is not categorical")
        return param.values.index(value)

    def decode_categorical(self, param_name: str, code: int) -> str:
        """
        Decode integer to categorical value.

        Args:
            param_name: Name of the parameter
            code: Integer encoding

        Returns:
            Categorical value
        """
        param = self.parameters[param_name]
        if param.param_type != "categorical":
            raise ValueError(f"{param_name} is not categorical")
        return param.values[code]

    def vector_to_dict(self, x: np.ndarray) -> Dict:
        """
        Convert a parameter vector to a dictionary.

        For GP optimization, we use integer encoding for categorical variables
        and normalized continuous variables in [0, 1].

        Args:
            x: Vector of shape (n_params,) with encoded values

        Returns:
            Dictionary with parameter names and actual values
        """
        param_list = list(self.parameters.keys())
        config = {}

        for i, param_name in enumerate(param_list):
            param = self.parameters[param_name]
            value = x[i]

            if param.param_type == "categorical":
                config[param_name] = self.decode_categorical(
                    param_name, int(round(value * (len(param.values) - 1)))
                )
            else:  # continuous
                config[param_name] = self.denormalize_continuous(param_name, value)

        config["response_time_s"] = self.RESPONSE_TIME_S
        return config

    def dict_to_vector(self, config: Dict) -> np.ndarray:
        """
        Convert a parameter dictionary to a vector.

        Args:
            config: Dictionary with parameter names and values

        Returns:
            Vector of shape (n_params,) with encoded values
        """
        param_list = list(self.parameters.keys())
        x = np.zeros(self.n_params, dtype=np.float32)

        for i, param_name in enumerate(param_list):
            param = self.parameters[param_name]
            value = config[param_name]

            if param.param_type == "categorical":
                code = self.encode_categorical(param_name, value)
                x[i] = code / (len(param.values) - 1)
            else:  # continuous
                x[i] = self.normalize_continuous(param_name, value)

        return x

    def sample_random(self, rng: np.random.RandomState = None) -> Dict:
        """
        Sample a random point in the search space.

        Args:
            rng: Random number generator (default: np.random)

        Returns:
            Dictionary with parameter values
        """
        if rng is None:
            rng = np.random.RandomState()

        config = {}

        for param_name, param in self.parameters.items():
            if param.param_type == "categorical":
                config[param_name] = rng.choice(param.values)
            else:  # continuous
                if param.scale == "log":
                    # Log-uniform sampling
                    log_lower = np.log10(param.lower)
                    log_upper = np.log10(param.upper)
                    log_value = rng.uniform(log_lower, log_upper)
                    config[param_name] = 10.0 ** log_value
                else:
                    # Linear uniform sampling
                    config[param_name] = rng.uniform(param.lower, param.upper)

        config["response_time_s"] = self.RESPONSE_TIME_S
        return config

    def is_valid(self, config: Dict) -> Tuple[bool, str]:
        """
        Check if a configuration is valid.

        Enforces domain-specific constraints.

        Args:
            config: Parameter dictionary

        Returns:
            Tuple (is_valid, error_message)
        """
        # Kd must maintain signal separation
        kd = config["kd_nm"]
        if kd < 0.05 or kd > 20.0:
            return (
                False,
                f"kd_nm={kd} outside valid range [0.05, 20.0]",
            )

        return True, ""

    def summary(self) -> str:
        """Return a summary of the search space."""
        summary = "Biosensor Search Space:\n"
        summary += "=" * 60 + "\n"

        for name, param in self.parameters.items():
            if param.param_type == "continuous":
                summary += f"{name:20s} [continuous, {param.scale:6s}]: "
                summary += f"[{param.lower:8.3f}, {param.upper:8.3f}]\n"
            else:
                summary += f"{name:20s} [categorical]: {param.values}\n"

        return summary
