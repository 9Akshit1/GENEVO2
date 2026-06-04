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
    Six-dimensional search space for biosensor design optimization.

    Parameters:
    1. biosensor_type (categorical): {direct_binding, amplifying}
    2. kd_nm (continuous, log): [0.1, 10.0] nM
    3. sensitivity (continuous, log): [0.5, 5.0]
    4. response_time_s (continuous, log): [100, 3600] s
    5. noise_preset (categorical): {low, medium, high}
    6. target_scenario (categorical): {healthy, pmo, ckd_mbd}
    """

    # Sclerostin concentrations at sensor (biochemical basis)
    SCLEROSTIN_CONC_NM = {
        "healthy": 0.375,
        "pmo": 0.875,
        "ckd_mbd": 2.0,
    }

    # Noise presets from models/noise.py
    NOISE_FRACTIONS = {
        "low": {"additive": 0.01, "multiplicative": 0.005},
        "medium": {"additive": 0.02, "multiplicative": 0.01},
        "high": {"additive": 0.03, "multiplicative": 0.015},
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
                values=["direct_binding", "amplifying"],
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
            "response_time_s": ParameterBounds(
                name="response_time_s",
                param_type="continuous",
                lower=100,
                upper=3600,
                scale="log",
            ),
            "noise_preset": ParameterBounds(
                name="noise_preset",
                param_type="categorical",
                values=["low", "medium", "high"],
            ),
            "target_scenario": ParameterBounds(
                name="target_scenario",
                param_type="categorical",
                values=["pmo", "ckd_mbd"],  # Removed "healthy" - not a disease scenario, always has DR≈0.0
            ),
        }

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
        # response_time_s is only used for amplifying sensors
        if config["biosensor_type"] == "direct_binding" and "response_time_s" in config:
            if config["response_time_s"] < 1000:
                return False, "response_time_s only applies to amplifying sensors"

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
