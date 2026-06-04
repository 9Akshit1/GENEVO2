#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Physics-based forward model for SNR estimation.

Estimates SNR_dB from biosensor design parameters without running full simulation.
Based on Langmuir and exponential kinetics formulas from models/biosensors.py
and noise model from models/noise.py.
"""

import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class PhysicsForwardModel:
    """
    Lightweight physics model for SNR estimation.

    Uses equilibrium biosensor equations and noise fractions to estimate
    expected SNR given design parameters and noise conditions.
    """

    # Sclerostin concentrations at biosensor sensor (nM)
    SCLEROSTIN_CONC_NM = {
        "healthy": 0.375,
        "pmo": 0.875,
        "ckd_mbd": 2.0,
    }

    # Noise fractions from models/noise.py (V4.1)
    NOISE_FRACTIONS = {
        "low": {
            "additive": 0.01,
            "multiplicative": 0.005,
            "shot": 0.0005,
        },
        "medium": {
            "additive": 0.02,
            "multiplicative": 0.01,
            "shot": 0.0015,
        },
        "high": {
            "additive": 0.03,
            "multiplicative": 0.015,
            "shot": 0.003,
        },
    }

    # SNR-dependent processing delays (from biosensor_engine.py)
    PROCESSING_DELAYS_S = {
        25: 25,
        15: 75,
        5: 175,
        -np.inf: 400,
    }

    def __init__(self):
        """Initialize the physics model."""
        pass

    def compute_signal(
        self,
        kd_nm: float,
        sensitivity: float,
        sclerostin_conc_nm: float,
        biosensor_type: str,
        response_time_s: float = None,
    ) -> float:
        """
        Compute biosensor signal using Langmuir (direct binding) or exponential (amplifying) kinetics.

        Args:
            kd_nm: Dissociation constant [nM]
            sensitivity: Signal transduction efficiency (unitless)
            sclerostin_conc_nm: Sclerostin concentration at sensor [nM]
            biosensor_type: 'direct_binding' or 'amplifying'
            response_time_s: Time constant for amplifying sensor [s]

        Returns:
            Expected signal amplitude (unitless)
        """
        # Langmuir occupancy at equilibrium
        occupancy = sclerostin_conc_nm / (kd_nm + sclerostin_conc_nm)

        if biosensor_type == "direct_binding":
            # Direct binding: signal proportional to occupancy
            signal = sensitivity * occupancy
        elif biosensor_type == "amplifying":
            # Amplifying sensor: exponential rise to steady state
            # At quasi-steady state (t >> response_time_s):
            # signal = sensitivity * occupancy * (1 - exp(-t/response_time_s))
            # For SNR estimation, assume near steady state (e.g., t = 2-3 * response_time_s)
            # Using factor ~0.9 for practical steady state
            steady_state_factor = 0.9
            signal = sensitivity * occupancy * steady_state_factor
        else:
            raise ValueError(f"Unknown biosensor type: {biosensor_type}")

        return signal

    def compute_noise_std(
        self,
        signal_amplitude: float,
        noise_preset: str,
    ) -> float:
        """
        Compute noise standard deviation from signal amplitude and noise preset.

        Combines additive, multiplicative, and shot noise contributions.

        Args:
            signal_amplitude: Expected signal level (unitless)
            noise_preset: 'low', 'medium', or 'high'

        Returns:
            Total noise standard deviation
        """
        if noise_preset not in self.NOISE_FRACTIONS:
            raise ValueError(f"Unknown noise preset: {noise_preset}")

        noise_params = self.NOISE_FRACTIONS[noise_preset]

        # Additive noise (constant std)
        additive_std = noise_params["additive"] * signal_amplitude

        # Multiplicative noise (proportional to signal)
        multiplicative_std = noise_params["multiplicative"] * signal_amplitude

        # Shot noise (Poisson-like, sqrt of signal)
        shot_std = noise_params["shot"] * np.sqrt(np.abs(signal_amplitude) + 1e-10)

        # Combine in quadrature
        total_noise_std = np.sqrt(
            additive_std ** 2 + multiplicative_std ** 2 + shot_std ** 2
        )

        return total_noise_std

    def compute_snr_db(
        self,
        signal_delta: float,
        noise_std: float,
    ) -> float:
        """
        Compute SNR in dB from signal and noise.

        Args:
            signal_delta: AC signal amplitude (signal change between conditions)
            noise_std: Total noise standard deviation

        Returns:
            SNR in dB
        """
        if noise_std < 1e-12:
            return 50.0  # Cap at 50 dB if noise is negligible

        snr_linear = (signal_delta + 1e-10) / (noise_std + 1e-10)
        snr_db = 10.0 * np.log10(snr_linear)

        # Clip to realistic range
        snr_db = np.clip(snr_db, -50.0, 50.0)

        return snr_db

    def estimate_snr(
        self,
        biosensor_type: str,
        kd_nm: float,
        sensitivity: float,
        response_time_s: float,
        noise_preset: str,
        target_scenario: str,
    ) -> float:
        """
        Estimate SNR_dB for a given biosensor design and environmental condition.

        The SNR is computed as the ratio of AC signal power (signal change from healthy to disease)
        to noise power. This reflects the temporal signal-to-noise ratio in a typical detection scenario.

        Args:
            biosensor_type: 'direct_binding' or 'amplifying'
            kd_nm: Kd [nM]
            sensitivity: Sensitivity [unitless]
            response_time_s: Response time [s], only used for amplifying
            noise_preset: 'low', 'medium', 'high'
            target_scenario: 'pmo', 'ckd_mbd', or 'both'
                If 'both', use average SNR across both disease scenarios

        Returns:
            Estimated SNR in dB
        """
        # Get disease scenario concentrations
        if target_scenario == "both":
            scenarios = ["pmo", "ckd_mbd"]
        else:
            scenarios = [target_scenario]

        snr_values = []

        for scenario in scenarios:
            # Get sclerostin concentration for this scenario
            disease_conc = self.SCLEROSTIN_CONC_NM[scenario]
            healthy_conc = self.SCLEROSTIN_CONC_NM["healthy"]

            # Compute signals at healthy and disease states
            signal_healthy = self.compute_signal(
                kd_nm=kd_nm,
                sensitivity=sensitivity,
                sclerostin_conc_nm=healthy_conc,
                biosensor_type=biosensor_type,
                response_time_s=response_time_s,
            )

            signal_disease = self.compute_signal(
                kd_nm=kd_nm,
                sensitivity=sensitivity,
                sclerostin_conc_nm=disease_conc,
                biosensor_type=biosensor_type,
                response_time_s=response_time_s,
            )

            # AC signal is the difference (signal change over time)
            signal_delta = abs(signal_disease - signal_healthy)

            # For noise, use the disease signal level (more conservative)
            signal_amplitude = signal_disease

            # Compute noise
            noise_std = self.compute_noise_std(signal_amplitude, noise_preset)

            # Compute SNR
            snr_db = self.compute_snr_db(signal_delta, noise_std)
            snr_values.append(snr_db)

        # Return average SNR across scenarios
        return float(np.mean(snr_values))

    def estimate_snr_matrix(
        self,
        config: Dict,
    ) -> np.ndarray:
        """
        Estimate SNR across all scenario × noise combinations.

        Useful for robustness analysis.

        Args:
            config: Parameter dictionary with keys:
                biosensor_type, kd_nm, sensitivity, response_time_s,
                and optionally noise_preset, target_scenario (both ignored here)

        Returns:
            Array of shape (3 scenarios, 3 noise levels) with SNR estimates
        """
        scenarios = ["healthy", "pmo", "ckd_mbd"]
        noise_presets = ["low", "medium", "high"]

        snr_matrix = np.zeros((len(scenarios), len(noise_presets)))

        for i, scenario in enumerate(scenarios):
            for j, noise_preset in enumerate(noise_presets):
                snr_matrix[i, j] = self.estimate_snr(
                    biosensor_type=config["biosensor_type"],
                    kd_nm=config["kd_nm"],
                    sensitivity=config["sensitivity"],
                    response_time_s=config.get("response_time_s", 500),
                    noise_preset=noise_preset,
                    target_scenario=scenario,
                )

        return snr_matrix

    @staticmethod
    def get_processing_delay(snr_db: float) -> float:
        """
        Get processing delay from SNR using data-driven thresholds.

        Based on SNR-dependent detection delays from biosensor_engine.py.

        Args:
            snr_db: SNR in dB

        Returns:
            Processing delay in seconds
        """
        if snr_db > 25:
            return 25.0
        elif snr_db > 15:
            return 75.0
        elif snr_db > 5:
            return 175.0
        else:
            return 400.0
