#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Objective function for BO - RL-Based Surrogate Approach (v2)

KEY CHANGE: Uses RL-style features [SNR, biosensor_type, noise] instead of raw parameters.

Process:
1. Take candidate configuration (biosensor params + environment)
2. Compute SNR using physics_forward_model (actual simulator output)
3. Encode categorical variables
4. Build feature vector [SNR, biosensor_type_enc, noise_enc]
5. Pass to v2_rl surrogates
6. Return composite objective

This approach is MORE RELIABLE because:
- SNR is computed from physics, not regressed
- Surrogates trained on [SNR, biosensor_type, noise] which are reliable
- FNR model improved from R²=-0.44 to R²=0.58
- Avoids trying to predict from raw parameters (kd, sensitivity) which aren't in training data
"""

import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class ObjectiveFunctionV2RL:
    """
    RL-based objective function using physics-derived SNR.

    Two-layer evaluation:
    Layer A: Hard clinical constraints (non-negotiable)
    Layer B: Weighted composite optimization (if constraints pass)
    """

    # Objective weights
    WEIGHT_DR = 0.45
    WEIGHT_FNR = 0.25
    WEIGHT_TTD = 0.15
    WEIGHT_SNR = 0.15

    # Hard clinical constraints
    MIN_DETECTION_RATE = 0.70
    MAX_FALSE_NEGATIVE_RATE = 0.20
    MIN_SNR_DB = 0.0
    CATASTROPHIC_PENALTY = -100.0

    TTD_MAX = 9000.0

    def __init__(
        self,
        physics_model,
        surrogate_loader_v2,
        weight_dr: float = 0.45,
        weight_fnr: float = 0.25,
        weight_ttd: float = 0.15,
        weight_snr: float = 0.15,
        min_detection_rate: float = 0.70,
        max_false_negative_rate: float = 0.20,
        min_snr_db: float = 0.0,
        apply_constraints: bool = True,
    ):
        """
        Initialize objective function.

        Args:
            physics_model: PhysicsForwardModel for SNR computation
            surrogate_loader_v2: Loader for v2_rl surrogates [SNR, biosensor, noise]
            (other args as before)
        """
        self.physics_model = physics_model
        self.surrogate_loader = surrogate_loader_v2

        self.weight_dr = weight_dr
        self.weight_fnr = weight_fnr
        self.weight_ttd = weight_ttd
        self.weight_snr = weight_snr

        self.min_detection_rate = min_detection_rate
        self.max_false_negative_rate = max_false_negative_rate
        self.min_snr_db = min_snr_db
        self.apply_constraints = apply_constraints

        # Normalize weights
        total_weight = weight_dr + weight_fnr + weight_ttd + weight_snr
        self.weight_dr /= total_weight
        self.weight_fnr /= total_weight
        self.weight_ttd /= total_weight
        self.weight_snr /= total_weight

        constraint_status = "ENABLED" if apply_constraints else "DISABLED"
        logger.info(
            f"Objective (RL v2 approach) weights: DR={self.weight_dr:.3f}, "
            f"FNR={self.weight_fnr:.3f}, TTD={self.weight_ttd:.3f}, SNR={self.weight_snr:.3f}"
        )
        logger.info(
            f"Hard constraints ({constraint_status}): "
            f"DR>={min_detection_rate:.2f}, FNR<={max_false_negative_rate:.2f}, SNR>={min_snr_db:.1f}dB"
        )

    def __call__(self, config: Dict) -> float:
        """
        Evaluate objective for a configuration.

        Process:
        1. Compute SNR from physics
        2. Get surrogate predictions using [SNR, biosensor_type, noise]
        3. Check hard constraints
        4. Compute weighted objective
        """
        try:
            # Compute SNR from physics model
            snr_db = self.physics_model.estimate_snr(
                biosensor_type=config["biosensor_type"],
                kd_nm=config.get("kd_nm", 1.0),  # Physics doesn't use these directly
                sensitivity=config.get("sensitivity", 1.0),
                response_time_s=config.get("response_time_s", 500.0),
                noise_preset=config["noise_preset"],
                target_scenario=config["target_scenario"],
            )

            # Get surrogate predictions using RL-style features
            dr_pred, fnr_pred, ttd_pred = self._get_surrogate_predictions(config, snr_db)

            # Log if OOD (SNR outside training range)
            training_bounds = self.surrogate_loader.get_training_bounds()
            snr_min = training_bounds.get('snr_min', -100.0)
            snr_max = training_bounds.get('snr_max', 100.0)

            if snr_db < snr_min or snr_db > snr_max:
                logger.debug(
                    f"Out-of-distribution SNR: {snr_db:.2f} dB "
                    f"(training range: [{snr_min:.2f}, {snr_max:.2f}])"
                )

            # LAYER A: Hard constraints
            if self.apply_constraints:
                if dr_pred < self.min_detection_rate:
                    return self.CATASTROPHIC_PENALTY

                if fnr_pred > self.max_false_negative_rate:
                    return self.CATASTROPHIC_PENALTY

                if snr_db < self.min_snr_db:
                    return self.CATASTROPHIC_PENALTY

            # LAYER B: Weighted composite objective
            snr_norm = np.clip((snr_db + 10.0) / 20.0, 0.0, 1.0)

            objective = (
                self.weight_dr * dr_pred
                + self.weight_fnr * (1.0 - fnr_pred)
                + self.weight_ttd * (1.0 - ttd_pred / self.TTD_MAX)
                + self.weight_snr * snr_norm
            )

            objective = float(np.clip(objective, 0.0, 1.0))
            return objective

        except Exception as e:
            logger.error(f"Error evaluating objective: {e}")
            return 0.0

    def _get_surrogate_predictions(self, config: Dict, snr_db: float) -> Tuple[float, float, float]:
        """
        Get predictions from v2_rl surrogates using [SNR, biosensor_type, noise].

        Args:
            config: Configuration dict
            snr_db: Computed SNR in dB

        Returns:
            (detection_rate, false_negative_rate, time_to_detection)
        """
        # Encode categoricals
        biosensor_enc, noise_enc = self.surrogate_loader.encode_categorical_v2(
            config["biosensor_type"],
            config["noise_preset"]
        )

        # Build feature vector [SNR, biosensor_encoded, noise_encoded]
        X_raw = np.array([[
            snr_db,
            biosensor_enc,
            noise_enc,
        ]], dtype=np.float32)

        # Scale using v2 scaler
        X_scaled = self.surrogate_loader.scaler.transform(X_raw)

        # Get predictions
        preds = self.surrogate_loader.predict_metrics_v2(X_scaled)
        dr_pred = float(preds[0])
        fnr_pred = float(preds[1])
        ttd_pred = float(preds[2])

        return dr_pred, fnr_pred, ttd_pred

    def evaluate_with_details(
        self, config: Dict
    ) -> Tuple[float, Dict]:
        """
        Evaluate and return detailed breakdown for debugging.

        Returns field names compatible with BO pipeline:
        - snr_db_est (not snr_db)
        - dr_pred
        - fnr_pred
        - ttd_pred_s (not ttd_pred)
        """
        try:
            snr_db = self.physics_model.estimate_snr(
                biosensor_type=config["biosensor_type"],
                kd_nm=config.get("kd_nm", 1.0),
                sensitivity=config.get("sensitivity", 1.0),
                response_time_s=config.get("response_time_s", 500.0),
                noise_preset=config["noise_preset"],
                target_scenario=config["target_scenario"],
            )

            dr_pred, fnr_pred, ttd_pred = self._get_surrogate_predictions(config, snr_db)

            constraint_violations = []
            if self.apply_constraints:
                if dr_pred < self.min_detection_rate:
                    constraint_violations.append(f"DR={dr_pred:.4f} < {self.min_detection_rate}")
                if fnr_pred > self.max_false_negative_rate:
                    constraint_violations.append(f"FNR={fnr_pred:.4f} > {self.max_false_negative_rate}")
                if snr_db < self.min_snr_db:
                    constraint_violations.append(f"SNR={snr_db:.2f}dB < {self.min_snr_db}dB")

            if constraint_violations and self.apply_constraints:
                objective = self.CATASTROPHIC_PENALTY
            else:
                snr_norm = np.clip((snr_db + 10.0) / 20.0, 0.0, 1.0)
                objective = (
                    self.weight_dr * dr_pred
                    + self.weight_fnr * (1.0 - fnr_pred)
                    + self.weight_ttd * (1.0 - ttd_pred / self.TTD_MAX)
                    + self.weight_snr * snr_norm
                )
                objective = float(np.clip(objective, 0.0, 1.0))

            return objective, {
                'snr_db_est': snr_db,  # Field name expected by pipeline
                'dr_pred': dr_pred,
                'fnr_pred': fnr_pred,
                'ttd_pred_s': ttd_pred,  # Field name expected by pipeline (in seconds)
                'constraint_violations': constraint_violations,
            }

        except Exception as e:
            logger.error(f"Error in evaluate_with_details: {e}")
            return 0.0, {
                'snr_db_est': 0.0,
                'dr_pred': 0.0,
                'fnr_pred': 1.0,
                'ttd_pred_s': 9000.0,
                'error': str(e)
            }
