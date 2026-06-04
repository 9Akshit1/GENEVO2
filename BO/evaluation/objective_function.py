#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Composite objective function for BO.

Combines physics forward model and surrogate predictions into a single
weighted objective that BO maximizes.
"""

import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class ObjectiveFunction:
    """
    Composite objective combining detection_rate, FNR, and TTD.

    Architecture:
    =============
    Layer A — HARD CLINICAL CONSTRAINTS (non-negotiable)
      if DR < min_dr:    return CATASTROPHIC_PENALTY
      if FNR > max_fnr:  return CATASTROPHIC_PENALTY
      if SNR < min_snr:  return CATASTROPHIC_PENALTY

    Layer B — OPTIMIZATION OBJECTIVE (only if constraints pass)
      Objective = 0.45 * DR + 0.25 * (1 - FNR) + 0.15 * (1 - TTD/9000) + 0.15 * SNR_norm

    This two-layer structure prevents the optimizer from finding pathological solutions
    (e.g., DR=0.28% with good FNR/TTD scores).

    Clinical thresholds are conservative for exploratory research:
      - DR >= 0.70 (70% sensitivity)
      - FNR <= 0.20 (20% false negative rate)
      - SNR >= 0 dB (basic signal requirement)
    """

    # Objective function weights (renormalized in __init__)
    WEIGHT_DR = 0.45    # Detection rate (primary objective)
    WEIGHT_FNR = 0.25   # False negative rate (secondary - model performance is limited)
    WEIGHT_TTD = 0.15   # Time to detection
    WEIGHT_SNR = 0.15   # Signal quality bonus

    # NOTE: FNR model has poor performance (median R²≈-0.46) due to extreme class imbalance
    # (90%+ of FNR values are 0.0). The FNR constraint is still enforced in Layer A,
    # but the FNR optimization term in Layer B should be treated as secondary guidance only.

    # TTD normalization: max value in dataset
    TTD_MAX = 9000.0

    # HARD CLINICAL CONSTRAINTS
    # For exploratory research
    MIN_DETECTION_RATE = 0.70  # Biosensor must detect ≥70% of disease cases
    MAX_FALSE_NEGATIVE_RATE = 0.20  # Cannot miss >20% of disease cases
    MIN_SNR_DB = 0.0  # SNR must be above noise floor
    CATASTROPHIC_PENALTY = -100.0  # Returned if hard constraints violated

    def __init__(
        self,
        physics_model,
        surrogate_loader,
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
        Initialize the objective function.

        Args:
            physics_model: PhysicsForwardModel instance
            surrogate_loader: SurrogateLoader instance with loaded models
            weight_dr: Weight for detection_rate term
            weight_fnr: Weight for (1-FNR) term
            weight_ttd: Weight for (1-TTD/TTD_MAX) term
            weight_snr: Weight for SNR term
            min_detection_rate: Hard constraint on minimum DR
            max_false_negative_rate: Hard constraint on maximum FNR
            min_snr_db: Hard constraint on minimum SNR
            apply_constraints: Whether to enforce hard constraints (set False for debugging)
        """
        self.physics_model = physics_model
        self.surrogate_loader = surrogate_loader
        self.weight_dr = weight_dr
        self.weight_fnr = weight_fnr
        self.weight_ttd = weight_ttd
        self.weight_snr = weight_snr

        # Hard constraints
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
            f"Objective weights: DR={self.weight_dr:.3f}, "
            f"FNR={self.weight_fnr:.3f}, TTD={self.weight_ttd:.3f}, SNR={self.weight_snr:.3f}"
        )
        logger.info(
            f"Hard constraints ({constraint_status}): "
            f"DR≥{min_detection_rate:.2f}, FNR≤{max_false_negative_rate:.2f}, SNR≥{min_snr_db:.1f}dB"
        )

    def __call__(self, config: Dict) -> float:
        """
        Evaluate objective for a configuration.

        Three-layer evaluation:
          1. Check input validity and bounds
          2. Check hard clinical constraints
          3. Compute weighted composite objective with OOD penalty

        Args:
            config: Parameter dictionary from search_space.vector_to_dict()

        Returns:
            Composite objective score in [-100, 1]
            (Catastrophic penalty if constraints violated or input invalid)
        """
        try:
            # LAYER 0: Validate input bounds
            dr_pred, fnr_pred, ttd_pred, snr_db, ood_penalty = self._evaluate_with_ood_check(config)

            if ood_penalty > 0:
                logger.warning(
                    f"Out-of-distribution parameters: DR={dr_pred:.4f}, FNR={fnr_pred:.4f}, "
                    f"OOD_penalty={ood_penalty:.4f}"
                )

            # LAYER A: Check hard clinical constraints
            if self.apply_constraints:
                if dr_pred < self.min_detection_rate:
                    return self.CATASTROPHIC_PENALTY

                if fnr_pred > self.max_false_negative_rate:
                    return self.CATASTROPHIC_PENALTY

                if snr_db < self.min_snr_db:
                    return self.CATASTROPHIC_PENALTY

            # LAYER B: Compute weighted composite objective (only if constraints pass)
            # Normalize SNR contribution: 10 dB is "good", -10 dB is "bad"
            snr_norm = np.clip((snr_db + 10.0) / 20.0, 0.0, 1.0)

            objective = (
                self.weight_dr * dr_pred
                + self.weight_fnr * (1.0 - fnr_pred)
                + self.weight_ttd * (1.0 - ttd_pred / self.TTD_MAX)
                + self.weight_snr * snr_norm
            )

            # Apply OOD penalty - penalize extrapolation
            objective = objective * (1.0 - ood_penalty)

            # Clip to [0, 1]
            objective = float(np.clip(objective, 0.0, 1.0))

            return objective

        except Exception as e:
            logger.error(f"Error evaluating objective: {e}")
            return 0.0

    def _evaluate_with_ood_check(self, config: Dict) -> tuple:
        """
        Evaluate with OOD detection.

        Returns:
            Tuple of (dr_pred, fnr_pred, ttd_pred, snr_db, ood_penalty)
        """
        # Encode categorical variables
        biosensor_encoded, noise_encoded, scenario_encoded = self.surrogate_loader.encode_categorical(
            config["biosensor_type"],
            config["noise_preset"],
            config["target_scenario"],
        )

        # Prepare input for surrogates [kd, sensitivity, biosensor_type_enc, noise_preset_enc, scenario_enc]
        X_raw = np.array([[
            config["kd_nm"],
            config["sensitivity"],
            biosensor_encoded,
            noise_encoded,
            scenario_encoded,
        ]], dtype=np.float32)

        # Scale using the fitted scaler
        X_scaled = self.surrogate_loader.scaler.transform(X_raw)

        # Check if input is OOD using scaled features
        ood_penalty = self._compute_ood_penalty(X_scaled)

        # Get surrogate predictions (handles both v1 and v2 models)
        # v2 returns: (dr_prob, fnr_median, fnr_lower, fnr_upper, ttd_median, ttd_lower, ttd_upper)
        # v1 returns: (dr_pred, fnr_pred, ttd_pred, fnr_dummy1, fnr_dummy2, ttd_dummy1, ttd_dummy2)
        preds = self.surrogate_loader.predict_metrics(X_scaled)
        dr_pred = preds[0]
        fnr_pred = preds[1]  # Use median for v2, single value for v1
        ttd_pred = preds[4]  # Use median for v2, single value for v1

        # Estimate SNR for constraint checking and visualization
        snr_db = self.physics_model.estimate_snr(
            biosensor_type=config["biosensor_type"],
            kd_nm=config["kd_nm"],
            sensitivity=config["sensitivity"],
            response_time_s=config.get("response_time_s", 500.0),
            noise_preset=config["noise_preset"],
            target_scenario=config["target_scenario"],
        )

        return dr_pred, fnr_pred, ttd_pred, snr_db, ood_penalty

    def _compute_ood_penalty(self, X_scaled: np.ndarray) -> float:
        """
        Compute OOD penalty for scaled features.

        Detects if input falls significantly outside typical training distribution.
        Uses Mahalanobis distance or simple standard deviation thresholding.

        Args:
            X_scaled: Scaled feature array from StandardScaler

        Returns:
            Penalty in [0, 1] where 0 = in-distribution, 1 = severe extrapolation
        """
        # For standardized data, values with |z| > 3 are ~0.3% of population
        # We use a soft penalty that increases with distance from origin
        abs_z = np.abs(X_scaled[0])

        # Compute penalty as fraction of features in extreme region
        n_extreme = np.sum(abs_z > 3.0)
        penalty_from_extremeness = n_extreme / len(abs_z)

        # Also penalize average distance from training center
        avg_distance = np.mean(abs_z)
        penalty_from_distance = max(0.0, (avg_distance - 2.0) / 3.0)

        # Combine penalties
        ood_penalty = max(penalty_from_extremeness, penalty_from_distance)
        ood_penalty = np.clip(ood_penalty, 0.0, 1.0)

        return float(ood_penalty)

    def evaluate_with_details(
        self, config: Dict
    ) -> Tuple[float, Dict]:
        """
        Evaluate objective and return detailed breakdown.

        Includes constraint violation info and OOD status for debugging.

        Args:
            config: Parameter dictionary

        Returns:
            Tuple of (objective_score, details_dict)
        """
        try:
            # Get predictions with OOD check
            dr_pred, fnr_pred, ttd_pred, snr_db, ood_penalty = self._evaluate_with_ood_check(config)

            # Check constraints
            constraint_violations = []
            if self.apply_constraints:
                if dr_pred < self.min_detection_rate:
                    constraint_violations.append(f"DR={dr_pred:.4f} < {self.min_detection_rate}")
                if fnr_pred > self.max_false_negative_rate:
                    constraint_violations.append(f"FNR={fnr_pred:.4f} > {self.max_false_negative_rate}")
                if snr_db < self.min_snr_db:
                    constraint_violations.append(f"SNR={snr_db:.2f}dB < {self.min_snr_db}dB")
                if ood_penalty > 0.1:
                    constraint_violations.append(f"OOD_penalty={ood_penalty:.4f} (extrapolation risk)")

            # Compute objective
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
                # Apply OOD penalty
                objective = objective * (1.0 - ood_penalty)
                objective = float(np.clip(objective, 0.0, 1.0))

            # For v2 surrogates, also get uncertainty estimates
            biosensor_encoded_unc, noise_encoded_unc, scenario_encoded_unc = self.surrogate_loader.encode_categorical(
                config["biosensor_type"],
                config["noise_preset"],
                config["target_scenario"],
            )
            X_raw_unc = np.array([[
                config["kd_nm"],
                config["sensitivity"],
                biosensor_encoded_unc,
                noise_encoded_unc,
                scenario_encoded_unc,
            ]], dtype=np.float32)
            X_scaled_unc = self.surrogate_loader.scaler.transform(X_raw_unc)
            preds = self.surrogate_loader.predict_metrics(X_scaled_unc)
            fnr_lower = preds[2] if len(preds) > 2 else fnr_pred
            fnr_upper = preds[3] if len(preds) > 3 else fnr_pred
            ttd_lower = preds[5] if len(preds) > 5 else ttd_pred
            ttd_upper = preds[6] if len(preds) > 6 else ttd_pred

            details = {
                "snr_db_est": snr_db,
                "dr_pred": dr_pred,
                "fnr_pred": fnr_pred,
                "fnr_lower": fnr_lower,
                "fnr_upper": fnr_upper,
                "ttd_pred_s": ttd_pred,
                "ttd_lower": ttd_lower,
                "ttd_upper": ttd_upper,
                "ood_penalty": float(ood_penalty),
                "dr_term": self.weight_dr * dr_pred,
                "fnr_term": self.weight_fnr * (1.0 - fnr_pred),
                "ttd_term": self.weight_ttd * (1.0 - ttd_pred / self.TTD_MAX),
                "snr_norm": np.clip((snr_db + 10.0) / 20.0, 0.0, 1.0),
                "snr_term": self.weight_snr * np.clip((snr_db + 10.0) / 20.0, 0.0, 1.0),
                "composite_score": objective,
                "constraint_violations": constraint_violations if constraint_violations else "PASS",
            }

            return objective, details

        except Exception as e:
            logger.error(f"Error in evaluate_with_details: {e}", exc_info=True)
            return 0.0, {}

    def get_description(self) -> str:
        """Return description of the objective function."""
        desc = "Composite Objective Function (Two-Layer Architecture)\n"
        desc += "=" * 70 + "\n"
        desc += "\nLAYER A — HARD CLINICAL CONSTRAINTS (Violations → Catastrophic Penalty)\n"
        if self.apply_constraints:
            desc += f"  ✓ Detection Rate (DR)       must be ≥ {self.min_detection_rate:.2f}\n"
            desc += f"  ✓ False Negative Rate (FNR) must be ≤ {self.max_false_negative_rate:.2f}\n"
            desc += f"  ✓ Signal-to-Noise Ratio     must be ≥ {self.min_snr_db:.1f} dB\n"
        else:
            desc += "  ⚠ Constraints are DISABLED (for debugging only)\n"
        desc += "\nLAYER B — WEIGHTED OPTIMIZATION OBJECTIVE (If constraints pass)\n"
        desc += f"  Detection Rate (DR):        weight={self.weight_dr:.3f}, goal=maximize\n"
        desc += f"  False Negative Rate (FNR):  weight={self.weight_fnr:.3f}, goal=minimize\n"
        desc += f"  Time to Detection (TTD):    weight={self.weight_ttd:.3f}, goal=minimize\n"
        desc += f"  Signal Quality (SNR):       weight={self.weight_snr:.3f}, goal=maximize\n"
        desc += "\n"
        desc += "Composite = weight_DR * DR + weight_FNR * (1-FNR) + weight_TTD * (1-TTD/9000)\n"
        return desc
