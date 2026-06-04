#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Robustness analysis across noise presets and disease scenarios.

Evaluates how well a biosensor design performs across different
environmental conditions (noise levels and disease states).
"""

import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class RobustnessAnalyzer:
    """Analyze robustness of biosensor designs across noise and scenario variations."""

    def __init__(self, objective_function):
        """
        Initialize robustness analyzer.

        Args:
            objective_function: ObjectiveFunction instance
        """
        self.objective = objective_function

    def evaluate_robustness(self, config: Dict) -> Dict:
        """
        Evaluate robustness across all noise presets and scenarios.

        Args:
            config: Base configuration (without noise_preset and target_scenario)

        Returns:
            Dictionary with robustness metrics
        """
        noise_presets = ["low", "medium", "high"]
        scenarios = ["healthy", "pmo", "ckd_mbd"]

        scores = {}
        scores_matrix = np.zeros((len(scenarios), len(noise_presets)))

        # Evaluate across all combinations
        for i, scenario in enumerate(scenarios):
            for j, noise_preset in enumerate(noise_presets):
                test_config = config.copy()
                test_config["noise_preset"] = noise_preset
                test_config["target_scenario"] = scenario

                score, _ = self.objective.evaluate_with_details(test_config)
                scores[(scenario, noise_preset)] = score
                scores_matrix[i, j] = score

        # Compute robustness metrics
        mean_score = float(np.mean(scores_matrix))
        min_score = float(np.min(scores_matrix))
        max_score = float(np.max(scores_matrix))
        std_score = float(np.std(scores_matrix))

        # Robustness: mean score with penalty for variance
        # Higher mean and lower variance = more robust
        robustness_score = mean_score - 0.1 * std_score  # Penalize high variance

        return {
            "mean_score": mean_score,
            "min_score": min_score,
            "max_score": max_score,
            "std_score": std_score,
            "robustness_score": float(np.clip(robustness_score, 0.0, 1.0)),
            "scores_matrix": scores_matrix,
            "scores_dict": scores,
        }

    def analyze_sensitivity(self, config: Dict, param_name: str) -> Dict:
        """
        Analyze sensitivity to a single parameter.

        Evaluate objective as a parameter varies while others remain fixed.

        Args:
            config: Base configuration
            param_name: Name of parameter to vary

        Returns:
            Dictionary with sensitivity analysis results
        """
        # Get parameter bounds from search space
        from .biosensor_space import BiosensorSearchSpace

        space = BiosensorSearchSpace()
        param = space.parameters[param_name]

        if param.param_type != "continuous":
            return {"error": f"Parameter {param_name} is not continuous"}

        # Evaluate at 10 points in parameter range
        n_points = 10
        if param.scale == "log":
            log_lower = np.log10(param.lower)
            log_upper = np.log10(param.upper)
            param_values = 10.0 ** np.linspace(log_lower, log_upper, n_points)
        else:
            param_values = np.linspace(param.lower, param.upper, n_points)

        objectives = []
        for pv in param_values:
            test_config = config.copy()
            test_config[param_name] = pv

            obj, _ = self.objective.evaluate_with_details(test_config)
            objectives.append(obj)

        objectives = np.array(objectives)

        return {
            "param_name": param_name,
            "param_values": param_values.tolist(),
            "objectives": objectives.tolist(),
            "gradient": float(np.gradient(objectives).mean()),
            "max_objective": float(np.max(objectives)),
            "min_objective": float(np.min(objectives)),
        }

    def get_worst_case_scenario(self, config: Dict) -> Tuple[str, str, float]:
        """
        Find the worst-case noise × scenario combination.

        Args:
            config: Configuration to analyze

        Returns:
            Tuple of (scenario, noise_preset, worst_score)
        """
        noise_presets = ["low", "medium", "high"]
        scenarios = ["pmo", "ckd_mbd", "both"]

        worst_score = 1.0
        worst_combo = None

        for scenario in scenarios:
            for noise_preset in noise_presets:
                test_config = config.copy()
                test_config["noise_preset"] = noise_preset
                test_config["target_scenario"] = scenario

                score, _ = self.objective.evaluate_with_details(test_config)

                if score < worst_score:
                    worst_score = score
                    worst_combo = (scenario, noise_preset)

        return worst_combo[0], worst_combo[1], worst_score

    def get_best_case_scenario(self, config: Dict) -> Tuple[str, str, float]:
        """
        Find the best-case noise × scenario combination.

        Args:
            config: Configuration to analyze

        Returns:
            Tuple of (scenario, noise_preset, best_score)
        """
        noise_presets = ["low", "medium", "high"]
        scenarios = ["pmo", "ckd_mbd", "both"]

        best_score = 0.0
        best_combo = None

        for scenario in scenarios:
            for noise_preset in noise_presets:
                test_config = config.copy()
                test_config["noise_preset"] = noise_preset
                test_config["target_scenario"] = scenario

                score, _ = self.objective.evaluate_with_details(test_config)

                if score > best_score:
                    best_score = score
                    best_combo = (scenario, noise_preset)

        return best_combo[0], best_combo[1], best_score
