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

    def evaluate_robustness(self, config: Dict, n_trials: int = 10) -> Dict:
        """
        Evaluate robustness using ACTUAL SIMULATOR across all noise × scenario conditions.

        Each (scenario, noise) combination is run n_trials times with stochastic
        patient variability (apply_variability=True).  Results are reported as
        mean ± std so a single unlucky draw can no longer mask a clinically
        relevant PMO or CKD failure.

        Args:
            config: Biosensor configuration (kd_nm, sensitivity, response_time_s, etc.)
            n_trials: Number of stochastic simulator runs per (scenario, noise) cell.
                      Default 10 gives a reasonable estimate without taking too long.

        Returns:
            Dictionary with actual robustness metrics including per-cell mean/std.
        """
        noise_presets = ["low", "medium", "high"]
        scenarios = ["healthy", "pmo", "ckd_mbd"]

        logger.info(
            f"\n[ROBUSTNESS] Evaluating {len(scenarios)}×{len(noise_presets)} conditions "
            f"× {n_trials} trials using ACTUAL SIMULATOR"
        )

        try:
            from simulation.dataset.generator import DatasetGenerator
            import tempfile

            circuit_type = config["biosensor_type"]
            kd = config.get("kd_nm", 1.0)
            sensitivity = config.get("sensitivity", 1.0)
            response_time = config.get("response_time_s", 500.0)

            if circuit_type == "array":
                # Array sensors require the full multi-channel config.
                # Use the threshold already embedded in the BO config (computed at
                # search/evaluate time via generate_random_array_config margin=1.25).
                # _compute_calibrated_threshold does not support array type and falls
                # back to 0.5, which is wrong and inflates FP to ~100%.
                kd_ctx  = config.get("kd_ctx_nm",  kd)
                kd_p1np = config.get("kd_p1np_nm", kd)
                w_ctx   = config.get("w_ctx",  0.0)
                w_p1np  = config.get("w_p1np", 0.0)
                w_scl   = max(0.0, 1.0 - w_ctx - w_p1np)

                H = {"scl": 0.375, "ctx": 0.200, "p1np": 0.350}
                P = {"scl": 0.875, "ctx": 0.500, "p1np": 0.525}

                def _occ(c, kd_v): return c / (kd_v + c + 1e-12)
                ref_scl  = _occ(H["scl"],  kd)
                ref_ctx  = _occ(H["ctx"],  kd_ctx)
                ref_p1np = _occ(H["p1np"], kd_p1np)

                def _composite(conc_dict):
                    n_s = _occ(conc_dict["scl"], kd)     / (ref_scl  + 1e-12)
                    n_c = _occ(conc_dict["ctx"], kd_ctx) / (ref_ctx  + 1e-12)
                    n_p = _occ(conc_dict["p1np"], kd_p1np) / (ref_p1np + 1e-12)
                    return sensitivity * (w_scl * n_s + w_ctx * n_c + w_p1np * n_p)

                sig_h = _composite(H)
                sig_p = _composite(P)
                hp_gap_ref = (sig_p / sensitivity) - 1.0
                threshold = float(sig_h + 1.25 * hp_gap_ref)

                biosensor_config = {
                    "circuit_type": "array",
                    "kd_scl":  kd,
                    "kd_ctx":  kd_ctx,
                    "kd_p1np": kd_p1np,
                    "w_scl":   w_scl,
                    "w_ctx":   w_ctx,
                    "w_p1np":  w_p1np,
                    "sensitivity": sensitivity,
                    "threshold": threshold,
                    "dynamic_range": (0.0, sensitivity * 3.0),
                    "kd": kd,
                }
            else:
                # Single-channel: use a simple default threshold (Langmuir mid-point)
                c_healthy = 0.375  # healthy SOST [nM]
                occ_h = c_healthy / (kd + c_healthy + 1e-12)
                threshold = float(sensitivity * occ_h * 1.25)
                biosensor_config = {
                    "circuit_type": circuit_type,
                    "kd": kd,
                    "sensitivity": sensitivity,
                    "response_time": response_time,
                    "threshold": threshold,
                    "dynamic_range": (0.0, sensitivity * 3.0),
                }

            results_dict = {}
            all_dr_values = []
            all_fnr_values = []
            all_snr_values = []
            all_ttd_values = []

            logger.debug(f"Biosensor config for robustness eval: {biosensor_config}")

            for scenario in scenarios:
                for noise_preset in noise_presets:
                    trial_drs, trial_fnrs, trial_snrs, trial_ttds = [], [], [], []

                    try:
                        with tempfile.TemporaryDirectory() as tmpdir:
                            generator = DatasetGenerator(
                                antimony_model_path="simulation/models/bone_environment.ant",
                                output_dir=tmpdir,
                                seed=None,
                            )

                            for _trial in range(n_trials):
                                result = generator.generate_single_simulation_instrumented(
                                    scenario_name=scenario,
                                    biosensor_config=biosensor_config,
                                    noise_preset=noise_preset,
                                    duration=3600.0,
                                    num_points=361,
                                    apply_variability=True,
                                    instrument=False,
                                )

                                if result is None:
                                    continue

                                trial_drs.append(float(result['measurement']['detection_rate']))
                                trial_fnrs.append(float(result['measurement']['false_negative_rate']))
                                trial_snrs.append(float(result['measurement']['snr_db']))
                                trial_ttds.append(float(result['measurement']['time_to_detection']))

                    except Exception as e:
                        logger.debug(
                            f"  {scenario}+{noise_preset}: Error: {type(e).__name__}: {str(e)[:100]}"
                        )
                        continue

                    if not trial_drs:
                        continue

                    mean_dr  = float(np.mean(trial_drs))
                    std_dr   = float(np.std(trial_drs))
                    mean_fnr = float(np.mean(trial_fnrs))
                    mean_snr = float(np.mean(trial_snrs))
                    mean_ttd = float(np.mean(trial_ttds))

                    results_dict[(scenario, noise_preset)] = {
                        'dr':      mean_dr,
                        'dr_std':  std_dr,
                        'fnr':     mean_fnr,
                        'snr':     mean_snr,
                        'ttd':     mean_ttd,
                        'n_trials': len(trial_drs),
                    }

                    all_dr_values.extend(trial_drs)
                    all_fnr_values.extend(trial_fnrs)
                    all_snr_values.extend(trial_snrs)
                    all_ttd_values.extend(trial_ttds)

                    logger.info(
                        f"  {scenario:10s}+{noise_preset:6s}: "
                        f"DR={mean_dr:.3f}±{std_dr:.3f} ({len(trial_drs)} trials), "
                        f"SNR={mean_snr:.1f}dB"
                    )

            if not all_dr_values:
                return {
                    "mean_score": -100.0,
                    "min_score": -100.0,
                    "max_score": -100.0,
                    "std_score": 0.0,
                    "robustness_score": 0.0,
                    "results_dict": {},
                }

            disease_scenarios = ["pmo", "ckd_mbd"]
            disease_dr_means, disease_fnr_means = [], []
            healthy_dr_means = []

            for (sc, _noise), vals in results_dict.items():
                if sc in disease_scenarios:
                    disease_dr_means.append(vals["dr"])
                    disease_fnr_means.append(vals["fnr"])
                elif sc == "healthy":
                    healthy_dr_means.append(vals["dr"])

            disease_dr_arr  = np.array(disease_dr_means)  if disease_dr_means  else np.array([0.0])
            disease_fnr_arr = np.array(disease_fnr_means) if disease_fnr_means else np.array([1.0])
            healthy_dr_arr  = np.array(healthy_dr_means)  if healthy_dr_means  else np.array([0.0])

            mean_dr         = float(np.mean(disease_dr_arr))
            min_dr          = float(np.min(disease_dr_arr))
            std_dr          = float(np.std(disease_dr_arr))
            max_fnr         = float(np.max(disease_fnr_arr))
            mean_snr        = float(np.mean(all_snr_values))
            min_snr         = float(np.min(all_snr_values))
            mean_healthy_dr = float(np.mean(healthy_dr_arr))

            robustness_score = (
                0.60 * min_dr
                + 0.20 * mean_dr
                + 0.10 * (1.0 - min(max_fnr, 1.0))
                + 0.10 * (1.0 - mean_healthy_dr)
            )

            logger.info(f"  Disease DR: mean={mean_dr:.3f}, min={min_dr:.3f}")
            logger.info(f"  Healthy DR: mean={mean_healthy_dr:.3f} (false positive rate)")
            logger.info(f"  SNR: mean={mean_snr:.1f}dB, min={min_snr:.1f}dB")
            logger.info(f"  Robustness Score: {robustness_score:.4f}")

            return {
                "mean_score": mean_dr,
                "min_score": min_dr,
                "max_score": float(np.max(disease_dr_arr)),
                "std_score": std_dr,
                "healthy_fp_rate": float(mean_healthy_dr),
                "robustness_score": float(np.clip(robustness_score, 0.0, 1.0)),
                "results_dict": results_dict,
            }

        except Exception as e:
            logger.warning(f"Robustness evaluation error: {e}")
            return {
                "mean_score": 0.0,
                "min_score": 0.0,
                "max_score": 0.0,
                "std_score": 0.0,
                "robustness_score": 0.0,
                "results_dict": {},
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
