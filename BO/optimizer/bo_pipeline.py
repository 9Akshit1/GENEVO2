#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Complete BO pipeline orchestrator.

Handles initialization, optimization, logging, visualization, and result saving.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import logging
import pickle

logger = logging.getLogger(__name__)


class BOPipeline:
    """Orchestrates the complete BO optimization workflow."""

    def __init__(
        self,
        optimizer,
        objective_fn,
        search_space,
        robustness_analyzer,
        output_dir: Path,
    ):
        """
        Initialize BO pipeline.

        Args:
            optimizer: GaussianProcessBO instance
            objective_fn: ObjectiveFunction instance
            search_space: BiosensorSearchSpace instance
            robustness_analyzer: RobustnessAnalyzer instance
            output_dir: Root output directory
        """
        self.optimizer = optimizer
        self.objective_fn = objective_fn
        self.search_space = search_space
        self.robustness_analyzer = robustness_analyzer
        self.output_dir = Path(output_dir)

        # Create output subdirectories
        self.log_dir = self.output_dir / "logs"
        self.result_dir = self.output_dir / "results"
        self.plot_dir = self.output_dir / "plots"
        self.model_dir = self.output_dir / "models"

        for d in [self.log_dir, self.result_dir, self.plot_dir, self.model_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict:
        """
        Execute the full BO pipeline.

        Returns:
            Dictionary with results and best configuration
        """
        logger.info("=" * 80)
        logger.info("GENEVO2 Bayesian Optimization Pipeline")
        logger.info("=" * 80)
        logger.info(f"Output directory: {self.output_dir}")

        # Run BO optimization
        logger.info("\n[1/4] Running BO Optimization")
        bo_result = self.optimizer.optimize()

        # Analyze robustness
        logger.info("\n[2/4] Analyzing Robustness")
        config_best = bo_result["config_best"]
        robustness_result = self.robustness_analyzer.evaluate_robustness(config_best)

        # Get detailed predictions
        logger.info("\n[3/4] Getting Detailed Predictions")
        score, details = self.objective_fn.evaluate_with_details(config_best)

        # Save results
        logger.info("\n[4/4] Saving Results")
        self._save_results(bo_result, robustness_result, details)
        self._save_iteration_log(bo_result["iteration_history"])
        self._save_gp_model(bo_result["gp"])
        self._save_best_config(config_best, bo_result, robustness_result, details)

        # Generate visualizations
        logger.info("\nGenerating Visualizations")
        self._plot_convergence(bo_result)
        self._plot_robustness_heatmap(robustness_result)

        logger.info("\n" + "=" * 80)
        logger.info("BO Pipeline Complete")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {self.result_dir}")
        logger.info(f"Logs saved to: {self.log_dir}")
        logger.info(f"Plots saved to: {self.plot_dir}")

        return {
            "bo_result": bo_result,
            "robustness_result": robustness_result,
            "details": details,
        }

    def _save_results(self, bo_result: dict, robustness_result: dict, details: dict) -> None:
        """Save comprehensive results to JSON."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "best_y": float(bo_result["y_best"]),
            "best_config": {
                k: (v if not isinstance(v, np.ndarray) else v.tolist())
                for k, v in bo_result["config_best"].items()
            },
            "uncertainty": {
                "gp_mean": float(bo_result["gp_mean_best"]),
                "gp_std": float(bo_result["gp_std_best"]),
                "ci_lower": float(bo_result["ci_lower"]),
                "ci_upper": float(bo_result["ci_upper"]),
            },
            "predictions": {
                "snr_db_est": float(details["snr_db_est"]),
                "detection_rate": float(details["dr_pred"]),
                "false_negative_rate": float(details["fnr_pred"]),
                "time_to_detection_s": float(details["ttd_pred_s"]),
            },
            "robustness": {
                "mean_score": float(robustness_result["mean_score"]),
                "min_score": float(robustness_result["min_score"]),
                "max_score": float(robustness_result["max_score"]),
                "std_score": float(robustness_result["std_score"]),
                "robustness_score": float(robustness_result["robustness_score"]),
            },
            "n_evaluations": len(bo_result["y_observed"]),
            "n_init": self.optimizer.n_init,
            "n_iter": self.optimizer.n_iter,
        }

        results_file = self.result_dir / "optimization_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {results_file.name}")

    def _save_iteration_log(self, iteration_history: list) -> None:
        """Save per-iteration log as CSV."""
        rows = []
        y_best_list = []
        y_best = -np.inf

        for iter_info in iteration_history:
            y = iter_info["y"]
            y_best = max(y_best, y)
            y_best_list.append(y_best)

            config = iter_info["config"]
            row = {
                "iteration": iter_info["iteration"],
                "score": y,
                "best_so_far": y_best,
                "gp_mean": iter_info["gp_mean"],
                "gp_std": iter_info["gp_std"],
                "biosensor_type": config["biosensor_type"],
                "kd_nm": config["kd_nm"],
                "sensitivity": config["sensitivity"],
                "response_time_s": config.get("response_time_s", 0.0),
                "noise_preset": config["noise_preset"],
                "target_scenario": config["target_scenario"],
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        log_file = self.log_dir / "iteration_log.csv"
        df.to_csv(log_file, index=False)
        logger.info(f"Saved iteration log to {log_file.name}")

    def _save_gp_model(self, gp) -> None:
        """Save fitted GP model."""
        model_file = self.model_dir / "gp_surrogate.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(gp, f)
        logger.info(f"Saved GP model to {model_file.name}")

    def _save_best_config(
        self,
        config_best: dict,
        bo_result: dict,
        robustness_result: dict,
        details: dict,
    ) -> None:
        """Save best configuration in user-friendly format."""
        best_config = {
            "biosensor_design": {
                "type": config_best["biosensor_type"],
                "kd_nm": float(config_best["kd_nm"]),
                "sensitivity": float(config_best["sensitivity"]),
                "response_time_s": float(config_best.get("response_time_s", 0.0)),
            },
            "measurement_environment": {
                "noise_preset": config_best["noise_preset"],
                "target_scenario": config_best["target_scenario"],
            },
            "predicted_performance": {
                "detection_rate": float(details["dr_pred"]),
                "false_negative_rate": float(details["fnr_pred"]),
                "time_to_detection_s": float(details["ttd_pred_s"]),
                "estimated_snr_db": float(details["snr_db_est"]),
            },
            "optimization_metrics": {
                "composite_score": float(bo_result["y_best"]),
                "gp_uncertainty": {
                    "mean": float(bo_result["gp_mean_best"]),
                    "std": float(bo_result["gp_std_best"]),
                    "ci_95": [
                        float(bo_result["ci_lower"]),
                        float(bo_result["ci_upper"]),
                    ],
                },
            },
            "robustness_analysis": {
                "mean_score_across_conditions": float(robustness_result["mean_score"]),
                "worst_case_score": float(robustness_result["min_score"]),
                "best_case_score": float(robustness_result["max_score"]),
                "score_std_dev": float(robustness_result["std_score"]),
                "robustness_index": float(robustness_result["robustness_score"]),
            },
        }

        config_file = self.result_dir / "best_config.json"
        with open(config_file, "w") as f:
            json.dump(best_config, f, indent=2)
        logger.info(f"Saved best configuration to {config_file.name}")

    def _plot_convergence(self, bo_result: dict) -> None:
        """Plot convergence curve."""
        try:
            import matplotlib.pyplot as plt

            y_observed = bo_result["y_observed"]
            y_best = np.maximum.accumulate(y_observed)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(y_observed, "o-", alpha=0.5, label="Observed", markersize=4)
            ax.plot(y_best, "r-", linewidth=2, label="Best so far")
            ax.set_xlabel("Evaluation #")
            ax.set_ylabel("Objective Score")
            ax.set_title("BO Convergence Curve")
            ax.grid(True, alpha=0.3)
            ax.legend()

            plot_file = self.plot_dir / "convergence_curve.png"
            plt.savefig(plot_file, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved convergence plot to {plot_file.name}")
        except Exception as e:
            logger.warning(f"Failed to plot convergence: {e}")

    def _plot_robustness_heatmap(self, robustness_result: dict) -> None:
        """Plot robustness heatmap across scenario × noise."""
        try:
            import matplotlib.pyplot as plt

            scores_matrix = robustness_result["scores_matrix"]
            scenarios = ["pmo", "ckd_mbd", "both"]
            noise_presets = ["low", "medium", "high"]

            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(scores_matrix.T, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

            ax.set_xticks(range(len(scenarios)))
            ax.set_yticks(range(len(noise_presets)))
            ax.set_xticklabels(scenarios)
            ax.set_yticklabels(noise_presets)
            ax.set_xlabel("Target Scenario")
            ax.set_ylabel("Noise Preset")
            ax.set_title("Robustness Heatmap: Score across Scenario × Noise")

            # Add text annotations
            for i in range(len(scenarios)):
                for j in range(len(noise_presets)):
                    text = ax.text(
                        i,
                        j,
                        f"{scores_matrix[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="black",
                        fontweight="bold",
                    )

            plt.colorbar(im, ax=ax, label="Score")
            fig.tight_layout()

            plot_file = self.plot_dir / "robustness_heatmap.png"
            plt.savefig(plot_file, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved robustness heatmap to {plot_file.name}")
        except Exception as e:
            logger.warning(f"Failed to plot robustness heatmap: {e}")
