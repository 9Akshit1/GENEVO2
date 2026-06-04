#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare BO results with RL baseline.

Loads RL training results and compares with BO performance.
"""

import pandas as pd
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BOVsRLComparison:
    """Compare Bayesian Optimization with Reinforcement Learning baseline."""

    def __init__(self, rl_results_dir: Path):
        """
        Initialize comparison.

        Args:
            rl_results_dir: Path to RL results directory (e.g., rl_results_v7)
        """
        self.rl_results_dir = Path(rl_results_dir)

    def load_rl_results(self) -> dict:
        """
        Load RL training results.

        Returns:
            Dictionary with RL metrics
        """
        logger.info("Loading RL results...")

        try:
            # Try to load episode summary
            episode_log = self.rl_results_dir / "rl_logs" / "episode_summary.csv"
            if not episode_log.exists():
                logger.warning(f"Episode log not found: {episode_log}")
                return {}

            df = pd.read_csv(episode_log)
            logger.info(f"Loaded {len(df)} episodes from RL training")

            # Get best reward
            if "episode_return" in df.columns:
                best_return = df["episode_return"].max()
                mean_return = df["episode_return"].mean()
                final_return = df["episode_return"].iloc[-1]
            else:
                logger.warning("episode_return column not found in episode_summary.csv")
                return {}

            return {
                "best_return": float(best_return),
                "mean_return": float(mean_return),
                "final_return": float(final_return),
                "n_episodes": len(df),
                "episode_log": df,
            }

        except Exception as e:
            logger.error(f"Failed to load RL results: {e}")
            return {}

    def compare(self, bo_result: dict) -> dict:
        """
        Compare BO with RL.

        Args:
            bo_result: BO optimization result dictionary

        Returns:
            Comparison summary
        """
        logger.info("Comparing BO vs RL...")

        rl_results = self.load_rl_results()
        if not rl_results:
            logger.warning("Could not load RL results for comparison")
            return {"error": "RL results not available"}

        # Map RL reward to composite score
        # RL reward composition: 0.50 * DR + 0.25 * (1-FNR) + 0.25 * (1-TTD/5000)
        # This is close but not exactly the same as our BO objective
        # For comparison, we'll use the RL values as-is

        rl_best_score = rl_results.get("best_return", 0.0)
        bo_best_score = bo_result.get("y_best", 0.0)

        # Sample efficiency: score achieved per evaluation
        rl_n_evals = rl_results.get("n_episodes", 0) * 2048  # steps per episode
        bo_n_evals = bo_result.get("n_evaluations", 0)

        rl_sample_efficiency = rl_best_score / max(rl_n_evals, 1)
        bo_sample_efficiency = bo_best_score / max(bo_n_evals, 1)

        comparison = {
            "best_score": {
                "RL": float(rl_best_score),
                "BO": float(bo_best_score),
                "improvement": float((bo_best_score - rl_best_score) / max(rl_best_score, 0.01) * 100),
            },
            "sample_efficiency": {
                "RL_score_per_eval": float(rl_sample_efficiency),
                "BO_score_per_eval": float(bo_sample_efficiency),
                "BO_speedup": float(bo_sample_efficiency / max(rl_sample_efficiency, 1e-10)),
            },
            "evaluation_budget": {
                "RL_evaluations": int(rl_n_evals),
                "BO_evaluations": int(bo_n_evals),
            },
            "rl_metrics": {
                "mean_reward": float(rl_results.get("mean_return", 0.0)),
                "final_reward": float(rl_results.get("final_return", 0.0)),
                "n_episodes": int(rl_results.get("n_episodes", 0)),
            },
        }

        logger.info(f"\nComparison Results:")
        logger.info(f"  RL best score:    {rl_best_score:.4f}")
        logger.info(f"  BO best score:    {bo_best_score:.4f}")
        logger.info(f"  BO improvement:   {comparison['best_score']['improvement']:.1f}%")
        logger.info(f"  RL sample eff:    {rl_sample_efficiency:.6f}")
        logger.info(f"  BO sample eff:    {bo_sample_efficiency:.6f}")
        logger.info(f"  BO speedup:       {comparison['sample_efficiency']['BO_speedup']:.1f}x")

        return comparison

    def save_comparison(self, comparison: dict, output_file: Path) -> None:
        """
        Save comparison results to JSON.

        Args:
            comparison: Comparison dictionary
            output_file: Path to save JSON
        """
        with open(output_file, "w") as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"Saved comparison to {output_file.name}")
