#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RL Training Results Evaluator
Automatically analyzes training outcomes and provides diagnostic feedback
"""

import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class TrainingEvaluator:
    """Evaluates RL training results and provides diagnostic feedback"""

    def __init__(self):
        self.issues = []
        self.warnings = []
        self.successes = []

    def evaluate(self, results: Dict) -> Dict:
        """
        Evaluate training results and provide comprehensive feedback

        Args:
            results: Training results dict from RL trainer

        Returns:
            Evaluation dict with diagnostics, issues, and recommendations
        """
        self.issues = []
        self.warnings = []
        self.successes = []

        # Extract key metrics
        total_episodes = results.get('total_episodes', 0)
        mean_return = results.get('mean_episode_return', 0)
        max_return = results.get('max_episode_return', 0)
        min_return = results.get('min_episode_return', 0)
        std_return = results.get('std_return', 0) if 'std_return' in results else 0
        episode_returns = results.get('episode_returns', [])
        training_time = results.get('training_time', 0)
        learning_summary = results.get('learning_summary', {})

        logger.info("\n" + "="*80)
        logger.info("TRAINING RESULTS EVALUATION")
        logger.info("="*80)

        # Check 1: Episode tracking
        if total_episodes > 0:
            self.successes.append(f"✓ Episode tracking working: {total_episodes} episodes detected")
        else:
            self.issues.append("✗ CRITICAL: No episodes detected - episode tracking failed")

        # Check 2: Return magnitude
        if mean_return < 10:
            self.issues.append(
                f"✗ CRITICAL: Mean episode return too low ({mean_return:.2f}) - "
                "rewards may not be computing correctly or reward function is broken"
            )
        elif mean_return < 50:
            self.warnings.append(
                f"⚠ Low mean return ({mean_return:.2f}) - agent may not be learning effectively"
            )
        else:
            self.successes.append(f"✓ Reasonable mean return: {mean_return:.2f}")

        # Check 3: Return variance
        if len(episode_returns) > 1:
            std_return = np.std(episode_returns)
            if std_return > mean_return * 0.8:
                self.warnings.append(
                    f"⚠ High variance in returns (std={std_return:.2f} vs mean={mean_return:.2f}) - "
                    "training is unstable"
                )
            else:
                self.successes.append(f"✓ Stable returns: variance is reasonable")

        # Check 4: Learning trend
        if len(episode_returns) > 2:
            early_returns = episode_returns[:len(episode_returns)//2]
            late_returns = episode_returns[len(episode_returns)//2:]
            early_mean = np.mean(early_returns)
            late_mean = np.mean(late_returns)

            if late_mean < early_mean * 0.9:
                self.issues.append(
                    f"✗ LEARNING FAILURE: Return is declining ({early_mean:.2f} → {late_mean:.2f}) - "
                    "agent is getting worse, not better. This suggests:\n"
                    "    - Surrogate models may be inaccurate/diverging\n"
                    "    - Reward function may be misaligned with goals\n"
                    "    - Policy may be overfitting to noisy surrogate predictions"
                )
            elif late_mean > early_mean * 1.1:
                self.successes.append(
                    f"✓ Learning improvement detected: {early_mean:.2f} → {late_mean:.2f}"
                )
            else:
                self.warnings.append(
                    f"⚠ No clear learning trend: returns stable but flat"
                )

        # Check 5: Min/max spread
        if max_return > min_return:
            spread_ratio = (max_return - min_return) / max_return
            if spread_ratio > 0.9:
                self.warnings.append(
                    f"⚠ Huge spread in returns: {min_return:.2f} to {max_return:.2f} - "
                    "suggests very inconsistent performance"
                )

        # Check 6: Episode length
        mean_episode_length = results.get('mean_length', 0)
        if mean_episode_length >= 2000:  # Likely hitting max_steps
            self.warnings.append(
                f"⚠ Episodes hitting max_steps ({mean_episode_length:.0f}) - "
                "may not be terminating naturally"
            )

        # Check 7: Computational efficiency
        total_timesteps = results.get('total_timesteps', 1)
        fps = results.get('fps', 0)
        if fps < 20:
            self.warnings.append(f"⚠ Slow training: {fps:.0f} FPS (expect 50+)")
        elif fps > 20:
            self.successes.append(f"✓ Good training speed: {fps:.0f} FPS")

        # Generate report
        logger.info("\n" + "-"*80)
        logger.info("DIAGNOSTIC SUMMARY")
        logger.info("-"*80)

        for item in self.successes:
            logger.info(item)

        for item in self.warnings:
            logger.warning(item)

        for item in self.issues:
            logger.error(item)

        # Overall assessment
        logger.info("\n" + "-"*80)
        if self.issues:
            assessment = "FAILED - Critical issues detected"
            status = "❌"
        elif self.warnings and len(self.warnings) > 1:
            assessment = "POOR - Multiple concerns"
            status = "⚠️"
        elif self.warnings:
            assessment = "ACCEPTABLE - Minor issues"
            status = "⚠️"
        else:
            assessment = "SUCCESSFUL - Training completed successfully"
            status = "✅"

        logger.info(f"{status} OVERALL ASSESSMENT: {assessment}")
        logger.info("="*80)

        return {
            'status': status,
            'assessment': assessment,
            'issues': self.issues,
            'warnings': self.warnings,
            'successes': self.successes,
            'is_successful': len(self.issues) == 0 and len(self.warnings) <= 1,
        }

    def print_recommendations(self):
        """Print actionable recommendations based on evaluation"""
        logger.info("\nRECOMMENDATIONS:")
        logger.info("-"*80)

        if any("declining" in issue for issue in self.issues):
            logger.info("""
If returns are declining, consider:
1. Check surrogate model accuracy on validation data
2. Verify reward function is correctly implemented
3. Try reducing learning rate (currently 3e-4)
4. Increase n_steps for more stable gradient estimates
5. Use Bayesian Optimization instead (documented in ARCHITECTURAL_ASSESSMENT.md)
""")

        if any("unstable" in issue for issue in self.warnings):
            logger.info("""
If training is unstable, consider:
1. Reduce learning rate
2. Increase batch_size
3. Use gradient clipping
4. Verify feature scaling is correct
""")

        if not self.issues and not self.warnings:
            logger.info("""
Training appears successful! Next steps:
1. Run on full production dataset
2. Validate learned parameters on real biosensor data
3. Compare with Bayesian Optimization baseline
4. Deploy optimized parameters
""")
