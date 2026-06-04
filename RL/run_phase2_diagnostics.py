#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 2 Diagnostic Orchestrator

Runs both critical Phase 2 diagnostics in sequence:
1. Action Sensitivity Map - measures if actions affect rewards
2. Random Baseline - establishes baseline for PPO comparison

This will answer the fundamental question:
"Is this problem actually learnable with RL?"
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime
import json

# Add RL module to path
sys.path.insert(0, str(Path(__file__).parent))

from logging_config import configure_logging


def main():
    """Run Phase 2 diagnostics"""

    data_dir = Path("data")
    output_dir = Path("RL/rl_results_diagnostic")

    # Setup logging
    log_dir = output_dir / "diagnostic_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(log_dir, "phase2_orchestrator", verbose=True)

    logger.info("=" * 80)
    logger.info("PHASE 2: ENVIRONMENT ASSUMPTION VALIDATION")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    logger.info("This phase answers: Is this problem learnable with RL?")
    logger.info("")

    # Validate data directory
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return 1

    try:
        # ====================================================================
        # Diagnostic 2.1: Action Sensitivity
        # ====================================================================
        logger.info("=" * 80)
        logger.info("DIAGNOSTIC 2.1: ACTION SENSITIVITY MAP")
        logger.info("=" * 80)
        logger.info("Running action sensitivity analysis...")
        logger.info("This measures how reward responds to each action dimension")
        logger.info("")

        result = subprocess.run(
            [sys.executable, "diagnostic_action_sensitivity.py",
             "--data-dir", str(data_dir),
             "--output-dir", str(output_dir)],
            cwd=Path(__file__).parent,
            capture_output=False,
            text=True
        )

        if result.returncode != 0:
            logger.error("Action sensitivity diagnostic failed")
            return 1

        logger.info("\n✓ Action sensitivity diagnostic complete")
        logger.info("  Check plots in: RL/rl_results_diagnostic/diagnostic_plots/")

        # ====================================================================
        # Diagnostic 2.2: Random Baseline
        # ====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("DIAGNOSTIC 2.2: RANDOM BASELINE")
        logger.info("=" * 80)
        logger.info("Running random baseline for comparison...")
        logger.info("This establishes what PPO needs to beat")
        logger.info("")

        result = subprocess.run(
            [sys.executable, "diagnostic_random_baseline.py",
             "--data-dir", str(data_dir),
             "--output-dir", str(output_dir)],
            cwd=Path(__file__).parent,
            capture_output=False,
            text=True
        )

        if result.returncode != 0:
            logger.error("Random baseline diagnostic failed")
            return 1

        logger.info("\n✓ Random baseline diagnostic complete")
        logger.info("  Results saved to: RL/rl_results_diagnostic/diagnostic_logs/")

        # ====================================================================
        # Summary Report
        # ====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 2 COMPLETE")
        logger.info("=" * 80)

        logger.info("\nGenerated Outputs:")
        logger.info("  Plots:")
        logger.info("    - action_sensitivity_dim*.png (action response curves)")
        logger.info("    - random_vs_ppo_comparison.png (baseline comparison)")
        logger.info("")
        logger.info("  Data:")
        logger.info("    - action_sensitivity_results.json")
        logger.info("    - random_baseline_results.json")
        logger.info("")

        logger.info("Next Steps Based on Results:")
        logger.info("")
        logger.info("1. CHECK ACTION SENSITIVITY:")
        logger.info("   - If reward_range < 0.05 for all actions:")
        logger.info("     → Environment is INSENSITIVE, RL will NOT work")
        logger.info("     → SWITCH to Bayesian Optimization or CMA-ES")
        logger.info("")
        logger.info("   - If reward_range > 0.3 for most actions:")
        logger.info("     → Environment is RESPONSIVE, continue with RL")
        logger.info("")

        logger.info("2. CHECK RANDOM BASELINE:")
        logger.info("   - If PPO mean >> Random mean (>20% improvement):")
        logger.info("     → RL is learning effectively, continue")
        logger.info("")
        logger.info("   - If PPO mean ≈ Random mean (+5% to +20%):")
        logger.info("     → RL is learning weakly, review reward scaling")
        logger.info("")
        logger.info("   - If PPO mean < Random mean:")
        logger.info("     → SOMETHING IS WRONG, RL is broken")
        logger.info("")

        logger.info("3. RECOMMENDATIONS:")
        logger.info("   Based on sensitivity + baseline comparison, decide:")
        logger.info("")
        logger.info("   ✓ CONTINUE with RL if:")
        logger.info("     - Actions are sensitive (range > 0.2)")
        logger.info("     - PPO beats random by >15%")
        logger.info("     → Move to Phase 3 (reward decomposition, RL metrics)")
        logger.info("")
        logger.info("   ❌ SWITCH to Optimization if:")
        logger.info("     - Actions are insensitive (range < 0.1)")
        logger.info("     - Problem seems like parameter search")
        logger.info("     → Use Bayesian Optimization or CMA-ES instead")
        logger.info("")

        logger.info("=" * 80)
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Phase 2 failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
