#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diagnostic script to investigate episode accounting issues.
Runs a short training session with detailed logging to understand:
1. Why only 8 episodes were detected when 50 were requested
2. Whether all 4 parallel environments are actually active
3. Per-environment episode completion rates
"""

import logging
import sys
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# Add RL module to path
sys.path.insert(0, str(Path(__file__).parent))

from data_processor import DataProcessor
from surrogate_trainer import SurrogateTrainer
from rl_trainer import RLTrainer
from logging_config import configure_logging


def run_diagnostic(data_dir: Path, output_dir: Path,
                   total_episodes: int = 5,  # Short test run
                   n_envs: int = 4):
    """Run short training with detailed diagnostics"""

    output_dir = Path(output_dir)
    log_dir = output_dir / "diagnostic_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup detailed logging
    logger = configure_logging(
        log_dir=log_dir,
        log_name="episode_diagnostic",
        verbose=True,  # Verbose to capture debug logs
        suppress_third_party=False  # See third-party logs too
    )

    logger.info("=" * 80)
    logger.info("EPISODE ACCOUNTING DIAGNOSTIC")
    logger.info("=" * 80)
    logger.info(f"Test configuration:")
    logger.info(f"  Total episodes: {total_episodes}")
    logger.info(f"  Parallel environments: {n_envs}")
    logger.info(f"  Steps per episode: 2048")
    logger.info(f"  Expected total timesteps: {total_episodes * 2048 * n_envs}")
    logger.info(f"  Expected episode completions: ~{total_episodes * n_envs}")
    logger.info("")

    try:
        # Load data
        logger.info("Loading data...")
        processor = DataProcessor(logger)
        df = processor.load_master_index(data_dir)
        processor.validate_data(df, processed=False)
        df = processor.process_data(df)
        processor.validate_data(df, processed=True)

        X_rl, rl_info = processor.prepare_rl_features(df)
        X_surr, y_dr, y_fnr, y_ttd = processor.prepare_surrogate_features(df)

        logger.info(f"Data loaded: {X_rl.shape[0]} samples")

        # Train surrogates quickly
        logger.info("\nTraining surrogates (quick mode)...")
        trainer = SurrogateTrainer(logger)
        surr_metrics = trainer.train_all_surrogates(X_surr, y_dr, y_fnr, y_ttd)
        surrogates = trainer.get_models()
        logger.info("Surrogates trained")

        # Train RL with diagnostics
        logger.info("\nStarting RL training with detailed diagnostics...")
        logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        rl_trainer = RLTrainer(logger)
        encoders = processor.get_encoders() if hasattr(processor, 'get_encoders') else {}
        surrogate_scaler = encoders.get('surrogate_feature_scaler', None)

        training_info = rl_trainer.train_agent(
            surrogates, X_rl, output_dir,
            total_episodes=total_episodes,
            steps_per_episode=2048,
            n_envs=n_envs,
            surrogate_feature_scaler=surrogate_scaler
        )

        logger.info("\n" + "=" * 80)
        logger.info("DIAGNOSTIC SUMMARY")
        logger.info("=" * 80)
        logger.info(f"\nExpected vs Actual Episodes:")
        logger.info(f"  Expected: {total_episodes} episodes requested")
        logger.info(f"  Expected per env: {total_episodes} episodes per {n_envs} envs")
        logger.info(f"  Actual detected: {training_info['total_episodes']} episodes")

        if 'episode_detections_per_env' in training_info:
            logger.info(f"\n  Per-environment breakdown:")
            for env_idx, count in training_info['episode_detections_per_env'].items():
                logger.info(f"    Environment {env_idx}: {count} episodes")

        total_detected = training_info.get('total_episode_detections', training_info['total_episodes'])
        expected_total = total_episodes * n_envs

        logger.info(f"\n  Diagnostic Assessment:")
        if total_detected < expected_total * 0.5:
            logger.warning(f"  ⚠️  ISSUE DETECTED: Only {total_detected}/{expected_total} episodes detected")
            logger.warning(f"     This is a critical accounting bug. Possible causes:")
            logger.warning(f"     1. Callback episode detection is broken")
            logger.warning(f"     2. Not all environments are resetting properly")
            logger.warning(f"     3. Vectorization is broken (only 1 env active)")
        elif total_detected < expected_total * 0.9:
            logger.warning(f"  ⚠️  PARTIAL ISSUE: Only {total_detected}/{expected_total} episodes detected")
            logger.warning(f"     Some environments may not be resetting")
        else:
            logger.info(f"  ✓ Episode accounting looks correct: {total_detected}/{expected_total}")

        logger.info(f"\nTraining Statistics:")
        logger.info(f"  Duration: {training_info['training_time']:.1f}s")
        logger.info(f"  FPS: {training_info['fps']:.0f}")
        logger.info(f"  Mean episode return: {training_info['mean_episode_return']:.4f}")
        logger.info(f"  Median episode length: 2048 (hardcoded max)")

        logger.info("\n" + "=" * 80)
        logger.info("Next Steps:")
        logger.info("1. Check the per-environment episode counts above")
        logger.info("2. If highly imbalanced (e.g., 5,0,0,0), vectorization is broken")
        logger.info("3. If all near 0, episode detection callback is broken")
        logger.info("4. If roughly balanced, increase episodes and run full diagnostic")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Diagnostic failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Episode accounting diagnostic')
    parser.add_argument('--data-dir', type=Path, default=Path("data"),
                       help='Path to data directory')
    parser.add_argument('--output-dir', type=Path,
                       default=Path("RL/rl_results_diagnostic"),
                       help='Path to output directory')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of test episodes (default: 5 for quick test)')
    parser.add_argument('--envs', type=int, default=4,
                       help='Number of parallel environments (default: 4)')

    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)

    sys.exit(run_diagnostic(args.data_dir, args.output_dir,
                           total_episodes=args.episodes, n_envs=args.envs))
