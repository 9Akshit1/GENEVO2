#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GENEVO2 Complete RL Training Pipeline
Master orchestrator for data processing, ML training, RL training, and visualization
"""

import argparse
import logging
import json
import sys
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime

# Suppress non-critical warnings
warnings.filterwarnings('ignore', message='.*sklearn.utils.parallel.delayed.*')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Import pipeline components
from data_processor import DataProcessor
from surrogate_trainer import SurrogateTrainer
from rl_trainer import RLTrainer
from visualizer import Visualizer
from cache_manager import CacheManager
from logging_config import configure_logging


def setup_logging(log_dir: Path):
    """Setup production-grade logging"""
    return configure_logging(
        log_dir=log_dir,
        log_name="pipeline",
        verbose=False,
        suppress_third_party=True
    )


def ask_user_cached(cache_manager: CacheManager, component: str) -> bool:
    """Ask user if they want to use cached component"""
    if not cache_manager.has_cached(component):
        return False

    response = input(f"\nUse cached {component}? (y/n): ").strip().lower()
    return response == 'y'


def run_pipeline(data_dir: Path, output_dir: Path, use_cached: bool = False,
                total_episodes: int = 50, clear_cache: bool = False):
    """Run complete RL training pipeline"""

    # Setup output directories
    output_dir = Path(output_dir)
    log_dir = output_dir / "logs"
    cache_dir = output_dir / ".cache"

    logger = setup_logging(log_dir)
    logger.info("=" * 80)
    logger.info("GENEVO2 RL Training Pipeline - Complete System")
    logger.info("=" * 80)
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Setup output subdirectories
    for subdir in ['models', 'plots', 'results', 'rl_logs', 'saved_ml']:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Initialize cache
    cache = CacheManager(cache_dir)
    if clear_cache:
        logger.info("Clearing cache...")
        cache.clear_cache()

    try:
        # =====================================================================
        # Step 1: Data Processing
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: Data Processing")
        logger.info("=" * 80)

        processor = DataProcessor(logger)

        # Check cache
        if use_cached and cache.has_cached('processed_data'):
            logger.info("Loading processed data from cache...")
            data_result = cache.get_cached('processed_data')
            if data_result:
                df = data_result['df']
                X_rl = data_result['X_rl']
                X_surr = data_result['X_surr']
                y_dr = data_result['y_dr']
                y_fnr = data_result['y_fnr']
                y_ttd = data_result['y_ttd']
            else:
                logger.warning("Failed to load from cache, reprocessing...")
                use_cached = False

        if not use_cached or not cache.has_cached('processed_data'):
            # Load and process data
            df = processor.load_master_index(data_dir)
            processor.validate_data(df, processed=False)
            df = processor.process_data(df)
            processor.validate_data(df, processed=True)

            # Prepare features
            X_rl, rl_info = processor.prepare_rl_features(df)
            X_surr, y_dr, y_fnr, y_ttd = processor.prepare_surrogate_features(df)

            # Cache processed data
            data_result = {
                'df': df,
                'X_rl': X_rl,
                'X_surr': X_surr,
                'y_dr': y_dr,
                'y_fnr': y_fnr,
                'y_ttd': y_ttd,
                'rl_info': rl_info,
                'encoders': processor.get_encoders(),
            }
            cache.save_cache('processed_data', data_result)

        logger.info(f"[OK] Data processing complete")
        logger.info(f"  Features shape: {X_rl.shape}")

        # =====================================================================
        # Step 2: Surrogate Model Training
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Surrogate Model Training")
        logger.info("=" * 80)

        trainer = SurrogateTrainer(logger)

        # Check cache
        if use_cached and cache.has_cached('surrogates'):
            logger.info("Loading surrogates from cache...")
            surr_result = cache.get_cached('surrogates')
            if surr_result:
                surrogates = surr_result['models']
                surr_metrics = surr_result['metrics']
            else:
                logger.warning("Failed to load from cache, retraining...")
                use_cached = False

        if not use_cached or not cache.has_cached('surrogates'):
            # Train surrogates
            surr_metrics = trainer.train_all_surrogates(X_surr, y_dr, y_fnr, y_ttd)
            surrogates = trainer.get_models()

            # Save surrogates to disk (persistent artifacts)
            trainer.save_surrogates(output_dir, version='v1')

            # Cache surrogates in memory
            surr_result = {
                'models': surrogates,
                'metrics': surr_metrics,
            }
            cache.save_cache('surrogates', surr_result)

        logger.info(f"[OK] Surrogate training complete")

        # Save metrics summary
        metrics_file = output_dir / "results" / "surrogate_metrics.json"
        with open(metrics_file, 'w') as f:
            # Convert numpy types to Python types for JSON (skip non-numeric values)
            json_metrics = {}
            for key, val in surr_metrics.items():
                json_metrics[key] = {}
                for k, v in val.items():
                    if isinstance(v, str):
                        json_metrics[key][k] = v  # Keep strings as-is
                    else:
                        json_metrics[key][k] = float(v)  # Convert numeric to float
            json.dump(json_metrics, f, indent=2)
        logger.info(f"  Metrics saved to: {metrics_file.name}")
        logger.info(f"  Models saved to: saved_ml/")

        # =====================================================================
        # Step 3: RL Agent Training
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: RL Agent Training")
        logger.info("=" * 80)

        rl_trainer = RLTrainer(logger)
        # Get the surrogate feature scaler for proper reward computation
        encoders = processor.get_encoders() if hasattr(processor, 'get_encoders') else {}
        surrogate_scaler = encoders.get('surrogate_feature_scaler', None)

        training_info = rl_trainer.train_agent(
            surrogates, X_rl, output_dir,
            total_episodes=total_episodes,
            steps_per_episode=2048,
            n_envs=4,  # Use 4 parallel environments for 4x faster rollout collection
            surrogate_feature_scaler=surrogate_scaler
        )

        logger.info(f"[OK] RL training complete")

        # =====================================================================
        # Step 4: Visualization
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: Generating Visualizations")
        logger.info("=" * 80)

        viz = Visualizer(output_dir, logger)

        # Dataset overview
        data_dict = {
            'detection_rate': y_dr,
            'fnr': y_fnr,
            'ttd': y_ttd,
            'snr': X_surr[:, 0],  # SNR is first feature
        }
        viz.plot_dataset_overview(data_dict)

        # Surrogate quality
        y_dict = {
            'detection_rate': y_dr,
            'fnr': y_fnr,
            'ttd': y_ttd,
        }
        viz.plot_surrogate_quality(surrogates, X_surr, y_dict)

        # Metric correlations
        viz.plot_metric_correlations(data_dict)

        # Training summary (use test-set R² for honest performance)
        training_info.update({
            'n_samples': len(df),
            'dr_r2_test': surr_metrics['detection_rate']['r2_test'],
            'fnr_r2_test': surr_metrics['fnr']['r2_test'],
            'ttd_r2_test': surr_metrics['ttd']['r2_test'],
            'dr_r2_train': surr_metrics['detection_rate']['r2_train'],
            'fnr_r2_train': surr_metrics['fnr']['r2_train'],
            'ttd_r2_train': surr_metrics['ttd']['r2_train'],
            'dr_mean': float(y_dr.mean()),
            'dr_std': float(y_dr.std()),
            'fnr_mean': float(y_fnr.mean()),
            'fnr_std': float(y_fnr.std()),
            'ttd_mean': float(y_ttd.mean()),
            'ttd_std': float(y_ttd.std()),
            'snr_min': float(X_surr[:, 0].min()),
            'snr_max': float(X_surr[:, 0].max()),
        })
        viz.plot_training_summary(training_info)

        logger.info(f"[OK] Visualization complete")

        # =====================================================================
        # Step 5: Summary and Verification
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 80)

        # Save comprehensive summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'status': 'COMPLETE',
            'data_dir': str(data_dir),
            'output_dir': str(output_dir),
            'dataset': {
                'n_samples': len(df),
                'detection_rate': {
                    'mean': float(y_dr.mean()),
                    'std': float(y_dr.std()),
                    'min': float(y_dr.min()),
                    'max': float(y_dr.max()),
                },
                'fnr': {
                    'mean': float(y_fnr.mean()),
                    'std': float(y_fnr.std()),
                    'min': float(y_fnr.min()),
                    'max': float(y_fnr.max()),
                },
                'ttd': {
                    'mean': float(y_ttd.mean()),
                    'std': float(y_ttd.std()),
                    'min': float(y_ttd.min()),
                    'max': float(y_ttd.max()),
                },
            },
            'surrogates': {k: {k2: float(v2) if isinstance(v2, (int, float, np.number)) else v2
                               for k2, v2 in v.items()} for k, v in surr_metrics.items()},
            'rl_training': {k: float(v) if isinstance(v, (int, float)) else v
                           for k, v in training_info.items()},
            'cache': {
                'cached_items': cache.list_cached(),
            },
        }

        summary_file = output_dir / "results" / "pipeline_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"\nSummary saved to: {summary_file}")

        # List output files
        logger.info("\nOutput Files:")
        for subdir in ['models', 'plots', 'results', 'rl_logs']:
            subdir_path = output_dir / subdir
            if subdir_path.exists():
                files = list(subdir_path.glob('*'))
                logger.info(f"  {subdir}/: {len(files)} items")

        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE SUCCESSFUL")
        logger.info("=" * 80)
        logger.info(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"\nResults available at: {output_dir}")
        logger.info(f"Logs available at: {log_dir}")

        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


def main():
    """Main entry point"""

    parser = argparse.ArgumentParser(
        description='GENEVO2 RL Training Pipeline - Complete System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with default directories
  python pipeline_main.py

  # Use cached components where available
  python pipeline_main.py --use-cached

  # Train for 100 episodes instead of 50
  python pipeline_main.py --episodes 100

  # Clear cache and retrain everything
  python pipeline_main.py --clear-cache

  # Custom data and output directories
  python pipeline_main.py --data-dir /path/to/data --output-dir /path/to/output
        """
    )

    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path("data"),
        help='Path to data directory (default: data/)'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path("RL/rl_results_production"),
        help='Path to output directory (default: RL/rl_results_production/)'
    )

    parser.add_argument(
        '--episodes',
        type=int,
        default=50,
        help='Number of RL training episodes (default: 50)'
    )

    parser.add_argument(
        '--use-cached',
        action='store_true',
        help='Use cached data and models if available'
    )

    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear cache before running'
    )

    args = parser.parse_args()

    # Validate paths
    if not args.data_dir.exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        return 1

    # Run pipeline
    return run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        use_cached=args.use_cached,
        total_episodes=args.episodes,
        clear_cache=args.clear_cache,
    )


if __name__ == "__main__":
    sys.exit(main())
