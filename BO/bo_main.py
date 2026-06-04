#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GENEVO2 Bayesian Optimization Main Entry Point

Complete command-line interface for running BO optimization on biosensor parameters.
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

# Ensure we can import from BO
sys.path.insert(0, str(Path(__file__).parent))

# BO modules
from core.surrogate_loader import SurrogateLoader
from core.build_surrogates import SurrogateBuilder
from search_space.biosensor_space import BiosensorSearchSpace
from evaluation.physics_forward_model import PhysicsForwardModel
from evaluation.objective_function import ObjectiveFunction
from evaluation.robustness_analyzer import RobustnessAnalyzer
from acquisition.acquisition_functions import ExpectedImprovement
from optimizer.gaussian_process_bo import GaussianProcessBO
from optimizer.bo_pipeline import BOPipeline
from diagnostics.bo_vs_rl_comparison import BOVsRLComparison

# Logging setup
def setup_logging(log_dir: Path, verbose: bool = False) -> logging.Logger:
    """Setup production-grade logging."""
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("bo_optimization")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # File handler
    fh = logging.FileHandler(log_dir / "bo_optimization.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO if not verbose else logging.DEBUG)
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)

    return logger


def main():
    """Main entry point with CLI arguments."""

    parser = argparse.ArgumentParser(
        description="GENEVO2 Bayesian Optimization for Biosensor Parameter Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (5 initial + 10 iterations)
  python bo_main.py --n-init 5 --n-iter 10

  # Full optimization (20 initial + 80 iterations = 100 evals)
  python bo_main.py --n-init 20 --n-iter 80

  # Custom data and surrogate directories
  python bo_main.py --data-dir /path/to/data --surrogate-dir /path/to/surrogates

  # Enable RL comparison
  python bo_main.py --compare-rl --rl-dir rl_results_v7

  # Save to custom output
  python bo_main.py --output-dir bo_results_custom
        """,
    )

    # Paths
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Path to data directory with master_index.csv",
    )

    parser.add_argument(
        "--surrogate-dir",
        type=Path,
        default=Path("BO/bo_results"),
        help="Path to surrogate models directory (default: BO/bo_results). "
             "Surrogates are stored in surrogate_dir/saved_ml/",
    )

    parser.add_argument(
        "--retrain-surrogates",
        action="store_true",
        help="Force retraining of surrogate models even if they exist",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("BO/bo_results"),
        help="Output directory for results (default: BO/bo_results)",
    )

    parser.add_argument(
        "--rl-dir",
        type=Path,
        default=Path("rl_results_v7"),
        help="RL results directory for comparison (default: rl_results_v7)",
    )

    # BO parameters
    parser.add_argument(
        "--n-init",
        type=int,
        default=20,
        help="Number of initial random samples (default: 20)",
    )

    parser.add_argument(
        "--n-iter",
        type=int,
        default=80,
        help="Number of BO iterations (default: 80)",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    # Flags
    parser.add_argument(
        "--compare-rl",
        action="store_true",
        help="Compare results with RL baseline",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Validate paths
    if not args.data_dir.exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        return 1

    # Setup output (create surrogate dir if using default)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.surrogate_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(log_dir, verbose=args.verbose)

    try:
        logger.info("=" * 80)
        logger.info("GENEVO2 Bayesian Optimization")
        logger.info("=" * 80)
        logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Data directory: {args.data_dir.resolve()}")
        logger.info(f"Surrogate directory: {args.surrogate_dir.resolve()}")
        logger.info(f"Output directory: {args.output_dir.resolve()}")

        # =====================================================================
        # Step 1: Build or Load Surrogates
        # =====================================================================
        logger.info("\n[1/5] Building/Loading Surrogate Models")
        logger.info("-" * 80)

        # Check if surrogates need to be built
        saved_ml_dir = args.surrogate_dir / "saved_ml"
        scaler_file = saved_ml_dir / "scaler_v1.pkl" if saved_ml_dir.exists() else None
        surrogates_exist = scaler_file and scaler_file.exists()

        if args.retrain_surrogates or not surrogates_exist:
            if args.retrain_surrogates:
                logger.info("Retraining surrogates (--retrain-surrogates flag set)...")
            else:
                logger.info("Surrogates not found. Building from data...")

            builder = SurrogateBuilder(logger)

            try:
                # Extract features from data
                X, _, df_results = builder.load_and_extract_features(args.data_dir)

                # Train surrogates
                metrics = builder.train_all_surrogates(X, df_results)
                logger.info(f"[OK] Surrogate training complete")

                # Log metrics (different types for DR classifier vs regressors)
                logger.info(f"  Detection Rate (Classifier):")
                logger.info(f"    Test ROC-AUC: {metrics['detection_rate']['test_auc']:.4f}")
                logger.info(f"    Test Brier: {metrics['detection_rate']['test_brier']:.4f}")
                logger.info(f"  FNR (Quantile Regression):")
                logger.info(f"    Test R²: {metrics['fnr']['r2_test']:.4f}")
                logger.info(f"    Test RMSE: {metrics['fnr']['rmse_test']:.4f}")
                logger.info(f"  TTD (Quantile Regression):")
                logger.info(f"    Test R²: {metrics['ttd']['r2_test']:.4f}")
                logger.info(f"    Test RMSE: {metrics['ttd']['rmse_test']:.4f}")

                # Save to surrogate dir
                builder.save_surrogates(args.surrogate_dir, version="v1")
                logger.info(f"[OK] Surrogates saved to {args.surrogate_dir}")
            except Exception as e:
                logger.error(f"Failed to train surrogates: {e}", exc_info=True)
                return 1
        else:
            logger.info(f"Found existing surrogate files at {saved_ml_dir}")

        # Load surrogates
        try:
            surrogate_loader = SurrogateLoader(str(args.surrogate_dir))
            if not surrogate_loader.is_initialized():
                logger.error("Surrogates loaded but not fully initialized")
                return 1
            logger.info(f"[OK] Loaded {len(surrogate_loader.surrogates)} surrogate models")
        except FileNotFoundError as e:
            logger.error(f"Failed to load surrogates: {e}")
            return 1
        except Exception as e:
            logger.error(f"Unexpected error loading surrogates: {e}", exc_info=True)
            return 1

        # =====================================================================
        # Step 2: Initialize Components
        # =====================================================================
        logger.info("\n[2/5] Initializing BO Components")
        logger.info("-" * 80)

        search_space = BiosensorSearchSpace()
        physics_model = PhysicsForwardModel()
        objective_fn = ObjectiveFunction(physics_model, surrogate_loader)
        robustness_analyzer = RobustnessAnalyzer(objective_fn)
        acquisition_fn = ExpectedImprovement(xi=0.01)

        logger.info(f"[OK] Initialized all components")
        logger.info(search_space.summary())

        # =====================================================================
        # Step 3: Run BO Optimization
        # =====================================================================
        logger.info("\n[3/5] Running Bayesian Optimization")
        logger.info("-" * 80)

        optimizer = GaussianProcessBO(
            objective_fn=objective_fn,
            search_space=search_space,
            acquisition_fn=acquisition_fn,
            n_init=args.n_init,
            n_iter=args.n_iter,
            random_state=args.random_state,
        )

        pipeline = BOPipeline(
            optimizer=optimizer,
            objective_fn=objective_fn,
            search_space=search_space,
            robustness_analyzer=robustness_analyzer,
            output_dir=args.output_dir,
        )

        result = pipeline.run()

        # =====================================================================
        # Step 4: Optional RL Comparison
        # =====================================================================
        if args.compare_rl:
            logger.info("\n[4/5] Comparing with RL Baseline")
            logger.info("-" * 80)

            comparator = BOVsRLComparison(args.rl_dir)
            comparison = comparator.compare(result["bo_result"])
            comparator.save_comparison(comparison, args.output_dir / "results" / "bo_vs_rl.json")

            logger.info("[OK] Comparison saved")
        else:
            logger.info("\n[4/5] Skipping RL Comparison (use --compare-rl to enable)")

        # =====================================================================
        # Step 5: Summary
        # =====================================================================
        logger.info("\n[5/5] Summary")
        logger.info("-" * 80)

        best_config = result["bo_result"]["config_best"]
        best_score = result["bo_result"]["y_best"]
        details = result["details"]

        logger.info(f"Best composite score: {best_score:.4f}")
        logger.info(f"Predicted Detection Rate: {details['dr_pred']:.4f}")
        logger.info(f"Predicted False Negative Rate: {details['fnr_pred']:.4f}")
        logger.info(f"Predicted Time to Detection: {details['ttd_pred_s']:.1f} s")
        logger.info(f"Estimated SNR: {details['snr_db_est']:.2f} dB")
        logger.info(f"\nBest Biosensor Design:")
        logger.info(f"  Type: {best_config['biosensor_type']}")
        logger.info(f"  Kd: {best_config['kd_nm']:.4f} nM")
        logger.info(f"  Sensitivity: {best_config['sensitivity']:.4f}")
        if best_config["biosensor_type"] == "amplifying":
            logger.info(f"  Response time: {best_config['response_time_s']:.1f} s")
        logger.info(f"  Noise preset: {best_config['noise_preset']}")
        logger.info(f"  Target scenario: {best_config['target_scenario']}")

        logger.info("\n" + "=" * 80)
        logger.info("BO OPTIMIZATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Results available at: {args.output_dir.resolve()}")
        logger.info(f"Best config: {args.output_dir / 'results' / 'best_config.json'}")

        return 0

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
