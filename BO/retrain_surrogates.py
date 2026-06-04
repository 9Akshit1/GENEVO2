"""
Retrain surrogate models from current training data.

This script uses the same SurrogateBuilder as bo_main.py to ensure
consistent feature engineering, encoding, and validation.

This ensures surrogates are:
- Properly trained with [kd, sensitivity, biosensor_type, noise_preset, scenario] features
- Using fitted LabelEncoders (not hardcoded mappings)
- Well-initialized with correct scalers
- Ready for BO to use
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from core.build_surrogates import SurrogateBuilder


def setup_logging(verbose: bool = True):
    logger = logging.getLogger("retrain_surrogates")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def main(
    data_dir: str = "data",
    surrogate_dir: str = "BO/bo_results",
    verbose: bool = True,
):
    logger = setup_logging(verbose)

    logger.info("=" * 80)
    logger.info("RETRAINING SURROGATE MODELS")
    logger.info("=" * 80)
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {surrogate_dir}")

    try:
        # Use SurrogateBuilder (same as bo_main.py)
        logger.info(f"\n[1/3] Loading and extracting features...")
        builder = SurrogateBuilder(logger)
        X, feature_names, df_results = builder.load_and_extract_features(Path(data_dir))

        # Train surrogates
        logger.info(f"\n[2/3] Training surrogate models...")
        metrics = builder.train_all_surrogates(X, df_results)

        logger.info(f"[OK] Surrogate training complete")
        logger.info(f"  Detection Rate ROC-AUC: {metrics['detection_rate']['test_auc']:.4f}")
        logger.info(f"  FNR R² (median): {metrics['fnr']['r2_test']:.4f}")
        logger.info(f"  TTD R² (median): {metrics['ttd']['r2_test']:.4f}")

        # Save surrogates
        logger.info(f"\n[3/3] Saving surrogates...")
        output_dir = Path(surrogate_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        builder.save_surrogates(output_dir, version="v1")

        logger.info("\n" + "=" * 80)
        logger.info("RETRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"✓ Surrogates saved to: {output_dir / 'saved_ml'}/")
        logger.info(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("\nNow you can:")
        logger.info(f"  1. Run: python BO/validate_surrogate_accuracy.py --surrogate-dir {surrogate_dir}")
        logger.info(f"  2. Run: python BO/bo_main.py --surrogate-dir {surrogate_dir} --n-init 20 --n-iter 80")
        logger.info(f"  3. Run: python BO/validate_bo_design.py")

        return True

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Retrain surrogate models from training data using SurrogateBuilder"
    )
    parser.add_argument("--data-dir", default="data", help="Path to training data (default: data)")
    parser.add_argument(
        "--surrogate-dir",
        default="BO/bo_results",
        help="Path to save surrogates (default: BO/bo_results). Surrogates will be saved to <surrogate_dir>/saved_ml/"
    )
    parser.add_argument("--verbose", action="store_true", default=True, help="Enable verbose logging")

    args = parser.parse_args()

    success = main(
        data_dir=args.data_dir,
        surrogate_dir=args.surrogate_dir,
        verbose=args.verbose,
    )

    sys.exit(0 if success else 1)
