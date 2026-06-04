"""
Surrogate Model Validation: Check accuracy on training domain.

CRITICAL: Before trusting BO, we must verify that surrogates are accurate
in the regions they were trained on. This script:

1. Loads RL training data (master_index.csv)
2. For each sample in training data:
   a. Extract: snr_db, biosensor_type, noise_preset (surrogates inputs)
   b. Extract: actual detection_rate, fnr, ttd (ground truth)
   c. Get surrogate predictions for those inputs
   d. Compare predicted vs actual
3. Reports whether surrogates are trustworthy

If surrogates are bad even on training data, BO cannot work.
"""

import json
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
import traceback

sys.path.insert(0, str(Path(__file__).parent.parent))

from BO.core.surrogate_loader import SurrogateLoader


def setup_logging(verbose: bool = True):
    logger = logging.getLogger("surrogate_validator")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def load_training_data(data_dir: str) -> pd.DataFrame:
    """Load RL training data from master_index.csv."""
    master_path = Path(data_dir) / "master_index.csv"
    if not master_path.exists():
        raise FileNotFoundError(f"master_index.csv not found at {master_path}")

    return pd.read_csv(master_path)


def compute_error(predicted: float, actual: float) -> float:
    """Compute relative error."""
    if actual == 0:
        return np.inf if predicted != 0 else 0.0
    return 100.0 * abs(predicted - actual) / abs(actual)


def main(
    data_dir: str = "data",
    surrogate_dir: str = "BO/bo_results",
    num_samples: int = 200,
    verbose: bool = True,
):
    logger = setup_logging(verbose)

    logger.info("=" * 80)
    logger.info("SURROGATE ACCURACY VALIDATION")
    logger.info("=" * 80)
    logger.info("WARNING: This tests surrogates on training domain samples.")
    logger.info("For true validation, surrogates should be tested on held-out test set.")
    logger.info("-" * 80)

    # Load training data
    logger.info(f"\n[1/4] Loading RL training data from {data_dir}...")
    try:
        training_data = load_training_data(data_dir)
        logger.info(f"✓ Loaded {len(training_data)} training samples")
        logger.info(f"  Columns: {list(training_data.columns)}")
    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        return False

    # Sample designs from training set
    logger.info(f"\n[2/4] Sampling {num_samples} designs from training set...")
    sample_indices = np.random.choice(len(training_data), min(num_samples, len(training_data)), replace=False)
    sample_data = training_data.iloc[sample_indices]
    logger.info(f"✓ Selected {len(sample_data)} designs")

    # Initialize surrogates
    logger.info(f"\n[3/4] Initializing surrogates...")
    try:
        surrogate_loader = SurrogateLoader(surrogate_dir)

        if not surrogate_loader.is_initialized():
            logger.error("Failed to initialize surrogates")
            return False

        logger.info(f"✓ Loaded surrogates with scaler and encoders")
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        traceback.print_exc()
        return False

    # Validate each sample
    logger.info(f"\n[4/4] Testing surrogate accuracy on each sample...")
    logger.info("\n" + "-" * 80)

    dr_errors = []
    fnr_errors = []
    ttd_errors = []

    for idx, (i, row) in enumerate(sample_data.iterrows(), 1):
        try:
            # Load metadata to get original design parameters
            metadata_file = Path(data_dir) / row['metadata_file']
            if not metadata_file.exists():
                logger.warning(f"  Metadata not found: {metadata_file}")
                continue

            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Extract design parameters from metadata
            biosensor_cfg = metadata['biosensor_config']
            kd = float(biosensor_cfg['kd'])
            sensitivity = float(biosensor_cfg['sensitivity'])
            biosensor_type = biosensor_cfg['circuit_type']
            noise_preset = metadata['noise_preset']
            scenario = metadata['scenario']

            # Extract ground truth outputs
            actual_dr = float(row['detection_rate'])
            actual_fnr = float(row['false_negative_rate'])
            actual_ttd = float(row['time_to_detection'])

            logger.info(f"\nSample {idx}/{len(sample_data)}:")
            logger.info(f"  Design: {biosensor_type}, Kd={kd:.4f}, Sens={sensitivity:.4f}, Noise={noise_preset}, Scenario={scenario}")

            # Get surrogate predictions
            try:
                # Encode categorical variables
                biosensor_encoded, noise_encoded, scenario_encoded = surrogate_loader.encode_categorical(
                    biosensor_type, noise_preset, scenario
                )

                # Prepare input [kd, sensitivity, biosensor_type_enc, noise_preset_enc, scenario_enc]
                X_raw = np.array([[kd, sensitivity, biosensor_encoded, noise_encoded, scenario_encoded]], dtype=np.float32)
                X_scaled = surrogate_loader.scaler.transform(X_raw)

                # Predict
                pred_dr = float(np.clip(surrogate_loader.surrogates['detection_rate'].predict(X_scaled)[0], 0, 1))
                pred_fnr = float(np.clip(surrogate_loader.surrogates['fnr'].predict(X_scaled)[0], 0, 1))
                pred_ttd = float(np.clip(surrogate_loader.surrogates['ttd'].predict(X_scaled)[0], 400, 9000))

            except Exception as e:
                logger.warning(f"  Surrogate prediction failed: {e}")
                continue

            # Compute errors
            dr_err = compute_error(pred_dr, actual_dr)
            fnr_err = compute_error(pred_fnr, actual_fnr)
            ttd_err = compute_error(pred_ttd, actual_ttd)

            logger.info(f"  Actual:    DR={actual_dr:.4f}, FNR={actual_fnr:.4f}, TTD={actual_ttd:.1f}s")
            logger.info(f"  Predicted: DR={pred_dr:.4f}, FNR={pred_fnr:.4f}, TTD={pred_ttd:.1f}s")
            logger.info(f"  Error:     DR={dr_err:.1f}%, FNR={fnr_err:.1f}%, TTD={ttd_err:.1f}%")

            # Track errors (only finite values)
            if np.isfinite(dr_err):
                dr_errors.append(dr_err)
            if np.isfinite(fnr_err):
                fnr_errors.append(fnr_err)
            if np.isfinite(ttd_err):
                ttd_errors.append(ttd_err)

        except Exception as e:
            logger.warning(f"Sample {idx} failed: {e}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SURROGATE ACCURACY SUMMARY")
    logger.info("=" * 80)

    if dr_errors:
        avg_dr_err = float(np.mean(dr_errors))
        max_dr_err = float(np.max(dr_errors))
        logger.info(f"\nDetection Rate Error:")
        logger.info(f"  Mean: {avg_dr_err:.1f}%")
        logger.info(f"  Max:  {max_dr_err:.1f}%")

    if fnr_errors:
        avg_fnr_err = float(np.mean(fnr_errors))
        max_fnr_err = float(np.max(fnr_errors))
        logger.info(f"\nFalse Negative Rate Error:")
        logger.info(f"  Mean: {avg_fnr_err:.1f}%")
        logger.info(f"  Max:  {max_fnr_err:.1f}%")

    if ttd_errors:
        avg_ttd_err = float(np.mean(ttd_errors))
        max_ttd_err = float(np.max(ttd_errors))
        logger.info(f"\nTime-to-Detection Error:")
        logger.info(f"  Mean: {avg_ttd_err:.1f}%")
        logger.info(f"  Max:  {max_ttd_err:.1f}%")

    # Assessment
    logger.info("\n" + "=" * 80)
    logger.info("ASSESSMENT")
    logger.info("=" * 80)

    avg_all_errors = []
    for errs in [dr_errors, fnr_errors, ttd_errors]:
        if errs:
            avg_all_errors.extend(errs)

    if avg_all_errors:
        overall_avg = float(np.mean(avg_all_errors))

        if overall_avg < 10:
            logger.info(f"\n✓ EXCELLENT: Surrogates are very accurate ({overall_avg:.1f}% avg error)")
            logger.info("  → BO should work well — surrogates are trustworthy")
            logger.info("  → Issue is extrapolation, not surrogate accuracy")
            logger.info("  → Solution: Restrict BO search space to training bounds")
        elif overall_avg < 30:
            logger.info(f"\n✓ GOOD: Surrogates are reasonably accurate ({overall_avg:.1f}% avg error)")
            logger.info("  → BO should work, but be cautious of extrapolation")
        elif overall_avg < 50:
            logger.warning(f"\n⚠ MODERATE: Surrogates have notable error ({overall_avg:.1f}% avg error)")
            logger.warning("  → BO results should be validated carefully")
            logger.warning("  → Consider retraining surrogates")
        else:
            logger.error(f"\n✗ POOR: Surrogates are inaccurate ({overall_avg:.1f}% avg error)")
            logger.error("  → BO cannot be trusted")
            logger.error("  → Surrogates must be retrained or replaced")

    else:
        logger.warning("\n⚠ Could not compute error metrics")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate surrogate accuracy on training domain")
    parser.add_argument("--data-dir", default="data", help="Path to training data (default: data)")
    parser.add_argument(
        "--surrogate-dir",
        default="BO/bo_results",
        help="Path to surrogate models (default: BO/bo_results). Surrogates should be in <surrogate_dir>/saved_ml/"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="Number of training samples to test (default: 200). Higher is better for statistical confidence."
    )
    parser.add_argument("--verbose", action="store_true", default=True, help="Enable verbose logging")

    args = parser.parse_args()

    success = main(
        data_dir=args.data_dir,
        surrogate_dir=args.surrogate_dir,
        num_samples=args.num_samples,
        verbose=args.verbose,
    )

    sys.exit(0 if success else 1)
