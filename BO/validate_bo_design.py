"""
BO Design Validator: Run best BO design through actual simulator.

This script performs the most critical validation:
  1. Load best BO configuration
  2. Run through actual biosensor simulator (real simulation, not surrogates)
  3. Compute real DR, FNR, TTD
  4. Compare against surrogate predictions
  5. Quantify prediction error and extrapolation issues

This is essential for determining whether BO is scientifically trustworthy.
"""

import json
import sys
import os
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import traceback

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.generator import DatasetGenerator
from models.biosensors import create_biosensor


logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = True) -> logging.Logger:
    """Setup logging for validation."""
    logger_instance = logging.getLogger("bo_validator")
    logger_instance.setLevel(logging.DEBUG if verbose else logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    ch.setFormatter(formatter)
    logger_instance.addHandler(ch)

    return logger_instance


def load_best_config(results_dir: str) -> Dict:
    """Load best BO configuration from results/best_config.json."""
    config_path = Path(results_dir) / "results" / "best_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Best config not found at {config_path}")

    with open(config_path, 'r') as f:
        return json.load(f)


def convert_bo_config_to_biosensor(bo_config: Dict) -> Tuple[Dict, Dict]:
    """
    Convert BO configuration to biosensor_config dict.

    BO config has structure:
      - biosensor_design: {type, kd_nm, sensitivity, response_time_s}
      - measurement_environment: {noise_preset, target_scenario}
      - predicted_performance: {detection_rate, false_negative_rate, ...}

    Returns:
      - biosensor_config: Dict for create_biosensor()
      - test_params: Dict with noise_preset, target_scenario, etc.

    CRITICAL: Ensures all required fields are present and validated.
    """
    try:
        design = bo_config['biosensor_design']
        env = bo_config['measurement_environment']
    except KeyError as e:
        raise ValueError(f"BO config missing required key: {e}")

    # Validate design fields
    for field in ['type', 'kd_nm', 'sensitivity']:
        if field not in design:
            raise ValueError(f"biosensor_design missing field: {field}")

    # Validate environment fields
    for field in ['noise_preset', 'target_scenario']:
        if field not in env:
            raise ValueError(f"measurement_environment missing field: {field}")

    # Validate parameter ranges
    kd = float(design['kd_nm'])
    sensitivity = float(design['sensitivity'])

    if not (0.05 <= kd <= 20.0):
        raise ValueError(f"kd_nm={kd} outside valid range [0.05, 20.0] nM")
    if not (0.1 <= sensitivity <= 10.0):
        raise ValueError(f"sensitivity={sensitivity} outside valid range [0.1, 10.0]")

    # Build biosensor config
    biosensor_config = {
        'circuit_type': design['type'],
        'sensitivity': sensitivity,
        'kd': kd,
        'threshold': 0.6,  # Default threshold for detection
        'dynamic_range': (0.0, 5.0),  # Will adjust based on circuit type
    }

    # Adjust parameters based on circuit type
    if design['type'] == 'amplifying':
        if 'response_time_s' not in design:
            raise ValueError("amplifying biosensor requires response_time_s")
        response_time = float(design['response_time_s'])
        if not (50 <= response_time <= 5000):
            raise ValueError(f"response_time_s={response_time} outside valid range [50, 5000]")
        biosensor_config['response_time'] = response_time
        biosensor_config['dynamic_range'] = (0.0, 8.0)
    elif design['type'] == 'direct_binding':
        biosensor_config['dynamic_range'] = (0.0, 5.0)
    else:
        raise ValueError(f"Unknown biosensor type: {design['type']}")

    test_params = {
        'noise_preset': env['noise_preset'],
        'target_scenario': env['target_scenario'],
        'predicted_metrics': bo_config.get('predicted_performance', {}),
    }

    logger.info(f"✓ BO config converted successfully")
    logger.info(f"  Biosensor: {biosensor_config['circuit_type']}, "
                f"Kd={kd:.4f} nM, Sensitivity={sensitivity:.4f}")

    return biosensor_config, test_params


def run_simulation_for_scenario(
    gen: DatasetGenerator,
    scenario_name: str,
    biosensor_config: Dict,
    noise_preset: str,
    duration: float = 3600.0,
    num_points: int = 361,
) -> Optional[Dict]:
    """
    Run a single simulation for a specific scenario.

    Returns:
      Dict with simulation results including metrics, or None if failed
    """
    try:
        result = gen.generate_single_simulation_instrumented(
            scenario_name=scenario_name,
            biosensor_config=biosensor_config,
            noise_preset=noise_preset,
            duration=duration,
            num_points=num_points,
            apply_variability=False,  # Fixed params for reproducibility
            instrument=True,
        )
        return result
    except Exception as e:
        logger.error(f"Simulation failed for {scenario_name}: {e}")
        traceback.print_exc()
        return None


def extract_metrics_from_simulation(
    result: Dict,
    scenario_name: str
) -> Tuple[float, float, float, float]:
    """
    Extract DR, FNR, TTD, SNR from simulation result.

    Returns:
      (detection_rate, false_negative_rate, time_to_detection_s, snr_db)
    """
    try:
        # Navigate result structure (depends on generator v5.0 output format)
        if 'metrics' in result:
            metrics = result['metrics']
        else:
            # Fall back to direct fields
            metrics = result

        # Extract fields with sensible defaults
        dr = float(metrics.get('detection_rate', 0.0))
        fnr = float(metrics.get('false_negative_rate', 1.0))
        ttd = float(metrics.get('time_to_detection_threshold', 3600.0))
        snr = float(metrics.get('estimated_snr_db', 0.0))

        return dr, fnr, ttd, snr

    except Exception as e:
        logger.warning(f"Failed to extract metrics from {scenario_name} result: {e}")
        return 0.0, 1.0, 3600.0, 0.0


def compute_prediction_error(predicted: float, actual: float) -> float:
    """Compute percentage error: |predicted - actual| / |actual + eps|."""
    if actual == 0:
        return np.inf if predicted != 0 else 0.0
    return 100.0 * abs(predicted - actual) / abs(actual)


def validate_biosensor_separability(biosensor_config: Dict) -> bool:
    """
    Sanity check: does this biosensor separate disease states?
    Tests: H_output < PMO_output < CKD_output
    """
    try:
        biosensor = create_biosensor(biosensor_config)

        # Test at steady state (1800s)
        time_test = np.array([1800.0])
        h_out = float(biosensor.measure(np.array([0.375]), time_test)[0])
        p_out = float(biosensor.measure(np.array([0.875]), time_test)[0])
        c_out = float(biosensor.measure(np.array([2.0]), time_test)[0])

        valid = (h_out < p_out) and (p_out < c_out)
        logger.info(f"Separability check: H={h_out:.3f}, P={p_out:.3f}, C={c_out:.3f} — {'PASS' if valid else 'FAIL'}")

        return valid

    except Exception as e:
        logger.warning(f"Separability check failed: {e}")
        return False


def main(
    bo_results_dir: str = "BO/bo_results",
    antimony_model: str = "models/bone_environment.ant",
    data_dir: str = "data",
    verbose: bool = True,
):
    """
    Main validation pipeline.
    """
    global logger
    logger = setup_logging(verbose)

    logger.info("=" * 70)
    logger.info("BO DESIGN VALIDATION PIPELINE")
    logger.info("=" * 70)

    # 1. Load best BO config
    logger.info("\n[1/5] Loading best BO configuration...")
    try:
        bo_config = load_best_config(bo_results_dir)
        logger.info(f"✓ Loaded best_config.json")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return False

    # 2. Convert to biosensor config
    logger.info("\n[2/5] Converting BO parameters to biosensor configuration...")
    try:
        biosensor_config, test_params = convert_bo_config_to_biosensor(bo_config)
        logger.info(f"✓ Biosensor type: {biosensor_config['circuit_type']}")
        logger.info(f"  Kd: {biosensor_config['kd']:.4f} nM")
        logger.info(f"  Sensitivity: {biosensor_config['sensitivity']:.4f}")
        if 'response_time' in biosensor_config:
            logger.info(f"  Response time: {biosensor_config['response_time']:.1f} s")
    except Exception as e:
        logger.error(f"Failed to convert config: {e}")
        return False

    # 3. Validate separability
    logger.info("\n[3/5] Testing biosensor disease separability...")
    if not validate_biosensor_separability(biosensor_config):
        logger.warning("⚠ Biosensor failed separability check — may be clinically unusable")

    # 4. Run simulations
    logger.info("\n[4/5] Running actual simulator for all scenarios...")

    # Initialize dataset generator
    try:
        gen = DatasetGenerator(antimony_model, output_dir=data_dir, seed=42)
    except Exception as e:
        logger.error(f"Failed to initialize simulator: {e}")
        return False

    results = {}
    scenarios = ['healthy', 'pmo', 'ckd_mbd']

    for scenario in scenarios:
        logger.info(f"\n  Running: {scenario}...")
        result = run_simulation_for_scenario(
            gen,
            scenario_name=scenario,
            biosensor_config=biosensor_config,
            noise_preset=test_params['noise_preset'],
        )

        if result:
            dr, fnr, ttd, snr = extract_metrics_from_simulation(result, scenario)
            results[scenario] = {
                'detection_rate': dr,
                'false_negative_rate': fnr,
                'time_to_detection_s': ttd,
                'estimated_snr_db': snr,
                'raw_result': result,  # Full result for debugging
            }
            logger.info(f"    DR={dr:.4f}, FNR={fnr:.4f}, TTD={ttd:.1f}s, SNR={snr:.2f}dB")
        else:
            logger.error(f"    Simulation failed for {scenario}")
            results[scenario] = None

    # 5. Compare predicted vs actual
    logger.info("\n[5/5] Comparing surrogate predictions vs actual simulation results...")
    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION REPORT")
    logger.info("=" * 70)

    if test_params['target_scenario'] == 'both':
        # Average across all scenarios
        all_scenarios = scenarios
    elif test_params['target_scenario'] == 'pmo':
        all_scenarios = ['pmo']
    elif test_params['target_scenario'] == 'ckd_mbd':
        all_scenarios = ['ckd_mbd']
    else:
        all_scenarios = scenarios

    logger.info(f"\nTarget scenario: {test_params['target_scenario']}")
    logger.info(f"Noise preset: {test_params['noise_preset']}")
    logger.info(f"Evaluating against scenarios: {all_scenarios}\n")

    # Compute average metrics across scenarios
    actual_metrics = {}
    for metric in ['detection_rate', 'false_negative_rate', 'time_to_detection_s', 'estimated_snr_db']:
        values = []
        for scenario in all_scenarios:
            if results[scenario] is not None:
                values.append(results[scenario][metric])
        if values:
            actual_metrics[metric] = float(np.mean(values))
        else:
            actual_metrics[metric] = 0.0

    predicted_metrics = test_params['predicted_metrics']

    # Display side-by-side comparison
    logger.info("PREDICTED (from surrogates):")
    logger.info(f"  Detection Rate:       {predicted_metrics.get('detection_rate', 0.0):.6f}")
    logger.info(f"  False Negative Rate:  {predicted_metrics.get('false_negative_rate', 0.0):.6f}")
    logger.info(f"  Time-to-Detection:    {predicted_metrics.get('time_to_detection_s', 0.0):.1f} s")
    logger.info(f"  Estimated SNR:        {predicted_metrics.get('estimated_snr_db', 0.0):.2f} dB")

    logger.info("\nACTUAL (from simulator):")
    logger.info(f"  Detection Rate:       {actual_metrics.get('detection_rate', 0.0):.6f}")
    logger.info(f"  False Negative Rate:  {actual_metrics.get('false_negative_rate', 0.0):.6f}")
    logger.info(f"  Time-to-Detection:    {actual_metrics.get('time_to_detection_s', 0.0):.1f} s")
    logger.info(f"  Estimated SNR:        {actual_metrics.get('estimated_snr_db', 0.0):.2f} dB")

    logger.info("\nPREDICTION ERRORS:")

    dr_error = compute_prediction_error(
        predicted_metrics.get('detection_rate', 0.0),
        actual_metrics.get('detection_rate', 0.0)
    )
    fnr_error = compute_prediction_error(
        predicted_metrics.get('false_negative_rate', 0.0),
        actual_metrics.get('false_negative_rate', 0.0)
    )
    ttd_error = compute_prediction_error(
        predicted_metrics.get('time_to_detection_s', 0.0),
        actual_metrics.get('time_to_detection_s', 0.0)
    )
    snr_error = compute_prediction_error(
        predicted_metrics.get('estimated_snr_db', 0.0),
        actual_metrics.get('estimated_snr_db', 0.0)
    )

    logger.info(f"  Detection Rate error:      {dr_error:.1f}%")
    logger.info(f"  False Negative Rate error: {fnr_error:.1f}%")
    logger.info(f"  Time-to-Detection error:   {ttd_error:.1f}%")
    logger.info(f"  SNR error:                 {snr_error:.1f}%")

    # Diagnostic assessment
    logger.info("\n" + "=" * 70)
    logger.info("DIAGNOSTIC ASSESSMENT")
    logger.info("=" * 70)

    avg_error = np.mean([dr_error, fnr_error, ttd_error, snr_error])

    if actual_metrics.get('detection_rate', 0.0) < 0.5:
        logger.warning("\n⚠ CRITICAL: Detection Rate < 50% — Design is clinically unusable")
        logger.warning("  → Optimizer found a pathological solution")
        logger.warning("  → Objective function is likely misaligned with clinical goals")

    if avg_error > 50.0:
        logger.warning(f"\n⚠ HIGH PREDICTION ERROR: {avg_error:.1f}% avg error")
        logger.warning("  → Surrogates are extrapolating beyond training distribution")
        logger.warning("  → BO may be exploiting surrogate weaknesses")
    elif avg_error > 20.0:
        logger.warning(f"\n⚠ MODERATE PREDICTION ERROR: {avg_error:.1f}% avg error")
        logger.warning("  → Surrogates have reasonable but not excellent accuracy")
    else:
        logger.info(f"\n✓ GOOD: Prediction error {avg_error:.1f}% — Surrogates are trustworthy")

    # Save detailed report
    report = {
        'bo_config': bo_config,
        'biosensor_config': biosensor_config,
        'test_params': test_params,
        'predicted_metrics': predicted_metrics,
        'actual_metrics': actual_metrics,
        'per_scenario_results': results,
        'prediction_errors': {
            'detection_rate_pct': dr_error,
            'false_negative_rate_pct': fnr_error,
            'time_to_detection_pct': ttd_error,
            'snr_pct': snr_error,
            'average_pct': avg_error,
        },
    }

    report_path = Path(bo_results_dir) / "validation_report.json"
    with open(report_path, 'w') as f:
        # Clean up for JSON
        def json_clean(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, float)):
                return float(obj) if not np.isnan(obj) else None
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: json_clean(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [json_clean(x) for x in obj]
            return obj

        json.dump(json_clean(report), f, indent=2)

    logger.info(f"\n✓ Validation report saved to: {report_path}")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate BO design against actual simulator")
    parser.add_argument("--bo-results-dir", default="BO/bo_results",
                       help="Path to BO results directory")
    parser.add_argument("--antimony-model", default="models/bone_environment.ant",
                       help="Path to Antimony model file")
    parser.add_argument("--data-dir", default="data",
                       help="Path to data directory")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Verbose logging")

    args = parser.parse_args()

    success = main(
        bo_results_dir=args.bo_results_dir,
        antimony_model=args.antimony_model,
        data_dir=args.data_dir,
        verbose=args.verbose,
    )

    sys.exit(0 if success else 1)
