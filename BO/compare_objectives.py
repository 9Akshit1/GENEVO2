"""
Objective Function Comparison Diagnostic.

Compares UNCONSTRAINED vs CONSTRAINED objective functions
on BO-optimized designs to show the impact of clinical constraints.

This reveals why the previous BO found a pathological solution
and how constraints fix it.
"""

import json
import sys
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from BO.core.surrogate_loader import SurrogateLoader
from BO.evaluation.physics_forward_model import PhysicsForwardModel
from BO.evaluation.objective_function import ObjectiveFunction


def setup_logging(verbose: bool = True):
    logger = logging.getLogger("objective_compare")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def main(
    bo_results_dir: str = "BO/bo_results",
    data_dir: str = "data",
    surrogate_dir: str = "BO/results",
    verbose: bool = True,
):
    logger = setup_logging(verbose)

    logger.info("=" * 80)
    logger.info("OBJECTIVE FUNCTION COMPARISON: Unconstrained vs Constrained")
    logger.info("=" * 80)

    # Load best BO config
    config_path = Path(bo_results_dir) / "results" / "best_config.json"
    if not config_path.exists():
        logger.error(f"Best config not found at {config_path}")
        return False

    with open(config_path, 'r') as f:
        best_config = json.load(f)

    # Extract BO parameters
    design = best_config['biosensor_design']
    env = best_config['measurement_environment']

    bo_params = {
        'biosensor_type': design['type'],
        'kd_nm': design['kd_nm'],
        'sensitivity': design['sensitivity'],
        'response_time_s': design.get('response_time_s', 500.0),
        'noise_preset': env['noise_preset'],
        'target_scenario': env['target_scenario'],
    }

    logger.info("\nBO-OPTIMIZED DESIGN:")
    logger.info(f"  Type:           {bo_params['biosensor_type']}")
    logger.info(f"  Kd:             {bo_params['kd_nm']:.4f} nM")
    logger.info(f"  Sensitivity:    {bo_params['sensitivity']:.4f}")
    logger.info(f"  Response time:  {bo_params['response_time_s']:.1f} s")
    logger.info(f"  Noise:          {bo_params['noise_preset']}")
    logger.info(f"  Target:         {bo_params['target_scenario']}")

    # Initialize objective functions
    try:
        surrogate_loader = SurrogateLoader(surrogate_dir, data_dir)
        surrogate_loader.load_surrogates(version='v1')
        surrogate_loader.refit_scaler()
        physics_model = PhysicsForwardModel()

        # UNCONSTRAINED version
        obj_unconstrained = ObjectiveFunction(
            physics_model,
            surrogate_loader,
            apply_constraints=False,  # Key difference
        )

        # CONSTRAINED version (with clinical thresholds)
        obj_constrained = ObjectiveFunction(
            physics_model,
            surrogate_loader,
            apply_constraints=True,  # Key difference
        )

    except Exception as e:
        logger.error(f"Failed to initialize objectives: {e}")
        return False

    logger.info("\n" + "=" * 80)
    logger.info("OBJECTIVE FUNCTION CONFIGURATIONS")
    logger.info("=" * 80)

    logger.info("\nUNCONSTRAINED (Original):")
    logger.info("  - No hard clinical thresholds")
    logger.info("  - Weights: DR=0.5, FNR=0.3, TTD=0.2")
    logger.info("  - Can find pathological solutions")

    logger.info("\nCONSTRAINED (Improved):")
    logger.info(f"  - Hard constraints: DR≥{obj_constrained.min_detection_rate:.2f}, FNR≤{obj_constrained.max_false_negative_rate:.2f}, SNR≥{obj_constrained.min_snr_db:.1f}dB")
    logger.info("  - Weights: DR=0.45, FNR=0.25, TTD=0.15, SNR=0.15")
    logger.info("  - Prevents pathological solutions")

    # Evaluate both
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)

    score_unconstrained, details_unconstrained = obj_unconstrained.evaluate_with_details(bo_params)
    score_constrained, details_constrained = obj_constrained.evaluate_with_details(bo_params)

    logger.info("\nUNCONSTRAINED Objective Evaluation:")
    logger.info(f"  Composite Score:           {score_unconstrained:.6f}")
    logger.info(f"  Detection Rate (DR):       {details_unconstrained['dr_pred']:.6f}")
    logger.info(f"  False Negative Rate (FNR): {details_unconstrained['fnr_pred']:.6f}")
    logger.info(f"  Time to Detection (TTD):   {details_unconstrained['ttd_pred_s']:.1f} s")
    logger.info(f"  SNR:                       {details_unconstrained['snr_db_est']:.2f} dB")

    logger.info("\nCONSTRAINED Objective Evaluation:")
    if score_constrained == obj_constrained.CATASTROPHIC_PENALTY:
        logger.info(f"  Composite Score:           {score_constrained} (CATASTROPHIC PENALTY)")
        logger.info(f"  Constraint Violations:     {details_constrained['constraint_violations']}")
    else:
        logger.info(f"  Composite Score:           {score_constrained:.6f}")
        logger.info(f"  Detection Rate (DR):       {details_constrained['dr_pred']:.6f}")
        logger.info(f"  False Negative Rate (FNR): {details_constrained['fnr_pred']:.6f}")
        logger.info(f"  Time to Detection (TTD):   {details_constrained['ttd_pred_s']:.1f} s")
        logger.info(f"  SNR:                       {details_constrained['snr_db_est']:.2f} dB")

    logger.info("\n" + "=" * 80)
    logger.info("DIAGNOSTIC ASSESSMENT")
    logger.info("=" * 80)

    if details_unconstrained['dr_pred'] < 0.50:
        logger.warning("\n⚠ CRITICAL FINDING: Unconstrained BO found pathological solution!")
        logger.warning(f"   DR = {details_unconstrained['dr_pred']:.6f} (only {details_unconstrained['dr_pred']*100:.2f}% detection)")
        logger.warning("   This is clinically UNUSABLE — sensor almost never detects disease")
        logger.warning("\n✓ SOLUTION: Constrained version REJECTS this design automatically")
        logger.warning("   by enforcing hard threshold: DR must be ≥ 0.70")

    if score_constrained == obj_constrained.CATASTROPHIC_PENALTY:
        logger.warning("\n→ RECOMMENDED ACTION:")
        logger.warning("   1. Re-run BO with CONSTRAINED objective function")
        logger.warning("   2. This will prevent pathological solutions from being found")
        logger.warning("   3. BO will search only in clinically viable parameter space")

    logger.info("\n" + "=" * 80)
    logger.info("NEXT STEPS")
    logger.info("=" * 80)
    logger.info("\n1. Run validation_bo_design.py to test BO design against actual simulator")
    logger.info("2. Update bo_main.py to use constrained ObjectiveFunction by default")
    logger.info("3. Re-run BO optimization: python BO/bo_main.py --n-init 20 --n-iter 80")
    logger.info("4. Compare new results to old results")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare constrained vs unconstrained objectives")
    parser.add_argument("--bo-results-dir", default="BO/bo_results")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--surrogate-dir", default="BO/results")
    parser.add_argument("--verbose", action="store_true", default=True)

    args = parser.parse_args()

    success = main(
        bo_results_dir=args.bo_results_dir,
        data_dir=args.data_dir,
        surrogate_dir=args.surrogate_dir,
        verbose=args.verbose,
    )

    sys.exit(0 if success else 1)
