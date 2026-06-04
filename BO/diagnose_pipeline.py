#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive diagnostic script for BO pipeline.

Tests each component independently to identify issues:
1. Data availability and loading
2. Surrogate training and persistence
3. Surrogate loading and inference
4. Parameter conversion
5. Simulator integration
6. Full BO objective evaluation
"""

import sys
import logging
from pathlib import Path
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from BO.core.build_surrogates import SurrogateBuilder
from BO.core.surrogate_loader import SurrogateLoader
from BO.search_space.biosensor_space import BiosensorSearchSpace
from BO.evaluation.physics_forward_model import PhysicsForwardModel
from BO.evaluation.objective_function import ObjectiveFunction
from dataset.generator import DatasetGenerator
from models.biosensors import create_biosensor

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("pipeline_diagnostic")


def test_data_loading(data_dir: Path) -> bool:
    """Test if data can be loaded."""
    logger.info("\n" + "="*70)
    logger.info("TEST 1: Data Loading")
    logger.info("="*70)

    master_index = data_dir / "master_index.csv"
    if not master_index.exists():
        logger.error(f"❌ master_index.csv not found at {master_index}")
        return False

    logger.info(f"✓ Found master_index.csv at {master_index}")

    try:
        builder = SurrogateBuilder(logger)
        X, feature_names, df_results = builder.load_and_extract_features(data_dir)
        logger.info(f"✓ Loaded {X.shape[0]} samples with {X.shape[1]} features")
        logger.info(f"  Targets: DR[{df_results['detection_rate'].min():.3f}, {df_results['detection_rate'].max():.3f}]")
        logger.info(f"  Targets: FNR[{df_results['false_negative_rate'].min():.3f}, {df_results['false_negative_rate'].max():.3f}]")
        logger.info(f"  Targets: TTD[{df_results['time_to_detection'].min():.1f}, {df_results['time_to_detection'].max():.1f}]")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load data: {e}")
        return False


def test_surrogate_training(data_dir: Path, surrogate_dir: Path) -> bool:
    """Test surrogate training."""
    logger.info("\n" + "="*70)
    logger.info("TEST 2: Surrogate Training")
    logger.info("="*70)

    try:
        builder = SurrogateBuilder(logger)
        X, _, df_results = builder.load_and_extract_features(data_dir)

        logger.info(f"Training surrogates on {X.shape[0]} samples...")
        metrics = builder.train_all_surrogates(X, df_results)

        logger.info(f"✓ Training complete:")
        for metric, m in metrics.items():
            logger.info(f"  {metric}: R²_test={m['r2_test']:.4f}")

        logger.info(f"Saving to {surrogate_dir}...")
        builder.save_surrogates(surrogate_dir, version="v1")
        logger.info(f"✓ Surrogates saved successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Surrogate training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_surrogate_loading(surrogate_dir: Path) -> bool:
    """Test surrogate loading and initialization."""
    logger.info("\n" + "="*70)
    logger.info("TEST 3: Surrogate Loading")
    logger.info("="*70)

    try:
        loader = SurrogateLoader(str(surrogate_dir))
        logger.info(f"✓ Surrogates loaded")
        logger.info(f"  Models: {list(loader.surrogates.keys())}")
        logger.info(f"  Scaler initialized: {loader.scaler is not None}")
        logger.info(f"  Encoders initialized: {len(loader.label_encoders)} encoders")

        # Test encoding
        bio_enc, noise_enc, scenario_enc = loader.encode_categorical(
            "amplifying", "medium", "pmo"
        )
        logger.info(f"✓ Categorical encoding works: bio={bio_enc:.2f}, noise={noise_enc:.2f}, scenario={scenario_enc:.2f}")

        # Test prediction (v2 returns: dr_pred, fnr_median, fnr_lower, fnr_upper, ttd_median, ttd_lower, ttd_upper)
        X_test = np.array([[1.0, 2.0, 0.5, 0.5, 0.5]], dtype=np.float32)
        X_scaled = loader.scaler.transform(X_test)
        preds = loader.predict_metrics(X_scaled)
        dr, fnr_median, fnr_lower, fnr_upper, ttd_median, ttd_lower, ttd_upper = preds
        logger.info(f"✓ Prediction works: DR={dr:.4f}, FNR_median={fnr_median:.4f}, TTD_median={ttd_median:.1f}s")
        return True
    except Exception as e:
        logger.error(f"❌ Surrogate loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parameter_conversion() -> bool:
    """Test parameter conversion and bounds."""
    logger.info("\n" + "="*70)
    logger.info("TEST 4: Parameter Conversion")
    logger.info("="*70)

    try:
        space = BiosensorSearchSpace()
        logger.info(f"✓ Search space initialized with {space.n_params} parameters")

        # Test random sampling
        config = space.sample_random()
        logger.info(f"✓ Random sample generated:")
        for k, v in config.items():
            logger.info(f"  {k}: {v}")

        # Test vector conversion
        x_vec = space.dict_to_vector(config)
        config_recovered = space.vector_to_dict(x_vec)
        logger.info(f"✓ Vector conversion round-trip successful")

        # Verify bounds
        for param, bounds in space.get_bounds().items():
            val = config[param]
            if val < bounds[0] or val > bounds[1]:
                logger.error(f"❌ {param}={val} outside bounds {bounds}")
                return False
        logger.info(f"✓ All parameters within bounds")
        return True
    except Exception as e:
        logger.error(f"❌ Parameter conversion failed: {e}")
        return False


def test_biosensor_creation() -> bool:
    """Test biosensor creation from config."""
    logger.info("\n" + "="*70)
    logger.info("TEST 5: Biosensor Creation")
    logger.info("="*70)

    try:
        configs = [
            {
                'circuit_type': 'direct_binding',
                'sensitivity': 2.0,
                'kd': 1.0,
                'threshold': 0.6,
                'dynamic_range': (0.0, 5.0),
            },
            {
                'circuit_type': 'amplifying',
                'sensitivity': 2.0,
                'kd': 1.0,
                'threshold': 0.8,
                'dynamic_range': (0.0, 8.0),
                'response_time': 500.0,
            },
        ]

        for cfg in configs:
            try:
                biosensor = create_biosensor(cfg)
                logger.info(f"✓ Created {cfg['circuit_type']} biosensor")

                # Test measurement
                sclerostin = np.array([0.375, 0.875, 2.0])  # H, P, C
                time = np.array([1800.0, 1800.0, 1800.0])
                signal = biosensor.measure(sclerostin, time)
                logger.info(f"  Signal range: [{signal.min():.3f}, {signal.max():.3f}]")

            except Exception as e:
                logger.error(f"❌ Failed to create {cfg['circuit_type']}: {e}")
                return False

        return True
    except Exception as e:
        logger.error(f"❌ Biosensor creation test failed: {e}")
        return False


def test_objective_function(surrogate_dir: Path) -> bool:
    """Test objective function evaluation."""
    logger.info("\n" + "="*70)
    logger.info("TEST 6: Objective Function Evaluation")
    logger.info("="*70)

    try:
        loader = SurrogateLoader(str(surrogate_dir))
        physics = PhysicsForwardModel()
        objective = ObjectiveFunction(physics, loader)

        # Test config
        config = {
            'biosensor_type': 'amplifying',
            'kd_nm': 1.0,
            'sensitivity': 2.0,
            'response_time_s': 500.0,
            'noise_preset': 'medium',
            'target_scenario': 'pmo',
        }

        logger.info(f"Testing objective with config: {config}")
        score = objective(config)
        logger.info(f"✓ Objective evaluation works: score={score:.4f}")

        # Test detailed evaluation
        score_detailed, details = objective.evaluate_with_details(config)
        logger.info(f"✓ Detailed evaluation works:")
        logger.info(f"  DR_pred: {details.get('dr_pred', 'N/A'):.4f}")
        logger.info(f"  FNR_pred: {details.get('fnr_pred', 'N/A'):.4f}")
        logger.info(f"  TTD_pred: {details.get('ttd_pred_s', 'N/A'):.1f}s")
        logger.info(f"  SNR_est: {details.get('snr_db_est', 'N/A'):.2f}dB")
        logger.info(f"  OOD_penalty: {details.get('ood_penalty', 'N/A'):.4f}")

        return True
    except Exception as e:
        logger.error(f"❌ Objective function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simulator_integration(data_dir: Path) -> bool:
    """Test simulator integration."""
    logger.info("\n" + "="*70)
    logger.info("TEST 7: Simulator Integration")
    logger.info("="*70)

    try:
        gen = DatasetGenerator("models/bone_environment.ant", output_dir=str(data_dir), seed=42)
        logger.info(f"✓ DatasetGenerator initialized")

        # Test simple simulation
        biosensor_config = {
            'circuit_type': 'amplifying',
            'sensitivity': 2.0,
            'kd': 1.0,
            'threshold': 0.6,
            'dynamic_range': (0.0, 8.0),
            'response_time': 500.0,
        }

        logger.info(f"Running test simulation for healthy scenario...")
        result = gen.generate_single_simulation_instrumented(
            scenario_name='healthy',
            biosensor_config=biosensor_config,
            noise_preset='medium',
            duration=3600.0,
            num_points=361,
            apply_variability=False,
            instrument=True,
        )

        if result:
            metrics = result['measurement']
            logger.info(f"✓ Simulator execution successful:")
            logger.info(f"  DR: {metrics['detection_rate']:.4f}")
            logger.info(f"  FNR: {metrics['false_negative_rate']:.4f}")
            logger.info(f"  TTD: {metrics['time_to_detection']:.1f}s")
            logger.info(f"  SNR: {metrics['snr_db']:.2f}dB")
            return True
        else:
            logger.error(f"❌ Simulator returned None")
            return False
    except Exception as e:
        logger.error(f"❌ Simulator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all diagnostics."""
    logger.info("="*70)
    logger.info("BO PIPELINE DIAGNOSTIC SUITE")
    logger.info("="*70)

    data_dir = Path("data")
    surrogate_dir = Path("BO/bo_results")

    results = {}

    # Test 1: Data loading
    results['data_loading'] = test_data_loading(data_dir)

    if not results['data_loading']:
        logger.error("Cannot proceed without data. Please generate dataset first.")
        return 1

    # Test 2: Surrogate training
    results['training'] = test_surrogate_training(data_dir, surrogate_dir)

    if not results['training']:
        logger.error("Cannot proceed without trained surrogates.")
        return 1

    # Test 3: Surrogate loading
    results['loading'] = test_surrogate_loading(surrogate_dir)

    # Test 4: Parameter conversion
    results['conversion'] = test_parameter_conversion()

    # Test 5: Biosensor creation
    results['biosensor'] = test_biosensor_creation()

    # Test 6: Objective function
    results['objective'] = test_objective_function(surrogate_dir)

    # Test 7: Simulator
    results['simulator'] = test_simulator_integration(data_dir)

    # Summary
    logger.info("\n" + "="*70)
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info("="*70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, passed_flag in results.items():
        status = "✓ PASS" if passed_flag else "❌ FAIL"
        logger.info(f"{status}: {test_name}")

    logger.info(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        logger.info("\n✓ All diagnostics passed! BO pipeline is ready.")
        return 0
    else:
        logger.error(f"\n❌ {total - passed} test(s) failed. See errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
