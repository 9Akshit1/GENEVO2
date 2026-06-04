#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dataset Expansion using Latin Hypercube Sampling.

The original dataset (10,000 samples) uses narrow parameter ranges [0.8-1.2] for Kd,
[1.0-3.0] for Sensitivity, etc. The BO search space is 10-25x wider.

This script generates additional simulations to cover the full BO search space using
Latin Hypercube Sampling. The expanded dataset allows surrogates to be trained
on the full parameter range without massive distribution shift.

Output: Appends N_SAMPLES new runs to data/master_index.csv
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dataset.generator import DatasetGenerator

# Configuration
N_SAMPLES = 5000  # Generate 5000 new samples
LOG_DIR = Path(__file__).parent / "logs"
DATA_DIR = Path(__file__).parent.parent.parent / "data"
RANDOM_SEED = 42


def setup_logging():
    """Setup logging to console and file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("expand_dataset")
    logger.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    log_file = LOG_DIR / f"expand_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def generate_lhs_samples(n_samples, random_state=42):
    """
    Generate Latin Hypercube Samples over the BO search space.

    Returns:
        List of dicts with parameters sampled from the full BO search space:
        - kd_nm: [0.1, 10.0] log-uniform
        - sensitivity: [0.5, 5.0] log-uniform
        - response_time_s: [100, 3600] log-uniform (amplifying only)
        - biosensor_type: {direct_binding, amplifying}
        - noise_preset: {low, medium, high}
        - scenario: {healthy, pmo, ckd_mbd}
    """
    from scipy.stats.qmc import LatinHypercube

    logger = logging.getLogger("expand_dataset")
    logger.info(f"Generating {n_samples} LHS samples...")

    # LHS over 5 continuous dimensions (+ discrete sampled separately)
    # Dimensions: [kd_nm_log, sensitivity_log, response_time_log, biosensor_type_idx, noise_preset_idx]
    sampler = LatinHypercube(d=5, seed=random_state)
    samples_unit = sampler.random(n=n_samples)

    # Convert unit [0,1] to actual parameter ranges
    kd_log_lower = np.log10(0.1)
    kd_log_upper = np.log10(10.0)
    kd_log = kd_log_lower + samples_unit[:, 0] * (kd_log_upper - kd_log_lower)
    kd_nm = 10.0 ** kd_log

    sens_log_lower = np.log10(0.5)
    sens_log_upper = np.log10(5.0)
    sens_log = sens_log_lower + samples_unit[:, 1] * (sens_log_upper - sens_log_lower)
    sensitivity = 10.0 ** sens_log

    rt_log_lower = np.log10(100)
    rt_log_upper = np.log10(3600)
    rt_log = rt_log_lower + samples_unit[:, 2] * (rt_log_upper - rt_log_lower)
    response_time = 10.0 ** rt_log

    # Biosensor type: 0 = direct_binding, 1 = amplifying
    biosensor_type_idx = (samples_unit[:, 3] > 0.5).astype(int)
    biosensor_types = np.array(['direct_binding', 'amplifying'])[biosensor_type_idx]

    # Noise preset: 0 = low, 1 = medium, 2 = high
    noise_preset_idx = np.floor(samples_unit[:, 4] * 3).astype(int)
    noise_preset_idx = np.clip(noise_preset_idx, 0, 2)
    noise_presets = np.array(['low', 'medium', 'high'])[noise_preset_idx]

    # Scenario: randomly weighted (30% healthy, 35% pmo, 35% ckd)
    rng = np.random.RandomState(random_state)
    scenario_choices = rng.choice(
        ['healthy', 'pmo', 'ckd_mbd'],
        size=n_samples,
        p=[0.30, 0.35, 0.35]
    )

    # Construct parameter dicts
    configs = []
    for i in range(n_samples):
        cfg = {
            'kd_nm': float(kd_nm[i]),
            'sensitivity': float(sensitivity[i]),
            'response_time_s': float(response_time[i]),
            'biosensor_type': biosensor_types[i],
            'noise_preset': noise_presets[i],
            'scenario': scenario_choices[i],
        }
        configs.append(cfg)

    logger.info(f"[OK] Generated {n_samples} LHS samples")
    logger.info(f"  Kd range: [{kd_nm.min():.4f}, {kd_nm.max():.4f}]")
    logger.info(f"  Sensitivity range: [{sensitivity.min():.4f}, {sensitivity.max():.4f}]")
    logger.info(f"  Response time range: [{response_time.min():.1f}, {response_time.max():.1f}]")
    logger.info(f"  Biosensor types: {np.bincount(biosensor_type_idx)}")
    logger.info(f"  Noise presets: {dict(zip(*np.unique(noise_presets, return_counts=True)))}")

    return configs


def main():
    """Main expansion routine."""
    logger = setup_logging()

    logger.info("=" * 80)
    logger.info("GENEVO2 DATASET EXPANSION")
    logger.info("=" * 80)
    logger.info(f"Target: {N_SAMPLES} new simulations covering full BO search space")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Random seed: {RANDOM_SEED}")

    # Check that master_index.csv exists
    master_index_path = DATA_DIR / "master_index.csv"
    if not master_index_path.exists():
        logger.error(f"master_index.csv not found at {master_index_path}")
        return False

    # Load existing data to get the row count
    logger.info(f"\n[1/3] Loading existing dataset...")
    df_existing = pd.read_csv(master_index_path)
    n_existing = len(df_existing)
    logger.info(f"[OK] Existing dataset: {n_existing} samples")

    # Generate LHS samples
    logger.info(f"\n[2/3] Generating LHS samples...")
    configs = generate_lhs_samples(N_SAMPLES, random_state=RANDOM_SEED)

    # Generate simulations
    logger.info(f"\n[3/3] Running simulations...")
    logger.info(f"This may take 10-30 minutes depending on CPU speed...")

    try:
        # Get the antimony model path (same as main.py)
        antimony_path = Path(__file__).parent.parent.parent / "models" / "bone_environment.ant"
        if not antimony_path.exists():
            logger.warning(f"Antimony model not found at {antimony_path}, will be generated")
            antimony_path = None

        generator = DatasetGenerator(
            antimony_model_path=str(antimony_path) if antimony_path else None,
            output_dir=str(DATA_DIR),
            seed=RANDOM_SEED
        )

        # Generate each simulation with override config
        new_results = []
        start_time = datetime.now()

        for i, config in enumerate(configs):
            if (i + 1) % 100 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = (i + 1) / (elapsed + 1e-6)
                eta_remaining = (N_SAMPLES - i - 1) / (rate + 1e-6)
                logger.info(
                    f"Progress: {i+1}/{N_SAMPLES} samples ({100*(i+1)/N_SAMPLES:.1f}%) | "
                    f"Rate: {rate:.1f} sim/s | ETA: {eta_remaining/60:.1f} min"
                )

            try:
                # Create biosensor config with LHS-sampled parameters
                biosensor_config = {
                    'kd': config['kd_nm'],
                    'sensitivity': config['sensitivity'],
                    'response_time_s': config['response_time_s'],
                    'circuit_type': config['biosensor_type'],
                }

                # Generate one simulation with override biosensor config
                run_result = generator.generate_single_simulation_instrumented(
                    scenario_name=config['scenario'],
                    biosensor_config=biosensor_config,
                    noise_preset=config['noise_preset'],
                    duration=3600.0,
                    num_points=361,
                    apply_variability=False,  # Use exact parameters from LHS
                    instrument=True,
                    rng_seed=RANDOM_SEED + i
                )

                if run_result:
                    new_results.append(run_result)
            except Exception as e:
                logger.warning(f"Simulation {i} failed: {e}")
                continue

        logger.info(f"\n[OK] Successfully generated {len(new_results)} new samples")

        # Format new results into master_index rows
        logger.info(f"\nFormatting results into master_index rows...")
        new_index_rows = []
        for record in new_results:
            if record:
                row = {
                    'run_id': record['run_id'],
                    'timestamp': record['timestamp'],
                    'scenario': record['scenario'],
                    'biosensor_type': record['biosensor_config']['circuit_type'],
                    'noise_preset': record['noise_preset'],
                    'snr_db': float(record['measurement'].get('snr_db', 0.0)),
                    'n_detections': int(record['measurement'].get('n_detections', 0)),
                    'detection_rate': float(record['measurement'].get('detection_rate', 0.0)),
                    'time_to_detection': float(record['measurement'].get('time_to_detection', 0.0)),
                    'false_negative_rate': float(record['measurement'].get('false_negative_rate', 0.0)),
                    'sclerostin_mean': record['sclerostin_mean'],
                    'sclerostin_std': record['sclerostin_std'],
                    'metadata_file': record['metadata_file'],
                    'timeseries_file': record['timeseries_file']
                }
                new_index_rows.append(row)

        # Append to master_index.csv
        logger.info(f"\nAppending to master_index.csv...")
        if new_index_rows:
            df_new = pd.DataFrame(new_index_rows)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(master_index_path, index=False, encoding='utf-8')
            logger.info(f"[OK] Updated master_index.csv: {len(df_existing)} -> {len(df_combined)} samples")

        elapsed_total = (datetime.now() - start_time).total_seconds()
        logger.info(f"\n" + "=" * 80)
        logger.info(f"EXPANSION COMPLETE")
        logger.info(f"=" * 80)
        logger.info(f"Total time: {elapsed_total/60:.1f} minutes")
        logger.info(f"New samples: {len(new_results)}")
        logger.info(f"Total dataset size: {len(df_combined)}")
        logger.info(f"\nNext steps:")
        logger.info(f"  1. python BO/retrain_surrogates.py")
        logger.info(f"  2. python BO/bo_main.py --retrain-surrogates --n-init 20 --n-iter 80")

        return True

    except Exception as e:
        logger.error(f"Expansion failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
