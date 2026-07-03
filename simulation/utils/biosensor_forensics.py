"""
Population-Level Biosensor Compression Analysis

Correctly computes disease separability compression at the population level
by analyzing stage_0_ode vs stage_1_biosensor_clean metrics across all
simulations, stratified by scenario.

V2.0: Fixed to use correct population-level metrics from instrumentation.
"""

import json
from pathlib import Path
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def analyze_biosensor_compression(data_dir: str):
    """
    Analyze biosensor compression at population level.

    Key insight: Each metadata file is ONE simulation of ONE scenario.
    To compute compression, we must:
    1. Load all metadata files
    2. Group by scenario (healthy, pmo, ckd_mbd)
    3. Extract stage_0_ode.mean_signal and stage_1_biosensor_clean.mean_signal
    4. Compute population means per stage per scenario
    5. Compare input_range (ODE) vs output_range (biosensor)
    """

    master_path = Path(data_dir) / 'master_index.csv'

    logger.info(f"\n{'='*80}")
    logger.info(f"BIOSENSOR COMPRESSION ANALYSIS - POPULATION LEVEL")
    logger.info(f"{'='*80}\n")

    master_df = pd.read_csv(master_path)
    logger.info(f"Loading {len(master_df)} simulations...\n")

    # Collect stage metrics per scenario
    stage_metrics = {
        'healthy': {'stage_0': [], 'stage_1': []},
        'pmo': {'stage_0': [], 'stage_1': []},
        'ckd_mbd': {'stage_0': [], 'stage_1': []},
    }

    biosensor_configs = {
        'healthy': [],
        'pmo': [],
        'ckd_mbd': [],
    }

    failed_count = 0

    for _, row in master_df.iterrows():
        scenario = row['scenario']
        metadata_path = Path(data_dir) / row['metadata_file']

        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            instr = metadata.get('instrumentation', {})
            stages = instr.get('stages', {})

            # Extract mean_signal from each stage
            stage_0 = stages.get('stage_0_ode', {})
            stage_1 = stages.get('stage_1_biosensor_clean', {})

            s0_mean = stage_0.get('mean_signal')
            s1_mean = stage_1.get('mean_signal')

            if s0_mean is not None and s1_mean is not None:
                stage_metrics[scenario]['stage_0'].append(float(s0_mean))
                stage_metrics[scenario]['stage_1'].append(float(s1_mean))

                # Also collect biosensor config for diagnostics
                bio_config = metadata.get('biosensor_config', {})
                biosensor_configs[scenario].append({
                    'circuit_type': bio_config.get('circuit_type'),
                    'kd': bio_config.get('kd'),
                    'sensitivity': bio_config.get('sensitivity'),
                })

        except Exception:
            failed_count += 1
            continue

    logger.info(f"Successfully loaded instrumentation from {len(master_df) - failed_count} simulations")
    logger.info(f"Failed to load: {failed_count}\n")

    # Compute population means and ranges
    logger.info(f"{'='*80}")
    logger.info(f"POPULATION-LEVEL ANALYSIS")
    logger.info(f"{'='*80}\n")

    results = {}

    for scenario in ['healthy', 'pmo', 'ckd_mbd']:
        s0_vals = np.array(stage_metrics[scenario]['stage_0'])
        s1_vals = np.array(stage_metrics[scenario]['stage_1'])

        if len(s0_vals) == 0:
            logger.warning(f"{scenario}: No data")
            continue

        s0_mean = float(np.mean(s0_vals))
        s0_std = float(np.std(s0_vals))
        s1_mean = float(np.mean(s1_vals))
        s1_std = float(np.std(s1_vals))

        results[scenario] = {
            'n_sims': len(s0_vals),
            'stage_0_ode_mean': s0_mean,
            'stage_0_ode_std': s0_std,
            'stage_1_clean_mean': s1_mean,
            'stage_1_clean_std': s1_std,
        }

        logger.info(f"{scenario.upper():10s}")
        logger.info(f"  n={len(s0_vals):5d}")
        logger.info(f"  ODE output:           {s0_mean:.6f} ± {s0_std:.6f} nM")
        logger.info(f"  Biosensor output:     {s1_mean:.6f} ± {s1_std:.6f}")
        logger.info(f"  Configs: {len(biosensor_configs[scenario])} unique")
        logger.info("")

    # Compute compression ratio
    logger.info(f"{'='*80}")
    logger.info(f"COMPRESSION ANALYSIS")
    logger.info(f"{'='*80}\n")

    h_s0 = results['healthy']['stage_0_ode_mean']
    c_s0 = results['ckd_mbd']['stage_0_ode_mean']

    h_s1 = results['healthy']['stage_1_clean_mean']
    c_s1 = results['ckd_mbd']['stage_1_clean_mean']

    ode_range = c_s0 - h_s0
    biosensor_range = c_s1 - h_s1
    compression_ratio = biosensor_range / ode_range if ode_range > 0 else 0

    logger.info(f"ODE disease separation (H → C):")
    logger.info(f"  Healthy:   {h_s0:.6f} nM")
    logger.info(f"  CKD-MBD:   {c_s0:.6f} nM")
    logger.info(f"  Range:     {ode_range:.6f} nM\n")

    logger.info(f"Biosensor disease separation (H → C):")
    logger.info(f"  Healthy:   {h_s1:.6f}")
    logger.info(f"  CKD-MBD:   {c_s1:.6f}")
    logger.info(f"  Range:     {biosensor_range:.6f}\n")

    logger.info(f"Compression ratio: {compression_ratio:.4f}×")
    logger.info(f"  (biosensor_range / ode_range)")

    if compression_ratio < 0.5:
        logger.info(f"\n⚠️  CRITICAL FINDING:")
        logger.info(f"   Disease signal COMPRESSED by {1/compression_ratio:.2f}×")
        logger.info(f"   Biosensors are attenuating disease separability")
        logger.info(f"\n   LIKELY ROOT CAUSES:")
        logger.info(f"   • Kd values misaligned with disease range [{h_s0:.3f}, {c_s0:.3f}]")
        logger.info(f"   • Sensitivity too low or dynamic range too large")
        logger.info(f"   • Hill kinetics creating non-monotonic response")
        logger.info(f"   • Output saturation or clipping")
    elif compression_ratio > 0.9:
        logger.info(f"\n✓ HEALTHY COMPRESSION:")
        logger.info(f"   Signal preservation is good")
        logger.info(f"   Biosensors are amplifying or preserving disease info")
    else:
        logger.info(f"\n⚠️  MODERATE COMPRESSION:")
        logger.info(f"   Some signal loss ({(1-compression_ratio)*100:.1f}%)")
        logger.info(f"   Acceptable but worth investigating")

    logger.info(f"\n{'='*80}\n")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python biosensor_forensics.py <data_dir>")
        print("\nExample:")
        print("  python biosensor_forensics.py ./RL/sclerostin_rl_results/data")
        sys.exit(1)

    data_dir = sys.argv[1]
    analyze_biosensor_compression(data_dir)