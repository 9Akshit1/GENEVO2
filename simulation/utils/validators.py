"""
Validation functions for simulation models and parameters.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def validate_parameters(params: Dict[str, Any], 
                       param_ranges: Dict[str, Tuple[float, float]]) -> bool:
    """
    Validate that parameters are within acceptable ranges.
    
    Args:
        params: Dictionary of parameter values
        param_ranges: Dictionary of (min, max) ranges
    
    Returns:
        True if all parameters valid
    
    Raises:
        ValueError: If any parameter out of range
    """
    for param_name, value in params.items():
        if param_name in param_ranges:
            min_val, max_val = param_ranges[param_name]
            if not (min_val <= value <= max_val):
                raise ValueError(
                    f"Parameter {param_name}={value} outside valid range [{min_val}, {max_val}]"
                )
    
    logger.debug(f"Validated {len(params)} parameters")
    return True


def validate_simulation_output(time: np.ndarray,
                              species_data: Dict[str, np.ndarray],
                              check_nans: bool = True,
                              check_negatives: bool = True,
                              check_infinities: bool = True) -> bool:
    """
    Validate simulation output for common numerical issues.
    
    Args:
        time: Time array
        species_data: Dictionary of species name → concentration array
        check_nans: Check for NaN values
        check_negatives: Check for negative concentrations
        check_infinities: Check for infinite values
    
    Returns:
        True if valid
    
    Raises:
        ValueError: If validation fails
    """
    issues = []
    
    for species_name, values in species_data.items():
        if check_nans and np.any(np.isnan(values)):
            issues.append(f"{species_name} contains NaN values")
        
        if check_infinities and np.any(np.isinf(values)):
            issues.append(f"{species_name} contains infinite values")
        
        if check_negatives and np.any(values < 0):
            n_negative = np.sum(values < 0)
            issues.append(
                f"{species_name} contains {n_negative} negative values "
                f"(min={np.min(values):.2e})"
            )
    
    if issues:
        error_msg = "Simulation validation failed:\n" + "\n".join(issues)
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.debug(f"Validated output for {len(species_data)} species")
    return True


def check_model_stability(roadrunner_model, 
                         test_duration: float = 100.0) -> bool:
    """
    Check if ODE model is numerically stable.
    
    Args:
        roadrunner_model: RoadRunner model instance
        test_duration: Duration to test (seconds)
    
    Returns:
        True if stable
    """
    try:
        # Run short simulation
        result = roadrunner_model.simulate(0, test_duration, 50)
        
        # Check for numerical issues
        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
            logger.warning("Model stability test failed: NaN or Inf detected")
            return False
        
        logger.debug("Model stability test passed")
        return True
        
    except Exception as e:
        logger.error(f"Model stability test failed with exception: {e}")
        return False


def validate_biosensor_config(config: Dict[str, Any]) -> bool:
    """
    Validate biosensor configuration.
    
    Args:
        config: Biosensor configuration dictionary
    
    Returns:
        True if valid
    
    Raises:
        ValueError: If configuration invalid
    """
    required_keys = ['circuit_type', 'sensitivity', 'threshold', 
                    'dynamic_range', 'kd', 'response_type']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required biosensor config key: {key}")
    
    # Validate specific parameters
    if config['sensitivity'] <= 0:
        raise ValueError("Sensitivity must be positive")
    
    if config['threshold'] < 0:
        raise ValueError("Threshold must be non-negative")
    
    if config['kd'] <= 0:
        raise ValueError("Kd must be positive")
    
    if config['dynamic_range'][0] >= config['dynamic_range'][1]:
        raise ValueError("Dynamic range min must be < max")
    
    logger.debug(f"Validated biosensor config: {config['circuit_type']}")
    return True

# EXTRA CHECKS (run when this command is ran: python utils/validators.py)
if __name__ == "__main__":   
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    print("="*80)
    print("COMPREHENSIVE DATASET VALIDATION")
    print("="*80)
    
    # Load master index
    master = pd.read_csv('data/master_index.csv')
    
    # ========================================================================
    # CHECK 1: SAMPLE VISUALIZATION (Stratified Random Sampling)
    # ========================================================================
    print("\n[1] Generating sample visualizations...")
    
    # Sample one from each scenario
    for scenario in ['healthy', 'pmo', 'ckd_mbd']:
        scenario_runs = master[master['scenario'] == scenario]
        if len(scenario_runs) > 0:
            run_id = scenario_runs.sample(1).iloc[0]['run_id']
            biosensor_type = scenario_runs.sample(1).iloc[0]['biosensor_type']
            
            ts = pd.read_csv(f'data/timeseries/{run_id}.csv')
            
            print(f"\n  {scenario.upper()} / {biosensor_type}")
            print(f"    Sclerostin: {ts['sclerostin_sensor'].min():.4f} - {ts['sclerostin_sensor'].max():.4f} nM")
            print(f"    Biosensor: {ts['biosensor_signal'].min():.4f} - {ts['biosensor_signal'].max():.4f}")
            print(f"    Detections: {ts['detection_event'].sum()}/{len(ts)}")
            
            plt.figure(figsize=(15, 4))
            plt.subplot(141)
            plt.plot(ts['time'], ts['sclerostin_sensor'])
            plt.title(f'{scenario.upper()}: Sclerostin')
            plt.xlabel('Time (s)')
            plt.ylabel('Concentration (nM)')
            plt.grid(alpha=0.3)
            
            plt.subplot(142)
            plt.plot(ts['time'], ts['biosensor_signal'])
            plt.axhline(y=scenario_runs.sample(1).iloc[0].get('threshold', 0), 
                       color='r', linestyle='--', label='Threshold')
            plt.title('Biosensor Signal')
            plt.xlabel('Time (s)')
            plt.legend()
            plt.grid(alpha=0.3)
            
            plt.subplot(143)
            plt.plot(ts['time'], ts['rankl_bone'], label='RANKL')
            plt.plot(ts['time'], ts['opg_bone'], label='OPG')
            plt.title('RANKL/OPG')
            plt.xlabel('Time (s)')
            plt.ylabel('Concentration (pM)')
            plt.legend()
            plt.grid(alpha=0.3)
            
            plt.subplot(144)
            ratio = ts['rankl_bone'] / (ts['opg_bone'] + 1e-9)
            plt.plot(ts['time'], ratio)
            plt.title('RANKL:OPG Ratio')
            plt.xlabel('Time (s)')
            plt.axhline(y=0.1, color='g', linestyle='--', alpha=0.5, label='Healthy (~0.1)')
            plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Disease (>0.5)')
            plt.legend()
            plt.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'utils/validation_{scenario}.png', dpi=150)
            plt.close()
    
    print("\n  ✓ Saved: validation_*.png")
    
    # ========================================================================
    # CHECK 2: PHYSIOLOGICAL RANGE VALIDATION
    # ========================================================================
    print("\n[2] Checking physiological ranges...")
    
    # Relaxed ranges accounting for model units vs serum measurements
    # Allow up to 10× serum values (local bone microenvironment)
    expected_ranges = {
        'healthy': {
            'sclerostin': (0.02, 1.0),     # Relaxed upper bound
            'rankl': (0.1, 2.0),
            'opg': (1.0, 10.0),
            'ratio': (0.01, 0.5)
        },
        'pmo': {
            'sclerostin': (0.04, 1.5),     # Relaxed upper bound
            'rankl': (0.3, 5.0),
            'opg': (0.5, 5.0),
            'ratio': (0.2, 2.0)
        },
        'ckd_mbd': {
            'sclerostin': (0.08, 2.5),     # Relaxed upper bound
            'rankl': (0.2, 3.0),
            'opg': (1.0, 8.0),
            'ratio': (0.05, 1.0)
        }
    }
    
    issues = []
    for scenario in ['healthy', 'pmo', 'ckd_mbd']:
        scenario_data = master[master['scenario'] == scenario]
        mean_scl = scenario_data['sclerostin_mean'].mean()
        
        exp_min, exp_max = expected_ranges[scenario]['sclerostin']
        
        if mean_scl < exp_min:
            issues.append(f"  ⚠️  {scenario}: Mean sclerostin TOO LOW ({mean_scl:.4f} nM < {exp_min} nM)")
        elif mean_scl > exp_max:
            issues.append(f"  ⚠️  {scenario}: Mean sclerostin TOO HIGH ({mean_scl:.4f} nM > {exp_max} nM)")
        else:
            print(f"  ✓ {scenario}: Sclerostin in range ({mean_scl:.4f} nM)")
    
    if issues:
        print("\n  ISSUES FOUND:")
        for issue in issues:
            print(issue)
    
    # ========================================================================
    # CHECK 3: SCENARIO DISCRIMINATION
    # ========================================================================
    print("\n[3] Checking scenario discrimination...")
    
    scenario_means = master.groupby('scenario')['sclerostin_mean'].mean()
    print(f"  Sclerostin by scenario:")
    for scenario, value in scenario_means.items():
        print(f"    {scenario}: {value:.4f} nM")
    
    # Check if ordering is correct: healthy < pmo/ckd_mbd
    if scenario_means['healthy'] < min(scenario_means['pmo'], scenario_means['ckd_mbd']):
        print("  ✓ Healthy has lowest sclerostin (correct)")
    else:
        print("  ⚠️  Healthy does NOT have lowest sclerostin (WRONG)")
    
    # Check statistical separation
    from scipy.stats import f_oneway
    healthy_scl = master[master['scenario']=='healthy']['sclerostin_mean']
    pmo_scl = master[master['scenario']=='pmo']['sclerostin_mean']
    ckd_scl = master[master['scenario']=='ckd_mbd']['sclerostin_mean']
    
    f_stat, p_value = f_oneway(healthy_scl, pmo_scl, ckd_scl)
    print(f"\n  ANOVA test (scenarios differ?): F={f_stat:.2f}, p={p_value:.2e}")
    if p_value < 0.001:
        print("  ✓ Scenarios are SIGNIFICANTLY different")
    else:
        print("  ⚠️  Scenarios NOT significantly different (problem!)")
    
    # ========================================================================
    # CHECK 4: BIOSENSOR SATURATION ANALYSIS
    # ========================================================================
    print("\n[4] Checking biosensor saturation...")
    
    # Load a sample of time-series to check saturation
    sample_runs = master.sample(min(50, len(master)))
    saturation_count = 0
    
    for _, row in sample_runs.iterrows():
        ts = pd.read_csv(f"data/timeseries/{row['run_id']}.csv")
        signal = ts['biosensor_signal'].values
        
        # Check if >80% of points are within 10% of max
        max_signal = signal.max()
        if max_signal > 0:
            near_max = np.sum(signal > 0.9 * max_signal) / len(signal)
            if near_max > 0.8:
                saturation_count += 1
    
    saturation_rate = saturation_count / len(sample_runs) * 100
    print(f"  Saturation rate: {saturation_rate:.1f}% (sample of {len(sample_runs)})")
    
    if saturation_rate > 50:
        print("  ⚠️  >50% of sensors are saturating - reduce dynamic ranges or scale sclerostin")
    elif saturation_rate > 20:
        print("  ⚠️  20-50% saturation - some biosensors may need retuning")
    else:
        print("  ✓ Low saturation rate (good)")

    # ========================================================================
    # CHECK 4.5: TEMPORAL STABILITY (No Runaway Accumulation)
    # ========================================================================
    print("\n[4.5] Checking temporal stability (no runaway growth)...")
    
    # Sample runs and check if sclerostin grows exponentially
    sample_runs = master.sample(min(30, len(master)))
    runaway_count = 0
    
    for _, row in sample_runs.iterrows():
        ts = pd.read_csv(f"data/timeseries/{row['run_id']}.csv")
        scl = ts['sclerostin_sensor'].values
        
        # Check if final value is >10x initial value (indicates runaway)
        if len(scl) > 10:
            initial = np.mean(scl[1:5])  # Skip t=0
            final = np.mean(scl[-5:])
            
            if initial > 0 and final / initial > 10:
                runaway_count += 1
                print(f"    ⚠️  Run {row['run_id'][:8]}: {initial:.4f}→{final:.4f} nM ({final/initial:.1f}x growth)")
    
    if runaway_count == 0:
        print(f"  ✓ No runaway accumulation detected")
    else:
        print(f"  ⚠️  {runaway_count}/{len(sample_runs)} runs show >10x growth (reduce production rates)")
    
    # ========================================================================
    # CHECK 5: DETECTION CAPABILITY
    # ========================================================================
    print("\n[5] Analyzing detection capability...")
    
    no_detection = master[master['n_detections'] == 0]
    print(f"  Zero-detection runs: {len(no_detection)} ({len(no_detection)/len(master)*100:.1f}%)")
    
    if len(no_detection) > 0:
        print(f"  Breakdown by scenario:")
        for scenario in ['healthy', 'pmo', 'ckd_mbd']:
            scenario_zero = no_detection[no_detection['scenario']==scenario]
            print(f"    {scenario}: {len(scenario_zero)} ({len(scenario_zero)/len(no_detection)*100:.1f}%)")
    
    # Check for pathological cases: high sclerostin + zero detection
    high_scl_no_detect = no_detection[no_detection['sclerostin_mean'] > 
                                      master['sclerostin_mean'].median()]
    if len(high_scl_no_detect) > 0:
        print(f"\n  ⚠️  PATHOLOGICAL: {len(high_scl_no_detect)} runs with HIGH sclerostin but ZERO detections")
        print(f"      Biosensor types: {high_scl_no_detect['biosensor_type'].value_counts().to_dict()}")
    else:
        print(f"  ✓ No pathological zero-detection cases")
    
    # ========================================================================
    # CHECK 6: SNR ANALYSIS
    # ========================================================================
    print("\n[6] SNR distribution analysis...")
    
    low_snr = master[master['snr_db'] < 0]
    print(f"  Negative SNR: {len(low_snr)} ({len(low_snr)/len(master)*100:.1f}%)")
    print(f"  SNR by noise level:")
    print(master.groupby('noise_preset')['snr_db'].describe()[['mean', '50%', 'min', 'max']])
    
    # Check if negative SNR correlates with high noise (expected)
    high_noise_neg_snr = low_snr[low_snr['noise_preset']=='high']
    print(f"\n  Negative SNR in 'high' noise: {len(high_noise_neg_snr)}/{len(low_snr)} ({len(high_noise_neg_snr)/len(low_snr)*100:.1f}%)")
    if len(high_noise_neg_snr)/len(low_snr) > 0.6:
        print("  ✓ Negative SNR mostly in high-noise runs (expected)")
    else:
        print("  ⚠️  Negative SNR not concentrated in high-noise (check noise model)")
    
    # ========================================================================
    # CHECK 7: VARIANCE & DIVERSITY
    # ========================================================================
    print("\n[7] Checking dataset diversity...")
    
    # Coefficient of variation for sclerostin within scenarios
    for scenario in ['healthy', 'pmo', 'ckd_mbd']:
        scenario_data = master[master['scenario']==scenario]['sclerostin_mean']
        cv = scenario_data.std() / scenario_data.mean() * 100
        print(f"  {scenario} sclerostin CV: {cv:.1f}%", end="")
        if 10 < cv < 40:
            print(" ✓ (good diversity)")
        elif cv < 10:
            print(" ⚠️  (low diversity - increase variability)")
        else:
            print(" ⚠️  (very high diversity - check for outliers)")
    
    # Check biosensor parameter diversity
    print(f"\n  Biosensor type distribution:")
    print(master['biosensor_type'].value_counts())
    
    # Check if distribution is reasonably uniform
    type_counts = master['biosensor_type'].value_counts()
    if type_counts.max() / type_counts.min() < 1.5:
        print("  ✓ Biosensor types well-balanced")
    else:
        print("  ⚠️  Biosensor type distribution is imbalanced")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    # Count issues
    critical_issues = []
    warnings = []
    
    # Check sclerostin ranges
    for scenario in ['healthy', 'pmo', 'ckd_mbd']:
        mean_scl = master[master['scenario']==scenario]['sclerostin_mean'].mean()
        exp_min, exp_max = expected_ranges[scenario]['sclerostin']
        if mean_scl < exp_min * 0.5 or mean_scl > exp_max * 2:
            critical_issues.append(f"Sclerostin in {scenario} outside 2x expected range")
    
    # Check saturation
    if saturation_rate > 50:
        critical_issues.append("Biosensor saturation >50%")
    elif saturation_rate > 20:
        warnings.append("Biosensor saturation 20-50%")
    
    # Check pathological detections
    if len(high_scl_no_detect) > 10:
        critical_issues.append(f"{len(high_scl_no_detect)} pathological zero-detection cases")
    
    # Check scenario separation
    if p_value > 0.001:
        critical_issues.append("Scenarios not statistically different")

    print(f"\n🔴 CRITICAL ISSUES: {len(critical_issues)}")
    for issue in critical_issues:
        print(f"   - {issue}")

    print(f"\n WARNINGS: {len(warnings)}")
    for warning in warnings:
        print(f"   - {warning}")

    if len(critical_issues) == 0:
        print("\n✅ DATASET PASSED VALIDATION")
        print("   Ready for RL training!")
    else:
        print("\n❌ DATASET HAS CRITICAL ISSUES")
        print("   Apply fixes above and regenerate!")

    print("="*80)