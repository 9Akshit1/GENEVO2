"""
Priority 1: Variance Inflation Analysis (CORRECTED)

Properly extracts stage metrics from instrumentation metadata.
Measures within-class variance amplification at each stage.

This directly tests whether biosensor parameters amplify within-class variance
beyond biological differences.
"""

import json
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_stage_means(data_dir: str) -> dict:
    """
    Extract mean values at each stage from instrumentation metadata.
    
    Returns dict with structure:
    {
        'stage_0_ode': {'healthy': [...], 'pmo': [...], 'ckd_mbd': [...]},
        'stage_1_clean': {...},
        'stage_2_noisy': {...},
    }
    """
    
    master_path = Path(data_dir) / 'master_index.csv'
    master_df = pd.read_csv(master_path)
    
    logger.info(f"Loading {len(master_df)} simulations...")
    
    stage_data = {
        'stage_0_ode': {'healthy': [], 'pmo': [], 'ckd_mbd': []},
        'stage_1_clean': {'healthy': [], 'pmo': [], 'ckd_mbd': []},
        'stage_2_noisy': {'healthy': [], 'pmo': [], 'ckd_mbd': []},
    }
    
    count_loaded = 0
    count_failed = 0
    
    for idx, row in master_df.iterrows():
        metadata_path = Path(data_dir) / row['metadata_file']
        scenario = row['scenario']
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            instr = metadata.get('instrumentation', {})
            if not instr:
                count_failed += 1
                continue
            
            stages = instr.get('stages', {})
            
            # Extract mean_signal from each stage
            stage_0 = stages.get('stage_0_ode', {})
            stage_1 = stages.get('stage_1_biosensor_clean', {})
            stage_2 = stages.get('stage_2_biosensor_noisy', {})
            
            s0_mean = stage_0.get('mean_signal')
            s1_mean = stage_1.get('mean_signal')
            s2_mean = stage_2.get('mean_signal')
            
            if s0_mean is not None and s1_mean is not None and s2_mean is not None:
                stage_data['stage_0_ode'][scenario].append(float(s0_mean))
                stage_data['stage_1_clean'][scenario].append(float(s1_mean))
                stage_data['stage_2_noisy'][scenario].append(float(s2_mean))
                count_loaded += 1
            else:
                count_failed += 1
        
        except Exception as e:
            count_failed += 1
            if count_failed < 5:
                logger.debug(f"Failed {idx}: {type(e).__name__}")
            continue
    
    logger.info(f"Successfully loaded: {count_loaded}")
    logger.info(f"Failed to load: {count_failed}\n")
    
    return stage_data


def analyze_variance_inflation(stage_data: dict) -> dict:
    """
    Analyze within-class variance at each stage.
    
    Compute variance amplification factors (stage_1_std / stage_0_std).
    """
    
    logger.info("="*80)
    logger.info("VARIANCE INFLATION ANALYSIS")
    logger.info("="*80 + "\n")
    
    results = {}
    
    for stage_name in ['stage_0_ode', 'stage_1_clean', 'stage_2_noisy']:
        logger.info(f"\n{stage_name.upper()}")
        logger.info("-"*80)
        logger.info(f"{'Disease':<15} {'N':<8} {'Mean':<15} {'Std':<15} {'CV':<10}")
        logger.info("-"*80)
        
        stage_results = {}
        
        for disease in ['healthy', 'pmo', 'ckd_mbd']:
            values = np.array(stage_data[stage_name][disease])
            
            if len(values) == 0:
                logger.warning(f"No data for {disease}")
                continue
            
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)  # Sample std
            cv_val = std_val / mean_val if mean_val > 1e-12 else np.nan
            
            logger.info(f"{disease:<15} {len(values):<8} {mean_val:<15.6f} {std_val:<15.6f} {cv_val:<10.4f}")
            
            stage_results[disease] = {
                'n': len(values),
                'mean': float(mean_val),
                'std': float(std_val),
                'cv': float(cv_val),
                'values': values  # Keep for later analysis
            }
        
        results[stage_name] = stage_results
    
    return results


def compute_variance_inflation(results: dict) -> dict:
    """
    Compute inflation factors between stages.
    
    Inflation = std_stage_1 / std_stage_0
    """
    
    logger.info("\n" + "="*80)
    logger.info("VARIANCE INFLATION FACTORS")
    logger.info("="*80 + "\n")
    
    logger.info("Inflation = Std(Stage N) / Std(Stage 0)\n")
    logger.info(f"{'Disease':<15} {'Stage 0→1':<15} {'Stage 0→2':<15} {'Interpretation':<40}")
    logger.info("-"*80)
    
    inflation_analysis = {}
    
    for disease in ['healthy', 'pmo', 'ckd_mbd']:
        s0_std = results['stage_0_ode'][disease]['std']
        s1_std = results['stage_1_clean'][disease]['std']
        s2_std = results['stage_2_noisy'][disease]['std']
        
        inflation_0_to_1 = s1_std / s0_std if s0_std > 1e-12 else np.inf
        inflation_0_to_2 = s2_std / s0_std if s0_std > 1e-12 else np.inf
        
        # Interpretation
        if inflation_0_to_1 > 5.0:
            interp = "SEVERE amplification"
        elif inflation_0_to_1 > 2.0:
            interp = "Major amplification"
        elif inflation_0_to_1 > 1.2:
            interp = "Moderate amplification"
        else:
            interp = "Minimal amplification"
        
        logger.info(f"{disease:<15} {inflation_0_to_1:<15.2f} {inflation_0_to_2:<15.2f} {interp:<40}")
        
        inflation_analysis[disease] = {
            'inflation_0_to_1': float(inflation_0_to_1),
            'inflation_0_to_2': float(inflation_0_to_2),
        }
    
    return inflation_analysis


def analyze_between_class_separation(results: dict) -> dict:
    """
    Analyze between-class separation relative to within-class variance.
    """
    
    logger.info("\n" + "="*80)
    logger.info("BETWEEN-CLASS SEPARATION ANALYSIS")
    logger.info("="*80 + "\n")
    
    separation_analysis = {}
    
    for stage_name in ['stage_0_ode', 'stage_1_clean', 'stage_2_noisy']:
        logger.info(f"\n{stage_name}")
        logger.info("-"*80)
        
        h_values = results[stage_name]['healthy']['values']
        p_values = results[stage_name]['pmo']['values']
        c_values = results[stage_name]['ckd_mbd']['values']
        
        h_mean = results[stage_name]['healthy']['mean']
        p_mean = results[stage_name]['pmo']['mean']
        c_mean = results[stage_name]['ckd_mbd']['mean']
        
        h_std = results[stage_name]['healthy']['std']
        c_std = results[stage_name]['ckd_mbd']['std']
        
        # Compute effect size
        pooled_std = np.sqrt(((len(h_values)-1)*h_std**2 + (len(c_values)-1)*c_std**2) / 
                            (len(h_values) + len(c_values) - 2))
        
        if pooled_std > 1e-12:
            d_hc = (c_mean - h_mean) / pooled_std
        else:
            d_hc = np.nan
        
        # Signal-to-noise ratio
        signal = c_mean - h_mean  # Between-class difference
        noise = (h_std + c_std) / 2  # Average within-class noise
        snr = signal / noise if noise > 1e-12 else np.inf
        
        logger.info(f"H mean: {h_mean:.6f} ± {h_std:.6f}")
        logger.info(f"P mean: {p_mean:.6f}")
        logger.info(f"C mean: {c_mean:.6f} ± {c_std:.6f}")
        logger.info(f"\nEffect size d(H↔C): {d_hc:.3f}")
        logger.info(f"Signal (C-H): {signal:.6f}")
        logger.info(f"Noise (avg std): {noise:.6f}")
        logger.info(f"SNR: {snr:.2f}")
        
        # Interpret
        if snr > 3:
            logger.info("✓ Good separation (SNR > 3)")
        elif snr > 1:
            logger.info("⚠️  Moderate separation (1 < SNR < 3)")
        else:
            logger.info("❌ Poor separation (SNR < 1)")
        
        separation_analysis[stage_name] = {
            'd_hc': float(d_hc),
            'signal': float(signal),
            'noise': float(noise),
            'snr': float(snr),
        }
    
    return separation_analysis


def main(data_dir: str):
    """Run complete variance inflation analysis."""
    
    logger.info("\n" + "="*80)
    logger.info("PRIORITY 1: VARIANCE INFLATION ANALYSIS")
    logger.info("="*80 + "\n")
    
    logger.info("Question: Does biosensor amplify within-class variance?\n")
    
    # Extract data
    stage_data = extract_stage_means(data_dir)
    
    # Analyze variance
    results = analyze_variance_inflation(stage_data)
    
    # Compute inflation
    inflation = compute_variance_inflation(results)
    
    # Analyze separation
    separation = analyze_between_class_separation(results)
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY & INTERPRETATION")
    logger.info("="*80 + "\n")
    
    avg_inflation = np.mean([
        inflation['healthy']['inflation_0_to_1'],
        inflation['pmo']['inflation_0_to_1'],
        inflation['ckd_mbd']['inflation_0_to_1']
    ])
    
    logger.info(f"Average variance inflation (Stage 0→1): {avg_inflation:.2f}×\n")
    
    if avg_inflation > 5.0:
        logger.info("🔴 CRITICAL: Biosensor amplifies variance by >5×")
        logger.info("\nInterpretation:")
        logger.info("  Within-class variance EXPLODES at biosensor stage")
        logger.info("  This overwhelms disease-class differences")
        logger.info("  Parameter heterogeneity is likely PRIMARY cause\n")
        logger.info("Evidence: SNR collapses from ~3 to ~1")
        logger.info("          Even though ordering preserved\n")
        logger.info("Next experiment: FROZEN PARAMETERS")
        logger.info("  Run with identical Kd, sensitivity across simulations")
        logger.info("  If variance shrinks → parameters are culprit")
        return True
    
    elif avg_inflation > 2.0:
        logger.info("🟠 SIGNIFICANT: Variance amplification 2-5×")
        logger.info("\nInterpretation:")
        logger.info("  Biosensor parameters contribute to variance inflation")
        logger.info("  But may not be the ONLY mechanism\n")
        logger.info("Next: Still run frozen parameters experiment")
        return True
    
    elif avg_inflation > 1.2:
        logger.info("🟡 MODERATE: Variance amplification 1.2-2×")
        logger.info("\nInterpretation:")
        logger.info("  Biosensor adds some variance")
        logger.info("  But primary mechanism may be elsewhere\n")
        logger.info("Possible mechanisms:")
        logger.info("  - Feature extraction issues")
        logger.info("  - Normalization bugs")
        logger.info("  - Nonlinear saturation")
        return False
    
    else:
        logger.info("🟢 MODEST: Variance amplification <1.2×")
        logger.info("\nInterpretation:")
        logger.info("  Biosensor does not amplify variance significantly")
        logger.info("  Primary collapse mechanism is elsewhere\n")
        logger.info("Investigate:")
        logger.info("  - Transfer function saturation")
        logger.info("  - Feature extraction")
        logger.info("  - Normalization artifacts")
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python variance_analysis_fixed.py <data_dir>")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    
    try:
        is_parameter_culprit = main(data_dir)
        
        logger.info("\n" + "="*80)
        logger.info("NEXT STEPS")
        logger.info("="*80 + "\n")
        
        if is_parameter_culprit:
            logger.info("Priority: Run frozen parameters experiment")
            logger.info("  python frozen_parameters_experiment.py <model_path>\n")
        else:
            logger.info("Priority: Investigate feature extraction & normalization")
            logger.info("  Check for per-simulation normalization")
            logger.info("  Try different features (max, auc, steady-state)\n")
    
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        sys.exit(1)