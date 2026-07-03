"""
Population-level analysis of paired instrumentation runs.

CRITICAL:
  - Analyzes scalar metrics across simulation population
  - NOT trajectory points (auto-correlated)
  - Modular separability metrics (extensible)
  - Explicit degradation quantification
  - No artificial distributions
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# SEPARABILITY METRICS (Modular, Extensible)
# ============================================================================

@dataclass
class SeparabilityResult:
    """Result from a separability metric calculation."""
    metric_name: str
    healthy_to_pmo: float
    pmo_to_ckd: float
    healthy_to_ckd: float
    n_samples: int
    is_valid: bool
    notes: str = ""


class SeparabilityMetrics:
    """
    Modular collection of separability metrics.
    
    CRITICAL: Each metric is independent and can be plugged in.
    Later, additional metrics can be added without breaking existing code.
    """
    
    @staticmethod
    def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
        """
        Cohen's d effect size (population-level scalar metrics).
        
        Valid because each value is a summary statistic computed
        once per simulation (independent observation).
        
        NOT auto-correlated trajectory points.
        """
        group1 = np.array(group1).flatten()
        group2 = np.array(group2).flatten()
        
        # Remove NaN values
        group1 = group1[~np.isnan(group1)]
        group2 = group2[~np.isnan(group2)]
        
        n1, n2 = len(group1), len(group2)
        if n1 < 2 or n2 < 2:
            return np.nan
        
        mean1 = np.mean(group1)
        mean2 = np.mean(group2)
        
        var1 = np.var(group1, ddof=1)
        var2 = np.var(group2, ddof=1)
        
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        
        if pooled_std < 1e-12:
            return np.nan
        
        return (mean1 - mean2) / pooled_std
    
    @staticmethod
    def overlap_coefficient(group1: np.ndarray, group2: np.ndarray) -> float:
        """
        Overlap coefficient: fraction of distribution overlap.
        
        Range: [0, 1]
        0 = no overlap (perfect separation)
        1 = identical distributions
        """
        group1 = np.array(group1).flatten()
        group2 = np.array(group2).flatten()
        
        group1 = group1[~np.isnan(group1)]
        group2 = group2[~np.isnan(group2)]
        
        if len(group1) == 0 or len(group2) == 0:
            return np.nan
        
        # Compute KDEs and numerical overlap integral
        from scipy.stats import gaussian_kde
        
        try:
            kde1 = gaussian_kde(group1)
            kde2 = gaussian_kde(group2)
        except:
            return np.nan
        
        # Overlap over union of supports
        min_val = min(np.min(group1), np.min(group2))
        max_val = max(np.max(group1), np.max(group2))
        
        x = np.linspace(min_val, max_val, 100)
        
        # Overlap integral: min(pdf1, pdf2)
        overlap = np.trapz(np.minimum(kde1(x), kde2(x)), x)
        
        return float(overlap)
    
    @staticmethod
    def classifier_accuracy(group1: np.ndarray, group2: np.ndarray) -> float:
        """
        Perfect linear classifier separability.
        
        Finds optimal threshold on scalar feature that
        maximizes binary classification accuracy.
        
        Range: [0.5, 1.0]
        0.5 = random guessing
        1.0 = perfect separation
        """
        group1 = np.array(group1).flatten()
        group2 = np.array(group2).flatten()
        
        group1 = group1[~np.isnan(group1)]
        group2 = group2[~np.isnan(group2)]
        
        if len(group1) < 2 or len(group2) < 2:
            return np.nan
        
        # Labels
        y_true = np.concatenate([
            np.zeros(len(group1)),
            np.ones(len(group2))
        ])
        
        # Values
        x = np.concatenate([group1, group2])
        
        # Find threshold that maximizes accuracy
        thresholds = np.linspace(np.min(x), np.max(x), 50)
        accuracies = []
        
        for threshold in thresholds:
            y_pred = (x >= threshold).astype(int)
            accuracy = np.mean(y_true == y_pred)
            accuracies.append(accuracy)
        
        return float(np.max(accuracies))


class PopulationAnalyzer:
    """
    Population-level analysis of instrumented runs.
    
    Computes effect sizes across many simulations using
    scalar summary metrics (statistically valid).
    """
    
    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: Directory containing master_index.csv and metadata
        """
        self.data_dir = Path(data_dir)
        self.master_df = None
        self.instrument_data = []
        self._load_data()
    
    def _load_data(self):
        """Load instrumentation data from all simulations."""
        
        master_path = self.data_dir / 'master_index.csv'
        self.master_df = pd.read_csv(master_path)
        
        logger.info(f"Loading instrumentation from {len(self.master_df)} simulations...")
        
        for idx, row in self.master_df.iterrows():
            metadata_path = self.data_dir / row['metadata_file']
            
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                instrumentation = metadata.get('instrumentation', {})
                if not instrumentation:
                    continue
                
                # Build record from all stage metrics
                record = {
                    'run_id': row['run_id'],
                    'scenario': row['scenario'],
                    'biosensor_type': row['biosensor_type'],
                    'noise_preset': row['noise_preset'],
                    'snr_db': row['snr_db'],
                    'ttd': row['time_to_detection'],
                    'is_sentinel': (row['time_to_detection'] == 9000),
                }
                
                # Extract all stage metrics as flattened columns
                stages = instrumentation.get('stages', {})
                for stage_name, metrics in stages.items():
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            key = f"{stage_name}_{metric_name}"
                            record[key] = value
                
                # Extract stage deltas
                deltas = instrumentation.get('stage_deltas', {})
                for delta_name, delta_metrics in deltas.items():
                    for metric_name, value in delta_metrics.items():
                        if isinstance(value, (int, float)):
                            key = f"{delta_name}_{metric_name}"
                            record[key] = value
                
                # Biological metadata
                bio_realization = instrumentation.get('biological_realization', {})
                record['realization_hash'] = bio_realization.get('realization_hash')
                record['rng_seed'] = bio_realization.get('rng_seed')
                
                self.instrument_data.append(record)
            
            except Exception as e:
                logger.warning(f"Failed to load {row['metadata_file']}: {e}")
                continue
        
        self.df = pd.DataFrame(self.instrument_data)
        logger.info(f"Successfully loaded {len(self.df)} instrumented simulations")
    
    def analyze_separability_degradation(self) -> Dict:
        """
        Compute separability (using multiple metrics) at each stage.
        
        Returns degradation across pipeline.
        """
        
        print("\n" + "="*80)
        print("STAGEWISE DISEASE SEPARABILITY DEGRADATION")
        print("="*80)
        
        stages = ['stage_0_ode', 'stage_1_biosensor_clean', 
                  'stage_2_biosensor_noisy', 'stage_3_thresholded']
        
        # Use 'mean_signal' as primary feature
        # (works for all stages)
        feature = 'mean_signal'
        
        stage_results = {}
        
        for stage in stages:
            feature_col = f'{stage}_{feature}'
            
            if feature_col not in self.df.columns:
                logger.warning(f"Feature {feature_col} not found")
                continue
            
            print(f"\n{stage}:")
            
            # Separate by disease state
            h_vals = self.df[self.df['scenario'] == 'healthy'][feature_col].dropna().values
            p_vals = self.df[self.df['scenario'] == 'pmo'][feature_col].dropna().values
            c_vals = self.df[self.df['scenario'] == 'ckd_mbd'][feature_col].dropna().values
            
            if len(h_vals) < 2 or len(p_vals) < 2 or len(c_vals) < 2:
                logger.warning(f"Insufficient data for {stage}")
                continue
            
            # Compute multiple separability metrics
            metrics_obj = SeparabilityMetrics()
            
            d_hc = metrics_obj.cohens_d(c_vals, h_vals)
            d_hp = metrics_obj.cohens_d(p_vals, h_vals)
            d_pc = metrics_obj.cohens_d(c_vals, p_vals)
            
            overlap_hp = metrics_obj.overlap_coefficient(h_vals, p_vals)
            overlap_pc = metrics_obj.overlap_coefficient(p_vals, c_vals)
            
            acc_hp = metrics_obj.classifier_accuracy(h_vals, p_vals)
            acc_pc = metrics_obj.classifier_accuracy(p_vals, c_vals)
            
            stage_results[stage] = {
                'n_h': len(h_vals),
                'n_p': len(p_vals),
                'n_c': len(c_vals),
                'h_mean': float(np.mean(h_vals)),
                'p_mean': float(np.mean(p_vals)),
                'c_mean': float(np.mean(c_vals)),
                'd_h_to_p': d_hp,
                'd_p_to_c': d_pc,
                'd_h_to_c': d_hc,
                'overlap_h_to_p': overlap_hp,
                'overlap_p_to_c': overlap_pc,
                'classifier_acc_h_to_p': acc_hp,
                'classifier_acc_p_to_c': acc_pc,
            }
            
            # Pretty print
            print(f"  d(H<->P)={d_hp:7.3f}  d(P<->C)={d_pc:7.3f}  d(H<->C)={d_hc:7.3f}")
            print(f"  Overlap(H,P)={overlap_hp:.3f}  Overlap(P,C)={overlap_pc:.3f}")
            print(f"  Classifier accuracy H vs P: {acc_hp:.3f}  P vs C: {acc_pc:.3f}")
            print(f"  Healthy: {np.mean(h_vals):.4f}  PMO: {np.mean(p_vals):.4f}  CKD: {np.mean(c_vals):.4f}")
        
        # Compute total degradation
        print("\n" + "-"*80)
        print("DEGRADATION SUMMARY")
        print("-"*80)
        
        stages_with_data = [s for s in stages if s in stage_results]
        
        if len(stages_with_data) >= 2:
            d_first = stage_results[stages_with_data[0]]['d_h_to_c']
            d_last = stage_results[stages_with_data[-1]]['d_h_to_c']
            
            if d_first > 0:
                total_loss = max(0, (d_first - d_last) / d_first)
                print(f"\nTotal separability degradation: {100*total_loss:.1f}%")
                print(f"  d_initial: {d_first:.3f}")
                print(f"  d_final:   {d_last:.3f}")
                
                if total_loss > 0.7:
                    print("\n  [!!] SEVERE DEGRADATION")
                    print("      Information loss is critical")
                elif total_loss > 0.4:
                    print("\n  [!] MODERATE DEGRADATION")
                    print("      Significant information loss")
                else:
                    print("\n  [OK] ACCEPTABLE DEGRADATION")
                    print("    Signal preservation is reasonable")
        
        return stage_results
    
    def analyze_noise_impact(self) -> Dict:
        """
        Analyze impact of noise on separability.
        
        Stratifies by SNR level.
        """
        
        print("\n" + "="*80)
        print("NOISE IMPACT ANALYSIS")
        print("="*80)
        
        # Bin by SNR
        snr_bins = pd.cut(self.df['snr_db'],
                          bins=[-100, -5, 0, 5, 10, 30],
                          labels=['<-5dB', '-5:0dB', '0:5dB', '5:10dB', '>10dB'])
        
        results = {}
        
        for bin_label in ['<-5dB', '-5:0dB', '0:5dB', '5:10dB', '>10dB']:
            subset = self.df[snr_bins == bin_label]
            
            if len(subset) == 0:
                continue
            
            # Detection rate
            detection_rate = 1.0 - subset['is_sentinel'].mean()
            
            # Separability if available
            feature_col = 'stage_2_biosensor_noisy_mean_signal'
            if feature_col in subset.columns:
                h_vals = subset[subset['scenario'] == 'healthy'][feature_col].dropna().values
                c_vals = subset[subset['scenario'] == 'ckd_mbd'][feature_col].dropna().values
                
                if len(h_vals) > 1 and len(c_vals) > 1:
                    d_noisy = SeparabilityMetrics.cohens_d(c_vals, h_vals)
                else:
                    d_noisy = np.nan
            else:
                d_noisy = np.nan
            
            results[str(bin_label)] = {
                'n_sims': len(subset),
                'detection_rate': float(detection_rate),
                'd_noisy': float(d_noisy),
                'mean_snr': float(subset['snr_db'].mean()),
            }
            
            print(f"\nSNR {bin_label}: {len(subset)} simulations")
            print(f"  Detection rate: {detection_rate:.1%}")
            print(f"  d(H<->C):       {d_noisy:.3f}")
            print(f"  Mean SNR:       {subset['snr_db'].mean():.1f}dB")
        
        # Correlation analysis
        corr = self.df['snr_db'].corr(~self.df['is_sentinel'])

        print(f"\n" + "-"*80)
        print(f"Correlation (SNR <-> detection): {corr:.3f}")
        print("-"*80)
        
        if abs(corr) > 0.5:
            print("  [*] STRONG: Noise is MAJOR failure factor")
        elif abs(corr) > 0.3:
            print("  [*] MODERATE: Noise contributes significantly")
        elif abs(corr) > 0.1:
            print("  [*] WEAK: Noise is secondary factor")
        else:
            print("  [*] NEGLIGIBLE: Threshold or other factors dominant")
        
        return results
    
    def analyze_threshold_positioning(self) -> Dict:
        """
        Analyze threshold positioning relative to population-level signal distributions.

        FIXED: Correctly compares each sim's threshold against the population-level
        healthy and CKD signal ranges (not against the same run's min/max).
        """

        print("\n" + "="*80)
        print("THRESHOLD POSITIONING ANALYSIS")
        print("="*80)

        # Compute population-level means for healthy and CKD
        h_pop_mean = self.df[self.df['scenario'] == 'healthy'][
            'stage_1_biosensor_clean_mean_signal'
        ].mean()

        c_pop_mean = self.df[self.df['scenario'] == 'ckd_mbd'][
            'stage_1_biosensor_clean_mean_signal'
        ].mean()

        p_pop_mean = self.df[self.df['scenario'] == 'pmo'][
            'stage_1_biosensor_clean_mean_signal'
        ].mean()

        print(f"\nPopulation-level biosensor outputs:")
        print(f"  Healthy:  {h_pop_mean:.6f}")
        print(f"  PMO:      {p_pop_mean:.6f}")
        print(f"  CKD-MBD:  {c_pop_mean:.6f}")
        print(f"  H->C span: {c_pop_mean - h_pop_mean:.6f}\n")

        # Normalize each threshold relative to population range
        positions = []

        for _, row in self.df.iterrows():
            threshold = row.get('stage_3_thresholded_threshold')

            if pd.notna(threshold) and c_pop_mean > h_pop_mean:
                # Normalized position: 0 = healthy level, 1 = CKD level
                norm_pos = (threshold - h_pop_mean) / (c_pop_mean - h_pop_mean)
                positions.append({
                    'norm_pos': norm_pos,
                    'scenario': row.get('scenario'),
                    'is_sentinel': row.get('is_sentinel', False),
                    'snr_db': row.get('snr_db'),
                    'threshold': threshold,
                })

        if positions:
            df_pos = pd.DataFrame(positions)

            print(f"Threshold positioning relative to population [H=0, C=1]:")
            print(f"  Mean:     {df_pos['norm_pos'].mean():.3f}")
            print(f"  Std:      {df_pos['norm_pos'].std():.3f}")
            print(f"  Median:   {df_pos['norm_pos'].median():.3f}")
            print(f"  Range:    [{df_pos['norm_pos'].min():.3f}, {df_pos['norm_pos'].max():.3f}]")

            # Ideal positioning: threshold between healthy and CKD (0 < pos < 1)
            ideal_mask = (df_pos['norm_pos'] > 0) & (df_pos['norm_pos'] < 1)
            ideal_rate = ideal_mask.sum() / len(df_pos)

            print(f"\nQuality check:")
            print(f"  Thresholds in ideal range [H, C]: {ideal_rate:.1%}")

            if ideal_rate < 0.7:
                print(f"  [!] Only {ideal_rate:.1%} thresholds properly positioned")
                print(f"      Many thresholds likely too high or too low")
            else:
                print(f"  [OK] {ideal_rate:.1%} thresholds in reasonable range")

            # Stratify by detection outcome
            sentinel_pos = df_pos[df_pos['is_sentinel']]['norm_pos']
            detected_pos = df_pos[~df_pos['is_sentinel']]['norm_pos']

            if len(sentinel_pos) > 0:
                print(f"\nNon-detected (sentinel) sims:")
                print(f"  Count: {len(sentinel_pos)}")
                print(f"  Mean threshold position: {sentinel_pos.mean():.3f}")
                if sentinel_pos.mean() > 1.0:
                    print(f"  [!] Thresholds above CKD range -> detection impossible")

            if len(detected_pos) > 0:
                print(f"\nDetected sims:")
                print(f"  Count: {len(detected_pos)}")
                print(f"  Mean threshold position: {detected_pos.mean():.3f}")

            return {
                'n_positions': len(df_pos),
                'mean_position': float(df_pos['norm_pos'].mean()),
                'ideal_rate': float(ideal_rate),
                'population_h_mean': float(h_pop_mean),
                'population_c_mean': float(c_pop_mean),
            }
        else:
            print("  (Insufficient threshold data in metadata)")
            return {}
    
    def generate_summary_report(self) -> Dict:
        """Generate complete diagnostic report."""
        
        print("\n" + "="*80)
        print("INSTRUMENTATION ANALYSIS COMPLETE")
        print("="*80)
        
        return {
            'n_simulations': len(self.df),
            'n_scenarios': self.df['scenario'].nunique(),
            'n_biosensors': self.df['biosensor_type'].nunique(),
            'detection_rate': float(1.0 - self.df['is_sentinel'].mean()),
            'ttd_sentinel_rate': float(self.df['is_sentinel'].mean()),
        }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_instrumentation.py <data_dir>")
        sys.exit(1)
    
    analyzer = PopulationAnalyzer(sys.argv[1])
    
    # Run comprehensive analysis
    degradation = analyzer.analyze_separability_degradation()
    noise_impact = analyzer.analyze_noise_impact()
    threshold_analysis = analyzer.analyze_threshold_positioning()
    summary = analyzer.generate_summary_report()
    
    print("\n" + "="*80)
    print("READY FOR INTERPRETATION")
    print("="*80)
    print("\nNext steps:")
    print("  1. Review stagewise degradation metrics")
    print("  2. Assess noise impact correlation")
    print("  3. Evaluate threshold positioning quality")
    print("  4. Determine primary root cause")
    print("  5. Design targeted fix")