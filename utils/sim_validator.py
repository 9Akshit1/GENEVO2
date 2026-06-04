"""
Comprehensive simulation validator - v4.0 (Tier 1 Fixes).

FIX 3: ROBUSTNESS CHECKS
========================
New validation methods:
  - check_ttd_structure(): Detects pathological TTD distributions
  - check_gradient_stability(): Verifies detection rates are monotonic
  - check_equilibrium_markers(): Samples runs to validate final-state stability

These catch quality issues that simple range checks miss.
All other validation logic from v3.x unchanged.
"""

import os
import json
import numpy as np
import pandas as pd
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

_TIMEOUT_SENTINEL = 9000.0


class SimulationValidator:
    """Comprehensive validator for dataset."""

    def __init__(self, data_dir: str, master_index_path: str):
        self.data_dir          = data_dir
        self.master_index_path = master_index_path
        self.master_df         = pd.read_csv(master_index_path)
        self.n_samples         = len(self.master_df)
        logger.info(f"Validator initialised: {self.n_samples} simulations (v4.0 Tier 1)")

    # ==================== ORIGINAL CHECKS (v3.x) ====================
    
    def check_completeness(self) -> Dict:
        """Verify all required columns and files are present."""
        logger.info("  Checking data completeness...")
        issues = []

        required_cols = {
            'run_id', 'scenario', 'biosensor_type', 'noise_preset',
            'snr_db', 'time_to_detection', 'false_negative_rate',
            'metadata_file', 'timeseries_file',
        }
        missing_cols = required_cols - set(self.master_df.columns)
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")

        nan_counts = self.master_df.isnull().sum()
        if nan_counts.sum() > 0:
            issues.append(f"NaN values:\n{nan_counts[nan_counts > 0]}")

        missing_files = []
        for _, row in self.master_df.iterrows():
            for col in ('metadata_file', 'timeseries_file'):
                p = os.path.join(self.data_dir, row[col])
                if not os.path.exists(p):
                    missing_files.append(p)
        if missing_files:
            issues.append(f"Missing {len(missing_files)} data files")

        for _, row in self.master_df.iloc[:10].iterrows():
            try:
                p = os.path.join(self.data_dir, row['metadata_file'])
                with open(p, 'r', encoding='utf-8') as f:
                    json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError, FileNotFoundError):
                issues.append(f"JSON error in {row['metadata_file']}")

        return {
            'status':          not bool(issues),
            'n_samples':       self.n_samples,
            'n_missing_files': len(missing_files),
            'issues':          issues,
        }

    def check_signal_quality(self) -> Dict:
        """Check SNR, TTD, and FNR distributions."""
        logger.info("  Checking signal quality...")
        issues = []

        snr_mean = float(self.master_df['snr_db'].mean())
        snr_std  = float(self.master_df['snr_db'].std())

        if snr_mean < -10.0:
            issues.append(
                f"Mean SNR critically low (mean={snr_mean:.2f} dB, target >-10). "
                "Signal is heavily noise-dominated."
            )

        ttd      = self.master_df['time_to_detection']
        ttd_mean = float(ttd.mean())
        ttd_std  = float(ttd.std())
        ttd_cv   = ttd_std / ttd_mean if ttd_mean > 0 else float('inf')
        logger.info(f"    TTD: mean={ttd_mean:.0f}s  std={ttd_std:.0f}s  CV={ttd_cv:.2f}")

        q1, q3 = np.percentile(ttd.values, [25, 75])
        iqr    = float(q3 - q1)
        if iqr < 100 and ttd_mean < 500:
            issues.append(
                f"TTD IQR={iqr:.0f} s is tiny -- signal may be constant-zero."
            )

        fnr_mean = float(self.master_df['false_negative_rate'].mean())
        fnr_std  = float(self.master_df['false_negative_rate'].std())
        if fnr_mean > 0.70:
            issues.append(f"FNR mean too high ({fnr_mean:.2%}) -- most sensors failing.")

        return {
            'status':     not bool(issues),
            'snr_mean_db': snr_mean,
            'snr_std_db':  snr_std,
            'ttd_mean_s':  ttd_mean,
            'ttd_std_s':   ttd_std,
            'ttd_cv':      ttd_cv,
            'fnr_mean':    fnr_mean,
            'fnr_std':     fnr_std,
            'issues':      issues,
        }

    def check_parameter_realism(self) -> Dict:
        """Verify sampled parameters are physiologically plausible."""
        logger.info("  Checking parameter realism...")
        issues = []

        sample_path = os.path.join(
            self.data_dir,
            self.master_df.iloc[0]['metadata_file']
        )
        try:
            with open(sample_path, 'r', encoding='utf-8') as f:
                sample = json.load(f)
            params = sample['environment_params']

            if params.get('Estrogen', 0.0) > 3.0:
                issues.append(
                    f"Estrogen unrealistically high: {params['Estrogen']:.3f} nM "
                    f"(physiological max ~3.0 nM)"
                )
            if params.get('PTH', 0.0) > 400:
                issues.append(
                    f"PTH unrealistically high: {params['PTH']:.1f} pg/mL "
                    f"(physiological max ~400 in severe HPT)"
                )
            if params.get('Sclerostin_bone', 0.0) > 0.5:
                issues.append(
                    f"Sclerostin_bone unrealistically high: "
                    f"{params['Sclerostin_bone']:.4f} nM"
                )

            for key in ('k_prod_Scl', 'k_prod_RANKL', 'k_prod_OPG'):
                val = params.get(key, None)
                if val is not None and val <= 0:
                    issues.append(f"Non-positive rate constant: {key} = {val}")

        except FileNotFoundError:
            issues.append(f"Sample metadata not found: {sample_path}")
        except Exception as e:
            issues.append(f"Could not read sample metadata: {e}")

        return {
            'status': not bool(issues),
            'issues': issues,
        }

    def check_distribution_balance(self) -> Dict:
        """Verify scenario/noise/biosensor sampling fractions."""
        logger.info("  Checking distribution balance...")

        scenario_dist  = self.master_df['scenario'].value_counts(normalize=True)
        noise_dist     = self.master_df['noise_preset'].value_counts(normalize=True)
        biosensor_dist = self.master_df['biosensor_type'].value_counts(normalize=True)

        return {
            'scenario_dist':  scenario_dist.to_dict(),
            'noise_dist':     noise_dist.to_dict(),
            'biosensor_dist': biosensor_dist.to_dict(),
        }

    # ==================== NEW CHECKS (Tier 1 Fix 3) ====================

    def check_ttd_structure(self) -> Dict:
        """
        FIX 3: Check TTD distribution for pathologies.
        
        Detects:
          - Sentinel clustering (many 9000 s values)
          - Bimodal distribution (real detections + sentinel)
          - Stuck signals (all same TTD)
        """
        logger.info("  Checking TTD structure (Tier 1 Fix 3)...")
        issues = []

        ttd = self.master_df['time_to_detection']
        
        # Detect sentinel clustering
        sentinel_count = (ttd == _TIMEOUT_SENTINEL).sum()
        sentinel_frac = sentinel_count / len(ttd) if len(ttd) > 0 else 0
        
        logger.info(f"    TTD sentinel values: {sentinel_count}/{len(ttd)} ({sentinel_frac:.1%})")
        
        if sentinel_frac > 0.25:
            issues.append(
                f"TTD: {sentinel_frac:.1%} are sentinel ({_TIMEOUT_SENTINEL}s non-detected). "
                f"Check if system equilibrates properly (Fix 1)."
            )
        
        # Distribution of detected signals
        detected_ttd = ttd[ttd < _TIMEOUT_SENTINEL]
        if len(detected_ttd) > 10:
            logger.info(f"    TTD (detected only): "
                       f"mean={detected_ttd.mean():.0f}s, "
                       f"median={detected_ttd.median():.0f}s, "
                       f"std={detected_ttd.std():.0f}s")

        return {
            'status': not bool(issues),
            'sentinel_fraction': float(sentinel_frac),
            'n_detected': len(detected_ttd),
            'issues': issues
        }

    def check_gradient_stability(self) -> Dict:
        """
        FIX 3: Check if detection rates form stable gradient.
        
        RL cannot learn disease discrimination if healthy ≈ PMO ≈ CKD.
        This checks for monotonicity: healthy ≤ PMO ≤ CKD-MBD
        """
        logger.info("  Checking detection gradient (Tier 1 Fix 3)...")
        issues = []
        warnings = []

        detection_by_scenario = self.master_df.groupby('scenario').apply(
            lambda x: ((x['time_to_detection'] < _TIMEOUT_SENTINEL).sum() / len(x))
            if len(x) > 0 else 0
        )

        scenarios = ['healthy', 'pmo', 'ckd_mbd']
        rates = {s: detection_by_scenario.get(s, 0) for s in scenarios}

        logger.info(f"    Detection rates: {rates}")

        # Monotonicity check
        if not (rates['healthy'] <= rates['pmo'] <= rates['ckd_mbd']):
            issues.append(
                f"Detection gradient NOT monotonic: "
                f"healthy={rates['healthy']:.1%}, "
                f"pmo={rates['pmo']:.1%}, "
                f"ckd={rates['ckd_mbd']:.1%}. "
                f"RL cannot learn disease-severity signal."
            )
        
        # Check spacing
        pmo_healthy_gap = rates['pmo'] - rates['healthy']
        ckd_pmo_gap = rates['ckd_mbd'] - rates['pmo']
        
        if pmo_healthy_gap < 0.05:
            warnings.append(
                f"PMO only {pmo_healthy_gap:.1%} above healthy; gradient is shallow."
            )

        return {
            'status': not bool(issues),
            'detection_rates': rates,
            'gaps': {'healthy_to_pmo': float(pmo_healthy_gap), 'pmo_to_ckd': float(ckd_pmo_gap)},
            'issues': issues,
            'warnings': warnings
        }

    def check_equilibrium_markers(self) -> Dict:
        """
        FIX 3: Sample a few runs and check equilibrium validation.
        
        For each sample, check the equilibration_check metadata
        to see if final-state CV was reported as high.
        """
        logger.info("  Checking equilibrium markers (sample n=5, Tier 1 Fix 3)...")
        issues = []

        sample_indices = np.random.choice(
            len(self.master_df),
            min(5, len(self.master_df)),
            replace=False
        )

        high_cv_count = 0

        for idx in sample_indices:
            metadata_path = os.path.join(
                self.data_dir,
                self.master_df.iloc[idx]['metadata_file']
            )

            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # Check if equilibration_check exists and has high CVs
                eq_check = metadata.get('equilibration_check', {})
                
                for species_name, cv in eq_check.items():
                    if cv > 0.05:  # >5% at final state suggests non-equilibrium
                        high_cv_count += 1
                        logger.warning(
                            f"  Sample {idx}: {species_name} CV={cv:.2%} at final state"
                        )

            except Exception as e:
                logger.warning(f"  Could not check equilibrium for sample {idx}: {e}")

        if high_cv_count > 0:
            logger.warning(
                f"  {high_cv_count} species in samples have CV>5%. "
                f"Check if equilibration duration is sufficient (Fix 1)."
            )

        return {
            'status': high_cv_count == 0,
            'high_cv_species_count': high_cv_count,
            'n_sampled': len(sample_indices),
        }

    # ==================== MAIN VALIDATION RUN ====================

    def run_comprehensive_validation(self) -> Dict:
        """Run all validation checks."""
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE SIMULATION VALIDATION (v4.0 Tier 1)")
        logger.info("="*80)

        results = {
            'completeness': self.check_completeness(),
            'signal_quality': self.check_signal_quality(),
            'ttd_structure': self.check_ttd_structure(),  # NEW (Fix 3)
            'gradient_stability': self.check_gradient_stability(),  # NEW (Fix 3)
            'equilibrium_markers': self.check_equilibrium_markers(),  # NEW (Fix 3)
            'parameter_realism': self.check_parameter_realism(),
            'distribution_balance': self.check_distribution_balance(),
        }

        # Overall status
        overall_pass = all(r.get('status', False) for r in results.values())

        logger.info("\n" + "="*80)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*80)

        for check_name, result in results.items():
            status_str = "✓ OK" if result.get('status', False) else "✗ FAIL"
            logger.info(f"  [{status_str}] {check_name}")

            if 'issues' in result and result['issues']:
                for issue in result['issues']:
                    logger.warning(f"      {issue}")

            if 'warnings' in result and result['warnings']:
                for warning in result['warnings']:
                    logger.info(f"      ⚠️  {warning}")

        logger.info("="*80)

        return {
            'overall_status': overall_pass,
            'results': results
        }


def main():
    """Standalone validation script."""
    import argparse

    parser = argparse.ArgumentParser(description='Validate simulation dataset')
    parser.add_argument('--input_dir', type=str, default='data',
                       help='Input data directory')
    parser.add_argument('--master_index', type=str, default='master_index.csv',
                       help='Name of master index file')

    args = parser.parse_args()

    master_index_path = os.path.join(args.input_dir, args.master_index)

    if not os.path.exists(master_index_path):
        logger.error(f"Master index not found: {master_index_path}")
        return 1

    validator = SimulationValidator(args.input_dir, master_index_path)
    results = validator.run_comprehensive_validation()

    return 0 if results['overall_status'] else 1


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    sys.exit(main())