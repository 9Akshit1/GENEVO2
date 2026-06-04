#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data processing module for RL training
Handles loading, validation, normalization, and feature engineering
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """Process raw data for RL training"""

    def __init__(self, logger_obj=None):
        self.logger = logger_obj or logger
        self.scenario_encoder = LabelEncoder()
        self.biosensor_encoder = LabelEncoder()
        self.noise_encoder = LabelEncoder()
        # SEPARATE scalers for RL and surrogate features (different feature dimensions)
        self.rl_feature_scaler = StandardScaler()  # For 4D: snr, scenario, biosensor, noise
        self.surrogate_feature_scaler = StandardScaler()  # For 3D: snr, biosensor, noise

    def load_master_index(self, data_dir: Path) -> pd.DataFrame:
        """Load and validate master index"""
        self.logger.info("Loading master index...")

        index_path = data_dir / "master_index.csv"
        if not index_path.exists():
            raise FileNotFoundError(f"Master index not found: {index_path}")

        df = pd.read_csv(index_path)
        self.logger.info(f"Loaded {len(df)} samples")

        # Validate required columns
        required_cols = ['scenario', 'biosensor_type', 'noise_preset', 'snr_db',
                        'detection_rate', 'time_to_detection', 'false_negative_rate']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        return df

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process raw data with feature engineering and normalization"""
        self.logger.info("Processing data...")

        processed = df.copy()

        # Encode categorical variables
        processed['scenario_encoded'] = self.scenario_encoder.fit_transform(df['scenario'])
        processed['biosensor_encoded'] = self.biosensor_encoder.fit_transform(df['biosensor_type'])
        processed['noise_encoded'] = self.noise_encoder.fit_transform(df['noise_preset'])

        # Target metrics: use directly from data (already properly computed by simulator)
        # detection_rate is already normalized [0, 1] from generator.py
        processed['detection_rate'] = df['detection_rate'].astype(np.float32)
        processed['fnr'] = processed['false_negative_rate']
        processed['ttd'] = processed['time_to_detection']

        # Feature engineering: SNR-dependent metrics
        processed['snr_squared'] = processed['snr_db'] ** 2
        processed['snr_abs'] = np.abs(processed['snr_db'])
        processed['snr_log'] = np.log1p(np.abs(processed['snr_db']) + 1)

        # Interaction features
        processed['snr_x_scenario'] = processed['snr_db'] * (processed['scenario_encoded'] - 1)
        processed['snr_x_biosensor'] = processed['snr_db'] * (processed['biosensor_encoded'] - 1.5)

        # TTD-related feature engineering
        processed['ttd_log'] = np.log1p(processed['ttd'])
        processed['ttd_inverse'] = 1.0 / (1.0 + processed['ttd'] / 1000.0)

        self.logger.info("Data processing complete")
        return processed

    def prepare_rl_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """Prepare features for RL agent observation space (4D: SNR, scenario, biosensor, noise)

        Returns BOTH raw and scaled features:
        - Raw: for environment state management (allows proper scaling for surrogates)
        - Scaled: for neural network input (optional, stored in info for reference)
        """
        rl_feature_cols = ['snr_db', 'scenario_encoded', 'biosensor_encoded', 'noise_encoded']
        X_raw = df[rl_feature_cols].values.astype(np.float32)

        # Fit scaler for reference (not used here, but available for consistency)
        self.rl_feature_scaler.fit(X_raw)

        self.logger.info(f"RL features shape: {X_raw.shape}")
        self.logger.info(f"RL Feature raw means: {X_raw.mean(axis=0)}")
        self.logger.info(f"RL Feature raw stds: {X_raw.std(axis=0)}")

        return X_raw, {
            'feature_cols': rl_feature_cols,
            'rl_scaler_mean': self.rl_feature_scaler.mean_.tolist(),
            'rl_scaler_scale': self.rl_feature_scaler.scale_.tolist(),
        }

    def prepare_surrogate_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare features for surrogate models (3D: SNR, biosensor, noise)

        Excludes scenario_encoded to prevent leakage.
        Surrogate predicts outcomes based ONLY on hardware/condition parameters,
        not on experimental scenario grouping.
        """
        feature_cols = ['snr_db', 'biosensor_encoded', 'noise_encoded']
        X = df[feature_cols].values.astype(np.float32)

        # Use dedicated surrogate scaler (handles 3 features, separate from RL scaler)
        X_norm = self.surrogate_feature_scaler.fit_transform(X)

        # Target metrics (surrogates trained on raw targets, not normalized)
        y_dr = df['detection_rate'].values.astype(np.float32)
        y_fnr = df['fnr'].values.astype(np.float32)
        y_ttd = df['ttd'].values.astype(np.float32)

        self.logger.info(f"Surrogate features shape: {X_norm.shape}")
        self.logger.info(f"Detection rate: mean={y_dr.mean():.4f}, std={y_dr.std():.4f}")
        self.logger.info(f"FNR: mean={y_fnr.mean():.4f}, std={y_fnr.std():.4f}")
        self.logger.info(f"TTD: mean={y_ttd.mean():.1f}, std={y_ttd.std():.1f}")

        return X_norm, y_dr, y_fnr, y_ttd

    def get_encoders(self) -> Dict:
        """Get fitted encoders and scalers for later use"""
        return {
            'scenario': self.scenario_encoder,
            'biosensor': self.biosensor_encoder,
            'noise': self.noise_encoder,
            'rl_feature_scaler': self.rl_feature_scaler,
            'surrogate_feature_scaler': self.surrogate_feature_scaler,
        }

    def validate_data(self, df: pd.DataFrame, processed: bool = False):
        """Validate data quality and report statistics

        Args:
            df: DataFrame to validate
            processed: If True, expects processed data columns; if False, expects raw columns
        """
        self.logger.info("Validating data...")

        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            self.logger.warning(f"Missing values found:\n{missing[missing > 0]}")

        # Check scenario distribution
        if 'scenario' in df.columns:
            scenario_counts = df['scenario'].value_counts()
            self.logger.info(f"Scenario distribution:\n{scenario_counts}")

        # For raw data validation
        if not processed:
            if 'snr_db' in df.columns:
                snr_min, snr_max = df['snr_db'].min(), df['snr_db'].max()
                self.logger.info(f"SNR range: [{snr_min:.2f}, {snr_max:.2f}] dB")

            # detection_rate is now stored directly in CSV (not computed from n_detections)
            if 'detection_rate' in df.columns:
                dr_min, dr_max = df['detection_rate'].min(), df['detection_rate'].max()
                dr_mean = df['detection_rate'].mean()
                self.logger.info(f"Detection rate range: [{dr_min:.6f}, {dr_max:.6f}], mean={dr_mean:.6f}")

                # Warn if all zeros (data problem)
                if dr_max == 0:
                    self.logger.warning(f"WARNING: All detection rates are 0! Check simulator output.")

            if 'time_to_detection' in df.columns:
                ttd_min, ttd_max = df['time_to_detection'].min(), df['time_to_detection'].max()
                self.logger.info(f"TTD range: [{ttd_min:.1f}, {ttd_max:.1f}]")

            if 'false_negative_rate' in df.columns:
                fnr_min, fnr_max = df['false_negative_rate'].min(), df['false_negative_rate'].max()
                self.logger.info(f"FNR range: [{fnr_min:.4f}, {fnr_max:.4f}]")

        # For processed data validation
        else:
            if 'detection_rate' in df.columns:
                dr_min, dr_max = df['detection_rate'].min(), df['detection_rate'].max()
                dr_mean = df['detection_rate'].mean()
                dr_std = df['detection_rate'].std()
                self.logger.info(f"Detection rate: min={dr_min:.6f}, max={dr_max:.6f}, mean={dr_mean:.6f}, std={dr_std:.6f}")

                # Sanity check: detection_rate should be in [0, 1]
                if dr_min < 0 or dr_max > 1:
                    self.logger.warning(f"WARNING: detection_rate outside [0,1] range!")

            if 'fnr' in df.columns:
                fnr_min, fnr_max = df['fnr'].min(), df['fnr'].max()
                self.logger.info(f"FNR range: [{fnr_min:.4f}, {fnr_max:.4f}]")

            if 'ttd' in df.columns:
                ttd_min, ttd_max = df['ttd'].min(), df['ttd'].max()
                self.logger.info(f"TTD range: [{ttd_min:.1f}, {ttd_max:.1f}]")

                # Check for outliers in TTD
                q1, q3 = df['ttd'].quantile([0.25, 0.75])
                iqr = q3 - q1
                outliers = df[(df['ttd'] < q1 - 1.5*iqr) | (df['ttd'] > q3 + 1.5*iqr)]
                self.logger.info(f"Outliers in TTD: {len(outliers)} ({100*len(outliers)/len(df):.1f}%)")
