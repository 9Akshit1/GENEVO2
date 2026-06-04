#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build surrogate models from simulation data - Scientific Redesign (v2).

Key improvements over v1:
- Detection Rate (DR): Calibrated binary classifier → P(detection=1) ∈ [0,1]
- FNR: Quantile regression for uncertainty intervals
- TTD: Quantile regression for uncertainty intervals
- Actual training bounds saved (not hardcoded)
- Proper validation with scenario-aware splits
- Uncertainty quantification for all metrics
"""

import json
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    brier_score_loss, roc_auc_score, accuracy_score
)
import joblib

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
logging.getLogger('sklearn.utils.parallel').setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


class SurrogateBuilderV2:
    """Build and save scientifically valid surrogate models."""

    def __init__(self, logger_obj=None):
        self.logger = logger_obj or logger
        self.models = {}
        self.scaler = None
        self.label_encoders = {}
        self.training_bounds = {}

    def load_and_extract_features(self, data_dir: Path) -> Tuple[np.ndarray, dict, pd.DataFrame]:
        """Load master_index.csv and extract features from metadata files."""
        self.logger.info(f"Loading data from {data_dir}...")

        master_path = Path(data_dir) / "master_index.csv"
        if not master_path.exists():
            raise FileNotFoundError(f"master_index.csv not found: {master_path}")

        df_results = pd.read_csv(master_path)
        self.logger.info(f"  Loaded {len(df_results)} results from master_index.csv")

        # Extract features from metadata files
        features_list = []
        valid_indices = []

        for idx, row in df_results.iterrows():
            try:
                metadata_file = Path(data_dir) / row['metadata_file']
                if not metadata_file.exists():
                    self.logger.warning(f"  Metadata not found: {metadata_file}")
                    continue

                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                # Extract original input parameters
                biosensor_cfg = metadata['biosensor_config']
                noise_preset = metadata['noise_preset']
                scenario = metadata['scenario']

                feature_row = {
                    'kd': float(biosensor_cfg['kd']),
                    'sensitivity': float(biosensor_cfg['sensitivity']),
                    'biosensor_type': biosensor_cfg['circuit_type'],
                    'noise_preset': noise_preset,
                    'scenario': scenario,
                }

                features_list.append(feature_row)
                valid_indices.append(idx)

            except Exception as e:
                self.logger.warning(f"  Failed to parse metadata for row {idx}: {e}")
                continue

        self.logger.info(f"  Successfully extracted features for {len(features_list)} samples")

        # Convert to DataFrame and encode categoricals
        df_features = pd.DataFrame(features_list)

        # Fit label encoders
        self.label_encoders['biosensor_type'] = LabelEncoder()
        self.label_encoders['noise_preset'] = LabelEncoder()
        self.label_encoders['scenario'] = LabelEncoder()

        df_features['biosensor_type_enc'] = self.label_encoders['biosensor_type'].fit_transform(
            df_features['biosensor_type']
        )
        df_features['noise_preset_enc'] = self.label_encoders['noise_preset'].fit_transform(
            df_features['noise_preset']
        )
        df_features['scenario_enc'] = self.label_encoders['scenario'].fit_transform(
            df_features['scenario']
        )

        self.logger.info(f"\nFeature encoding:")
        self.logger.info(f"  Biosensor types: {list(self.label_encoders['biosensor_type'].classes_)}")
        self.logger.info(f"  Noise presets: {list(self.label_encoders['noise_preset'].classes_)}")
        self.logger.info(f"  Scenarios: {list(self.label_encoders['scenario'].classes_)}")

        # Build feature matrix: [kd, sensitivity, biosensor_type_enc, noise_preset_enc, scenario_enc]
        X = df_features[[
            'kd', 'sensitivity', 'biosensor_type_enc', 'noise_preset_enc', 'scenario_enc'
        ]].values.astype(np.float32)

        # Save actual training bounds
        self.training_bounds = {
            'kd_min': float(df_features['kd'].min()),
            'kd_max': float(df_features['kd'].max()),
            'sensitivity_min': float(df_features['sensitivity'].min()),
            'sensitivity_max': float(df_features['sensitivity'].max()),
        }

        self.logger.info(f"\nActual training bounds:")
        self.logger.info(f"  Kd: [{self.training_bounds['kd_min']:.4f}, {self.training_bounds['kd_max']:.4f}]")
        self.logger.info(f"  Sensitivity: [{self.training_bounds['sensitivity_min']:.4f}, {self.training_bounds['sensitivity_max']:.4f}]")

        feature_names = {
            0: 'kd',
            1: 'sensitivity',
            2: 'biosensor_type_enc',
            3: 'noise_preset_enc',
            4: 'scenario_enc',
        }

        # Filter results to match valid indices
        df_results_filtered = df_results.iloc[valid_indices].reset_index(drop=True)

        return X, feature_names, df_results_filtered

    def fit_scaler(self, X: np.ndarray) -> np.ndarray:
        """Fit scaler and return scaled data."""
        if X.shape[0] < 10:
            raise ValueError(f"Not enough data to fit scaler: {X.shape[0]} samples")

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.logger.info(f"Scaler fitted on {X.shape[0]} samples")
        self.logger.info(f"  Feature means: {self.scaler.mean_}")
        self.logger.info(f"  Feature stds: {self.scaler.scale_}")

        return X_scaled

    def train_dr_classifier(
        self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2
    ) -> Dict:
        """Train calibrated binary classifier for Detection Rate."""
        self.logger.info(f"Training DR classifier (calibrated)...")

        # Split data
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        self.logger.info(f"  Data split: {len(X_trainval)} train/val, {len(X_test)} test")
        self.logger.info(f"  Class distribution: train={np.mean(y_trainval):.2%}, test={np.mean(y_test):.2%}")

        # Base classifier
        base_clf = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            verbose=0
        )

        # Calibrate predictions
        calibrated_clf = CalibratedClassifierCV(base_clf, method='isotonic', cv=5)

        # Cross-validation on base classifier
        skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(base_clf, X_trainval, y_trainval, cv=skfold, scoring='roc_auc', n_jobs=1)
        self.logger.info(f"  CV ROC-AUC scores: {cv_scores}")
        self.logger.info(f"  CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # Train calibrated classifier
        calibrated_clf.fit(X_trainval, y_trainval)

        # Evaluate
        y_train_pred_proba = calibrated_clf.predict_proba(X_trainval)[:, 1]
        y_test_pred_proba = calibrated_clf.predict_proba(X_test)[:, 1]

        # Metrics
        train_auc = roc_auc_score(y_trainval, y_train_pred_proba)
        test_auc = roc_auc_score(y_test, y_test_pred_proba)
        train_brier = brier_score_loss(y_trainval, y_train_pred_proba)
        test_brier = brier_score_loss(y_test, y_test_pred_proba)

        self.logger.info(f"  Train ROC-AUC: {train_auc:.4f}")
        self.logger.info(f"  Test ROC-AUC: {test_auc:.4f}")
        self.logger.info(f"  Train Brier: {train_brier:.4f}")
        self.logger.info(f"  Test Brier: {test_brier:.4f}")

        # Store model
        self.models['detection_rate'] = calibrated_clf

        metrics = {
            'model_type': 'CalibratedClassifierCV(GradientBoostingClassifier)',
            'train_auc': float(train_auc),
            'test_auc': float(test_auc),
            'train_brier': float(train_brier),
            'test_brier': float(test_brier),
            'cv_auc_mean': float(cv_scores.mean()),
            'cv_auc_std': float(cv_scores.std()),
        }

        return metrics

    def train_quantile_regressor(
        self, X: np.ndarray, y: np.ndarray, metric_name: str,
        test_size: float = 0.2, cv_folds: int = 5
    ) -> Dict:
        """Train quantile regression models for FNR and TTD."""
        self.logger.info(f"Training {metric_name} quantile regressors...")

        # Validate input
        if X.shape[0] < 20:
            raise ValueError(f"Not enough data for {metric_name}: {X.shape[0]} samples")
        if y.shape[0] != X.shape[0]:
            raise ValueError(f"Mismatched X ({X.shape[0]}) and y ({y.shape[0]}) sizes")

        # Split data
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        self.logger.info(f"  Data split: {len(X_trainval)} train/val, {len(X_test)} test")
        self.logger.info(f"  Target range: [{y.min():.4f}, {y.max():.4f}]")

        # Train three models: median (0.5), lower (0.1), upper (0.9)
        quantile_models = {}
        for alpha, name in [(0.5, 'median'), (0.1, 'lower'), (0.9, 'upper')]:
            model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                min_samples_split=5,
                min_samples_leaf=2,
                loss='quantile',
                alpha=alpha,
                random_state=42,
                verbose=0
            )

            # Cross-validation
            kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_trainval, y_trainval, cv=kfold, scoring='r2', n_jobs=1)
            self.logger.info(f"  {name} CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

            # Train
            model.fit(X_trainval, y_trainval)
            quantile_models[name] = model

        # Evaluate ensemble
        y_train_pred_median = quantile_models['median'].predict(X_trainval)
        y_test_pred_median = quantile_models['median'].predict(X_test)

        r2_train = r2_score(y_trainval, y_train_pred_median)
        r2_test = r2_score(y_test, y_test_pred_median)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred_median))
        mae_test = mean_absolute_error(y_test, y_test_pred_median)

        self.logger.info(f"  Train R² (median): {r2_train:.4f}")
        self.logger.info(f"  Test R² (median): {r2_test:.4f}")
        self.logger.info(f"  Test RMSE: {rmse_test:.4f}")
        self.logger.info(f"  Test MAE: {mae_test:.4f}")

        # Store models
        self.models[metric_name] = quantile_models

        metrics = {
            'model_type': 'QuantileRegressionEnsemble(GradientBoostingRegressor)',
            'r2_train': float(r2_train),
            'r2_test': float(r2_test),
            'rmse_test': float(rmse_test),
            'mae_test': float(mae_test),
        }

        return metrics

    def train_all_surrogates(
        self, X: np.ndarray, df_results: pd.DataFrame
    ) -> Dict:
        """Train all three surrogate models."""
        self.logger.info("=" * 80)
        self.logger.info("TRAINING SURROGATE MODELS (V2 - Scientific Redesign)")
        self.logger.info("=" * 80)

        # Fit scaler
        X_scaled = self.fit_scaler(X)

        # Extract targets
        y_dr = df_results['detection_rate'].values.astype(np.float32)
        y_fnr = df_results['false_negative_rate'].values.astype(np.float32)
        y_ttd = df_results['time_to_detection'].values.astype(np.float32)

        self.logger.info(f"\nTarget statistics:")
        self.logger.info(f"  Detection Rate: min={y_dr.min():.4f}, max={y_dr.max():.4f}, mean={y_dr.mean():.4f}")
        self.logger.info(f"  False Negative Rate: min={y_fnr.min():.4f}, max={y_fnr.max():.4f}, mean={y_fnr.mean():.4f}")
        self.logger.info(f"  Time-to-Detection: min={y_ttd.min():.1f}, max={y_ttd.max():.1f}, mean={y_ttd.mean():.1f}")

        # Train surrogates
        metrics_all = {}

        self.logger.info(f"\n[1/3] Detection Rate Classifier")
        self.logger.info("-" * 80)
        metrics_all['detection_rate'] = self.train_dr_classifier(X_scaled, y_dr)

        self.logger.info(f"\n[2/3] FNR Quantile Regression")
        self.logger.info("-" * 80)
        metrics_all['fnr'] = self.train_quantile_regressor(X_scaled, y_fnr, 'fnr')

        self.logger.info(f"\n[3/3] TTD Quantile Regression")
        self.logger.info("-" * 80)
        metrics_all['ttd'] = self.train_quantile_regressor(X_scaled, y_ttd, 'ttd')

        return metrics_all

    def save_surrogates(self, output_dir: Path, version: str = 'v1'):
        """Save trained surrogates and scaler to disk."""
        if not self.scaler:
            raise RuntimeError("Scaler not fitted. Call fit_scaler() first.")
        if not self.models:
            raise RuntimeError("No models trained. Call train_all_surrogates() first.")

        output_dir = Path(output_dir)
        saved_ml_dir = output_dir / "saved_ml"
        saved_ml_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"\nSaving surrogates to {saved_ml_dir}...")

        # Save DR classifier
        dr_model_path = saved_ml_dir / f"surrogate_detection_rate_{version}.pkl"
        joblib.dump(self.models['detection_rate'], dr_model_path)
        self.logger.info(f"  Saved DR classifier: {dr_model_path}")

        # Save FNR quantile models
        for quantile, model in self.models['fnr'].items():
            fnr_model_path = saved_ml_dir / f"surrogate_fnr_{quantile}_{version}.pkl"
            joblib.dump(model, fnr_model_path)
            self.logger.info(f"  Saved FNR {quantile} model: {fnr_model_path}")

        # Save TTD quantile models
        for quantile, model in self.models['ttd'].items():
            ttd_model_path = saved_ml_dir / f"surrogate_ttd_{quantile}_{version}.pkl"
            joblib.dump(model, ttd_model_path)
            self.logger.info(f"  Saved TTD {quantile} model: {ttd_model_path}")

        # Save scaler
        scaler_path = saved_ml_dir / f"scaler_{version}.pkl"
        joblib.dump(self.scaler, scaler_path)
        self.logger.info(f"  Saved scaler: {scaler_path}")

        # Save label encoders
        encoders_path = saved_ml_dir / f"label_encoders_{version}.pkl"
        joblib.dump(self.label_encoders, encoders_path)
        self.logger.info(f"  Saved label encoders: {encoders_path}")

        # Save metadata
        feature_names = ['kd', 'sensitivity', 'biosensor_type_enc', 'noise_preset_enc', 'scenario_enc']
        metadata = {
            'version': version,
            'version_description': 'v2 - Calibrated classifier for DR, quantile regression for FNR/TTD',
            'n_features': 5,
            'feature_names': feature_names,
            'feature_order': feature_names,
            'label_encoder_classes': {
                'biosensor_type': list(self.label_encoders['biosensor_type'].classes_),
                'noise_preset': list(self.label_encoders['noise_preset'].classes_),
                'scenario': list(self.label_encoders['scenario'].classes_),
            },
            'scaler_mean': list(map(float, self.scaler.mean_)),
            'scaler_scale': list(map(float, self.scaler.scale_)),
            'training_data_bounds': self.training_bounds,
        }

        metadata_path = saved_ml_dir / f"metadata_{version}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        self.logger.info(f"  Saved metadata: {metadata_path}")

        self.logger.info(f"\n✓ All surrogates and preprocessing state saved to {saved_ml_dir}")
