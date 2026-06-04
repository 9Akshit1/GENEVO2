#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build surrogate models for BO - RL-based approach (FIXED VERSION)

KEY IMPROVEMENTS OVER V1:
1. Use RL's proven data features [SNR, biosensor_type, noise] instead of raw parameters
2. Extract SNR directly from simulation data (not trying to predict from kd/sensitivity)
3. Exclude scenario to prevent data leakage
4. Stricter train/val/test split (no leakage)
5. Better hyperparameter tuning
6. Regression for all metrics (including DR)
7. Rigorous validation on holdout test set
8. Clear quality checks with actionable warnings

This mirrors RL's surrogate_trainer.py approach which works reliably.
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
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
logging.getLogger('sklearn.utils.parallel').setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


class SurrogateBuilderV2RL:
    """
    Build surrogate models using RL's proven approach.

    PHILOSOPHY: Train on simulation-derived features (SNR, biosensor, noise)
    not raw input parameters. SNR is the actual outcome we measure.
    """

    def __init__(self, logger_obj=None):
        self.logger = logger_obj or logger
        self.models = {}
        self.scaler = None
        self.label_encoders = {}
        self.training_bounds = {}

    def load_and_prepare_data(self, data_dir: Path) -> Tuple[np.ndarray, dict, pd.DataFrame]:
        """
        Load master_index and prepare RL-style features.

        Returns:
            X: Feature matrix [SNR, biosensor_encoded, noise_encoded]
            feature_names: Dict mapping column indices to names
            df_results: Original results dataframe
        """
        self.logger.info(f"Loading data from {data_dir}...")

        master_path = Path(data_dir) / "master_index.csv"
        if not master_path.exists():
            raise FileNotFoundError(f"master_index.csv not found: {master_path}")

        df = pd.read_csv(master_path)
        self.logger.info(f"  Loaded {len(df)} results")

        # Validate required columns (direct from master_index)
        required_cols = ['scenario', 'biosensor_type', 'noise_preset', 'snr_db',
                         'detection_rate', 'false_negative_rate', 'time_to_detection']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Encode categorical variables
        self.label_encoders['biosensor_type'] = LabelEncoder()
        self.label_encoders['noise_preset'] = LabelEncoder()

        df['biosensor_encoded'] = self.label_encoders['biosensor_type'].fit_transform(
            df['biosensor_type']
        )
        df['noise_encoded'] = self.label_encoders['noise_preset'].fit_transform(
            df['noise_preset']
        )

        self.logger.info(f"\nFeature encoding:")
        self.logger.info(f"  Biosensor types: {list(self.label_encoders['biosensor_type'].classes_)}")
        self.logger.info(f"  Noise presets: {list(self.label_encoders['noise_preset'].classes_)}")

        # Build feature matrix: [SNR, biosensor_encoded, noise_encoded]
        # NOTE: Scenario intentionally excluded to prevent leakage
        feature_cols = ['snr_db', 'biosensor_encoded', 'noise_encoded']
        X = df[feature_cols].values.astype(np.float32)

        self.logger.info(f"\nFeature matrix:")
        self.logger.info(f"  Shape: {X.shape}")
        self.logger.info(f"  SNR range: [{X[:, 0].min():.2f}, {X[:, 0].max():.2f}] dB")
        self.logger.info(f"  Biosensor type encoding: {sorted(set(X[:, 1].astype(int)))}")
        self.logger.info(f"  Noise preset encoding: {sorted(set(X[:, 2].astype(int)))}")

        # Training bounds (for OOD detection in BO)
        self.training_bounds = {
            'snr_min': float(X[:, 0].min()),
            'snr_max': float(X[:, 0].max()),
        }

        feature_names = {
            0: 'snr_db',
            1: 'biosensor_type_encoded',
            2: 'noise_preset_encoded',
        }

        return X, feature_names, df

    def train_surrogate(
        self, X: np.ndarray, y: np.ndarray, metric_name: str,
        test_size: float = 0.2, cv_folds: int = 5
    ) -> Dict:
        """
        Train single surrogate with proper train/val/test split.

        No data leakage: Cross-validation only on training set.
        """
        self.logger.info(f"\n[{metric_name.upper()}] Training surrogate...")

        # Split: 80% train/val, 20% test (holdout)
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        self.logger.info(f"  Data split: {len(X_trainval)} train/val, {len(X_test)} test")

        # Select model and hyperparameters (tuned for biosensor metrics)
        if metric_name == 'detection_rate':
            # Regression (not classification!) to predict actual detection RATE
            model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=7,
                subsample=0.8,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                verbose=0
            )
        elif metric_name == 'fnr':
            # FNR also regression (not just binary classification)
            model = GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.08,
                max_depth=6,
                subsample=0.8,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                verbose=0
            )
        elif metric_name == 'ttd':
            # TTD benefits from RandomForest
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=1
            )
        else:
            raise ValueError(f"Unknown metric: {metric_name}")

        # Cross-validation on training set only
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_trainval, y_trainval, cv=kfold, scoring='r2', n_jobs=1)

        self.logger.info(f"  CV R2 scores: {[f'{x:.4f}' for x in cv_scores]}")
        self.logger.info(f"  CV R2: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # Train on full train/val set
        model.fit(X_trainval, y_trainval)

        # Evaluate on test set (unseen)
        y_test_pred = model.predict(X_test)
        r2_test = r2_score(y_test, y_test_pred)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
        mae_test = mean_absolute_error(y_test, y_test_pred)

        # Train performance (for overfitting check)
        y_train_pred = model.predict(X_trainval)
        r2_train = r2_score(y_trainval, y_train_pred)

        self.logger.info(f"  Train R2: {r2_train:.4f}")
        self.logger.info(f"  Test R2: {r2_test:.4f}")
        self.logger.info(f"  Test RMSE: {rmse_test:.4f}")
        self.logger.info(f"  Test MAE: {mae_test:.4f}")

        # Overfitting check
        overfit_gap = r2_train - r2_test
        status = "[OK]" if overfit_gap < 0.10 else "[WARN]" if overfit_gap < 0.15 else "[BAD]"
        self.logger.info(f"  Overfitting gap: {overfit_gap:.4f} {status}")

        if overfit_gap > 0.15:
            self.logger.warning(f"  WARNING: Significant overfitting (gap={overfit_gap:.3f})")

        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            features = ['SNR', 'Biosensor', 'Noise']
            top_features = sorted(zip(features, importance), key=lambda x: x[1], reverse=True)
            self.logger.info(f"  Feature importance: {top_features}")

        # Store
        self.models[metric_name] = model
        metrics = {
            'metric_name': metric_name,
            'r2_train': float(r2_train),
            'r2_test': float(r2_test),
            'rmse_test': float(rmse_test),
            'mae_test': float(mae_test),
            'cv_r2_mean': float(cv_scores.mean()),
            'cv_r2_std': float(cv_scores.std()),
            'overfitting_gap': float(overfit_gap),
            'model_type': type(model).__name__,
        }

        return metrics

    def train_all_surrogates(
        self, X: np.ndarray, y_dr: np.ndarray,
        y_fnr: np.ndarray, y_ttd: np.ndarray
    ) -> Dict:
        """Train all three surrogate models with validation."""

        self.logger.info("=" * 80)
        self.logger.info("TRAINING SURROGATE MODELS (RL-Based Approach)")
        self.logger.info("=" * 80)

        metrics = {}
        metrics['detection_rate'] = self.train_surrogate(X, y_dr, 'detection_rate')
        metrics['fnr'] = self.train_surrogate(X, y_fnr, 'fnr')
        metrics['ttd'] = self.train_surrogate(X, y_ttd, 'ttd')

        # Summary
        self.logger.info("\n" + "=" * 80)
        self.logger.info("SURROGATE TRAINING SUMMARY")
        self.logger.info("=" * 80)

        self.logger.info(f"\nTest Set Performance (Primary Metric):")
        self.logger.info(f"  Detection Rate R2: {metrics['detection_rate']['r2_test']:.4f}")
        self.logger.info(f"  FNR R2: {metrics['fnr']['r2_test']:.4f}")
        self.logger.info(f"  TTD R2: {metrics['ttd']['r2_test']:.4f}")

        avg_r2 = np.mean([
            metrics['detection_rate']['r2_test'],
            metrics['fnr']['r2_test'],
            metrics['ttd']['r2_test'],
        ])
        self.logger.info(f"  Average Test R2: {avg_r2:.4f}")

        # Quality thresholds
        self.logger.info(f"\nQuality Checks:")
        for metric_name in ['detection_rate', 'fnr', 'ttd']:
            r2 = metrics[metric_name]['r2_test']
            if r2 < 0.5:
                status = "[CRITICAL]"
            elif r2 < 0.7:
                status = "[WARNING]"
            else:
                status = "[OK]"
            self.logger.info(f"  {metric_name}: R2={r2:.4f} {status}")

        return metrics

    def fit_scaler(self, X: np.ndarray):
        """Fit and store StandardScaler for feature normalization."""
        self.scaler = StandardScaler()
        self.scaler.fit(X)

        self.logger.info(f"\nFeature Scaler:")
        self.logger.info(f"  Means: {self.scaler.mean_}")
        self.logger.info(f"  Scales: {self.scaler.scale_}")

    def save_surrogates(self, output_dir: Path, version: str = 'v2_rl'):
        """Save models and metadata."""
        import joblib

        saved_ml_dir = Path(output_dir) / 'saved_ml'
        saved_ml_dir.mkdir(parents=True, exist_ok=True)

        for metric_name, model in self.models.items():
            model_path = saved_ml_dir / f'surrogate_{metric_name}_{version}.pkl'
            joblib.dump(model, model_path)
            self.logger.info(f"  Saved {metric_name}: {model_path}")

        # Save scaler
        scaler_path = saved_ml_dir / f'scaler_{version}.pkl'
        joblib.dump(self.scaler, scaler_path)
        self.logger.info(f"  Saved scaler: {scaler_path}")

        # Collect metrics for all models (exclude model objects which aren't JSON serializable)
        model_metrics = {}
        for metric_name in ['detection_rate', 'fnr', 'ttd']:
            if metric_name in self.models:
                # Find metrics dict by searching through all trained models
                model_metrics[metric_name] = {
                    'model_type': type(self.models[metric_name]).__name__,
                }

        # Save metadata
        metadata = {
            'version': version,
            'approach': 'RL-based (features: SNR, biosensor_type, noise_preset)',
            'feature_names': ['snr_db', 'biosensor_type_encoded', 'noise_preset_encoded'],
            'feature_note': 'Scenario intentionally excluded to prevent leakage',
            'n_features': 3,
            'label_encoder_classes': {
                'biosensor_type': list(self.label_encoders['biosensor_type'].classes_),
                'noise_preset': list(self.label_encoders['noise_preset'].classes_),
            },
            'scaler_mean': self.scaler.mean_.tolist(),
            'scaler_scale': self.scaler.scale_.tolist(),
            'training_bounds': self.training_bounds,
            'model_types': model_metrics,
        }

        meta_path = saved_ml_dir / f'metadata_{version}.json'
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        self.logger.info(f"  Saved metadata: {meta_path}")


def main():
    """Build surrogates using RL-based approach."""
    logging.basicConfig(level=logging.INFO)

    builder = SurrogateBuilderV2RL()

    # Load data
    X, feature_names, df = builder.load_and_prepare_data(Path('data'))

    # Fit scaler
    builder.fit_scaler(X)

    # Prepare targets
    y_dr = df['detection_rate'].values.astype(np.float32)
    y_fnr = df['false_negative_rate'].values.astype(np.float32)
    y_ttd = df['time_to_detection'].values.astype(np.float32)

    # Train
    metrics = builder.train_all_surrogates(X, y_dr, y_fnr, y_ttd)

    # Save
    builder.save_surrogates(Path('BO/bo_results'), version='v2_rl')

    logger.info("\n[OK] Surrogate training complete (RL-based approach)")


if __name__ == '__main__':
    main()
