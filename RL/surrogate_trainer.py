#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Surrogate model training module
Trains ML models for reward computation with rigorous validation and model selection
"""

import numpy as np
import joblib
import json
import warnings
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging
from typing import Dict, Tuple

# Suppress scikit-learn joblib warnings at module level
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, module='joblib')
# Specifically suppress the delayed warning that comes from cross_val_score
import logging
logging.getLogger('sklearn.utils.parallel').setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


class SurrogateTrainer:
    """Train and validate surrogate models"""

    def __init__(self, logger_obj=None):
        self.logger = logger_obj or logger
        self.models = {}
        self.metrics = {}

    def train_surrogate(self, X: np.ndarray, y: np.ndarray, metric_name: str,
                       use_cross_val: bool = True, cv_folds: int = 5,
                       test_size: float = 0.2) -> Dict:
        """Train a single surrogate model with proper train/val/test split.

        Uses stratified train/test split to prevent leakage, then cross-validation
        on training set for robust hyperparameter assessment.
        """

        self.logger.info(f"Training {metric_name} surrogate...")

        # Split: 80% train/val, 20% test (holdout)
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        self.logger.info(f"  Data split: {len(X_trainval)} train/val, {len(X_test)} test")

        # Hyperparameters tuned for biosensor optimization
        if metric_name == 'detection_rate':
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
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=1  # Use single thread for inference (no parallelism overhead for single-sample predict)
            )
        else:
            raise ValueError(f"Unknown metric: {metric_name}")

        # Cross-validation on training set only (no test leakage)
        if use_cross_val:
            kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            # Use single thread to avoid joblib parallelism warnings
            cv_scores = cross_val_score(model, X_trainval, y_trainval, cv=kfold, scoring='r2', n_jobs=1)
            self.logger.info(f"  CV R² scores: {cv_scores}")
            self.logger.info(f"  CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        else:
            cv_scores = np.array([0.0])

        # Train on full train/val set
        model.fit(X_trainval, y_trainval)

        # Evaluate on test set (unseen)
        y_test_pred = model.predict(X_test)
        r2_test = r2_score(y_test, y_test_pred)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
        mae_test = mean_absolute_error(y_test, y_test_pred)

        # Also report train performance for comparison
        y_train_pred = model.predict(X_trainval)
        r2_train = r2_score(y_trainval, y_train_pred)

        self.logger.info(f"  Train R²: {r2_train:.4f}")
        self.logger.info(f"  Test R²: {r2_test:.4f}")
        self.logger.info(f"  Test RMSE: {rmse_test:.4f}")
        self.logger.info(f"  Test MAE: {mae_test:.4f}")

        # Check for overfitting
        overfit_gap = r2_train - r2_test
        if overfit_gap > 0.15:
            self.logger.warning(f"  WARNING: Significant overfitting detected (gap={overfit_gap:.3f})")

        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            features = ['SNR', 'Biosensor', 'Noise']  # Updated to match new features
            top_features = sorted(zip(features, importance), key=lambda x: x[1], reverse=True)
            self.logger.info(f"  Top features: {top_features}")

        # Store model and metrics
        self.models[metric_name] = model
        self.metrics[metric_name] = {
            'r2_train': float(r2_train),
            'r2_test': float(r2_test),
            'rmse_test': float(rmse_test),
            'mae_test': float(mae_test),
            'cv_r2_mean': float(cv_scores.mean()) if use_cross_val else r2_test,
            'cv_r2_std': float(cv_scores.std()) if use_cross_val else 0.0,
            'overfitting_gap': float(overfit_gap),
            'model_type': type(model).__name__,
        }

        return self.metrics[metric_name]

    def train_all_surrogates(self, X: np.ndarray, y_dr: np.ndarray,
                            y_fnr: np.ndarray, y_ttd: np.ndarray) -> Dict:
        """Train all surrogate models with proper validation"""

        self.logger.info("=" * 80)
        self.logger.info("Training Surrogate Models (with Train/Val/Test Split)")
        self.logger.info("=" * 80)

        metrics_dr = self.train_surrogate(X, y_dr, 'detection_rate')
        metrics_fnr = self.train_surrogate(X, y_fnr, 'fnr')
        metrics_ttd = self.train_surrogate(X, y_ttd, 'ttd')

        # Summary (test-set performance)
        self.logger.info("\nSurrogate Training Summary (Test Set):")
        self.logger.info(f"  Detection Rate Test R²: {metrics_dr['r2_test']:.4f}")
        self.logger.info(f"  FNR Test R²: {metrics_fnr['r2_test']:.4f}")
        self.logger.info(f"  TTD Test R²: {metrics_ttd['r2_test']:.4f}")
        avg_r2 = np.mean([metrics_dr['r2_test'], metrics_fnr['r2_test'], metrics_ttd['r2_test']])
        self.logger.info(f"  Average Test R²: {avg_r2:.4f}")

        # Quality checks (based on test set)
        if metrics_dr['r2_test'] < 0.7:
            self.logger.warning(f"[WARN] Detection Rate test R² is low: {metrics_dr['r2_test']:.4f}")
        if metrics_fnr['r2_test'] < 0.6:
            self.logger.warning(f"[WARN] FNR test R² is low: {metrics_fnr['r2_test']:.4f}")
        if metrics_ttd['r2_test'] < 0.5:
            self.logger.warning(f"[WARN] TTD test R² is low: {metrics_ttd['r2_test']:.4f}")

        # Overfitting check
        self.logger.info("\nOverfitting Check (Train vs Test Gap):")
        for metric, data in [('Detection Rate', metrics_dr), ('FNR', metrics_fnr), ('TTD', metrics_ttd)]:
            gap = data['overfitting_gap']
            status = "OK" if gap < 0.10 else "WARN" if gap < 0.15 else "BAD"
            self.logger.info(f"  {metric}: {gap:.4f} [{status}]")

        return self.metrics

    def get_models(self) -> Dict:
        """Get trained models"""
        return self.models

    def get_metrics(self) -> Dict:
        """Get training metrics"""
        return self.metrics

    def validate_surrogates(self, X_test: np.ndarray, y_test_dict: Dict[str, np.ndarray]):
        """Validate surrogates on test set"""
        self.logger.info("Validating surrogates on test set...")

        for metric_name, y_test in y_test_dict.items():
            if metric_name not in self.models:
                continue

            model = self.models[metric_name]
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            self.logger.info(f"  {metric_name}: R²={r2:.4f}, RMSE={rmse:.4f}")

    def save_surrogates(self, output_dir: Path, version: str = 'v1'):
        """Save trained surrogate models to disk with metadata.

        Creates:
        - saved_ml/surrogate_{metric}_*.pkl (trained models)
        - saved_ml/surrogate_{metric}_*_metadata.json (model info)
        """
        saved_ml_dir = Path(output_dir) / 'saved_ml'
        saved_ml_dir.mkdir(parents=True, exist_ok=True)

        for metric_name, model in self.models.items():
            # Save model
            model_path = saved_ml_dir / f'surrogate_{metric_name}_{version}.pkl'
            joblib.dump(model, model_path)
            self.logger.info(f"  Saved {metric_name} model: {model_path}")

            # Save metadata
            metadata = {
                'metric_name': metric_name,
                'version': version,
                'timestamp': datetime.now().isoformat(),
                'model_type': self.metrics[metric_name]['model_type'],
                'r2_train': self.metrics[metric_name]['r2_train'],
                'r2_test': self.metrics[metric_name]['r2_test'],
                'rmse_test': self.metrics[metric_name]['rmse_test'],
                'mae_test': self.metrics[metric_name]['mae_test'],
                'overfitting_gap': self.metrics[metric_name]['overfitting_gap'],
                'cv_r2_mean': self.metrics[metric_name]['cv_r2_mean'],
                'cv_r2_std': self.metrics[metric_name]['cv_r2_std'],
                'input_features': ['SNR', 'Biosensor Type', 'Noise Preset'],
                'notes': 'scenario_encoded excluded to prevent leakage'
            }
            meta_path = saved_ml_dir / f'surrogate_{metric_name}_{version}_metadata.json'
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            self.logger.info(f"  Saved metadata: {meta_path.name}")

    @staticmethod
    def load_surrogates(output_dir: Path, version: str = 'v1') -> Dict:
        """Load pre-trained surrogates from disk.

        Returns: dict with 'detection_rate', 'fnr', 'ttd' models
        """
        saved_ml_dir = Path(output_dir) / 'saved_ml'
        models = {}

        for metric in ['detection_rate', 'fnr', 'ttd']:
            model_path = saved_ml_dir / f'surrogate_{metric}_{version}.pkl'
            if model_path.exists():
                models[metric] = joblib.load(model_path)
            else:
                raise FileNotFoundError(f"Surrogate model not found: {model_path}")

        return models
