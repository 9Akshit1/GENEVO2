#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Surrogate loader for v2_rl models (RL-based approach).

Loads models trained on [SNR, biosensor_type, noise] features.
Features are much simpler than v1 (no kd, sensitivity, scenario).
"""

import json
import joblib
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)


class SurrogateLoaderV2RL:
    """Load and use v2_rl surrogates trained on [SNR, biosensor_type, noise]."""

    def __init__(self, results_dir: Path = None):
        """
        Initialize loader.

        Args:
            results_dir: Path to BO/bo_results directory
        """
        if results_dir is None:
            results_dir = Path(__file__).parent.parent / 'bo_results'
        else:
            results_dir = Path(results_dir)

        self.results_dir = results_dir
        self.saved_ml_dir = results_dir / 'saved_ml'

        # Load models
        self._load_models()

    def _load_models(self):
        """Load v2_rl models and metadata."""
        logger.info(f"Loading v2_rl surrogates from {self.saved_ml_dir}...")

        # Load models
        self.models = {}
        for metric in ['detection_rate', 'fnr', 'ttd']:
            model_path = self.saved_ml_dir / f'surrogate_{metric}_v2_rl.pkl'
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            self.models[metric] = joblib.load(model_path)
            logger.info(f"  Loaded {metric}: {model_path.name}")

        # Load scaler
        scaler_path = self.saved_ml_dir / 'scaler_v2_rl.pkl'
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        self.scaler = joblib.load(scaler_path)
        logger.info(f"  Loaded scaler: {scaler_path.name}")

        # Load metadata
        meta_path = self.saved_ml_dir / 'metadata_v2_rl.json'
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")

        with open(meta_path) as f:
            self.metadata = json.load(f)

        logger.info(f"  Loaded metadata: {meta_path.name}")

        # Create label encoders from metadata
        self.label_encoders = {}
        for cat_name, classes in self.metadata['label_encoder_classes'].items():
            encoder = LabelEncoder()
            encoder.fit(classes)
            self.label_encoders[cat_name] = encoder

        logger.info(f"  Biosensor types: {self.metadata['label_encoder_classes']['biosensor_type']}")
        logger.info(f"  Noise presets: {self.metadata['label_encoder_classes']['noise_preset']}")

    def encode_categorical_v2(
        self, biosensor_type: str, noise_preset: str
    ) -> Tuple[float, float]:
        """
        Encode categorical variables (v2 style: no scenario).

        Args:
            biosensor_type: 'amplifying' or 'direct_binding'
            noise_preset: 'low', 'medium', or 'high'

        Returns:
            (biosensor_encoded, noise_encoded)
        """
        biosensor_enc = float(self.label_encoders['biosensor_type'].transform([biosensor_type])[0])
        noise_enc = float(self.label_encoders['noise_preset'].transform([noise_preset])[0])

        return biosensor_enc, noise_enc

    def predict_metrics_v2(self, X_scaled: np.ndarray) -> Tuple[float, float, float]:
        """
        Predict DR, FNR, TTD using v2_rl surrogates.

        Args:
            X_scaled: Scaled feature vector [SNR, biosensor_encoded, noise_encoded]

        Returns:
            (detection_rate, false_negative_rate, time_to_detection)
        """
        dr_pred = float(self.models['detection_rate'].predict(X_scaled)[0])
        fnr_pred = float(self.models['fnr'].predict(X_scaled)[0])
        ttd_pred = float(self.models['ttd'].predict(X_scaled)[0])

        # Clip to valid ranges
        dr_pred = float(np.clip(dr_pred, 0.0, 1.0))
        fnr_pred = float(np.clip(fnr_pred, 0.0, 1.0))
        ttd_pred = float(np.clip(ttd_pred, 400.0, 9000.0))

        return dr_pred, fnr_pred, ttd_pred

    def get_training_bounds(self) -> Dict:
        """Get training data bounds for OOD detection."""
        return self.metadata.get('training_bounds', {})

    def get_metadata(self) -> Dict:
        """Get model metadata."""
        return self.metadata


# Convenience function for compatibility
def load_surrogates_v2_rl(results_dir: str = "BO/bo_results") -> SurrogateLoaderV2RL:
    """Load v2_rl surrogates."""
    return SurrogateLoaderV2RL(Path(results_dir))
