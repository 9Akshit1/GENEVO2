#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Load and manage surrogate models (v1 and v2).
Loads surrogates, scalers, and label encoders built by build_surrogates.py.

CRITICAL: Ensures all preprocessing state is properly loaded and used consistently.

Supports:
- v1: GradientBoostingRegressor for all metrics
- v2: CalibratedClassifier for DR, QuantileRegression for FNR/TTD
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import logging

logger = logging.getLogger(__name__)


class SurrogateLoader:
    """Load surrogate models with proper initialization of scaler and encoders."""

    def __init__(self, surrogate_dir: str = "BO/bo_results"):
        """
        Initialize loader.

        Args:
            surrogate_dir: Path to directory with saved surrogate models (default: BO/bo_results)
        """
        self.surrogate_dir = Path(surrogate_dir)
        self.surrogates = {}
        self.scaler = None
        self.label_encoders = {}
        self.metadata = {}
        self.model_version = None
        self._initialize()

    def _initialize(self, version: str = "v1"):
        """
        Initialize by loading surrogates, scaler, and label encoders.

        Args:
            version: Model version string (default: 'v1')

        Raises:
            FileNotFoundError: If required files are missing
        """
        saved_ml_dir = self.surrogate_dir / "saved_ml"

        if not saved_ml_dir.exists():
            raise FileNotFoundError(
                f"Surrogate models directory not found at {saved_ml_dir}. "
                f"Run surrogate training first (e.g., python BO/bo_main.py)"
            )

        # Load metadata first
        metadata_path = saved_ml_dir / f"metadata_{version}.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            logger.debug(f"Loaded metadata from {metadata_path}")
            self.model_version = self.metadata.get('version', version)
        else:
            logger.warning(f"Metadata not found at {metadata_path}")
            self.model_version = version

        # Load scaler - CRITICAL
        scaler_path = saved_ml_dir / f"scaler_{version}.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(
                f"Scaler not found at {scaler_path}. "
                f"Surrogates are unusable without preprocessing state."
            )
        self.scaler = joblib.load(scaler_path)
        logger.info(f"Loaded scaler from {scaler_path}")

        # Load label encoders - CRITICAL
        encoders_path = saved_ml_dir / f"label_encoders_{version}.pkl"
        if not encoders_path.exists():
            raise FileNotFoundError(
                f"Label encoders not found at {encoders_path}. "
                f"Cannot encode categorical variables."
            )
        self.label_encoders = joblib.load(encoders_path)
        logger.info(f"Loaded label encoders from {encoders_path}")

        # Load surrogate models
        # Detection Rate
        dr_path = saved_ml_dir / f"surrogate_detection_rate_{version}.pkl"
        if dr_path.exists():
            self.surrogates['detection_rate'] = joblib.load(dr_path)
            logger.info(f"Loaded detection_rate surrogate from {dr_path}")
        else:
            raise FileNotFoundError(f"Detection rate surrogate not found: {dr_path}")

        # FNR: Check if v2 (quantile) or v1 (single model)
        fnr_median_path = saved_ml_dir / f"surrogate_fnr_median_{version}.pkl"
        fnr_path = saved_ml_dir / f"surrogate_fnr_{version}.pkl"

        if fnr_median_path.exists():
            # v2: Quantile models
            self.surrogates['fnr'] = {
                'median': joblib.load(saved_ml_dir / f"surrogate_fnr_median_{version}.pkl"),
                'lower': joblib.load(saved_ml_dir / f"surrogate_fnr_lower_{version}.pkl"),
                'upper': joblib.load(saved_ml_dir / f"surrogate_fnr_upper_{version}.pkl"),
            }
            logger.info(f"Loaded FNR quantile regressors")
        elif fnr_path.exists():
            # v1: Single model
            self.surrogates['fnr'] = joblib.load(fnr_path)
            logger.info(f"Loaded FNR regressor from {fnr_path}")
        else:
            raise FileNotFoundError(f"FNR surrogate not found")

        # TTD: Check if v2 (quantile) or v1 (single model)
        ttd_median_path = saved_ml_dir / f"surrogate_ttd_median_{version}.pkl"
        ttd_path = saved_ml_dir / f"surrogate_ttd_{version}.pkl"

        if ttd_median_path.exists():
            # v2: Quantile models
            self.surrogates['ttd'] = {
                'median': joblib.load(saved_ml_dir / f"surrogate_ttd_median_{version}.pkl"),
                'lower': joblib.load(saved_ml_dir / f"surrogate_ttd_lower_{version}.pkl"),
                'upper': joblib.load(saved_ml_dir / f"surrogate_ttd_upper_{version}.pkl"),
            }
            logger.info(f"Loaded TTD quantile regressors")
        elif ttd_path.exists():
            # v1: Single model
            self.surrogates['ttd'] = joblib.load(ttd_path)
            logger.info(f"Loaded TTD regressor from {ttd_path}")
        else:
            raise FileNotFoundError(f"TTD surrogate not found")

        logger.info(f"✓ All surrogates and preprocessing state loaded successfully (version={self.model_version})")

    def encode_categorical(self, biosensor_type: str, noise_preset: str, scenario: str = "pmo") -> tuple:
        """
        Encode categorical variables using fitted label encoders.

        Args:
            biosensor_type: One of {direct_binding, amplifying}
            noise_preset: One of {low, medium, high}
            scenario: One of {healthy, pmo, ckd_mbd}

        Returns:
            Tuple of (biosensor_enc, noise_enc, scenario_enc) as floats

        Raises:
            RuntimeError: If encoders not initialized
            ValueError: If categorical values are unknown
        """
        if not self.label_encoders:
            raise RuntimeError(
                "Encoders not initialized. Surrogates were not loaded properly. "
                "Check that BO/bo_results/saved_ml/ contains all required files."
            )

        try:
            biosensor_enc = float(self.label_encoders["biosensor_type"].transform([biosensor_type])[0])
        except (KeyError, ValueError) as e:
            known = list(self.label_encoders["biosensor_type"].classes_)
            raise ValueError(
                f"Unknown biosensor_type: {biosensor_type}. "
                f"Known types: {known}. Error: {e}"
            )

        try:
            noise_enc = float(self.label_encoders["noise_preset"].transform([noise_preset])[0])
        except (KeyError, ValueError) as e:
            known = list(self.label_encoders["noise_preset"].classes_)
            raise ValueError(
                f"Unknown noise_preset: {noise_preset}. "
                f"Known presets: {known}. Error: {e}"
            )

        try:
            scenario_enc = float(self.label_encoders["scenario"].transform([scenario])[0])
        except (KeyError, ValueError) as e:
            known = list(self.label_encoders["scenario"].classes_)
            raise ValueError(
                f"Unknown scenario: {scenario}. "
                f"Known scenarios: {known}. Error: {e}"
            )

        return biosensor_enc, noise_enc, scenario_enc

    def is_initialized(self) -> bool:
        """Check if surrogates and scaler are properly initialized."""
        return bool(
            self.surrogates
            and len(self.surrogates) == 3
            and self.scaler is not None
            and len(self.label_encoders) == 3
        )

    def predict_metrics(self, X_scaled: np.ndarray) -> tuple:
        """
        Predict all three metrics for a scaled input.

        Handles both v1 (single models) and v2 (calibrated/quantile) surrogates.

        Args:
            X_scaled: Scaled feature array (already processed by self.scaler)

        Returns:
            For v1: Tuple of (dr_pred, fnr_pred, ttd_pred)
            For v2: Tuple of (dr_prob, fnr_median, fnr_lower, fnr_upper, ttd_median, ttd_lower, ttd_upper)
        """
        if not self.is_initialized():
            raise RuntimeError("Surrogates not initialized")

        # Detection Rate
        dr_model = self.surrogates['detection_rate']
        if hasattr(dr_model, 'predict_proba'):
            # v2: Calibrated classifier
            dr_pred = float(dr_model.predict_proba(X_scaled)[0, 1])
        else:
            # v1: Regressor
            dr_pred = float(np.clip(dr_model.predict(X_scaled)[0], 0, 1))

        # FNR
        fnr_model = self.surrogates['fnr']
        if isinstance(fnr_model, dict):
            # v2: Quantile regression
            fnr_median = float(np.clip(fnr_model['median'].predict(X_scaled)[0], 0, 1))
            fnr_lower = float(np.clip(fnr_model['lower'].predict(X_scaled)[0], 0, 1))
            fnr_upper = float(np.clip(fnr_model['upper'].predict(X_scaled)[0], 0, 1))
        else:
            # v1: Single regressor
            fnr_pred = float(np.clip(fnr_model.predict(X_scaled)[0], 0, 1))
            fnr_median = fnr_lower = fnr_upper = fnr_pred

        # TTD
        ttd_model = self.surrogates['ttd']
        if isinstance(ttd_model, dict):
            # v2: Quantile regression
            ttd_median = float(np.clip(ttd_model['median'].predict(X_scaled)[0], 400, 9000))
            ttd_lower = float(np.clip(ttd_model['lower'].predict(X_scaled)[0], 400, 9000))
            ttd_upper = float(np.clip(ttd_model['upper'].predict(X_scaled)[0], 400, 9000))
        else:
            # v1: Single regressor
            ttd_pred = float(np.clip(ttd_model.predict(X_scaled)[0], 400, 9000))
            ttd_median = ttd_lower = ttd_upper = ttd_pred

        return (dr_pred, fnr_median, fnr_lower, fnr_upper, ttd_median, ttd_lower, ttd_upper)
