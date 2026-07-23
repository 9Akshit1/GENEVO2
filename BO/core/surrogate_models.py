#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Surrogate wrapper classes for the v3.2 hurdle architecture.
Must be in a stable module (not __main__) so joblib can deserialise .pkl files.
"""

import numpy as np


class DerivedFNRModel:
    """FNR = 1 - P_detect, derived from the calibrated DR classifier."""

    def __init__(self, dr_classifier):
        self.dr_classifier = dr_classifier

    def predict(self, X_scaled: np.ndarray) -> np.ndarray:
        p_detect = self.dr_classifier.predict_proba(X_scaled)[:, 1]
        return 1.0 - p_detect


class HurdleTTDModel:
    """Two-stage TTD model: P_detect × TTD_conditional + (1-P_detect) × TTD_MAX."""

    def __init__(self, dr_classifier, conditional_regressor, ttd_max: float = 9000.0):
        self.dr_classifier = dr_classifier
        self.conditional_regressor = conditional_regressor
        self.ttd_max = ttd_max

    def predict(self, X_scaled: np.ndarray) -> np.ndarray:
        p_detect = self.dr_classifier.predict_proba(X_scaled)[:, 1]
        ttd_cond_log = self.conditional_regressor.predict(X_scaled)
        ttd_cond = np.expm1(np.clip(ttd_cond_log, 0.0, None))
        ttd_cond = np.clip(ttd_cond, 0.0, self.ttd_max)
        return p_detect * ttd_cond + (1.0 - p_detect) * self.ttd_max
