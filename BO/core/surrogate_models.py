#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Serialisable surrogate wrapper classes for v3.2 hurdle architecture.

These classes must live in a stable, importable module (not __main__) so
that joblib can deserialise them when loading the saved .pkl files.

Imported by both build_surrogates.py and surrogate_loader.py.
"""

import numpy as np


class DerivedFNRModel:
    """FNR surrogate derived from the DR classifier.

    FNR = 1 − P_detect is mathematically exact at the row level since both
    metrics count the same 50 stochastic trials.  Using the calibrated DR
    classifier avoids fitting a separate GBM on the U-shaped FNR target
    (75.8% of values are exactly 0 or 1 in data_v19), which is a fundamentally
    ill-posed regression problem that caps R² at ~0.56.

    The `.predict()` interface is identical to a standard sklearn regressor
    so the surrogate loader requires no changes.
    """

    def __init__(self, dr_classifier):
        self.dr_classifier = dr_classifier

    def predict(self, X_scaled: np.ndarray) -> np.ndarray:
        p_detect = self.dr_classifier.predict_proba(X_scaled)[:, 1]
        return 1.0 - p_detect


class HurdleTTDModel:
    """Two-stage hurdle model for time-to-detection.

    Stage 1: P_detect from the DR classifier (already trained).
    Stage 2: Conditional regressor trained ONLY on detected rows (TTD < sentinel).
             Conditional TTD is log1p-transformed during training for stability.

    Inference:
        TTD_pred = P_detect × TTD_cond(X) + (1 − P_detect) × TTD_MAX

    This resolves the bimodal distribution problem: 53.9% of TTD values in
    data_v19 are at the non-detection sentinel (~9000s).  The conditional
    regressor only sees the unimodal detected distribution [170, ~5000s].

    The `.predict()` returns TTD in seconds (not log-transformed).
    Set metadata["log_transform_ttd"] = False so the loader does not apply
    an additional expm1 transformation.
    """

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
