#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Surrogate loader for v3 models — 15-feature set (physics-informed, no SNR leakage).

Features: [log_kd, log_sensitivity, log_response_time,
           biosensor_type_enc, noise_preset_enc, scenario_enc,
           log_kd_ctx, log_kd_p1np, w_ctx, w_p1np,
           delta_theta_sost, delta_theta_ctx, delta_theta_p1np,
           composite_signal_proxy, log_composite_signal_proxy]

Features 6-9,11-12 are zero for single-channel sensors.
Features 10-14 are physics-informed Langmuir occupancy features: zero data leakage
since all inputs are design parameters + ODE-calibrated scenario concentrations.

SNR is intentionally excluded (creates shortcut leakage).
"""

import json
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from sklearn.preprocessing import LabelEncoder
import logging
import sys

# These imports register DerivedFNRModel / HurdleTTDModel in sys.modules so
# joblib can deserialise the v3.2 hurdle surrogate pkl files correctly.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from BO.core.surrogate_models import DerivedFNRModel, HurdleTTDModel  # noqa: F401

logger = logging.getLogger(__name__)

# Nominal sensor-compartment concentrations from ODE calibration (v5/V6).
# Must stay in sync with build_surrogates.py _SCENARIO_CONC.
_SCENARIO_CONC = {
    "healthy":  {"sost": 0.375,  "ctx": 0.200, "p1np": 0.350},
    "pmo_mild": {"sost": 0.5625, "ctx": 0.300, "p1np": 0.385},
    "pmo":      {"sost": 0.875,  "ctx": 0.500, "p1np": 0.525},
    "ckd_mbd":  {"sost": 1.125,  "ctx": 0.500, "p1np": 0.625},
}
_HEALTHY_CONC = _SCENARIO_CONC["healthy"]


class SurrogateLoaderV3:
    """Load and use v3 surrogates (15-feature, physics-informed, no SNR)."""

    FEATURE_NAMES = [
        "log_kd", "log_sensitivity", "log_response_time",
        "biosensor_type_enc", "noise_preset_enc", "scenario_enc",
        # Array-specific features (zero for single-channel sensors)
        "log_kd_ctx", "log_kd_p1np", "w_ctx", "w_p1np",
        # Physics-informed Langmuir occupancy features
        "delta_theta_sost", "delta_theta_ctx", "delta_theta_p1np",
        "composite_signal_proxy", "log_composite_signal_proxy",
    ]

    def __init__(self, results_dir: Path = None, version: str = None):
        if results_dir is None:
            results_dir = Path(__file__).parent.parent / "bo_results"
        self.results_dir = Path(results_dir)
        self.saved_ml_dir = self.results_dir / "saved_ml"
        self._load_models()

    def _load_models(self):
        logger.info(f"Loading surrogates from {self.saved_ml_dir}...")

        self.models = {}
        for metric in ("detection_rate", "fnr", "ttd"):
            path = self.saved_ml_dir / f"surrogate_{metric}.pkl"
            if not path.exists():
                raise FileNotFoundError(f"Surrogate model not found: {path}\nRun: python BO/core/build_surrogates.py --data-dir data_v19")
            self.models[metric] = joblib.load(path)
            logger.info(f"  Loaded {metric}: {path.name}")

        scaler_path = self.saved_ml_dir / "scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        self.scaler = joblib.load(scaler_path)

        meta_path = self.saved_ml_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")
        with open(meta_path) as f:
            self.metadata = json.load(f)

        self.label_encoders: Dict[str, LabelEncoder] = {}
        for col, classes in self.metadata["label_encoder_classes"].items():
            enc = LabelEncoder()
            enc.fit(classes)
            self.label_encoders[col] = enc

        self.ttd_log_transform = self.metadata.get("log_transform_ttd", True)

        # Validate the stored feature count matches what we expect.
        # Emit a warning (not an error) so the loader can still serve old 6-feature
        # models while data_v8 training is in progress.
        stored_n = self.metadata.get("n_features", len(self.FEATURE_NAMES))
        if stored_n != len(self.FEATURE_NAMES):
            logger.warning(
                f"Stored surrogate has {stored_n} features but loader expects "
                f"{len(self.FEATURE_NAMES)} (multi-biomarker). "
                "Re-train surrogates on data_v8 for full array support. "
                "Padding missing features with zeros for compatibility."
            )
        self._stored_n_features = stored_n
        logger.info(f"  Surrogates loaded. Features: {self.FEATURE_NAMES}")

    def build_feature_vector(
        self,
        kd_nm: float,
        sensitivity: float,
        response_time: float,
        biosensor_type: str,
        noise_preset: str,
        scenario: str,
        is_direct_binding: bool = False,
        kd_ctx: float = 0.0,
        kd_p1np: float = 0.0,
        w_ctx: float = 0.0,
        w_p1np: float = 0.0,
    ) -> np.ndarray:
        """Build and scale a single 15-feature row.

        Array-specific parameters (kd_ctx, kd_p1np, w_ctx, w_p1np) default to
        0.0, which is the correct single-channel encoding (no secondary analytes).
        Physics features 10-14 are computed analytically from Langmuir kinetics
        using scenario-fixed ODE concentrations — no simulator call required.
        """
        log_kd   = np.log10(max(kd_nm,  1e-3))
        log_sens = np.log10(max(sensitivity, 1e-3))
        log_rt   = 0.0 if is_direct_binding else np.log10(max(response_time, 1.0))

        bt_enc = float(self.label_encoders["biosensor_type"].transform([biosensor_type])[0])
        np_enc = float(self.label_encoders["noise_preset"].transform([noise_preset])[0])
        sc_enc = float(self.label_encoders["scenario"].transform([scenario])[0])

        log_kd_ctx  = np.log10(max(kd_ctx,  1e-3)) if kd_ctx  > 0 else 0.0
        log_kd_p1np = np.log10(max(kd_p1np, 1e-3)) if kd_p1np > 0 else 0.0

        # Physics-informed Langmuir occupancy features
        conc = _SCENARIO_CONC.get(scenario, _HEALTHY_CONC)
        h    = _HEALTHY_CONC

        def _dtheta(c_d: float, c_h: float, kd: float) -> float:
            if kd <= 0:
                return 0.0
            return c_d / (kd + c_d) - c_h / (kd + c_h)

        dt_sost = _dtheta(conc["sost"], h["sost"], kd_nm)
        dt_ctx  = _dtheta(conc["ctx"],  h["ctx"],  kd_ctx)  if kd_ctx  > 0 else 0.0
        dt_p1np = _dtheta(conc["p1np"], h["p1np"], kd_p1np) if kd_p1np > 0 else 0.0

        w_scl = max(0.0, 1.0 - w_ctx - w_p1np)
        composite = sensitivity * (w_scl * dt_sost + w_ctx * dt_ctx + w_p1np * dt_p1np)
        log_composite = float(np.log1p(max(0.0, composite)))

        row = [log_kd, log_sens, log_rt, bt_enc, np_enc, sc_enc,
               log_kd_ctx, log_kd_p1np, w_ctx, w_p1np,
               dt_sost, dt_ctx, dt_p1np, composite, log_composite]

        # Backwards-compatibility: truncate to stored feature count for old models.
        stored_n = getattr(self, "_stored_n_features", len(self.FEATURE_NAMES))
        x = np.array([row[:stored_n]], dtype=np.float32)
        return self.scaler.transform(x)

    def predict(
        self,
        kd_nm: float,
        sensitivity: float,
        response_time: float,
        biosensor_type: str,
        noise_preset: str,
        scenario: str,
        kd_ctx: float = 0.0,
        kd_p1np: float = 0.0,
        w_ctx: float = 0.0,
        w_p1np: float = 0.0,
    ) -> Tuple[float, float, float]:
        """
        Predict (detection_rate, false_negative_rate, time_to_detection).

        No simulator call required — pure surrogate inference.
        DR is returned as a calibrated probability [0, 1].
        TTD is inverted from log space if log_transform was used during training.

        Array biosensor: pass kd_ctx, kd_p1np, w_ctx, w_p1np.
        Single-channel: leave defaults (0.0) — model treats as single-channel regime.
        """
        is_direct = biosensor_type in ("direct_binding", "array")
        X_scaled = self.build_feature_vector(
            kd_nm, sensitivity, response_time,
            biosensor_type, noise_preset, scenario,
            is_direct_binding=is_direct,
            kd_ctx=kd_ctx, kd_p1np=kd_p1np,
            w_ctx=w_ctx, w_p1np=w_p1np,
        )

        dr_pred = float(self.models["detection_rate"].predict_proba(X_scaled)[0, 1])
        fnr_pred = float(self.models["fnr"].predict(X_scaled)[0])

        ttd_raw = float(self.models["ttd"].predict(X_scaled)[0])
        if self.ttd_log_transform:
            ttd_pred = float(np.expm1(max(ttd_raw, 0.0)))
        else:
            ttd_pred = ttd_raw

        dr_pred = float(np.clip(dr_pred, 0.0, 1.0))
        fnr_pred = float(np.clip(fnr_pred, 0.0, 1.0))
        ttd_pred = float(np.clip(ttd_pred, 400.0, 9000.0))

        return dr_pred, fnr_pred, ttd_pred

    def get_training_bounds(self) -> Dict:
        return self.metadata.get("training_bounds", {})
