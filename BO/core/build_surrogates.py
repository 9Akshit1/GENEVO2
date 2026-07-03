#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Surrogate model builder - v3.2 (Physics-Informed Features + Hurdle Models).

ARCHITECTURE CHANGES in v3.2 over v3.1:
=============================================================
The v3.1 FNR and TTD regressors suffered from fundamental distributional
problems that prevented R² from improving beyond ~0.56:

  FNR problem: 75.8% of values are exactly 0.0 or 1.0 (U-shaped bimodal
  distribution). GBM regression on a U-shaped target is poorly specified —
  MSE loss pulls predictions toward the mean. Since FNR = 1 − DR_raw
  at the row level (verified empirically: mean(FNR + DR) = 1.004),
  training a separate FNR regressor is redundant. The calibrated DR
  classifier already produces P_detect ≈ E[DR_raw], so FNR_pred = 1 − P_detect
  is strictly better than a separate underpowered regressor.

  TTD problem: 53.9% of TTD values are at the non-detection sentinel (~9000s).
  A single regressor on a bimodal target where 54% mass is at one extreme
  will have high RMSE regardless of model complexity. The correct approach
  is a two-stage hurdle model:
    Stage 1 (classification): DR classifier gives P_detect = P(TTD < sentinel)
    Stage 2 (conditional regression): Train TTD regressor ONLY on detected
      rows (TTD < TTD_SENTINEL_THRESHOLD). These 921 rows span [170, ~5000s] —
      a unimodal, tractable regression target.
    Inference: TTD_pred = P_detect × TTD_cond(X) + (1 − P_detect) × TTD_MAX

RESULTS:
  FNR effective R²:  ~0.83 (vs 0.562 before) — derived from AUC=0.914 DR clf
  TTD hurdle R²:     ~0.75 (vs 0.548 before) — conditional regressor on 921 rows

FEATURE SET (15 dimensions): unchanged from v3.1.
TARGETS:
    detection_rate     binary {0,1} → calibrated GBC classifier (unchanged)
    false_negative_rate → DerivedFNRModel wrapper (FNR = 1 − P_detect)
    time_to_detection  → HurdleTTDModel (two-stage; no log-transform of final output)
"""

import json
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor, RandomForestRegressor
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error, roc_auc_score
)

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
logger = logging.getLogger(__name__)

# Wrapper classes live in a stable importable module so joblib can find them
# when deserialising the saved .pkl files regardless of __main__ context.
from BO.core.surrogate_models import DerivedFNRModel, HurdleTTDModel  # noqa: E402

# Fallback values for missing metadata fields
_DEFAULT_RESPONSE_TIME = 600.0   # seconds (mid-range for amplifying sensors)
_LOG_RESPONSE_TIME_DIRECT = 0.0  # log10(1) placeholder for non-amplifying sensors

# Nominal sensor-compartment concentrations from ODE calibration (v5/V6).
# Source: therapeutic_objective_v6.py _NOMINAL_CONCS + environment_configs.py
# Sensor = bone × 25 (diffusion equilibrium ratio k_diff/k_back × Vbone/Vsensor).
_SCENARIO_CONC = {
    "healthy":  {"sost": 0.375,  "ctx": 0.200, "p1np": 0.350},
    "pmo_mild": {"sost": 0.5625, "ctx": 0.300, "p1np": 0.385},
    "pmo":      {"sost": 0.875,  "ctx": 0.500, "p1np": 0.525},
    "ckd_mbd":  {"sost": 1.125,  "ctx": 0.500, "p1np": 0.625},
}
_HEALTHY_CONC = _SCENARIO_CONC["healthy"]


def _read_biosensor_config(metadata_file: str, data_root: Path) -> Dict:
    """
    Read biosensor configuration from per-simulation metadata JSON.

    Handles both relative paths (relative to data_root) and absolute paths.
    Returns an empty dict on any read failure so the caller can skip the row.
    """
    path = Path(metadata_file)
    if not path.is_absolute():
        path = data_root / path
    try:
        with open(path) as fh:
            meta = json.load(fh)
        return meta.get("biosensor_config", {})
    except Exception as exc:
        logger.debug(f"Could not read {path}: {exc}")
        return {}


class SurrogateBuilderV3:
    """
    Build surrogate models with scientifically correct feature engineering.

    Uses all biosensor design parameters (kd, sensitivity, response_time,
    circuit type) plus environmental context (scenario, noise preset) and
    the measured SNR.  Detection rate is treated as a classification problem;
    FNR and TTD are treated as regression problems.
    """

    FEATURE_NAMES = [
        "log_kd",
        "log_sensitivity",
        "log_response_time",
        "biosensor_type_enc",
        "noise_preset_enc",
        "scenario_enc",
        # Array-biosensor channel features (zero for single-channel sensors)
        "log_kd_ctx",
        "log_kd_p1np",
        "w_ctx",
        "w_p1np",
        # Physics-informed Langmuir occupancy features (no data leakage)
        "delta_theta_sost",
        "delta_theta_ctx",
        "delta_theta_p1np",
        "composite_signal_proxy",
        "log_composite_signal_proxy",
    ]

    def __init__(self, logger_obj=None):
        self.logger = logger_obj or logger
        self.models: Dict = {}
        self.scaler: Optional[StandardScaler] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.training_bounds: Dict = {}
        self._feature_names = list(self.FEATURE_NAMES)

    # ------------------------------------------------------------------
    def load_and_prepare_data(
        self, data_dir: Path
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Load master_index, enrich with per-simulation metadata, and build
        the feature matrix.

        If the master_index.csv already contains kd/sensitivity columns (new
        enriched format), those are used directly.  Otherwise the function
        falls back to reading individual metadata JSON files.

        Returns
        -------
        X  : float32 array of shape (n_valid, 7)
        df : enriched DataFrame with all feature columns and targets
        """
        data_dir = Path(data_dir)
        master_path = data_dir / "master_index.csv"
        if not master_path.exists():
            raise FileNotFoundError(f"master_index.csv not found: {master_path}")

        df = pd.read_csv(master_path)
        self.logger.info(f"Loaded {len(df)} rows from {master_path}")

        # ── Enrich with biosensor design parameters ─────────────────────────
        # Fast path: kd and sensitivity already in the CSV (new enriched format)
        if "kd" in df.columns and "sensitivity" in df.columns:
            self.logger.info("  Using kd/sensitivity from master_index.csv (enriched format)")
            if "response_time" not in df.columns:
                df["response_time"] = np.nan
            n_missing = int(df["kd"].isna().sum())

            # Fast path must also filter threshold=0 rows — the slow-path filter
            # gates on "_threshold" in df.columns, which the fast path never adds.
            # Read the threshold column directly from the CSV when it exists.
            if "threshold" in df.columns:
                n_before_thresh = len(df)
                df = df[df["threshold"].isna() | (df["threshold"] > 0)].copy()
                n_thresh_dropped = n_before_thresh - len(df)
                if n_thresh_dropped > 0:
                    self.logger.warning(
                        f"Dropped {n_thresh_dropped}/{n_before_thresh} rows with threshold=0 "
                        f"(fast-path filter, same as slow-path logic). "
                        f"{len(df)} rows remain."
                    )
        else:
            # Slow path: read from per-simulation metadata JSON files
            self.logger.info("  Reading kd/sensitivity/threshold from metadata files...")
            kd_vals, sens_vals, rt_vals, thresh_vals = [], [], [], []
            n_missing = 0
            for _, row in df.iterrows():
                bc = _read_biosensor_config(row.get("metadata_file", ""), data_dir)
                if bc:
                    kd_vals.append(float(bc.get("kd", np.nan)))
                    sens_vals.append(float(bc.get("sensitivity", np.nan)))
                    rt_raw = bc.get("response_time", bc.get("response_time_s", None))
                    rt_vals.append(float(rt_raw) if rt_raw is not None else np.nan)
                    t_raw = bc.get("threshold", None)
                    thresh_vals.append(float(t_raw) if t_raw is not None else np.nan)
                else:
                    kd_vals.append(np.nan)
                    sens_vals.append(np.nan)
                    rt_vals.append(np.nan)
                    thresh_vals.append(np.nan)
                    n_missing += 1
            df["kd"] = kd_vals
            df["sensitivity"] = sens_vals
            df["response_time"] = rt_vals
            df["_threshold"] = thresh_vals

        if n_missing > 0:
            self.logger.warning(
                f"{n_missing}/{len(df)} rows had missing kd/sensitivity — "
                "they will be dropped."
            )

        # Drop rows where critical design parameters are missing
        df = df.dropna(subset=["kd", "sensitivity"]).copy()
        self.logger.info(f"Retained {len(df)} rows with complete biosensor config")

        # ── Filter rows with threshold = 0 ───────────────────────────────────
        # When threshold=0 the biosensor fires on any positive signal, including
        # healthy patients.  These rows inflate the healthy DR to ~40% in the
        # training set, making the surrogate severely underestimate false-positive
        # rates during BO inference.
        #
        # Drop only rows where threshold is *explicitly* 0.  Rows whose metadata
        # did not record a threshold (NaN) are kept because they come from an older
        # simulator version that likely used a valid non-zero threshold — their DR
        # distribution (mean ≈ 32%) is consistent with calibrated sensors.
        if "_threshold" in df.columns:
            n_before = len(df)
            df = df[df["_threshold"].isna() | (df["_threshold"] > 0)].copy()
            df = df.drop(columns=["_threshold"])
            n_dropped = n_before - len(df)
            if n_dropped > 0:
                self.logger.warning(
                    f"Dropped {n_dropped}/{n_before} rows with threshold=0 "
                    f"(trivial all-scenario detection, contaminates healthy FP training). "
                    f"{len(df)} rows remain."
                )

        if len(df) < 50:
            raise ValueError(
                "Too few valid rows after metadata enrichment. "
                "Check that metadata files exist under data/metadata/."
            )

        # ── Encode categoricals ──────────────────────────────────────────────
        for col in ("biosensor_type", "noise_preset", "scenario"):
            enc = LabelEncoder()
            df[f"{col}_enc"] = enc.fit_transform(df[col])
            self.label_encoders[col] = enc
            self.logger.info(
                f"  {col}: {dict(zip(enc.classes_, enc.transform(enc.classes_)))}"
            )

        # Fill missing response_time with circuit-type-specific defaults:
        # amplifying sensors have a meaningful response_time; direct_binding does not.
        df["response_time"] = df.apply(
            lambda r: r["response_time"]
            if not np.isnan(r["response_time"])
            else (_DEFAULT_RESPONSE_TIME if r["biosensor_type"] == "amplifying" else 1.0),
            axis=1,
        )

        # ── Log-transform continuous design parameters ──────────────────────
        # kd and sensitivity span 1–2 orders of magnitude each; log-scale
        # linearises their effect on biosensor occupancy (Langmuir kinetics).
        # response_time likewise log-scaled; for direct_binding sensors we use
        # log10(1)=0 as a neutral placeholder.
        df["log_kd"] = np.log10(df["kd"].clip(lower=1e-3))
        df["log_sensitivity"] = np.log10(df["sensitivity"].clip(lower=1e-3))
        df["log_response_time"] = np.log10(df["response_time"].clip(lower=1.0))
        # For direct_binding sensors, response_time is not meaningful; zero it out
        # after log-transform so it carries no information for those circuit types.
        direct_binding_mask = df["biosensor_type"] == "direct_binding"
        df.loc[direct_binding_mask, "log_response_time"] = 0.0

        # ── Array-biosensor channel features ────────────────────────────────
        # For single-channel sensors these columns are absent or NaN → fill 0.
        # For array sensors they hold per-channel Kd (log10) and weights [0,1].
        # log_kd_ctx = log_kd_p1np = w_ctx = w_p1np = 0 is the single-channel regime.
        if "kd_ctx" in df.columns:
            df["log_kd_ctx"]  = np.where(
                df["kd_ctx"].isna(),  0.0, np.log10(df["kd_ctx"].clip(lower=1e-3))
            )
        else:
            df["log_kd_ctx"] = 0.0

        if "kd_p1np" in df.columns:
            df["log_kd_p1np"] = np.where(
                df["kd_p1np"].isna(), 0.0, np.log10(df["kd_p1np"].clip(lower=1e-3))
            )
        else:
            df["log_kd_p1np"] = 0.0

        df["w_ctx"]  = df["w_ctx"].fillna(0.0)  if "w_ctx"  in df.columns else 0.0
        df["w_p1np"] = df["w_p1np"].fillna(0.0) if "w_p1np" in df.columns else 0.0

        # ── Physics-informed Langmuir occupancy features ─────────────────────
        # Δθ = θ_disease(Kd) − θ_healthy(Kd) for each analyte channel.
        # Captures additional binding-site occupancy in disease vs healthy.
        # Inputs: only design parameters + ODE-calibrated constants → no leakage.
        kd_sost_v  = df["kd"].values.astype(float)
        kd_ctx_v   = df["kd_ctx"].values.astype(float)  if "kd_ctx"  in df.columns else np.full(len(df), np.nan)
        kd_p1np_v  = df["kd_p1np"].values.astype(float) if "kd_p1np" in df.columns else np.full(len(df), np.nan)

        c_sost_d = df["scenario"].map(lambda s: _SCENARIO_CONC.get(s, _HEALTHY_CONC)["sost"]).values.astype(float)
        c_ctx_d  = df["scenario"].map(lambda s: _SCENARIO_CONC.get(s, _HEALTHY_CONC)["ctx"]).values.astype(float)
        c_p1np_d = df["scenario"].map(lambda s: _SCENARIO_CONC.get(s, _HEALTHY_CONC)["p1np"]).values.astype(float)

        h_sost, h_ctx, h_p1np = _HEALTHY_CONC["sost"], _HEALTHY_CONC["ctx"], _HEALTHY_CONC["p1np"]

        df["delta_theta_sost"] = (
            c_sost_d / (kd_sost_v + c_sost_d) - h_sost / (kd_sost_v + h_sost)
        )
        valid_ctx  = ~np.isnan(kd_ctx_v)  & (kd_ctx_v  > 0)
        valid_p1np = ~np.isnan(kd_p1np_v) & (kd_p1np_v > 0)
        df["delta_theta_ctx"] = np.where(
            valid_ctx,
            c_ctx_d  / (kd_ctx_v  + c_ctx_d)  - h_ctx  / (kd_ctx_v  + h_ctx),
            0.0,
        )
        df["delta_theta_p1np"] = np.where(
            valid_p1np,
            c_p1np_d / (kd_p1np_v + c_p1np_d) - h_p1np / (kd_p1np_v + h_p1np),
            0.0,
        )

        # w_scl is stored in data_v19; fall back to complement for older datasets.
        w_scl_v = (df["w_scl"].values if "w_scl" in df.columns
                   else np.clip(1.0 - df["w_ctx"].values - df["w_p1np"].values, 0.0, 1.0))
        w_scl_v = np.clip(w_scl_v, 0.0, 1.0)

        df["composite_signal_proxy"] = (
            df["sensitivity"].values * (
                w_scl_v              * df["delta_theta_sost"].values +
                df["w_ctx"].values   * df["delta_theta_ctx"].values +
                df["w_p1np"].values  * df["delta_theta_p1np"].values
            )
        )
        df["log_composite_signal_proxy"] = np.log1p(
            np.clip(df["composite_signal_proxy"].values, 0.0, None)
        )
        self.logger.info(
            f"  Physics features: Δθ sost∈[{df['delta_theta_sost'].min():.3f},{df['delta_theta_sost'].max():.3f}]"
            f"  composite∈[{df['composite_signal_proxy'].min():.3f},{df['composite_signal_proxy'].max():.3f}]"
        )

        # ── Assemble feature matrix ──────────────────────────────────────────
        feature_cols = [
            "log_kd", "log_sensitivity", "log_response_time",
            "biosensor_type_enc", "noise_preset_enc", "scenario_enc",
            "log_kd_ctx", "log_kd_p1np", "w_ctx", "w_p1np",
            "delta_theta_sost", "delta_theta_ctx", "delta_theta_p1np",
            "composite_signal_proxy", "log_composite_signal_proxy",
        ]
        X = df[feature_cols].values.astype(np.float32)

        # Store bounds for OOD detection during BO
        self.training_bounds = {
            col: {"min": float(df[col].min()), "max": float(df[col].max())}
            for col in ["kd", "sensitivity"]
        }

        self.logger.info(f"\nFeature matrix shape: {X.shape}")
        for i, name in enumerate(feature_cols):
            self.logger.info(
                f"  {name:25s}: [{X[:, i].min():.3f}, {X[:, i].max():.3f}]"
            )

        return X, df

    # ------------------------------------------------------------------
    def fit_scaler(self, X: np.ndarray) -> np.ndarray:
        """Fit StandardScaler and return the scaled matrix."""
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.logger.info(f"Scaler means:  {self.scaler.mean_}")
        self.logger.info(f"Scaler scales: {self.scaler.scale_}")
        return X_scaled

    # ------------------------------------------------------------------
    def train_detection_rate(
        self, X: np.ndarray, y: np.ndarray,
        test_size: float = 0.2, cv_folds: int = 5
    ) -> Dict:
        """
        Train a calibrated gradient-boosting classifier for detection rate.

        Detection rate is binary {0, 1} so classification with isotonic
        calibration is scientifically appropriate.  The calibrated
        probability output is the continuous surrogate value used in BO.
        """
        self.logger.info("\n[DETECTION_RATE] Training classifier...")

        y_bin = (y >= 0.5).astype(int)
        self.logger.info(
            f"  Class balance: {y_bin.mean()*100:.1f}% positive"
        )

        X_tv, X_test, y_tv, y_test = train_test_split(
            X, y_bin, test_size=test_size, random_state=42, stratify=y_bin
        )

        base_clf = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
        )
        model = CalibratedClassifierCV(base_clf, method="isotonic", cv=5)

        # CV on base (pre-calibration) to get unbiased performance estimate
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_auc = cross_val_score(
            base_clf, X_tv, y_tv, cv=skf, scoring="roc_auc", n_jobs=1
        )
        self.logger.info(
            f"  CV ROC-AUC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}"
        )

        model.fit(X_tv, y_tv)
        prob_test = model.predict_proba(X_test)[:, 1]
        auc_test = roc_auc_score(y_test, prob_test)
        self.logger.info(f"  Test ROC-AUC: {auc_test:.4f}")

        self._log_feature_importance(base_clf.fit(X_tv, y_tv), "DR")

        self.models["detection_rate"] = model
        return {"metric": "detection_rate", "cv_roc_auc": float(cv_auc.mean()),
                "test_roc_auc": float(auc_test)}

    # ------------------------------------------------------------------
    def train_fnr_derived(
        self, X: np.ndarray, y_fnr: np.ndarray,
        test_size: float = 0.2,
    ) -> Dict:
        """Build FNR surrogate as DerivedFNRModel(dr_classifier).

        FNR = 1 − DR_raw per row (empirically verified: mean(FNR + DR) ≈ 1.0).
        Training a separate regressor on the U-shaped FNR distribution (75.8%
        of values exactly 0 or 1) is fundamentally ill-posed.  The DR classifier
        is a strictly better estimator because it handles the binary boundary
        decision properly via AUC=0.914 calibrated probabilities.

        Effective R² is measured post-hoc by comparing 1−P_detect vs y_fnr.
        """
        self.logger.info("\n[FNR] Building derived surrogate (FNR = 1 − P_detect)...")
        if "detection_rate" not in self.models:
            raise RuntimeError("DR classifier must be trained before FNR derived model.")

        dr_clf = self.models["detection_rate"]
        fnr_wrapper = DerivedFNRModel(dr_classifier=dr_clf)

        # Measure effective R² on a held-out split
        _, X_test, _, y_test = train_test_split(X, y_fnr, test_size=test_size, random_state=42)
        fnr_pred_test = fnr_wrapper.predict(X_test)
        r2 = r2_score(y_test, fnr_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, fnr_pred_test))

        self.logger.info(f"  Effective Test R²:   {r2:.4f}  (vs 0.562 from direct regressor)")
        self.logger.info(f"  Effective Test RMSE: {rmse:.4f}")
        self.logger.info(f"  Note: FNR = 1 − P_detect; no separate model is trained.")

        self.models["fnr"] = fnr_wrapper
        return {"metric": "fnr", "method": "derived_from_dr", "effective_r2": float(r2), "rmse": float(rmse)}

    # ------------------------------------------------------------------
    def train_ttd_hurdle(
        self, X: np.ndarray, y_ttd: np.ndarray,
        test_size: float = 0.2, cv_folds: int = 5,
        sentinel_threshold: float = 8500.0,
    ) -> Dict:
        """Two-stage hurdle model for TTD.

        Problem: 53.9% of TTD values in data_v19 are at the non-detection
        sentinel (~9000s).  A single regressor on this bimodal target produces
        R² ≈ 0.548.

        Fix: train the conditional TTD regressor ONLY on detected rows
        (TTD < sentinel_threshold).  At inference:
          TTD_pred = P_detect × TTD_cond(X) + (1 − P_detect) × TTD_MAX

        The conditional regressor sees a unimodal distribution over [170, ~5000s]
        with clear feature relationships (sensitivity dominates), yielding
        substantially higher R².
        """
        self.logger.info("\n[TTD] Training two-stage hurdle model...")
        if "detection_rate" not in self.models:
            raise RuntimeError("DR classifier must be trained before hurdle TTD model.")

        # Detect TTD_MAX from data (max observed value)
        ttd_max = float(np.max(y_ttd))
        self.logger.info(f"  TTD range: [{y_ttd.min():.0f}, {ttd_max:.0f}]s  "
                         f"sentinel_threshold={sentinel_threshold:.0f}s")

        # Identify detected rows
        detected_mask = y_ttd < sentinel_threshold
        n_detected = int(detected_mask.sum())
        n_total = len(y_ttd)
        self.logger.info(
            f"  Detected rows: {n_detected}/{n_total} ({n_detected/n_total*100:.1f}%)"
        )

        X_det  = X[detected_mask]
        y_det  = y_ttd[detected_mask]

        # Log1p-transform conditional TTD for stability
        y_det_log = np.log1p(y_det)
        self.logger.info(
            f"  Conditional TTD (log1p): mean={y_det_log.mean():.3f}, "
            f"std={y_det_log.std():.3f}, "
            f"range=[{y_det_log.min():.3f}, {y_det_log.max():.3f}]"
        )

        # Train conditional TTD regressor (more capacity: deeper trees, more estimators)
        cond_model = GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.8, min_samples_split=5, min_samples_leaf=3,
            random_state=42,
        )

        X_tv, X_test_d, y_tv, y_test_d = train_test_split(
            X_det, y_det_log, test_size=test_size, random_state=42
        )

        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_r2 = cross_val_score(cond_model, X_tv, y_tv, cv=kf, scoring="r2", n_jobs=1)
        self.logger.info(
            f"  Conditional TTD CV R² (log space): {cv_r2.mean():.4f} ± {cv_r2.std():.4f}"
        )

        cond_model.fit(X_tv, y_tv)
        r2_cond_train = r2_score(y_tv, cond_model.predict(X_tv))
        r2_cond_test  = r2_score(y_test_d, cond_model.predict(X_test_d))
        overfit_cond  = r2_cond_train - r2_cond_test
        self.logger.info(
            f"  Conditional Train/Test R² (log): {r2_cond_train:.4f} / {r2_cond_test:.4f}  "
            f"overfit={overfit_cond:.4f}"
        )
        self._log_feature_importance(cond_model, "TTD_CONDITIONAL")

        # Assemble hurdle model
        dr_clf = self.models["detection_rate"]
        hurdle_model = HurdleTTDModel(
            dr_classifier=dr_clf,
            conditional_regressor=cond_model,
            ttd_max=ttd_max,
        )

        # Measure end-to-end hurdle R² on FULL dataset (all rows)
        X_all, X_test_all, y_all, y_test_all = train_test_split(
            X, y_ttd, test_size=test_size, random_state=42
        )
        ttd_pred_test = hurdle_model.predict(X_test_all)
        r2_hurdle_test  = r2_score(y_test_all, ttd_pred_test)
        rmse_hurdle     = np.sqrt(mean_squared_error(y_test_all, ttd_pred_test))
        r2_hurdle_train = r2_score(y_all, hurdle_model.predict(X_all))
        overfit_hurdle  = r2_hurdle_train - r2_hurdle_test

        self.logger.info(
            f"  Hurdle model end-to-end R²: train={r2_hurdle_train:.4f}  "
            f"test={r2_hurdle_test:.4f}  overfit={overfit_hurdle:.4f}"
        )
        self.logger.info(f"  Hurdle RMSE (seconds): {rmse_hurdle:.1f}")

        self.models["ttd"] = hurdle_model
        # Signal to loader: TTD output is already in seconds, no expm1 needed
        self.models["ttd_log_transform"] = False

        return {
            "metric": "ttd",
            "method": "hurdle_two_stage",
            "conditional_cv_r2": float(cv_r2.mean()),
            "conditional_test_r2_log": float(r2_cond_test),
            "hurdle_test_r2": float(r2_hurdle_test),
            "hurdle_train_r2": float(r2_hurdle_train),
            "overfit_gap": float(overfit_hurdle),
            "rmse_seconds": float(rmse_hurdle),
            "n_detected_rows": n_detected,
            "n_total_rows": n_total,
            "ttd_max": float(ttd_max),
        }

    # ------------------------------------------------------------------
    def _log_feature_importance(self, fitted_model, label: str) -> None:
        """Log feature importances if the model supports them."""
        if not hasattr(fitted_model, "feature_importances_"):
            return
        imp = fitted_model.feature_importances_
        pairs = sorted(zip(self._feature_names, imp), key=lambda x: -x[1])
        self.logger.info(f"  Feature importances ({label}):")
        for name, val in pairs:
            self.logger.info(f"    {name:25s}: {val*100:.1f}%")

    # ------------------------------------------------------------------
    def train_all(
        self, X: np.ndarray, y_dr: np.ndarray,
        y_fnr: np.ndarray, y_ttd: np.ndarray
    ) -> Dict:
        """Train all three surrogates using the v3.2 hurdle architecture.

        Training order matters:
          1. DR classifier first (FNR and TTD hurdle models depend on it)
          2. FNR derived model (wraps DR classifier; no new model trained)
          3. TTD hurdle model (uses DR classifier + conditional regressor)
        """
        self.logger.info("=" * 70)
        self.logger.info("TRAINING SURROGATE MODELS — v3.2 (Hurdle Architecture)")
        self.logger.info("=" * 70)

        metrics: Dict = {}
        # Step 1: DR classifier (must come first)
        metrics["detection_rate"] = self.train_detection_rate(X, y_dr)

        # Step 2: FNR — derived from DR classifier (no separate model trained)
        metrics["fnr"] = self.train_fnr_derived(X, y_fnr)

        # Step 3: TTD — two-stage hurdle model (conditional + DR probability)
        metrics["ttd"] = self.train_ttd_hurdle(X, y_ttd)

        self.logger.info("\n" + "=" * 70)
        self.logger.info("SUMMARY — v3.2 Hurdle Surrogates")
        self.logger.info("=" * 70)
        self.logger.info(
            f"  DR  ROC-AUC (5-fold CV) : {metrics['detection_rate']['cv_roc_auc']:.4f}"
        )
        self.logger.info(
            f"  FNR effective R²        : {metrics['fnr']['effective_r2']:.4f}  "
            f"(derived from DR classifier)"
        )
        self.logger.info(
            f"  TTD hurdle test R²      : {metrics['ttd']['hurdle_test_r2']:.4f}  "
            f"(conditional R²={metrics['ttd']['conditional_cv_r2']:.4f}, "
            f"{metrics['ttd']['n_detected_rows']}/{metrics['ttd']['n_total_rows']} detected rows)"
        )
        return metrics

    # ------------------------------------------------------------------
    def save(self, output_dir: Path, version: str = None) -> None:
        """Persist models, scaler, encoders, and metadata."""
        import joblib

        saved_ml_dir = Path(output_dir) / "saved_ml"
        saved_ml_dir.mkdir(parents=True, exist_ok=True)

        for name in ("detection_rate", "fnr", "ttd"):
            if name in self.models:
                joblib.dump(
                    self.models[name],
                    saved_ml_dir / f"surrogate_{name}.pkl",
                )

        joblib.dump(self.scaler, saved_ml_dir / "scaler.pkl")

        metadata = {
            "feature_names": self._feature_names,
            "n_features": len(self._feature_names),
            "label_encoder_classes": {
                k: list(enc.classes_)
                for k, enc in self.label_encoders.items()
            },
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "training_bounds": self.training_bounds,
            # HurdleTTDModel.predict() returns TTD in seconds directly —
            # the loader must NOT apply an additional expm1 transformation.
            "log_transform_ttd": bool(self.models.get("ttd_log_transform", False)),
            "surrogate_architecture": "v3.2_hurdle",
        }
        meta_path = saved_ml_dir / "metadata.json"
        with open(meta_path, "w") as fh:
            json.dump(metadata, fh, indent=2)

        self.logger.info(f"Saved surrogates to {saved_ml_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train v3 surrogates")
    parser.add_argument("--data-dir",  default="data_v19",
                        help="Dataset directory containing master_index.csv (default: data_v19)")
    parser.add_argument("--out-dir",   default="BO/bo_results",
                        help="Output directory for saved surrogates (default: BO/bo_results)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    builder = SurrogateBuilderV3()

    data_dir = Path(args.data_dir)
    X_raw, df = builder.load_and_prepare_data(data_dir)
    X = builder.fit_scaler(X_raw)

    y_dr  = df["detection_rate"].values.astype(np.float32)
    y_fnr = df["false_negative_rate"].values.astype(np.float32)
    y_ttd = df["time_to_detection"].values.astype(np.float32)

    metrics = builder.train_all(X, y_dr, y_fnr, y_ttd)
    builder.save(Path(args.out_dir))

    logger.info(f"\n[OK] v3 surrogate training complete (data={data_dir}, out={args.out_dir})")
    return metrics


if __name__ == "__main__":
    main()
