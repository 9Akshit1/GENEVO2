#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Surrogate model inspector.

Loads the saved v4 surrogate models and prints:
  - Overall accuracy metrics (ROC-AUC, R², MAE)
  - Feature importances for each model
  - Per-scenario breakdown

Usage (run from project root):
    python BO/inspect_surrogates.py
    python BO/inspect_surrogates.py --surrogate-dir BO/bo_results_v14
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent.parent))


def get_feature_importances(model):
    """Extract feature importances from a model, including wrapped calibrated models."""
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    # CalibratedClassifierCV wraps the base estimator
    if hasattr(model, "calibrated_classifiers_"):
        fi_list = [
            c.estimator.feature_importances_
            for c in model.calibrated_classifiers_
            if hasattr(c.estimator, "feature_importances_")
        ]
        return np.mean(fi_list, axis=0) if fi_list else None
    if hasattr(model, "base_estimator") and hasattr(model.base_estimator, "feature_importances_"):
        return model.base_estimator.feature_importances_
    return None


def main():
    parser = argparse.ArgumentParser(description="Inspect surrogate model quality")
    parser.add_argument(
        "--surrogate-dir",
        type=Path,
        default=Path("BO/bo_results"),
        help="Directory containing saved_ml/ subfolder (default: BO/bo_results)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data_v16"),
        help="Data directory with master_index.csv (default: data_v16)",
    )
    parser.add_argument(
        "--version",
        default="v4",
        help="Surrogate version suffix (default: v4)",
    )
    args = parser.parse_args()

    ml_dir = args.surrogate_dir / "saved_ml"
    ver    = args.version

    # ── Load models ──────────────────────────────────────────────────────────
    meta   = json.load(open(ml_dir / f"metadata_{ver}.json"))
    scaler = joblib.load(ml_dir / f"scaler_{ver}.pkl")
    m_dr   = joblib.load(ml_dir / f"surrogate_detection_rate_{ver}.pkl")
    m_fnr  = joblib.load(ml_dir / f"surrogate_fnr_{ver}.pkl")
    m_ttd  = joblib.load(ml_dir / f"surrogate_ttd_{ver}.pkl")

    feat_names = meta["feature_names"]

    le_scenario = LabelEncoder()
    le_scenario.fit(meta["label_encoder_classes"]["scenario"])
    le_biosensor = LabelEncoder()
    le_biosensor.fit(meta["label_encoder_classes"]["biosensor_type"])
    le_noise = LabelEncoder()
    le_noise.fit(meta["label_encoder_classes"]["noise_preset"])

    # ── Load data and build feature matrix ───────────────────────────────────
    csv = args.data_dir / "master_index.csv"
    df  = pd.read_csv(csv)

    X_raw = pd.DataFrame({
        "log_kd":             np.log10(df["kd"].clip(lower=1e-3)),
        "log_sensitivity":    np.log10(df["sensitivity"].clip(lower=1e-3)),
        "log_response_time":  np.zeros(len(df)),
        "biosensor_type_enc": le_biosensor.transform(df["biosensor_type"]).astype(float),
        "noise_preset_enc":   le_noise.transform(df["noise_preset"]).astype(float),
        "scenario_enc":       le_scenario.transform(df["scenario"]).astype(float),
        "log_kd_ctx":         np.log10(df["kd_ctx"].clip(lower=1e-3)),
        "log_kd_p1np":        np.log10(df["kd_p1np"].clip(lower=1e-3)),
        "w_ctx":              df["w_ctx"],
        "w_p1np":             df["w_p1np"],
    })[feat_names].values

    X_scaled = scaler.transform(X_raw)

    y_dr  = (df["detection_rate"] > 0.5).astype(int).values
    y_fnr = df["false_negative_rate"].values
    y_ttd = np.log1p(df["time_to_detection"].values)

    dr_prob  = m_dr.predict_proba(X_scaled)[:, 1]
    fnr_pred = m_fnr.predict(X_scaled)
    ttd_pred = m_ttd.predict(X_scaled)

    # ── Overall metrics ───────────────────────────────────────────────────────
    print()
    print(f"=== SURROGATE QUALITY ({ver} — {csv}  N={len(df)}) ===")
    print()
    print("  NOTE: These are in-sample metrics (all rows). For held-out test")
    print("  metrics, check the training output from --retrain-surrogates.")
    print("  (Test metrics are lower: DR AUC~0.94, FNR R2~0.69, TTD R2~0.63)")
    print()
    print(f"  DR  ROC-AUC : {roc_auc_score(y_dr, dr_prob):.4f}")
    print(f"                 1.0 = perfect classifier | 0.5 = coin-flip")
    print()
    print(f"  FNR R2      : {r2_score(y_fnr, fnr_pred):.4f}")
    print(f"  FNR MAE     : {mean_absolute_error(y_fnr, fnr_pred):.4f}")
    print(f"                 MAE = avg absolute error in FNR (0.12 = 12 pp off)")
    print()
    print(f"  TTD R2      : {r2_score(y_ttd, ttd_pred):.4f}")
    print(f"  TTD MAE     : {mean_absolute_error(np.expm1(y_ttd), np.expm1(ttd_pred)):.0f} s")
    print(f"                 MAE = avg absolute error in time-to-detection")

    # ── Feature importances ───────────────────────────────────────────────────
    for label, model in [("DR (classifier)", m_dr), ("FNR (regressor)", m_fnr), ("TTD (regressor)", m_ttd)]:
        fi = get_feature_importances(model)
        if fi is None:
            print(f"\n  {label} feature importances: not available")
            continue
        print(f"\n  --- {label} Feature Importances ---")
        for name, imp in sorted(zip(feat_names, fi), key=lambda x: -x[1]):
            if imp < 0.0005:
                continue
            bar = "#" * int(imp * 50)
            print(f"    {name:<24} {imp*100:5.1f}%  {bar}")

    # ── Per-scenario breakdown ────────────────────────────────────────────────
    print()
    print("  --- Per-Scenario Breakdown ---")
    print(f"  {'Scenario':<12}  {'N':>4}  {'% positive':>10}  {'DR AUC':>7}  {'FNR MAE':>8}  {'TTD MAE':>10}")
    print("  " + "-" * 68)

    sc_codes = le_scenario.transform(df["scenario"].values)
    for sc_idx, sc_name in enumerate(le_scenario.classes_):
        mask = sc_codes == sc_idx
        n    = mask.sum()
        if n == 0:
            continue
        pos_rate = y_dr[mask].mean()
        n_uniq   = len(np.unique(y_dr[mask]))
        auc_str  = (
            f"{roc_auc_score(y_dr[mask], dr_prob[mask]):.4f}"
            if n_uniq > 1 else "   N/A"
        )
        fnr_mae = mean_absolute_error(y_fnr[mask], fnr_pred[mask])
        ttd_mae = mean_absolute_error(np.expm1(y_ttd[mask]), np.expm1(ttd_pred[mask]))
        print(
            f"  {sc_name:<12}  {n:4d}  {pos_rate*100:9.1f}%  "
            f"{auc_str:>7}  {fnr_mae:8.4f}  {ttd_mae:9.0f} s"
        )

    print()

    # ── Dead features check ───────────────────────────────────────────────────
    dr_fi = get_feature_importances(m_dr)
    dead = [n for n, f in zip(feat_names, dr_fi if dr_fi is not None else [])
            if f < 0.001]
    if dead:
        print(f"  Dead features (< 0.1% importance in DR model): {dead}")
        print(f"  These are excluded from the GP in BO (active dims only).")
    print()


if __name__ == "__main__":
    main()
