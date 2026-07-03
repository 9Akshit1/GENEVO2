#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Surrogate interpretability: permutation importance + partial dependence.

Computes two complementary interpretability measures for the v18 GBM surrogates:

  1. Permutation importance (PI)
     Shuffle each feature column independently, measure drop in ROC-AUC (DR)
     or R² (FNR, TTD).  PI is a model-agnostic, global measure of feature
     relevance.  Unlike tree-based feature_importances_, PI is not biased
     toward high-cardinality features (Breiman 2001; Strobl 2007).

  2. Partial dependence (PD)
     For each continuous feature, average the surrogate prediction over all
     other features while varying the target feature from its lower to upper
     bound.  Shows the marginal effect of each parameter on DR.

Both methods use the data_v18 dataset (1500 configs) as the background
distribution, so results reflect the realistic operating range.

Reference:
  Breiman L. (2001). Random forests. Machine Learning 45:5-32.
  Friedman J.H. (2001). Greedy function approximation: a gradient boosting
  machine. Annals of Statistics 29:1189-1232.

Usage:
    python BO/analysis/surrogate_shap.py
    python BO/analysis/surrogate_shap.py --n-permutations 20 --n-pd-points 30

Output:
    BO/bo_results/diagnostics/surrogate_interpretability.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import joblib

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "log_kd",
    "log_sensitivity",
    "log_response_time",
    "biosensor_type_enc",
    "noise_preset_enc",
    "scenario_enc",
    "log_kd_ctx",
    "log_kd_p1np",
    "w_ctx",
    "w_p1np",
    "delta_theta_sost",
    "delta_theta_ctx",
    "delta_theta_p1np",
    "composite_signal_proxy",
    "log_composite_signal_proxy",
]

FEATURE_DISPLAY_NAMES = {
    "log_kd":                      "SOST Kd (log)",
    "log_sensitivity":             "Sensitivity (log)",
    "log_response_time":           "Response time (log)",
    "biosensor_type_enc":          "Biosensor type",
    "noise_preset_enc":            "Noise preset",
    "scenario_enc":                "Disease scenario",
    "log_kd_ctx":                  "CTX Kd (log)",
    "log_kd_p1np":                 "P1NP Kd (log)",
    "w_ctx":                       "CTX weight",
    "w_p1np":                      "P1NP weight",
    "delta_theta_sost":            "dTheta SOST",
    "delta_theta_ctx":             "dTheta CTX",
    "delta_theta_p1np":            "dTheta P1NP",
    "composite_signal_proxy":      "Composite signal",
    "log_composite_signal_proxy":  "Log composite signal",
}

CONTINUOUS_FEATURES = [
    "log_kd", "log_sensitivity", "log_kd_ctx", "log_kd_p1np", "w_ctx", "w_p1np",
    "composite_signal_proxy", "log_composite_signal_proxy",
]

MODELS_TARGETS = {
    "detection_rate": "DR (disease detection rate)",
    "fnr":            "FNR (false negative rate)",
    "ttd":            "TTD (time to detection)",
}


def _load_data(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load master_index and build feature + target matrices via SurrogateBuilderV3."""
    from BO.core.build_surrogates import SurrogateBuilderV3

    builder = SurrogateBuilderV3()
    X_raw, df = builder.load_and_prepare_data(data_dir)

    y_dr  = df["detection_rate"].values.astype(np.float32)
    y_fnr = df["false_negative_rate"].values.astype(np.float32)
    y_ttd = df["time_to_detection"].values.astype(np.float32)

    return X_raw, y_dr, y_fnr, y_ttd


def permutation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_permutations: int = 10,
    rng: np.random.RandomState = None,
    is_classifier: bool = False,
) -> Dict[str, dict]:
    """
    Compute permutation importance for each feature.

    For classifiers: metric = ROC-AUC (requires model.predict_proba or predict).
    For regressors:  metric = R² (coefficient of determination).
    """
    if rng is None:
        rng = np.random.RandomState(42)

    from sklearn.metrics import roc_auc_score, r2_score

    if is_classifier:
        try:
            y_pred_base = model.predict_proba(X)[:, 1]
        except AttributeError:
            y_pred_base = model.predict(X)
        baseline = float(roc_auc_score(y, y_pred_base))
        def score_fn(X_):
            try:
                return float(roc_auc_score(y, model.predict_proba(X_)[:, 1]))
            except Exception:
                return float(roc_auc_score(y, model.predict(X_)))
    else:
        y_pred_base = model.predict(X)
        baseline = float(r2_score(y, y_pred_base))
        def score_fn(X_):
            return float(r2_score(y, model.predict(X_)))

    results: Dict[str, dict] = {}
    for j, fname in enumerate(feature_names):
        drops = []
        for _ in range(n_permutations):
            X_perm = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            score_perm = score_fn(X_perm)
            drops.append(baseline - score_perm)

        drops_arr = np.array(drops)
        results[fname] = {
            "importance_mean": float(drops_arr.mean()),
            "importance_std":  float(drops_arr.std()),
            "importance_min":  float(drops_arr.min()),
            "importance_max":  float(drops_arr.max()),
        }
        print(f"    {FEATURE_DISPLAY_NAMES.get(fname, fname):<30}  "
              f"importance={drops_arr.mean():.4f}±{drops_arr.std():.4f}")

    return {"baseline_score": baseline, "features": results}


def partial_dependence(
    model,
    X: np.ndarray,
    feature_names: List[str],
    continuous_features: List[str],
    n_points: int = 30,
    is_classifier: bool = False,
) -> Dict[str, dict]:
    """
    Compute partial dependence for each continuous feature.

    For each feature j:
      1. Sweep n_points values across [min, max] in X[:, j]
      2. For each sweep value v, replace X[:, j] with v for ALL samples
      3. Average model predictions → pd_curve[v]
    """
    pd_results: Dict[str, dict] = {}

    for fname in continuous_features:
        if fname not in feature_names:
            continue
        j = feature_names.index(fname)
        fmin, fmax = float(X[:, j].min()), float(X[:, j].max())

        sweep_vals = np.linspace(fmin, fmax, n_points)
        pd_curve = []

        for v in sweep_vals:
            X_sweep = X.copy()
            X_sweep[:, j] = float(v)
            if is_classifier:
                try:
                    preds = model.predict_proba(X_sweep)[:, 1]
                except AttributeError:
                    preds = model.predict(X_sweep)
            else:
                preds = model.predict(X_sweep)
            pd_curve.append(float(np.mean(preds)))

        pd_results[fname] = {
            "feature_values": sweep_vals.tolist(),
            "pd_values":      pd_curve,
            "display_name":   FEATURE_DISPLAY_NAMES.get(fname, fname),
            "trend": "increasing" if pd_curve[-1] > pd_curve[0] else "decreasing",
        }

    return pd_results


def main():
    parser = argparse.ArgumentParser(description="Surrogate interpretability analysis")
    parser.add_argument("--data-dir",       type=Path, default=Path("data_v19"))
    parser.add_argument("--surrogate-dir",  type=Path, default=Path("BO/bo_results"))
    parser.add_argument("--n-permutations", type=int,  default=15,
                        help="Permutation repeats per feature (default: 15)")
    parser.add_argument("--n-pd-points",    type=int,  default=30,
                        help="Points per feature in PD curves (default: 30)")
    parser.add_argument("--out", type=Path,
                        default=Path("BO/bo_results/diagnostics/surrogate_interpretability.json"))
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    print("=" * 70)
    print("GENEVO2 — Surrogate Interpretability Analysis")
    print(f"  Permutation importance  : {args.n_permutations} repeats/feature")
    print(f"  Partial dependence      : {args.n_pd_points} points/feature")
    print("=" * 70)

    # Load models
    saved_ml = args.surrogate_dir / "saved_ml"
    models: Dict[str, object] = {}
    is_classifier: Dict[str, bool] = {}

    for target in ("detection_rate", "fnr", "ttd"):
        model_path = saved_ml / f"surrogate_{target}.pkl"
        if not model_path.exists():
            print(f"[WARN] Model not found: {model_path} — skipping {target}")
            continue
        models[target] = joblib.load(model_path)
        is_classifier[target] = target == "detection_rate"
        print(f"  Loaded: {model_path.name}")

    if not models:
        print("[ERROR] No surrogate models found.")
        return 1

    # Also load scaler
    scaler_path = saved_ml / "scaler.pkl"
    if not scaler_path.exists():
        print(f"[ERROR] Scaler not found: {scaler_path}")
        return 1
    scaler = joblib.load(scaler_path)

    # Load data
    print(f"\n  Loading data from {args.data_dir}/master_index.csv ...")
    try:
        X_raw, y_dr, y_fnr, y_ttd = _load_data(args.data_dir)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return 1

    # Scale features
    n_stored = models["detection_rate"].n_features_in_ if hasattr(models["detection_rate"], "n_features_in_") else X_raw.shape[1]
    X_scaled = scaler.transform(X_raw[:, :n_stored])
    n_feat = X_scaled.shape[1]
    feat_names = FEATURE_NAMES[:n_feat]

    targets_y = {"detection_rate": y_dr, "fnr": y_fnr, "ttd": y_ttd}

    output: Dict[str, dict] = {}

    # ── Permutation Importance ──────────────────────────────────────────────
    print("\n--- Permutation Importance ---")
    pi_results: Dict[str, dict] = {}
    rng = np.random.RandomState(42)

    for target, model in models.items():
        y = targets_y[target]
        is_cls = is_classifier[target]
        print(f"\n  [{target.upper()}]  (baseline metric: {'ROC-AUC' if is_cls else 'R²'})")
        pi_results[target] = permutation_importance(
            model, X_scaled, y, feat_names,
            n_permutations=args.n_permutations,
            rng=rng,
            is_classifier=is_cls,
        )

    output["permutation_importance"] = pi_results

    # ── GBM built-in feature importance (for reference) ───────────────────
    print("\n--- GBM Native Feature Importances (for reference) ---")
    native_importance: Dict[str, dict] = {}
    for target, model in models.items():
        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_[:n_feat]
            ranked_idx = np.argsort(fi)[::-1]
            native_importance[target] = {
                feat_names[i]: float(fi[i]) for i in ranked_idx
            }
            top3 = [feat_names[i] for i in ranked_idx[:3]]
            print(f"  {target:<20}: top-3 = {top3}")

    output["native_feature_importance"] = native_importance

    # ── Partial Dependence for DR model ───────────────────────────────────
    if "detection_rate" in models:
        print("\n--- Partial Dependence (DR model) ---")
        cont_feats = [f for f in CONTINUOUS_FEATURES if f in feat_names]
        pd_results = partial_dependence(
            models["detection_rate"], X_scaled, feat_names, cont_feats,
            n_points=args.n_pd_points,
            is_classifier=True,
        )
        for fname, pd in pd_results.items():
            trend = pd["trend"]
            min_pd = min(pd["pd_values"])
            max_pd = max(pd["pd_values"])
            print(f"  {FEATURE_DISPLAY_NAMES.get(fname, fname):<30}  "
                  f"range=[{min_pd:.3f},{max_pd:.3f}]  trend={trend}")
        output["partial_dependence_dr"] = pd_results

    # ── Summary: top parameters by PI on DR ───────────────────────────────
    if "detection_rate" in pi_results:
        dr_pi = pi_results["detection_rate"]["features"]
        ranked = sorted(dr_pi.keys(), key=lambda k: dr_pi[k]["importance_mean"], reverse=True)
        print("\n--- Top Parameters by Permutation Importance (DR) ---")
        for i, fname in enumerate(ranked):
            imp = dr_pi[fname]["importance_mean"]
            print(f"  {i+1:2d}. {FEATURE_DISPLAY_NAMES.get(fname, fname):<30}  PI={imp:.4f}")
        output["top_features_dr"] = ranked

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\n[OK] Results saved: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
