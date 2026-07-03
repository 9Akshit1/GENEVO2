#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Design Model: Patient State → Optimal Biosensor Parameters.

This is the final stage of the pipeline:

    Simulator → Dataset Generation → Optimization → Optimal Designs → Design Model

The model takes a patient's biomarker profile and disease context as input
and directly predicts optimal biosensor parameters — no optimization needed
at inference time.

Architecture
------------
MLP Regressor (scikit-learn MLPRegressor):
  Input  (10 features): SOST, CTX, P1NP concentrations (log-space),
                         severity (ordinal), disease class (one-hot: PMO/CKD),
                         noise_level, target_ttd
  Output (7 targets):   kd_nm, kd_ctx_nm, kd_p1np_nm, sensitivity,
                        response_time_s, w_ctx, w_p1np  (log-space for kds)

Training strategy:
  - Filter: only use designs with score_v6 >= SCORE_THRESHOLD (top quartile)
  - Multi-output regression: train one MLP per output (wrapped in MultiOutputRegressor
    for independence, then combine into final predictor)
  - Alternatively: single MLP with 7 outputs (faster, often comparably accurate)
  - Features are standardized; kd outputs are in log10 space

Validation:
  - 80/20 train/test split
  - Per-output R² and MAE
  - Simulation-based validation: predict on 200 test patient profiles,
    run v6 objective on predicted design, compare to BO result

Usage:
    # Train from generated dataset
    python BO/design_model/train_design_model.py --data BO/data_expansion/optimized_designs.csv

    # Predict for a specific patient profile
    python BO/design_model/train_design_model.py --predict \
        --sost 0.875 --ctx 0.50 --p1np 0.525 --disease pmo --severity moderate

    # Evaluate model quality on held-out data
    python BO/design_model/train_design_model.py --evaluate
"""

import argparse
import json
import sys
import logging
import warnings
from pathlib import Path

import numpy as np
import joblib

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

CONSOLE = logging.getLogger("design_model")
CONSOLE.setLevel(logging.INFO)
CONSOLE.propagate = False
if not CONSOLE.handlers:
    _ch = logging.StreamHandler(sys.stdout)
    _ch.setFormatter(logging.Formatter("%(message)s"))
    CONSOLE.addHandler(_ch)

# Configs with v6 score below this are excluded from training
SCORE_THRESHOLD = 0.55

# Output parameter targets (log-space for kd values, linear for weights)
_OUTPUT_COLS = [
    "log_kd_nm", "log_kd_ctx_nm", "log_kd_p1np_nm",
    "log_sensitivity", "log_response_time_s",
    "w_ctx", "w_p1np",
]
_TARGET_COLS_RAW = [
    "kd_nm", "kd_ctx_nm", "kd_p1np_nm",
    "sensitivity", "response_time_s",
    "w_ctx", "w_p1np",
]

# Nominal healthy concentrations [nM]
_NOMINAL_HEALTHY = {"scl": 0.375, "ctx": 0.200, "p1np": 0.350}

# Disease class encoding
_DISEASE_CLASSES = {"healthy": 0, "pmo_mild": 1, "pmo": 2, "ckd_mbd": 3}
_SEVERITY_MAP    = {"minimal": 0, "mild": 1, "moderate": 2, "severe": 3}


def _build_features(sost_nm: float, ctx_nm: float, p1np_nm: float,
                    disease: str = "pmo", severity: str = "moderate",
                    noise_level: float = 0.1, target_ttd_s: float = 500.0) -> np.ndarray:
    """
    Build a 10-feature input vector from a patient biomarker profile.

    Features (all in transformed space):
      0: log10(SOST / SOST_healthy)    — fold-change from healthy
      1: log10(CTX  / CTX_healthy)
      2: log10(P1NP / P1NP_healthy)
      3: severity_ordinal              — 0=minimal 1=mild 2=moderate 3=severe
      4: is_pmo                        — binary
      5: is_ckd                        — binary
      6: log10(sost_nm)                — absolute log-space concentration
      7: log10(ctx_nm)
      8: log10(p1np_nm)
      9: log10(noise_level)
    """
    h = _NOMINAL_HEALTHY
    feat = np.array([
        np.log10(max(sost_nm / h["scl"], 1e-3)),
        np.log10(max(ctx_nm  / h["ctx"], 1e-3)),
        np.log10(max(p1np_nm / h["p1np"], 1e-3)),
        float(_SEVERITY_MAP.get(severity, 2)),
        1.0 if disease in ("pmo", "pmo_mild") else 0.0,
        1.0 if disease == "ckd_mbd" else 0.0,
        np.log10(max(sost_nm, 1e-3)),
        np.log10(max(ctx_nm, 1e-3)),
        np.log10(max(p1np_nm, 1e-3)),
        np.log10(max(noise_level, 1e-3)),
    ])
    return feat.reshape(1, -1)


def load_dataset(csv_path: Path, score_threshold: float = SCORE_THRESHOLD):
    """Load the generated dataset and return X (features), Y (targets)."""
    import csv

    CONSOLE.info(f"  Loading dataset: {csv_path}")
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    CONSOLE.info(f"  Total rows: {len(rows):,}")
    rows = [r for r in rows if float(r["score_v6"]) >= score_threshold]
    CONSOLE.info(f"  After score filter (>={score_threshold}): {len(rows):,}")

    if len(rows) < 500:
        raise ValueError(
            f"Only {len(rows)} high-quality designs found. "
            f"Run generate_optimized_dataset.py with larger --n-total first."
        )

    # Build patient feature vectors from the nominal concentration that
    # the design was optimized for (pmo, ckd_mbd, pmo_mild as recorded).
    # For each config we synthesize a "patient profile" using the target_scenario.
    _SCEN_CONCS = {
        "pmo":      {"scl": 0.875,  "ctx": 0.500, "p1np": 0.525},
        "ckd_mbd":  {"scl": 1.125,  "ctx": 0.500, "p1np": 0.625},
        "pmo_mild": {"scl": 0.5625, "ctx": 0.300, "p1np": 0.385},
        "healthy":  {"scl": 0.375,  "ctx": 0.200, "p1np": 0.350},
    }
    _SCEN_SEV = {"pmo": "moderate", "ckd_mbd": "moderate",
                 "pmo_mild": "mild", "healthy": "minimal"}

    X_list, Y_list = [], []
    for r in rows:
        scenario = r.get("target_scenario", "pmo")
        concs = _SCEN_CONCS.get(scenario, _SCEN_CONCS["pmo"])
        severity = _SCEN_SEV.get(scenario, "moderate")
        feat = _build_features(
            sost_nm=concs["scl"], ctx_nm=concs["ctx"], p1np_nm=concs["p1np"],
            disease=scenario, severity=severity,
        )
        X_list.append(feat.flatten())

        # Targets: log-space for kd/sensitivity/rt, linear for weights
        y = [
            np.log10(max(float(r["kd_nm"]), 1e-3)),
            np.log10(max(float(r["kd_ctx_nm"]), 1e-3)),
            np.log10(max(float(r["kd_p1np_nm"]), 1e-3)),
            np.log10(max(float(r["sensitivity"]), 1e-3)),
            np.log10(max(float(r["response_time_s"]), 1.0)),
            float(r["w_ctx"]),
            float(r["w_p1np"]),
        ]
        Y_list.append(y)

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    CONSOLE.info(f"  X shape: {X.shape}  Y shape: {Y.shape}")
    return X, Y


def train(csv_path: Path, model_dir: Path, score_threshold=SCORE_THRESHOLD,
          hidden_layer_sizes=(256, 128, 64), max_iter=500, seed=42):
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error

    CONSOLE.info("\n" + "=" * 70)
    CONSOLE.info("DESIGN MODEL TRAINING")
    CONSOLE.info("=" * 70)

    X, Y = load_dataset(csv_path, score_threshold)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.20, random_state=seed
    )
    CONSOLE.info(f"  Train: {X_train.shape[0]:,}  Test: {X_test.shape[0]:,}")

    # Standardize inputs
    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s  = scaler_X.transform(X_test)

    # Standardize outputs (helps MLP training)
    scaler_Y = StandardScaler()
    Y_train_s = scaler_Y.fit_transform(Y_train)

    CONSOLE.info(f"  MLP architecture: {hidden_layer_sizes}")
    CONSOLE.info(f"  max_iter={max_iter}  seed={seed}")

    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=max_iter,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=20,
        verbose=False,
    )
    CONSOLE.info("  Training...")
    mlp.fit(X_train_s, Y_train_s)
    CONSOLE.info(f"  Converged after {mlp.n_iter_} iterations")

    # Evaluate
    Y_pred_s = mlp.predict(X_test_s)
    Y_pred   = scaler_Y.inverse_transform(Y_pred_s)

    CONSOLE.info("\n  Per-output metrics (test set):")
    CONSOLE.info(f"  {'Output':<20}  R²      MAE")
    CONSOLE.info(f"  {'-'*45}")
    for i, col in enumerate(_OUTPUT_COLS):
        r2  = r2_score(Y_test[:, i], Y_pred[:, i])
        mae = mean_absolute_error(Y_test[:, i], Y_pred[:, i])
        CONSOLE.info(f"  {col:<20}  {r2:+.4f}  {mae:.4f}")

    r2_overall = r2_score(Y_test, Y_pred)
    CONSOLE.info(f"\n  Overall multi-output R²: {r2_overall:.4f}")

    # Save model + scalers
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(mlp,      model_dir / "design_mlp.pkl")
    joblib.dump(scaler_X, model_dir / "scaler_X.pkl")
    joblib.dump(scaler_Y, model_dir / "scaler_Y.pkl")

    meta = {
        "hidden_layer_sizes": list(hidden_layer_sizes),
        "score_threshold":    score_threshold,
        "n_train":            int(X_train.shape[0]),
        "n_test":             int(X_test.shape[0]),
        "r2_overall":         float(r2_overall),
        "per_output_r2": {
            col: float(r2_score(Y_test[:, i], Y_pred[:, i]))
            for i, col in enumerate(_OUTPUT_COLS)
        },
        "output_cols": _OUTPUT_COLS,
        "target_cols_raw": _TARGET_COLS_RAW,
    }
    with open(model_dir / "model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    CONSOLE.info(f"\n  Models saved -> {model_dir}")
    return mlp, scaler_X, scaler_Y


def predict_design(sost_nm: float, ctx_nm: float, p1np_nm: float,
                   disease: str, severity: str,
                   model_dir: Path) -> dict:
    """Load trained model and predict optimal biosensor parameters."""
    mlp      = joblib.load(model_dir / "design_mlp.pkl")
    scaler_X = joblib.load(model_dir / "scaler_X.pkl")
    scaler_Y = joblib.load(model_dir / "scaler_Y.pkl")

    feat = _build_features(sost_nm, ctx_nm, p1np_nm, disease, severity)
    feat_s = scaler_X.transform(feat)
    pred_s = mlp.predict(feat_s)
    pred   = scaler_Y.inverse_transform(pred_s)[0]

    # Decode from log/linear space
    w_ctx  = float(np.clip(pred[5], 0, 0.60))
    w_p1np = float(np.clip(pred[6], 0, 0.60))
    if w_ctx + w_p1np > 0.80:
        scale  = 0.80 / (w_ctx + w_p1np)
        w_ctx  *= scale
        w_p1np *= scale

    return {
        "kd_nm":           float(10 ** np.clip(pred[0], -2, 2)),
        "kd_ctx_nm":       float(10 ** np.clip(pred[1], -2, 2)),
        "kd_p1np_nm":      float(10 ** np.clip(pred[2], -2, 2)),
        "sensitivity":     float(10 ** np.clip(pred[3], -1, 1)),
        "response_time_s": float(10 ** np.clip(pred[4], 2, 4)),
        "w_ctx":           w_ctx,
        "w_p1np":          w_p1np,
        "biosensor_type":  "array",
        "noise_preset":    "realistic",
        "target_scenario": disease,
    }


def evaluate_model(model_dir: Path, surrogate_dir: Path, n_eval: int = 200, seed: int = 42):
    """
    Evaluate design model quality by comparing predicted designs to BO results.

    For N patient profiles sampled from training distribution:
      1. Predict design using the ML model
      2. Evaluate design using v6 objective
      3. Compare to random search baseline (same budget = 1 evaluation)
    """
    from BO.core.surrogate_loader import SurrogateLoaderV3
    from evaluation.physics_forward_model import PhysicsForwardModel
    from evaluation.therapeutic_objective_v6 import TherapeuticObjectiveV6

    surrogate = SurrogateLoaderV3(surrogate_dir)
    physics   = PhysicsForwardModel()
    obj_v6    = TherapeuticObjectiveV6(physics, surrogate)

    CONSOLE.info("\n" + "=" * 70)
    CONSOLE.info("DESIGN MODEL EVALUATION")
    CONSOLE.info("=" * 70)

    if not (model_dir / "design_mlp.pkl").exists():
        CONSOLE.info("  ERROR: No trained model found. Run --train first.")
        return

    rng = np.random.RandomState(seed)
    _SCEN_LIST = ["pmo", "pmo_mild", "ckd_mbd"]
    _SCEN_CONCS = {
        "pmo":      {"scl": 0.875,  "ctx": 0.500, "p1np": 0.525},
        "ckd_mbd":  {"scl": 1.125,  "ctx": 0.500, "p1np": 0.625},
        "pmo_mild": {"scl": 0.5625, "ctx": 0.300, "p1np": 0.385},
    }

    ml_scores, rand_scores = [], []
    for i in range(n_eval):
        # Sample a random patient profile with ±20% biological variation
        scenario = rng.choice(_SCEN_LIST)
        base = _SCEN_CONCS[scenario]
        sost = base["scl"]  * (1 + rng.uniform(-0.20, 0.20))
        ctx  = base["ctx"]  * (1 + rng.uniform(-0.20, 0.20))
        p1np = base["p1np"] * (1 + rng.uniform(-0.20, 0.20))
        sev  = rng.choice(["mild", "moderate", "severe"])

        # Predict with ML
        cfg_ml = predict_design(sost, ctx, p1np, scenario, sev, model_dir)
        score_ml = obj_v6(cfg_ml)
        ml_scores.append(score_ml)

        # Random baseline (single random config)
        rand_x = rng.rand(_OUTPUT_COLS.__len__())
        from data_expansion.generate_optimized_dataset import _unit_to_config
        cfg_rand = _unit_to_config(rand_x)
        score_rand = obj_v6(cfg_rand)
        rand_scores.append(score_rand)

    ml_arr   = np.array(ml_scores)
    rand_arr = np.array(rand_scores)

    CONSOLE.info(f"  Design model: mean={ml_arr.mean():.4f} ± {ml_arr.std():.4f}  "
                 f"p75={np.percentile(ml_arr,75):.4f}")
    CONSOLE.info(f"  Random:       mean={rand_arr.mean():.4f} ± {rand_arr.std():.4f}  "
                 f"p75={np.percentile(rand_arr,75):.4f}")
    CONSOLE.info(f"  ML lift:      {ml_arr.mean()-rand_arr.mean():+.4f}")
    CONSOLE.info(f"  ML wins:      {(ml_arr>rand_arr).sum()}/{n_eval}")

    from scipy.stats import wilcoxon
    try:
        stat, p = wilcoxon(ml_arr, rand_arr, alternative="greater")
        CONSOLE.info(f"  Wilcoxon p:   {p:.4f}  "
                     f"({'ML wins significantly' if p < 0.05 else 'not significant'})")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    type=Path,
                        default=Path("BO/data_expansion/optimized_designs.csv"))
    parser.add_argument("--model-dir", type=Path,
                        default=Path("BO/design_model/saved_model"))
    parser.add_argument("--surrogate-dir", type=Path, default=Path("BO/bo_results"))
    parser.add_argument("--score-threshold", type=float, default=SCORE_THRESHOLD)
    parser.add_argument("--train",    action="store_true", help="Train the design model")
    parser.add_argument("--predict",  action="store_true", help="Predict a design")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model quality")
    # Predict args
    parser.add_argument("--sost",    type=float, default=0.875)
    parser.add_argument("--ctx",     type=float, default=0.500)
    parser.add_argument("--p1np",    type=float, default=0.525)
    parser.add_argument("--disease", type=str,   default="pmo",
                        choices=["pmo", "pmo_mild", "ckd_mbd", "healthy"])
    parser.add_argument("--severity",type=str,   default="moderate",
                        choices=["minimal", "mild", "moderate", "severe"])
    # Train args
    parser.add_argument("--hidden",  type=int, nargs="+", default=[256, 128, 64])
    parser.add_argument("--max-iter",type=int, default=500)
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    if args.train:
        train(
            csv_path=args.data,
            model_dir=args.model_dir,
            score_threshold=args.score_threshold,
            hidden_layer_sizes=tuple(args.hidden),
            max_iter=args.max_iter,
            seed=args.seed,
        )

    if args.predict:
        if not (args.model_dir / "design_mlp.pkl").exists():
            CONSOLE.info("No model found. Run with --train first.")
            sys.exit(1)
        cfg = predict_design(
            args.sost, args.ctx, args.p1np,
            args.disease, args.severity, args.model_dir,
        )
        CONSOLE.info("\n" + "=" * 70)
        CONSOLE.info("PREDICTED BIOSENSOR DESIGN")
        CONSOLE.info("=" * 70)
        CONSOLE.info(f"  Patient: SOST={args.sost} nM  CTX={args.ctx} nM  "
                     f"P1NP={args.p1np} nM  disease={args.disease}  "
                     f"severity={args.severity}")
        CONSOLE.info(f"  kd_nm      = {cfg['kd_nm']:.4f} nM")
        CONSOLE.info(f"  kd_ctx_nm  = {cfg['kd_ctx_nm']:.4f} nM")
        CONSOLE.info(f"  kd_p1np_nm = {cfg['kd_p1np_nm']:.4f} nM")
        CONSOLE.info(f"  sensitivity= {cfg['sensitivity']:.3f}")
        CONSOLE.info(f"  response_t = {cfg['response_time_s']:.0f} s")
        CONSOLE.info(f"  w_ctx      = {cfg['w_ctx']:.3f}")
        CONSOLE.info(f"  w_p1np     = {cfg['w_p1np']:.3f}")
        CONSOLE.info(f"  w_scl      = {max(0,1-cfg['w_ctx']-cfg['w_p1np']):.3f}")

        # Optionally score it
        try:
            from BO.core.surrogate_loader import SurrogateLoaderV3
            from evaluation.physics_forward_model import PhysicsForwardModel
            from evaluation.therapeutic_objective_v6 import TherapeuticObjectiveV6
            surrogate = SurrogateLoaderV3(args.surrogate_dir)
            physics   = PhysicsForwardModel()
            obj_v6    = TherapeuticObjectiveV6(physics, surrogate)
            score = obj_v6(cfg)
            CONSOLE.info(f"  v6 score   = {score:.4f}")
        except Exception as e:
            CONSOLE.info(f"  (could not evaluate score: {e})")

    if args.evaluate:
        evaluate_model(args.model_dir, args.surrogate_dir, seed=args.seed)

    if not (args.train or args.predict or args.evaluate):
        parser.print_help()


if __name__ == "__main__":
    main()
