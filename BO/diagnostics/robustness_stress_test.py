#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Robustness stress tests for the best BO-v6 biosensor design.

Tests:
  1. Baseline (nominal conditions)
  2. 2× biological variability (doubled sigma on surrogate predictions)
  3. Perturbed biomarker concentrations (±20% shifts)
  4. Unseen severity levels (extra-mild, extra-severe PMO/CKD)
  5. Mixed disease (PMO+CKD hybrid biomarker profiles)
  6. Adversarial SOST fold-change sweep

Usage:
    python BO/diagnostics/robustness_stress_test.py
    python BO/diagnostics/robustness_stress_test.py --config-file BO/bo_results/results/best_config.json
"""

import argparse
import json
import sys
import logging
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from BO.core.surrogate_loader import SurrogateLoaderV3
from evaluation.physics_forward_model import PhysicsForwardModel
from evaluation.objective_function_v3 import ObjectiveFunctionV3
import evaluation.therapeutic_objective_v6 as _m6
from evaluation.therapeutic_objective_v6 import TherapeuticObjectiveV6

CONSOLE = logging.getLogger("robustness")
CONSOLE.setLevel(logging.INFO)
CONSOLE.propagate = False
if not CONSOLE.handlers:
    _ch = logging.StreamHandler(sys.stdout)
    _ch.setFormatter(logging.Formatter("%(message)s"))
    CONSOLE.addHandler(_ch)

_NOMINAL = {
    "healthy":  {"scl": 0.375, "ctx": 0.200, "p1np": 0.350},
    "pmo_mild": {"scl": 0.5625, "ctx": 0.300, "p1np": 0.385},
    "pmo":      {"scl": 0.875,  "ctx": 0.500, "p1np": 0.525},
    "ckd_mbd":  {"scl": 1.125,  "ctx": 0.500, "p1np": 0.625},
}


class PerturbedSurrogateLoader:
    """Wraps SurrogateLoaderV3 and adds Gaussian noise on predictions."""

    def __init__(self, base_loader, sigma_scale=1.0, seed=42):
        self.base = base_loader
        self.sigma_scale = sigma_scale
        self.rng = np.random.RandomState(seed)

    def predict(self, **kw):
        dr, fnr, ttd = self.base.predict(**kw)
        dr_noise  = self.sigma_scale * 0.05
        fnr_noise = self.sigma_scale * 0.05
        ttd_noise = self.sigma_scale * 0.08

        dr_p  = float(np.clip(dr  + self.rng.normal(0, dr_noise),  0, 1))
        fnr_p = float(np.clip(fnr + self.rng.normal(0, fnr_noise), 0, 1))
        ttd_p = float(np.clip(ttd + self.rng.normal(0, ttd_noise * ttd), 400, 9000))
        return dr_p, fnr_p, ttd_p


def _load_best_config(surrogate_dir: Path) -> dict:
    best_path = surrogate_dir / "results" / "best_config.json"
    if not best_path.exists():
        return None
    with open(best_path) as f:
        data = json.load(f)
    if "biosensor_design" in data:
        cfg = {**data["biosensor_design"], **data.get("measurement_environment", {})}
        cfg["biosensor_type"] = cfg.pop("type", cfg.get("biosensor_type", "array"))
    else:
        cfg = data
    return cfg


def _make_canonical_config():
    return {
        "kd_nm": 2.0, "kd_ctx_nm": 0.40, "kd_p1np_nm": 0.30,
        "sensitivity": 3.5, "w_ctx": 0.35, "w_p1np": 0.25,
        "biosensor_type": "array", "noise_preset": "realistic",
        "response_time_s": 500.0, "target_scenario": "pmo",
    }


def _analytical_R(cfg: dict, disease_concs: dict) -> float:
    """Langmuir occupancy ratio (v6 relative threshold) — no surrogate needed."""
    kd      = float(cfg.get("kd_nm", 1.0))
    kd_ctx  = float(cfg.get("kd_ctx_nm", 1.0))
    kd_p1np = float(cfg.get("kd_p1np_nm", 1.0))
    w_ctx   = float(cfg.get("w_ctx", 0.1))
    w_p1np  = float(cfg.get("w_p1np", 0.1))
    w_scl   = max(0.0, 1.0 - w_ctx - w_p1np)

    def occ(c, kd_): return c / (kd_ + c)
    def ratio(c_d, c_h, kd_): return occ(c_d, kd_) / max(occ(c_h, kd_), 1e-12)

    return (w_scl * ratio(disease_concs["scl"],  _NOMINAL["healthy"]["scl"],  kd)
            + w_ctx  * ratio(disease_concs["ctx"],  _NOMINAL["healthy"]["ctx"],  kd_ctx)
            + w_p1np * ratio(disease_concs["p1np"], _NOMINAL["healthy"]["p1np"], kd_p1np))


def _dose_and_bmd(obj: TherapeuticObjectiveV6, R: float) -> tuple:
    frac_margin = max(0.0, (R - obj.DRUG_THRESHOLD_FRAC) / obj.DRUG_THRESHOLD_FRAC)
    dose = obj.K_RELEASE * frac_margin
    overdose = max(0.0, dose / obj.D_SAFE - 1.0) ** 2 * obj.ALPHA_OVERDOSE
    bmd_gross = obj.BMD_GAIN_MAX * dose / (obj.D_HALF + dose) if dose > 0 else 0.0
    bmd_net = min(bmd_gross / obj.BMD_GAIN_REF, 2.0) - overdose
    return dose, overdose, bmd_net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--surrogate-dir", type=Path, default=Path("BO/bo_results"))
    parser.add_argument("--config-file",   type=Path, default=None)
    parser.add_argument("--n-trials",      type=int,  default=25)
    parser.add_argument("--out",           type=Path,
                        default=Path("BO/diagnostics/plots/robustness_stress_test.json"))
    args = parser.parse_args()

    CONSOLE.info("=" * 70)
    CONSOLE.info("GENEVO2 — Robustness Stress Test")
    CONSOLE.info("=" * 70)

    surrogate = SurrogateLoaderV3(args.surrogate_dir)
    physics   = PhysicsForwardModel()
    obj_v6    = TherapeuticObjectiveV6(physics, surrogate)
    obj_v3    = ObjectiveFunctionV3(physics, surrogate)

    if args.config_file and args.config_file.exists():
        with open(args.config_file) as f:
            data = json.load(f)
        if "biosensor_design" in data:
            cfg = {**data["biosensor_design"], **data.get("measurement_environment", {})}
            cfg["biosensor_type"] = cfg.pop("type", cfg.get("biosensor_type", "array"))
        else:
            cfg = data
        CONSOLE.info(f"  Config from : {args.config_file}")
    else:
        cfg = _load_best_config(args.surrogate_dir)
        if cfg is None:
            cfg = _make_canonical_config()
            CONSOLE.info("  Config : canonical fallback (no best_config.json)")
        else:
            CONSOLE.info(f"  Config from : {args.surrogate_dir}/results/best_config.json")

    CONSOLE.info(f"  sens={cfg.get('sensitivity',0):.3f}  kd={cfg.get('kd_nm',0):.3f}  "
                 f"kd_ctx={cfg.get('kd_ctx_nm',0):.3f}  kd_p1np={cfg.get('kd_p1np_nm',0):.3f}")
    CONSOLE.info(f"  w_ctx={cfg.get('w_ctx',0):.3f}  w_p1np={cfg.get('w_p1np',0):.3f}  "
                 f"n_trials={args.n_trials}")
    CONSOLE.info("")

    results = {}

    # ── Test 1: Baseline ──────────────────────────────────────────────────────
    CONSOLE.info("[TEST 1] Baseline (nominal conditions)")
    v3_base = obj_v3(cfg)
    v6_base, detail = obj_v6.evaluate_with_details(cfg)
    CONSOLE.info(f"  v3={v3_base:.4f}  v6={v6_base:.4f}")
    CONSOLE.info(f"  DR_pmo={detail.get('dr_pmo',0):.3f}  DR_mild={detail.get('dr_mild',0):.3f}  "
                 f"FNR={detail.get('fnr_mean',0):.3f}  TTD={detail.get('ttd_mean',0):.0f}s")
    CONSOLE.info(f"  R_mild={detail.get('R_pmo_mild',0):.4f}  R_pmo={detail.get('R_pmo',0):.4f}  "
                 f"R_ckd={detail.get('R_ckd',0):.4f}")
    CONSOLE.info(f"  bmd_mild={detail.get('bmd_net_mild',0):.4f}  "
                 f"bmd_pmo={detail.get('bmd_net_pmo',0):.4f}  "
                 f"bmd_ckd={detail.get('bmd_net_ckd',0):.4f}")
    results["baseline"] = {
        "v3": float(v3_base), "v6": float(v6_base),
        "detail": {k: float(v) for k, v in detail.items() if isinstance(v, (int, float))},
    }

    # ── Test 2: 2× Biological Variability ─────────────────────────────────────
    CONSOLE.info("\n[TEST 2] 2× Biological Variability")
    scores_1x, scores_2x = [], []
    for trial in range(args.n_trials):
        loader_1x = PerturbedSurrogateLoader(surrogate, sigma_scale=1.0, seed=trial)
        loader_2x = PerturbedSurrogateLoader(surrogate, sigma_scale=2.0, seed=trial)
        o1 = TherapeuticObjectiveV6(physics, loader_1x, apply_constraints=False)
        o2 = TherapeuticObjectiveV6(physics, loader_2x, apply_constraints=False)
        scores_1x.append(o1(cfg))
        scores_2x.append(o2(cfg))

    s1, s2 = np.array(scores_1x), np.array(scores_2x)
    delta = float(s1.mean() - s2.mean())
    robust = abs(delta) < 0.05
    CONSOLE.info(f"  1× variability: mean={s1.mean():.4f} ± {s1.std():.4f}  "
                 f"p5={np.percentile(s1,5):.4f}")
    CONSOLE.info(f"  2× variability: mean={s2.mean():.4f} ± {s2.std():.4f}  "
                 f"p5={np.percentile(s2,5):.4f}")
    CONSOLE.info(f"  Impact at 2×: {delta:+.4f}  {'ROBUST' if robust else 'FRAGILE'}")
    results["variability_2x"] = {
        "sigma_1x": {"mean": float(s1.mean()), "std": float(s1.std()),
                     "p5": float(np.percentile(s1, 5))},
        "sigma_2x": {"mean": float(s2.mean()), "std": float(s2.std()),
                     "p5": float(np.percentile(s2, 5))},
        "delta": delta, "robust": robust,
    }

    # ── Test 3: Perturbed Biomarker Concentrations ────────────────────────────
    CONSOLE.info("\n[TEST 3] Perturbed Biomarker Concentrations (±20%)")
    perturbations = {"nominal": 1.00, "+10%": 1.10, "+20%": 1.20, "-10%": 0.90, "-20%": 0.80}
    conc_results = {}
    orig_concs = _m6._NOMINAL_CONCS
    try:
        for label, factor in perturbations.items():
            _m6._NOMINAL_CONCS = {
                "healthy": _NOMINAL["healthy"].copy(),
                **{sc: {k: v * factor for k, v in _NOMINAL[sc].items()}
                   for sc in ("pmo_mild", "pmo", "ckd_mbd")}
            }
            score = obj_v6(cfg)
            conc_results[label] = float(score)
            CONSOLE.info(f"  {label:<12}: v6={score:.4f}")
    finally:
        _m6._NOMINAL_CONCS = orig_concs

    conc_range = max(conc_results.values()) - min(conc_results.values())
    CONSOLE.info(f"  Score range ±20%: {conc_range:.4f}  "
                 f"({'ROBUST' if conc_range < 0.10 else 'SENSITIVE'})")
    results["concentration_perturbation"] = conc_results

    # ── Test 4: Unseen Severity Levels ────────────────────────────────────────
    CONSOLE.info("\n[TEST 4] Unseen Severity Levels (therapeutic extrapolation)")
    severity_profiles = {
        "extra-mild  (1.1×)": {"scl": 0.375*1.10, "ctx": 0.200*1.10, "p1np": 0.350*1.05},
        "PMO-mild    (1.5×)": _NOMINAL["pmo_mild"].copy(),
        "PMO         (2.3×)": _NOMINAL["pmo"].copy(),
        "severe PMO  (3.0×)": {"scl": 0.375*3.00, "ctx": 0.200*3.00, "p1np": 0.350*2.00},
        "CKD-MBD     (3.0×)": _NOMINAL["ckd_mbd"].copy(),
        "severe CKD  (4.0×)": {"scl": 0.375*4.00, "ctx": 0.200*3.50, "p1np": 0.350*2.50},
    }
    sev_results = {}
    for label, concs in severity_profiles.items():
        R = _analytical_R(cfg, concs)
        dose, overdose, bmd_net = _dose_and_bmd(obj_v6, R)
        CONSOLE.info(f"  {label:<24}  R={R:.4f}  dose={dose:.4f}  "
                     f"od={overdose:.4f}  bmd={bmd_net:.4f}")
        sev_results[label] = {"R": R, "dose": dose, "overdose": overdose, "bmd_net": bmd_net}
    results["severity_extrapolation"] = sev_results

    # ── Test 5: Mixed Disease ─────────────────────────────────────────────────
    CONSOLE.info("\n[TEST 5] Mixed Disease (PMO+CKD hybrid)")
    mix_results = {}
    for alpha in [0.0, 0.25, 0.50, 0.75, 1.0]:
        mixed = {k: (1-alpha)*_NOMINAL["pmo"][k] + alpha*_NOMINAL["ckd_mbd"][k]
                 for k in ("scl", "ctx", "p1np")}
        R = _analytical_R(cfg, mixed)
        dose, overdose, bmd_net = _dose_and_bmd(obj_v6, R)
        label = f"PMO×{1-alpha:.2f}+CKD×{alpha:.2f}"
        CONSOLE.info(f"  {label:<22}  R={R:.4f}  dose={dose:.4f}  bmd={bmd_net:.4f}")
        mix_results[label] = {"alpha_ckd": alpha, "R": R,
                               "dose": dose, "overdose": overdose, "bmd_net": bmd_net}
    results["mixed_disease"] = mix_results

    # ── Test 6: SOST Fold-Change Sweep ───────────────────────────────────────
    CONSOLE.info("\n[TEST 6] Adversarial SOST Fold-Change Sweep")
    sweep_results = {}
    for fc in [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        test_conc = {"scl": 0.375*fc, "ctx": 0.200*fc, "p1np": 0.350*(fc**0.5)}
        R = _analytical_R(cfg, test_conc)
        dose, overdose, bmd_net = _dose_and_bmd(obj_v6, R)
        CONSOLE.info(f"  fold={fc:.1f}×  R={R:.4f}  dose={dose:.4f}  "
                     f"od={overdose:.4f}  bmd={bmd_net:.4f}")
        sweep_results[f"{fc:.1f}x"] = {
            "fold_change": fc, "R": R, "dose": dose,
            "overdose": overdose, "bmd_net": bmd_net,
        }
    results["sost_sweep"] = sweep_results

    # ── Summary ────────────────────────────────────────────────────────────────
    CONSOLE.info("\n" + "=" * 70)
    CONSOLE.info("ROBUSTNESS SUMMARY")
    CONSOLE.info("=" * 70)
    CONSOLE.info(f"  Baseline v6                  : {v6_base:.4f}")
    v2 = results["variability_2x"]
    CONSOLE.info(f"  2× variability impact        : {v2['delta']:+.4f}  "
                 f"({'ROBUST' if v2['robust'] else 'FRAGILE'})")
    CONSOLE.info(f"  ±20% conc perturbation range : {conc_range:.4f}  "
                 f"({'ROBUST' if conc_range < 0.10 else 'SENSITIVE'})")
    bmd_vals = [v["bmd_net"] for v in sev_results.values()]
    R_vals   = [v["R"]       for v in sev_results.values()]
    CONSOLE.info(f"  Severity BMD range           : {min(bmd_vals):.4f} – {max(bmd_vals):.4f}")
    CONSOLE.info(f"  Severity R range             : {min(R_vals):.4f} – {max(R_vals):.4f}")
    mix_bmds = [v["bmd_net"] for v in mix_results.values()]
    CONSOLE.info(f"  Mixed disease BMD range      : {min(mix_bmds):.4f} – {max(mix_bmds):.4f}")
    CONSOLE.info("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    CONSOLE.info(f"Results saved -> {args.out}")


if __name__ == "__main__":
    main()
