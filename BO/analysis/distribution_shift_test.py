#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Systematic distribution shift robustness evaluation.

Tests the best BO configuration against 7 distinct perturbation scenarios
that are not covered by the standard noise-preset robustness test:

  1. Biomarker concentration shift (±10%, ±20%, ±30%)
     — models population-level inter-site variability or systematic measurement drift
  2. Intra-patient variability scaling (1× to 4×)
     — models younger vs elderly patients or assay variability
  3. Sensor gain drift (gradual sensitivity degradation: 100% → 70%)
     — models implant aging over a 2-year deployment lifetime
  4. Missing biomarker (CTX channel fails → w_ctx = 0)
     — models partial sensor fouling
  5. Missing biomarker (P1NP channel fails → w_p1np = 0)
     — models partial sensor fouling
  6. Population shift: CKD-dominant (elevated creatinine alters CTX excretion)
     — CTX 20% higher than nominal in all patients
  7. PMO-CKD mixed phenotype (graded co-morbidity)
     — evaluates therapeutic safety when both conditions coexist

All evaluations use the v18 surrogate (fast, reproducible) plus the V6
therapeutic objective.  No real-simulator calls — this is a sensitivity
analysis, not a clinical validation.

Reference:
  Oakley & O'Hagan (2004) Probabilistic sensitivity analysis of complex models.
  JRSS-B 66(3):751-769.

Usage:
    python BO/analysis/distribution_shift_test.py
    python BO/analysis/distribution_shift_test.py --seed-config 1337

Output:
    BO/bo_results/diagnostics/distribution_shift_results.json
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_NOMINAL = {
    "healthy":  {"scl": 0.375, "ctx": 0.200, "p1np": 0.350},
    "pmo_mild": {"scl": 0.5625, "ctx": 0.300, "p1np": 0.385},
    "pmo":      {"scl": 0.875,  "ctx": 0.500, "p1np": 0.525},
    "ckd_mbd":  {"scl": 1.125,  "ctx": 0.500, "p1np": 0.625},
}


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _load_best_cfg(convergence_path: Path, seed: int) -> dict:
    with open(convergence_path) as f:
        report = json.load(f)
    for run in report["runs"]:
        if run["seed"] == seed:
            bd = run["best_config"]["biosensor_design"]
            me = run["best_config"].get("measurement_environment", {})
            cfg = {**bd, **me, "biosensor_type": bd.get("type", "array")}
            cfg["response_time_s"] = float(bd.get("response_time_s", 600.0))
            return cfg
    raise ValueError(f"Seed {seed} not found in {convergence_path}")


def _v6_score_with_concs(cfg: dict, perturbed_concs: dict, v6_obj, original_module) -> float:
    """Evaluate V6 with temporarily patched biomarker concentration table."""
    orig = original_module._NOMINAL_CONCS
    try:
        original_module._NOMINAL_CONCS = perturbed_concs
        score = float(v6_obj(cfg))
    finally:
        original_module._NOMINAL_CONCS = orig
    return score


def _R_surrogate(cfg: dict, concs: dict) -> float:
    """Langmuir occupancy ratio at given concentrations (analytical, no surrogate needed)."""
    kd      = float(cfg.get("kd_nm", 1.0))
    kd_ctx  = float(cfg.get("kd_ctx_nm", 1.0))
    kd_p1np = float(cfg.get("kd_p1np_nm", 1.0))
    w_ctx   = float(cfg.get("w_ctx", 0.1))
    w_p1np  = float(cfg.get("w_p1np", 0.1))
    w_scl   = max(0.0, 1.0 - w_ctx - w_p1np)

    def _occ(c, kd_v): return c / (kd_v + c + 1e-12)
    def _ratio(c_d, c_h, kd_v): return _occ(c_d, kd_v) / max(_occ(c_h, kd_v), 1e-12)

    h = _NOMINAL["healthy"]
    return (
        w_scl * _ratio(concs["scl"],  h["scl"],  kd)
        + w_ctx  * _ratio(concs["ctx"],  h["ctx"],  kd_ctx)
        + w_p1np * _ratio(concs["p1np"], h["p1np"], kd_p1np)
    )


def run_distribution_shift_test(
    cfg: dict,
    v6_obj,
    v6_module,
    loader,
    physics,
) -> Dict:
    """Execute all 7 perturbation scenarios. Return structured results."""
    results = {}

    # ── Test 1: Biomarker concentration shift ─────────────────────────────
    print("\n[1] Biomarker concentration shift (±10/20/30% uniform across all scenarios)")
    shifts = {
        "-30%": 0.70,
        "-20%": 0.80,
        "-10%": 0.90,
        "nominal": 1.00,
        "+10%": 1.10,
        "+20%": 1.20,
        "+30%": 1.30,
    }
    conc_shift_results: Dict[str, float] = {}
    for label, factor in shifts.items():
        perturbed = {
            sc: {mk: mv * factor for mk, mv in vals.items()}
            for sc, vals in _NOMINAL.items()
        }
        perturbed["healthy"] = dict(_NOMINAL["healthy"])  # don't shift healthy (calibration ref)
        score = _v6_score_with_concs(cfg, perturbed, v6_obj, v6_module)
        conc_shift_results[label] = score
        print(f"    {label:>8}: score={score:.4f}")

    nominal_score = conc_shift_results["nominal"]
    score_range   = max(conc_shift_results.values()) - min(conc_shift_results.values())
    robust_conc   = score_range < 0.05
    print(f"  Score range +-30%: {score_range:.4f}  -> {'ROBUST' if robust_conc else 'SENSITIVE'}")
    results["concentration_shift"] = {
        "scores": conc_shift_results,
        "nominal_score": nominal_score,
        "score_range_30pct": score_range,
        "robust": robust_conc,
    }

    # ── Test 2: Intra-patient variability scaling ──────────────────────────
    print("\n[2] Intra-patient variability scaling (σ multiplier 1× – 4×)")
    from BO.evaluation.therapeutic_objective_v6 import TherapeuticObjectiveV6

    class PerturbedLoader:
        def __init__(self, base, sigma_scale: float, rng):
            self.base = base
            self.sigma = sigma_scale * 0.05
            self.rng   = rng

        def predict(self, **kw):
            dr, fnr, ttd = self.base.predict(**kw)
            dr_p  = float(np.clip(dr  + self.rng.normal(0, self.sigma),         0, 1))
            fnr_p = float(np.clip(fnr + self.rng.normal(0, self.sigma),         0, 1))
            ttd_p = float(np.clip(ttd + self.rng.normal(0, self.sigma * ttd), 400, 9000))
            return dr_p, fnr_p, ttd_p

    variability_results: Dict[str, dict] = {}
    for scale in [1.0, 1.5, 2.0, 3.0, 4.0]:
        trial_scores = []
        rng = np.random.RandomState(42)
        for _ in range(25):
            perturbed_loader = PerturbedLoader(loader, scale, rng)
            obj = TherapeuticObjectiveV6(physics, perturbed_loader, apply_constraints=False)
            trial_scores.append(float(obj(cfg)))

        s = np.array(trial_scores)
        p5 = float(np.percentile(s, 5))
        label = "ROBUST" if s.std() < 0.05 and p5 > nominal_score * 0.85 else "SENSITIVE"
        print(f"    {scale:.1f}× variability: mean={s.mean():.4f}±{s.std():.4f}  p5={p5:.4f}  [{label}]")
        variability_results[f"{scale:.1f}x"] = {
            "mean": float(s.mean()), "std": float(s.std()), "p5": p5, "label": label,
        }
    results["variability_scaling"] = variability_results

    # ── Test 3: Sensor gain drift (sensitivity degradation) ───────────────
    print("\n[3] Sensor gain drift (sensitivity × factor; models implant aging)")
    nominal_sensitivity = float(cfg.get("sensitivity", 1.0))
    drift_results: Dict[str, dict] = {}
    for drift_pct in [100, 95, 90, 85, 80, 75, 70]:
        factor  = drift_pct / 100.0
        drifted = {**cfg, "sensitivity": nominal_sensitivity * factor}
        score_d = float(v6_obj(drifted))
        score_delta = score_d - nominal_score
        label = "OK" if abs(score_delta) < 0.05 else ("WARN" if abs(score_delta) < 0.10 else "FAIL")
        print(f"    Gain={drift_pct}%  sensitivity={nominal_sensitivity*factor:.2f}: "
              f"score={score_d:.4f}  Δ={score_delta:+.4f}  [{label}]")
        drift_results[f"{drift_pct}pct"] = {
            "sensitivity": float(nominal_sensitivity * factor),
            "score": float(score_d),
            "delta": float(score_delta),
            "label": label,
        }
    score_70 = drift_results["70pct"]["score"]
    print(f"  Score at 70% gain: {score_70:.4f} (nominal: {nominal_score:.4f}, "
          f"Δ={score_70-nominal_score:+.4f})")
    results["sensor_gain_drift"] = drift_results

    # ── Test 4 & 5: Missing biomarker channels ─────────────────────────────
    print("\n[4&5] Missing biomarker channels (sensor fouling)")
    missing_results: Dict[str, float] = {}

    # CTX channel fails
    cfg_no_ctx = {**cfg, "w_ctx": 0.0, "kd_ctx_nm": 1.0}
    score_no_ctx = float(v6_obj(cfg_no_ctx))
    delta_no_ctx = score_no_ctx - nominal_score
    print(f"    CTX channel failed (w_ctx=0):  score={score_no_ctx:.4f}  Δ={delta_no_ctx:+.4f}")
    missing_results["no_ctx"] = {"score": score_no_ctx, "delta": delta_no_ctx}

    # P1NP channel fails
    cfg_no_p1np = {**cfg, "w_p1np": 0.0, "kd_p1np_nm": 1.0}
    score_no_p1np = float(v6_obj(cfg_no_p1np))
    delta_no_p1np = score_no_p1np - nominal_score
    print(f"    P1NP channel failed (w_p1np=0): score={score_no_p1np:.4f}  Δ={delta_no_p1np:+.4f}")
    missing_results["no_p1np"] = {"score": score_no_p1np, "delta": delta_no_p1np}

    # Both CTX and P1NP fail (SOST only)
    cfg_sost_only = {**cfg, "w_ctx": 0.0, "w_p1np": 0.0, "kd_ctx_nm": 1.0, "kd_p1np_nm": 1.0}
    score_sost_only = float(v6_obj(cfg_sost_only))
    delta_sost_only = score_sost_only - nominal_score
    print(f"    SOST-only (both secondary fail): score={score_sost_only:.4f}  Δ={delta_sost_only:+.4f}")
    missing_results["sost_only"] = {"score": score_sost_only, "delta": delta_sost_only}
    results["missing_biomarker"] = missing_results

    # ── Test 6: Population shift — CKD-dominant ctx elevation ─────────────
    print("\n[6] Population shift: CKD-dominant (CTX +20% due to impaired renal excretion)")
    ckd_shift = {
        sc: {mk: mv * (1.20 if mk == "ctx" else 1.0) for mk, mv in vals.items()}
        for sc, vals in _NOMINAL.items()
    }
    score_ckd_shift = _v6_score_with_concs(cfg, ckd_shift, v6_obj, v6_module)
    delta_ckd_shift = score_ckd_shift - nominal_score
    print(f"    CTX +20% (CKD excretion effect): score={score_ckd_shift:.4f}  Δ={delta_ckd_shift:+.4f}")
    results["population_shift_ckd"] = {
        "score": float(score_ckd_shift),
        "delta": float(delta_ckd_shift),
        "description": "CTX 20% elevated due to impaired CKD renal excretion",
    }

    # ── Test 7: PMO-CKD mixed phenotype ───────────────────────────────────
    print("\n[7] PMO-CKD mixed phenotype (α×PMO + (1-α)×CKD biomarker profile)")
    pmo_ckd_mix_results: Dict[str, dict] = {}
    for alpha in [0.0, 0.25, 0.50, 0.75, 1.0]:
        # alpha=0 → pure PMO; alpha=1 → pure CKD
        pmo = _NOMINAL["pmo"]
        ckd = _NOMINAL["ckd_mbd"]
        mixed = {mk: (1-alpha)*pmo[mk] + alpha*ckd[mk] for mk in ("scl", "ctx", "p1np")}
        R = _R_surrogate(cfg, mixed)
        label = f"PMO×{1-alpha:.2f}+CKD×{alpha:.2f}"
        # Score using perturbed disease concentrations for a single "disease" scenario
        perturbed_pmo_concs = {**_NOMINAL, "pmo": mixed}
        score_mix = _v6_score_with_concs(cfg, perturbed_pmo_concs, v6_obj, v6_module)
        print(f"    {label:<20}  R={R:.4f}  score={score_mix:.4f}")
        pmo_ckd_mix_results[label] = {"alpha_ckd": alpha, "R": R, "score": float(score_mix)}
    results["pmo_ckd_mixed"] = pmo_ckd_mix_results

    return results


def main():
    parser = argparse.ArgumentParser(description="Distribution shift robustness test")
    parser.add_argument("--convergence-report", type=Path,
                        default=Path("BO/bo_results/convergence/convergence_report.json"))
    parser.add_argument("--surrogate-dir", type=Path, default=Path("BO/bo_results"))
    parser.add_argument("--seed-config",   type=int, default=888)
    parser.add_argument("--out", type=Path,
                        default=Path("BO/bo_results/diagnostics/distribution_shift_results.json"))
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    print("=" * 70)
    print("GENEVO2 — Distribution Shift Robustness Test")
    print("=" * 70)

    from BO.core.surrogate_loader import SurrogateLoaderV3
    from BO.evaluation.physics_forward_model import PhysicsForwardModel
    from BO.evaluation.therapeutic_objective_v6 import TherapeuticObjectiveV6
    import BO.evaluation.therapeutic_objective_v6 as v6_module

    loader  = SurrogateLoaderV3(args.surrogate_dir)
    physics = PhysicsForwardModel()
    v6_obj  = TherapeuticObjectiveV6(physics, loader)

    # Load config
    if args.convergence_report.exists():
        cfg = _load_best_cfg(args.convergence_report, args.seed_config)
        print(f"  Config: seed {args.seed_config}")
    else:
        print(f"[WARN] Convergence report not found. Using canonical fallback config.")
        cfg = {
            "kd_nm": 0.816, "kd_ctx_nm": 2.048, "kd_p1np_nm": 0.422,
            "sensitivity": 4.416, "w_ctx": 0.080, "w_p1np": 0.210,
            "biosensor_type": "array", "noise_preset": "realistic",
            "target_scenario": "pmo", "response_time_s": 600.0,
        }

    nominal_score = float(v6_obj(cfg))
    print(f"  Nominal V6 score: {nominal_score:.4f}")

    results = run_distribution_shift_test(cfg, v6_obj, v6_module, loader, physics)
    results["config_used"] = cfg
    results["nominal_score"] = nominal_score
    results["seed_config"] = args.seed_config

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DISTRIBUTION SHIFT SUMMARY")
    print("=" * 70)
    print(f"  Nominal score            : {nominal_score:.4f}")

    conc_range = results.get("concentration_shift", {}).get("score_range_30pct", 0)
    print(f"  ±30% conc shift range    : {conc_range:.4f}  "
          f"({'ROBUST' if conc_range < 0.05 else 'SENSITIVE'})")

    var_4x = results.get("variability_scaling", {}).get("4.0x", {})
    print(f"  4× variability mean      : {var_4x.get('mean', 0):.4f}  "
          f"(p5={var_4x.get('p5', 0):.4f})")

    drift_70 = results.get("sensor_gain_drift", {}).get("70pct", {})
    print(f"  Sensor at 70% gain       : {drift_70.get('score', 0):.4f}  "
          f"(Δ={drift_70.get('delta', 0):+.4f})")

    no_ctx = results.get("missing_biomarker", {}).get("no_ctx", {})
    no_p1np = results.get("missing_biomarker", {}).get("no_p1np", {})
    sost_only = results.get("missing_biomarker", {}).get("sost_only", {})
    print(f"  CTX channel fails        : {no_ctx.get('score', 0):.4f}  "
          f"(Δ={no_ctx.get('delta', 0):+.4f})")
    print(f"  P1NP channel fails       : {no_p1np.get('score', 0):.4f}  "
          f"(Δ={no_p1np.get('delta', 0):+.4f})")
    print(f"  SOST-only (both fail)    : {sost_only.get('score', 0):.4f}  "
          f"(Δ={sost_only.get('delta', 0):+.4f})")

    ckd_pop = results.get("population_shift_ckd", {})
    print(f"  CKD pop CTX+20%          : {ckd_pop.get('score', 0):.4f}  "
          f"(Δ={ckd_pop.get('delta', 0):+.4f})")

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n[OK] Results saved: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
