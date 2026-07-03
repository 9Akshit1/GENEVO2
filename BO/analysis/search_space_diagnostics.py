#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Search space boundary saturation diagnostics.

Analyses all 10 convergence seeds to detect:
  1. Which parameters cluster at or near the search-space boundary
  2. Whether boundary-hitting parameters are constrained by physics or by
     an artificial box that is too small (these have different fixes)
  3. What happens to the objective if bounds are extended (surrogate extrapolation)
  4. Parameter correlation structure across seeds

This answers the question from the audit:
  "BO repeatedly hits sensitivity=5.0, kd_ctx=10.0, kd_p1np=0.1 — is the
   physics pushing there, or is the search space too small?"

Usage:
    python BO/analysis/search_space_diagnostics.py
    python BO/analysis/search_space_diagnostics.py --bound-extension-factor 2.0

Output:
    BO/bo_results/diagnostics/search_space_diagnostics.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

# Current search space bounds
BOUNDS = {
    "kd_nm":       (0.1,  10.0,  "log"),
    "kd_ctx_nm":   (0.1,  10.0,  "log"),
    "kd_p1np_nm":  (0.1,  10.0,  "log"),
    "sensitivity": (0.5,   5.0,  "log"),
    "w_ctx":       (0.01,  0.49, "linear"),
    "w_p1np":      (0.01,  0.49, "linear"),
    "response_time_s": (100.0, 3600.0, "log"),
}

BOUND_TOLERANCE = 0.05  # fraction of range at which a value is considered "at bound"

FIXED_FOR_EXTRAPOLATION = {
    "biosensor_type":  "array",
    "noise_preset":    "realistic",
    "target_scenario": "pmo",
}


def _near_bound(val: float, lo: float, hi: float, scale: str, tol: float = BOUND_TOLERANCE) -> str:
    """Return 'lower', 'upper', 'interior', or 'at_lower', 'at_upper'."""
    if scale == "log":
        log_lo, log_hi = np.log10(lo), np.log10(hi)
        log_val = np.log10(max(val, 1e-12))
        frac = (log_val - log_lo) / (log_hi - log_lo + 1e-12)
    else:
        frac = (val - lo) / (hi - lo + 1e-12)

    if frac <= 0.001:
        return "AT_LOWER"
    if frac >= 0.999:
        return "AT_UPPER"
    if frac < tol:
        return "near_lower"
    if frac > (1.0 - tol):
        return "near_upper"
    return "interior"


def _boundary_extension_scan(
    param_name: str,
    base_cfg: dict,
    lo_ext: float,
    hi_ext: float,
    scale: str,
    n_points: int,
    objective_fn,
) -> Dict:
    """Vary param_name over [lo_ext, hi_ext] with all others fixed. Return scores."""
    if scale == "log":
        values = list(np.logspace(np.log10(lo_ext), np.log10(hi_ext), n_points))
    else:
        values = list(np.linspace(lo_ext, hi_ext, n_points))

    scores = []
    for v in values:
        cfg = {**base_cfg, param_name: float(v)}
        # Recompute response_time as fixed if not the target param
        if param_name != "response_time_s":
            cfg["response_time_s"] = 600.0
        try:
            scores.append(float(objective_fn(cfg)))
        except Exception:
            scores.append(float("nan"))

    return {
        "param": param_name,
        "values": [float(v) for v in values],
        "scores": scores,
        "lo_ext": lo_ext,
        "hi_ext": hi_ext,
        "optimal_value": float(values[int(np.nanargmax(scores))]),
        "optimal_score": float(np.nanmax(scores)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Search space boundary saturation diagnostics"
    )
    parser.add_argument(
        "--convergence-report", type=Path,
        default=Path("BO/bo_results/convergence/convergence_report.json"),
    )
    parser.add_argument("--surrogate-dir", type=Path, default=Path("BO/bo_results"))
    parser.add_argument("--bound-extension-factor", type=float, default=2.0,
                        help="Extend upper bounds by this factor for extrapolation scan (default: 2.0)")
    parser.add_argument("--n-points-scan", type=int, default=20,
                        help="Points per parameter in extension scan (default: 20)")
    parser.add_argument("--out", type=Path,
                        default=Path("BO/bo_results/diagnostics/search_space_diagnostics.json"))
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    print("=" * 70)
    print("GENEVO2 — Search Space Boundary Saturation Diagnostics")
    print("=" * 70)

    # Load all seed results
    if not args.convergence_report.exists():
        print(f"[ERROR] {args.convergence_report} not found")
        return 1

    with open(args.convergence_report) as f:
        report = json.load(f)

    # ── Part 1: Boundary hit analysis across all seeds ──────────────────────
    print("\n[1/3] Boundary hit analysis across all convergence seeds\n")

    seeds_data: Dict[str, Dict] = {}
    all_values: Dict[str, List[float]] = {p: [] for p in BOUNDS}

    for run in report["runs"]:
        seed = run["seed"]
        bd   = run["best_config"]["biosensor_design"]
        me   = run["best_config"].get("measurement_environment", {})
        score = run["best_score"]

        row = {}
        for param, (lo, hi, scale) in BOUNDS.items():
            if param == "noise_preset":
                continue
            val = bd.get(param) or me.get(param)
            if val is None:
                continue
            val = float(val)
            status = _near_bound(val, lo, hi, scale)
            row[param] = {"value": val, "status": status}
            all_values[param].append(val)

        seeds_data[str(seed)] = {"score": score, "params": row}

    # Summary: fraction of seeds at each bound per parameter
    print(f"  {'Parameter':<18}  {'Range':>20}  {'Mean':>8}  {'Std':>8}  "
          f"{'At lower':>9}  {'At upper':>9}  {'Near bound':>11}")
    print(f"  {'-'*18}  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*11}")

    boundary_summary: Dict[str, dict] = {}

    for param, (lo, hi, scale) in BOUNDS.items():
        vals = all_values.get(param, [])
        if not vals:
            continue

        statuses = []
        for run in report["runs"]:
            bd = run["best_config"]["biosensor_design"]
            me = run["best_config"].get("measurement_environment", {})
            val = bd.get(param) or me.get(param)
            if val is not None:
                statuses.append(_near_bound(float(val), lo, hi, scale))

        n = len(statuses)
        n_at_lower = sum(1 for s in statuses if s in ("AT_LOWER", "near_lower"))
        n_at_upper = sum(1 for s in statuses if s in ("AT_UPPER", "near_upper"))
        n_bound    = n_at_lower + n_at_upper

        frac_lower = n_at_lower / max(n, 1)
        frac_upper = n_at_upper / max(n, 1)
        frac_bound = n_bound / max(n, 1)

        vals_arr = np.array(vals)
        range_str = f"[{lo:.2f}, {hi:.2f}]" if scale == "linear" else f"log[{lo},{hi}]"

        print(f"  {param:<18}  {range_str:>20}  {vals_arr.mean():>8.3f}  {vals_arr.std():>8.3f}  "
              f"{frac_lower:>9.0%}  {frac_upper:>9.0%}  {frac_bound:>11.0%}")

        boundary_summary[param] = {
            "mean": float(vals_arr.mean()),
            "std":  float(vals_arr.std()),
            "frac_at_lower": float(frac_lower),
            "frac_at_upper": float(frac_upper),
            "frac_near_bound": float(frac_bound),
            "n_seeds": n,
        }

    # ── Part 2: Physics vs box interpretation ──────────────────────────────
    print("\n[2/3] Physics interpretation of boundary hits\n")

    interpretations: Dict[str, str] = {}

    # Sensitivity at upper bound (5.0)
    sens_at_upper = boundary_summary.get("sensitivity", {}).get("frac_at_upper", 0)
    if sens_at_upper > 0.3:
        interp = (
            "Physics-driven: higher sensitivity always helps until noise floor. "
            "Fabrication constraints (electrode fouling, drift) justify upper bound, "
            "not physics. Consider extending to 10.0 and adding noise floor penalty."
        )
        print(f"  sensitivity : {interp}")
        interpretations["sensitivity"] = interp

    # kd_ctx near upper bound (10.0)
    kd_ctx_vals = all_values.get("kd_ctx_nm", [])
    if kd_ctx_vals:
        kd_ctx_mean = np.mean(kd_ctx_vals)
        if kd_ctx_mean > 5.0:
            interp = (
                f"kd_ctx mean={kd_ctx_mean:.2f} nM → high kd means low affinity for CTX. "
                "This is physically meaningful: a low-affinity CTX channel avoids cross-saturation "
                "and lets the SOST channel dominate detection. The bound at 10 nM may cut off the "
                "true optimum. Extend to 20 nM and re-run."
            )
        else:
            interp = (
                f"kd_ctx mean={kd_ctx_mean:.2f} nM — within bounds, not constrained."
            )
        print(f"  kd_ctx_nm   : {interp}")
        interpretations["kd_ctx_nm"] = interp

    # kd_p1np near lower bound (0.1)
    kd_p1np_vals = all_values.get("kd_p1np_nm", [])
    if kd_p1np_vals:
        n_at_lower = sum(1 for v in kd_p1np_vals if v <= 0.15)
        if n_at_lower >= 3:
            interp = (
                f"{n_at_lower}/{len(kd_p1np_vals)} seeds have kd_p1np ≤ 0.15 nM (at lower bound). "
                "High P1NP affinity (low Kd) is physically beneficial: P1NP shows strongest absolute "
                "fold-change in early PMO. Current lower bound at 0.1 nM may cut off the optimum. "
                "Extend to 0.01 nM (requires aptamer engineering) or document constraint explicitly."
            )
            print(f"  kd_p1np_nm  : {interp}")
            interpretations["kd_p1np_nm"] = interp

    # response_time at upper bound (600s)
    rt_vals = all_values.get("response_time_s", [])
    if rt_vals:
        n_at_upper_rt = sum(1 for v in rt_vals if v >= 590.0)
        if n_at_upper_rt > len(rt_vals) * 0.5:
            interp = (
                f"{n_at_upper_rt}/{len(rt_vals)} seeds have response_time ≥ 590s. "
                "Longer response time gives the sensor more signal integration time. "
                "Upper bound at 600s (10 min) is clinically motivated — acceptable implant "
                "response window. This boundary is a design constraint, not a box artifact."
            )
            print(f"  response_time: {interp}")
            interpretations["response_time_s"] = interp

    # ── Part 3: Extended-bounds extrapolation scan ──────────────────────────
    print(f"\n[3/3] Extended-bounds extrapolation scan (factor={args.bound_extension_factor}×)\n")

    from BO.core.surrogate_loader import SurrogateLoaderV3
    from BO.evaluation.physics_forward_model import PhysicsForwardModel
    from BO.evaluation.therapeutic_objective_v6 import TherapeuticObjectiveV6

    loader  = SurrogateLoaderV3(args.surrogate_dir)
    physics = PhysicsForwardModel()
    v6      = TherapeuticObjectiveV6(physics, loader)

    # Use best seed (888) as base config
    best_seed_cfg = None
    for run in report["runs"]:
        if run["seed"] == 888 or run["seed"] == report.get("best_run", {}).get("seed"):
            bd = run["best_config"]["biosensor_design"]
            me = run["best_config"].get("measurement_environment", {})
            best_seed_cfg = {**bd, **me, "biosensor_type": bd.get("type", "array")}
            best_seed_cfg["response_time_s"] = float(bd.get("response_time_s", 600.0))
            break

    if best_seed_cfg is None:
        print("  [WARN] Could not load seed 888 config for extrapolation scan.")
        extrapolation_results = {}
    else:
        extrapolation_results = {}
        params_to_scan = [
            ("sensitivity", 0.5,   5.0 * args.bound_extension_factor, "log"),
            ("kd_ctx_nm",   0.1,  10.0 * args.bound_extension_factor, "log"),
            ("kd_p1np_nm",  0.01,  10.0,                              "log"),
        ]

        for param_name, lo_ext, hi_ext, scale in params_to_scan:
            print(f"  Scanning {param_name} in [{lo_ext:.3f}, {hi_ext:.3f}] ({args.n_points_scan} pts) ...",
                  flush=True)
            scan = _boundary_extension_scan(
                param_name, best_seed_cfg, lo_ext, hi_ext, scale,
                args.n_points_scan, v6,
            )
            extrapolation_results[param_name] = scan

            # Identify current-bound score vs optimal in extended range
            lo_curr, hi_curr = BOUNDS.get(param_name, (lo_ext, hi_ext, scale))[:2]
            score_at_upper_curr = None
            for v, s in zip(scan["values"], scan["scores"]):
                if abs(v - hi_curr) / (hi_curr + 1e-12) < 0.05:
                    score_at_upper_curr = s
                    break

            print(f"    Optimal in extended range: {param_name}={scan['optimal_value']:.4f} → score={scan['optimal_score']:.4f}")
            if score_at_upper_curr:
                gain = scan["optimal_score"] - score_at_upper_curr
                print(f"    Gain vs current upper bound: {gain:+.4f}")
                if gain > 0.01:
                    print(f"    *** Extending {param_name} upper bound to {hi_ext:.1f} may improve score by {gain:.4f} ***")
                else:
                    print(f"    Current bound sufficient (gain < 0.01)")

    # ── Save results ───────────────────────────────────────────────────────
    output = {
        "boundary_summary":       boundary_summary,
        "physics_interpretations": interpretations,
        "extrapolation_scans":    extrapolation_results,
        "seeds_data":             seeds_data,
        "recommendations": _build_recommendations(boundary_summary, interpretations),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\n[OK] Diagnostics saved: {args.out}")

    # Print recommendations
    recs = output["recommendations"]
    if recs:
        print("\nRECOMMENDATIONS:")
        for rec in recs:
            print(f"  - {rec}")
    return 0


def _build_recommendations(boundary_summary: dict, interpretations: dict) -> List[str]:
    recs = []

    kd_ctx = boundary_summary.get("kd_ctx_nm", {})
    if kd_ctx.get("mean", 0) > 5.0:
        recs.append(
            "kd_ctx upper bound (10 nM) may be binding: mean of optimal configs = "
            f"{kd_ctx.get('mean', 0):.2f} nM. Extend to 20 nM and re-run BO."
        )

    kd_p1np = boundary_summary.get("kd_p1np_nm", {})
    if kd_p1np.get("frac_at_lower", 0) > 0.3:
        recs.append(
            f"kd_p1np lower bound (0.1 nM) hit by {kd_p1np.get('frac_at_lower', 0):.0%} of seeds. "
            "Extend lower bound to 0.01 nM if sub-0.1 nM aptamers are fabricable."
        )

    sens = boundary_summary.get("sensitivity", {})
    if sens.get("frac_at_upper", 0) > 0.5:
        recs.append(
            f"sensitivity hits upper bound (5.0) in {sens.get('frac_at_upper', 0):.0%} of seeds. "
            "Consider extending to 10.0 with a fabrication penalty term in the objective."
        )

    rt = boundary_summary.get("response_time_s", {})
    if rt.get("frac_at_upper", 0) > 0.5:
        recs.append(
            "response_time_s consistently at 600s upper bound. This is the clinically-defined "
            "maximum implant response window — do NOT extend. Constraint is scientifically justified."
        )

    if not recs:
        recs.append("No clear bound violations detected at this threshold. Search space appears adequate.")

    return recs


if __name__ == "__main__":
    sys.exit(main())
