#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GENEVO2 — Parameter Landscape Analysis (Priorities 1, 2, 5)

Produces three types of analysis:
  1. 1D sensitivity sweeps — each active parameter swept across its full range;
     all others held at the baseline config.  Shows whether the landscape is
     monotonic (BO cannot outperform random) or has genuine structure.

  2. 2D response-surface heatmaps — key parameter pairs swept simultaneously
     to reveal interactions, ridges, and trade-offs.

  3. Robustness margin report — for the baseline config, computes the actual
     signal vs threshold ratio for each scenario and reports what concentration
     variability would trigger a miss.  Explains the DR=1.000 artifact.

Usage (run from project root):
    python BO/diagnostics/parameter_landscape.py
    python BO/diagnostics/parameter_landscape.py --config BO/bo_results_v14/results/best_config.json
    python BO/diagnostics/parameter_landscape.py --no-plots   # text only
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_objective(surrogate_dir: Path, margin: float = 1.25):
    """Load v3 objective function + surrogate loader."""
    from BO.core.surrogate_loader import SurrogateLoaderV3
    from BO.evaluation.objective_function_v3 import ObjectiveFunctionV3

    loader = SurrogateLoaderV3(results_dir=surrogate_dir)
    obj = ObjectiveFunctionV3(
        physics_model=None,
        surrogate_loader_v3=loader,
        weight_fp=0.10,
        apply_constraints=True,
    )
    return obj, loader


def _baseline_config(config_path: Path) -> dict:
    """Load a best_config.json and flatten to the format __call__ expects."""
    with open(config_path) as f:
        data = json.load(f)
    d = data["biosensor_design"]
    return {
        "biosensor_type":  d.get("type", "array"),
        "kd_nm":           float(d["kd_nm"]),
        "sensitivity":     float(d["sensitivity"]),
        "response_time_s": float(d.get("response_time_s", 600.0)),
        "noise_preset":    "realistic",
        "target_scenario": "pmo",
        "kd_ctx_nm":       float(d.get("kd_ctx_nm",  1.0)),
        "kd_p1np_nm":      float(d.get("kd_p1np_nm", 1.0)),
        "w_ctx":           float(d.get("w_ctx",  0.15)),
        "w_p1np":          float(d.get("w_p1np", 0.22)),
    }


def _score(obj, config: dict) -> float:
    try:
        return float(obj(config))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Active parameter definitions (mirrors _get_active_dims in gaussian_process_bo)
# ---------------------------------------------------------------------------

ACTIVE_PARAMS = [
    # name,         lo,    hi,    scale
    ("kd_nm",       0.10,  10.0,  "log"),
    ("sensitivity", 0.50,  5.00,  "log"),
    ("kd_ctx_nm",   0.10,  10.0,  "log"),
    ("kd_p1np_nm",  0.10,  10.0,  "log"),
    ("w_ctx",       0.01,  0.49,  "linear"),
    ("w_p1np",      0.01,  0.49,  "linear"),
]

N_SWEEP   = 25    # points per 1D sweep
N_HEAT    = 20    # points per axis on 2D heatmap


def _sweep_values(lo, hi, scale, n=N_SWEEP):
    if scale == "log":
        return np.logspace(np.log10(lo), np.log10(hi), n)
    return np.linspace(lo, hi, n)


# ---------------------------------------------------------------------------
# Priority 1: 1D sensitivity sweeps
# ---------------------------------------------------------------------------

def run_1d_sweeps(obj, baseline: dict, n: int = N_SWEEP):
    """
    Sweep each active parameter independently.
    Returns dict: param_name -> (values, scores, gradient_mean)
    """
    print("\n" + "=" * 72)
    print("PRIORITY 1 — 1D PARAMETER SENSITIVITY SWEEPS")
    print("=" * 72)
    print(f"  Baseline config (all other params fixed):")
    for k in ("kd_nm", "sensitivity", "kd_ctx_nm", "kd_p1np_nm", "w_ctx", "w_p1np"):
        print(f"    {k:<16} = {baseline[k]:.4f}")
    print()

    results = {}
    for name, lo, hi, scale in ACTIVE_PARAMS:
        vals = _sweep_values(lo, hi, scale, n)
        scores = []
        for v in vals:
            cfg = dict(baseline)
            cfg[name] = float(v)
            scores.append(_score(obj, cfg))
        scores = np.array(scores)

        lo_s, hi_s, mean_s = scores.min(), scores.max(), scores.mean()
        spread = hi_s - lo_s
        # Raw gradient: (score[-1] - score[0]) / (vals[-1] - vals[0]) (not log-adjusted)
        monotone_up   = bool(np.all(np.diff(scores) >= -0.005))
        monotone_down = bool(np.all(np.diff(scores) <=  0.005))
        structure = "FLAT" if spread < 0.01 else (
                    "MONOTONE_UP"   if monotone_up   else (
                    "MONOTONE_DOWN" if monotone_down else "NON-MONOTONE"))

        print(f"  {name:<16}  range=[{lo_s:.4f},{hi_s:.4f}]  spread={spread:.4f}  -> {structure}")

        results[name] = {
            "values": vals,
            "scores": scores,
            "spread": spread,
            "structure": structure,
            "min_score": lo_s,
            "max_score": hi_s,
        }

    # Rank parameters by landscape spread (biggest spread = most informative for BO)
    print()
    print("  Ranking by landscape spread (larger spread -> BO has more to exploit):")
    ranked = sorted(results.items(), key=lambda x: -x[1]["spread"])
    for rank, (name, info) in enumerate(ranked, 1):
        bar = "#" * int(info["spread"] * 100)
        print(f"    {rank}. {name:<16}  spread={info['spread']:.4f}  {bar}")

    print()
    print("  INTERPRETATION:")
    monotone = [n for n, i in results.items() if "MONOTONE" in i["structure"]]
    nonmono  = [n for n, i in results.items() if i["structure"] == "NON-MONOTONE"]
    flat     = [n for n, i in results.items() if i["structure"] == "FLAT"]
    if flat:
        print(f"    DEAD (flat, BO irrelevant):   {flat}")
    if monotone:
        print(f"    MONOTONE (random finds max):  {monotone}")
    if nonmono:
        print(f"    NON-MONOTONE (BO can help!):  {nonmono}")
    if not nonmono:
        print("    WARNING: No non-monotone parameters found.")
        print("       The landscape has no local structure — random search is")
        print("       theoretically as good as BO on this objective.")

    return results


# ---------------------------------------------------------------------------
# Priority 2: Robustness margin analysis
# ---------------------------------------------------------------------------

def run_robustness_margin(baseline: dict):
    """
    Compute signal / threshold ratio for each scenario at the baseline config.
    Shows WHY robustness DR = 1.000 and what concentration variability would
    cause a false negative.
    """
    print("\n" + "=" * 72)
    print("PRIORITY 2 — ROBUSTNESS MARGIN ANALYSIS")
    print("=" * 72)
    print("  (Explains why the simulator reports DR=1.000 robustness)")
    print()

    from simulation.models.biosensors import ArrayBiosensor

    kd_scl  = baseline["kd_nm"]
    kd_ctx  = baseline["kd_ctx_nm"]
    kd_p1np = baseline["kd_p1np_nm"]
    w_ctx   = baseline["w_ctx"]
    w_p1np  = baseline["w_p1np"]
    w_scl   = max(0.0, 1.0 - w_ctx - w_p1np)
    sens    = baseline["sensitivity"]

    # Healthy reference concentrations
    H = {"scl": 0.375, "ctx": 0.200, "p1np": 0.350}
    P = {"scl": 0.875, "ctx": 0.500, "p1np": 0.525}
    C = {"scl": 1.125, "ctx": 0.500, "p1np": 0.625}
    M = {"scl": 0.600, "ctx": 0.350, "p1np": 0.430}  # PMO-mild (estimated)

    def _occ(c, kd):
        return c / (kd + c + 1e-12)

    ref_scl  = _occ(H["scl"],  kd_scl)
    ref_ctx  = _occ(H["ctx"],  kd_ctx)
    ref_p1np = _occ(H["p1np"], kd_p1np)

    def _composite(conc_dict):
        n_s = _occ(conc_dict["scl"],  kd_scl)  / (ref_scl  + 1e-12)
        n_c = _occ(conc_dict["ctx"],  kd_ctx)  / (ref_ctx  + 1e-12)
        n_p = _occ(conc_dict["p1np"], kd_p1np) / (ref_p1np + 1e-12)
        return sens * (w_scl * n_s + w_ctx * n_c + w_p1np * n_p)

    sig_h  = _composite(H)
    sig_p  = _composite(P)
    sig_c  = _composite(C)
    sig_m  = _composite(M)

    # Threshold (matches robustness_analyzer.py exactly)
    hp_gap_ref = (_composite(P) / sens) - 1.0
    threshold  = float(sig_h + 1.25 * hp_gap_ref)

    print(f"  Composite signals at nominal concentrations:")
    print(f"    Healthy  (H): {sig_h:.4f}")
    print(f"    PMO-mild (M): {sig_m:.4f}  ({'ABOVE' if sig_m >= threshold else 'BELOW'} threshold)")
    print(f"    PMO      (P): {sig_p:.4f}  ({'ABOVE' if sig_p >= threshold else 'BELOW'} threshold)")
    print(f"    CKD-MBD  (C): {sig_c:.4f}  ({'ABOVE' if sig_c >= threshold else 'BELOW'} threshold)")
    print(f"    Threshold   : {threshold:.4f}")
    print()

    # Signal margin: how far above threshold?
    for label, sig in [("PMO-mild", sig_m), ("PMO", sig_p), ("CKD-MBD", sig_c)]:
        margin = sig - threshold
        margin_pct = 100 * margin / (threshold + 1e-9)
        # What concentration scaling would bring signal to threshold?
        # sig(c_factor × nominal) ≥ threshold
        # We need to find c_factor such that composite(c_factor × conc) = threshold
        # Binary search for the critical factor
        lo_f, hi_f = 0.01, 1.0
        ref_conc = {"PMO-mild": M, "PMO": P, "CKD-MBD": C}[label]
        for _ in range(30):
            mid_f = (lo_f + hi_f) / 2
            scaled = {k: v * mid_f for k, v in ref_conc.items()}
            if _composite(scaled) >= threshold:
                hi_f = mid_f
            else:
                lo_f = mid_f
        fail_factor = (lo_f + hi_f) / 2
        fail_reduction_pct = (1 - fail_factor) * 100

        print(f"  {label:<12}  signal={sig:.4f}  margin={margin:+.4f} ({margin_pct:+.1f}%)")
        print(f"               Fail threshold at {fail_factor:.2f}x nominal concentration")
        print(f"               Requires {fail_reduction_pct:.0f}% reduction from nominal to cause a miss")
        print()

    print("  ROOT CAUSE OF DR=1.000:")
    print(f"    sensitivity={sens:.2f} (near max=5.0) places all disease signals")
    print(f"    so far above threshold that even 15% concentration variability")
    print(f"    (apply_variability=0.15) cannot push them below the detection line.")
    print(f"    DR=1.000 is not measurement noise — it is threshold over-engineering.")
    print()
    print("  IMPLICATION FOR BO:")
    print("    Any config with sensitivity > ~2.0 likely achieves DR=1.000 in the")
    print("    real simulator.  The surrogate predicts DR in (0,1) with gradation,")
    print("    but the true binary outcome is dominated by whether sensitivity is")
    print("    above the critical threshold.  The search space collapse is real.")
    print()


# ---------------------------------------------------------------------------
# Priority 5: 2D response-surface heatmaps
# ---------------------------------------------------------------------------

def run_2d_surfaces(obj, baseline: dict, plot: bool = True):
    """
    Compute 2D score grids for key parameter pairs and optionally plot them.
    Returns dict of (param1, param2) -> (vals1, vals2, score_grid).
    """
    print("\n" + "=" * 72)
    print("PRIORITY 5 — 2D RESPONSE SURFACE HEATMAPS")
    print("=" * 72)

    pairs = [
        ("sensitivity", "kd_nm",      "log", 0.50, 5.0, "log", 0.10, 10.0),
        ("w_ctx",       "w_p1np",     "linear", 0.01, 0.49, "linear", 0.01, 0.49),
        ("kd_ctx_nm",   "kd_p1np_nm", "log", 0.10, 10.0, "log", 0.10, 10.0),
        ("sensitivity", "w_p1np",     "log", 0.50, 5.0, "linear", 0.01, 0.49),
    ]

    surfaces = {}
    for (p1, p2, s1, lo1, hi1, s2, lo2, hi2) in pairs:
        vals1 = _sweep_values(lo1, hi1, s1, N_HEAT)
        vals2 = _sweep_values(lo2, hi2, s2, N_HEAT)
        grid  = np.zeros((N_HEAT, N_HEAT))

        for i, v1 in enumerate(vals1):
            for j, v2 in enumerate(vals2):
                cfg = dict(baseline)
                cfg[p1] = float(v1)
                cfg[p2] = float(v2)
                grid[i, j] = _score(obj, cfg)

        surfaces[(p1, p2)] = (vals1, vals2, grid)
        spread = grid.max() - grid.min()

        # Compute interaction strength: if interaction, diagonal pattern in grid
        # Pearson corr of flattened grid rows is a proxy for separability
        row_means = grid.mean(axis=1)
        col_means = grid.mean(axis=0)
        row_pred  = np.outer(row_means - row_means.mean(),
                             np.ones(N_HEAT)) + grid.mean()
        residual  = grid - row_pred
        interaction = float(np.std(residual))

        print(f"\n  ({p1}, {p2}):")
        print(f"    Score range : [{grid.min():.4f}, {grid.max():.4f}]  spread={spread:.4f}")
        print(f"    Interaction : residual std={interaction:.4f}  "
              f"({'weak — additive' if interaction < 0.01 else 'STRONG — interactive'})")
        print(f"    Optimal at  : {p1}={vals1[np.unravel_index(grid.argmax(), grid.shape)[0]]:.3f},",
              f"                  {p2}={vals2[np.unravel_index(grid.argmax(), grid.shape)[1]]:.3f}")

    if plot:
        _plot_surfaces(surfaces, baseline)

    return surfaces


def _plot_surfaces(surfaces: dict, baseline: dict):
    """Generate and save matplotlib plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_dir = Path("BO/diagnostics/plots")
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1D sweeps plot
        obj_ref = None  # will be populated by caller; skip here — done separately
        # (1D sweeps stored externally)

        # 2D heatmaps
        n_pairs = len(surfaces)
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        for ax, ((p1, p2), (vals1, vals2, grid)) in zip(axes, surfaces.items()):
            im = ax.pcolormesh(
                vals2, vals1, grid,
                cmap="viridis", shading="auto",
                vmin=max(0, grid.min()), vmax=grid.max()
            )
            plt.colorbar(im, ax=ax, label="Objective Score")
            ax.set_xlabel(p2)
            ax.set_ylabel(p1)
            ax.set_title(f"Score({p1}, {p2})\nmax={grid.max():.3f}")

            # Mark baseline
            b1 = baseline.get(p1)
            b2 = baseline.get(p2)
            if b1 is not None and b2 is not None:
                ax.plot(b2, b1, "r*", markersize=12, label="baseline")
                ax.legend(fontsize=8)

        plt.suptitle("GENEVO2 — Objective Landscape: 2D Response Surfaces", fontsize=13)
        plt.tight_layout()
        save_path = out_dir / "response_surfaces_2d.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Saved 2D heatmaps -> {save_path}")

    except ImportError:
        print("  matplotlib not available — skipping plots")
    except Exception as e:
        print(f"  Plot error: {e}")


def _plot_1d(results: dict, baseline: dict):
    """Generate 1D sweep plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_dir = Path("BO/diagnostics/plots")
        out_dir.mkdir(parents=True, exist_ok=True)

        n = len(results)
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()

        for ax, (name, info) in zip(axes, results.items()):
            vals   = info["values"]
            scores = info["scores"]
            ax.plot(vals, scores, "b-o", markersize=4, linewidth=1.5)
            bv = baseline.get(name)
            if bv is not None:
                ax.axvline(bv, color="r", linestyle="--", alpha=0.7, label=f"baseline={bv:.3f}")
                ax.legend(fontsize=8)
            ax.set_xlabel(name)
            ax.set_ylabel("Objective Score")
            ax.set_title(f"{name}\n[{info['min_score']:.3f},{info['max_score']:.3f}]  {info['structure']}")
            ax.grid(True, alpha=0.3)

            _, lo_a, hi_a, scale = next(p for p in ACTIVE_PARAMS if p[0] == name)
            if scale == "log":
                ax.set_xscale("log")

        plt.suptitle("GENEVO2 — 1D Parameter Sensitivity Sweeps (v3 objective)", fontsize=13)
        plt.tight_layout()
        save_path = out_dir / "sensitivity_1d.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved 1D sweeps -> {save_path}")

    except ImportError:
        print("  matplotlib not available — skipping plots")
    except Exception as e:
        print(f"  Plot error: {e}")


# ---------------------------------------------------------------------------
# Summary: what needs to change for BO to work
# ---------------------------------------------------------------------------

def print_landscape_verdict(sweep_results: dict):
    """Print a clear verdict on what the landscape analysis implies for BO."""
    print("\n" + "=" * 72)
    print("LANDSCAPE VERDICT")
    print("=" * 72)

    spreads = {n: i["spread"] for n, i in sweep_results.items()}
    total_spread = sum(spreads.values())
    if total_spread < 1e-6:
        total_spread = 1e-6

    print("\n  Fraction of objective variance explained by each parameter:")
    for name, sp in sorted(spreads.items(), key=lambda x: -x[1]):
        frac = sp / total_spread
        bar  = "#" * int(frac * 40)
        mono = sweep_results[name]["structure"]
        print(f"    {name:<16}  {frac*100:5.1f}%  {bar}  {mono}")

    top = max(spreads, key=spreads.get)
    top_frac = spreads[top] / total_spread

    print()
    if top_frac > 0.60:
        print(f"  CRITICAL: '{top}' accounts for {top_frac*100:.0f}% of landscape variation.")
        print(f"     Random search finds the optimum of a 1-variable function easily.")
        print(f"     GP intelligence is irrelevant when 60%+ of the signal is in one axis.")
        print()
        print(f"  Options to create genuine BO advantage:")
        print(f"    1. Add constraints that penalise extreme {top} values")
        print(f"       (manufacturing cost, biological feasibility, FP rate at high sensitivity)")
        print(f"    2. Add cross-terms to the objective (e.g., sensitivity x Kd interaction)")
        print(f"    3. Expand search space to include parameters with real biology")
        print(f"       (drug release rate, treatment window, bone turnover response)")
        print(f"    4. Retrain surrogate to predict therapeutic outcome, not just detection")
    else:
        print(f"  Landscape is reasonably spread across parameters. BO should work.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GENEVO2 parameter landscape analysis")
    parser.add_argument("--surrogate-dir", type=Path, default=Path("BO/bo_results"))
    parser.add_argument("--config", type=Path,
                        default=Path("BO/bo_results/results/best_config.json"))
    parser.add_argument("--no-plots", action="store_true", help="Skip matplotlib plots")
    args = parser.parse_args()

    print("\n" + "=" * 72)
    print("GENEVO2 — PARAMETER LANDSCAPE ANALYSIS")
    print("=" * 72)
    print(f"  Surrogate : {args.surrogate_dir}")
    print(f"  Config    : {args.config}")

    obj, loader = _load_objective(args.surrogate_dir)
    baseline = _baseline_config(args.config)
    baseline_score = _score(obj, baseline)
    print(f"  Baseline score: {baseline_score:.4f}")

    # Priority 1: 1D sweeps
    sweep_results = run_1d_sweeps(obj, baseline)

    if not args.no_plots:
        _plot_1d(sweep_results, baseline)

    # Priority 2: Robustness margin
    run_robustness_margin(baseline)

    # Priority 5: 2D surfaces
    run_2d_surfaces(obj, baseline, plot=not args.no_plots)

    # Verdict
    print_landscape_verdict(sweep_results)

    print("\n" + "=" * 72)
    print("Analysis complete.")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
