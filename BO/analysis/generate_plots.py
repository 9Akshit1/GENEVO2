#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate publication-quality figures from existing BO result JSON files.

Reads JSON files already on disk and produces PNG figures without
re-running any simulation or surrogate evaluation.

Figures produced:
  1. kd_ctx_scan_plot.png       - DR per scenario vs K_{d,CTX}
  2. benchmark_boxplot.png      - Composite score distribution per optimizer
  3. sobol_barplot.png          - First-order and total Sobol indices
  4. realsim_validation.png     - Surrogate vs real-sim DR per scenario
  5. convergence_summary.png    - Final score distribution across 10 seeds
  6. langmuir_isotherms.png     - Binding isotherms for each aptamer channel
  7. robustness_analysis.png    - Missing-channel and gain-drift sensitivity
  8. closed_loop_learning.png   - Surrogate score and real DR across AL rounds
  9. scenario_rank_rho.png      - Per-scenario surrogate rank correlation
 10. [RETIRED] patient_subtypes.png — patient subtype analysis removed (see RESULTS_AUDIT.md §14)
 11. mobo_pareto_front.png      - Multi-objective Pareto front 2D projections
 12. mobo_hypervolume.png       - MOBO hypervolume convergence curve
 13. concentration_shift.png    - Score vs biomarker concentration perturbation

Usage:
    python BO/analysis/generate_plots.py
    python BO/analysis/generate_plots.py --out-dir BO/bo_results/diagnostics/plots
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "BO" / "bo_results"
DEFAULT_OUT  = RESULTS_DIR / "diagnostics" / "plots"


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 1: kd_ctx scan
# ---------------------------------------------------------------------------

def plot_kd_ctx_scan(out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    json_path = RESULTS_DIR / "diagnostics" / "kd_ctx_scan_results.json"
    if not json_path.exists():
        print(f"  [SKIP] {json_path} not found")
        return

    with open(json_path) as f:
        data = json.load(f)

    kd_vals      = [d["kd_ctx_nm"] for d in data]
    dr_mild      = [d["by_scenario"]["pmo_mild"]["mean"] for d in data]
    dr_pmo       = [d["by_scenario"]["pmo"]["mean"]      for d in data]
    dr_ckd       = [d["by_scenario"]["ckd_mbd"]["mean"]  for d in data]
    dr_composite = [d["dr_composite"] for d in data]

    std_mild = [d["by_scenario"]["pmo_mild"]["std"] / np.sqrt(20) for d in data]
    std_pmo  = [d["by_scenario"]["pmo"]["std"]      / np.sqrt(20) for d in data]
    std_ckd  = [d["by_scenario"]["ckd_mbd"]["std"]  / np.sqrt(20) for d in data]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    colors = {
        "PMO-mild": "#e07b39",
        "PMO":      "#3a7dc9",
        "CKD-MBD":  "#5ab05a",
        "Composite": "#9b59b6",
    }

    ax.errorbar(kd_vals, dr_mild, yerr=std_mild, marker="o", color=colors["PMO-mild"],
                label="PMO-mild", linewidth=1.8, capsize=3)
    ax.errorbar(kd_vals, dr_pmo, yerr=std_pmo, marker="s", color=colors["PMO"],
                label="PMO", linewidth=1.8, capsize=3)
    ax.errorbar(kd_vals, dr_ckd, yerr=std_ckd, marker="^", color=colors["CKD-MBD"],
                label="CKD-MBD", linewidth=1.8, capsize=3)
    ax.plot(kd_vals, dr_composite, marker="D", color=colors["Composite"],
            label="Composite", linewidth=1.8, linestyle="--")

    best_kd = 0.278
    ax.axvline(best_kd, color="gray", linestyle=":", linewidth=1.2, alpha=0.8)
    ax.annotate(
        f"$K_{{d,\\mathrm{{CTX}}}}$ = {best_kd} nM\n(composite peak)",
        xy=(best_kd, 0.884), xytext=(0.38, 0.78),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=8, color="gray",
    )

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.set_xlabel("$K_{d,\\mathrm{CTX}}$ (nM)", fontsize=11)
    ax.set_ylabel("Detection rate (n = 20 trials)", fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    _save(fig, out_dir, "kd_ctx_scan_plot.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Benchmark boxplot
# ---------------------------------------------------------------------------

def plot_benchmark(out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    json_path = RESULTS_DIR / "benchmark" / "benchmark_results.json"
    if not json_path.exists():
        print(f"  [SKIP] {json_path} not found")
        return

    with open(json_path) as f:
        data = json.load(f)

    order  = ["BO", "CMA-ES", "NSGA2-SE", "DE", "Random"]
    labels = ["BO\n(GP-EI)", "CMA-ES", "NSGA2-SE", "DE", "Random\nsearch"]
    scores = [data["scores"][o] for o in order]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bp = ax.boxplot(scores, tick_labels=labels, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))

    palette = ["#3a7dc9", "#5ab05a", "#e07b39", "#e74c3c", "#95a5a6"]
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    rng = np.random.RandomState(42)
    for i, (sc, color) in enumerate(zip(scores, palette), 1):
        jitter = rng.uniform(-0.15, 0.15, len(sc))
        ax.scatter([i + j for j in jitter], sc, color=color, alpha=0.5, s=18, zorder=3)

    means = [np.mean(s) for s in scores]
    for i, m in enumerate(means, 1):
        ax.scatter(i, m, marker="x", color="black", s=40, zorder=4, linewidths=1.5)

    ax.set_ylabel("Composite clinical score (n = 20 runs)", fontsize=11)
    ax.set_ylim(0.55, 0.73)
    ax.grid(True, axis="y", alpha=0.3)
    ax.annotate(
        f"BO mean: {means[0]:.3f}",
        xy=(1, means[0]), xytext=(2.3, means[0] + 0.005),
        fontsize=8, color="#3a7dc9",
        arrowprops=dict(arrowstyle="->", color="#3a7dc9", lw=0.8),
    )
    fig.tight_layout()

    _save(fig, out_dir, "benchmark_boxplot.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Sobol sensitivity bar chart
# ---------------------------------------------------------------------------

def plot_sobol(out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    json_path = RESULTS_DIR / "diagnostics" / "sobol_results.json"
    if not json_path.exists():
        print(f"  [SKIP] {json_path} not found")
        return

    with open(json_path) as f:
        data = json.load(f)

    param_display = {
        "sensitivity":  "Gain (alpha)",
        "kd_nm":        "$K_{d,\\mathrm{SOST}}$",
        "kd_ctx_nm":    "$K_{d,\\mathrm{CTX}}$",
        "w_p1np":       "$w_{\\mathrm{P1NP}}$",
        "w_ctx":        "$w_{\\mathrm{CTX}}$",
        "kd_p1np_nm":   "$K_{d,\\mathrm{P1NP}}$",
    }

    params = data["ranking_by_ST"]
    S1     = [data["S1"][p] for p in params]
    ST     = [data["ST"][p] for p in params]
    labels = [param_display.get(p, p) for p in params]

    x = np.arange(len(params))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x - w/2, S1, width=w, label="First-order ($S_1$)", color="#3a7dc9", alpha=0.8)
    ax.bar(x + w/2, ST, width=w, label="Total-order ($S_T$)",  color="#e07b39", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Sobol sensitivity index", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)
    fig.tight_layout()

    _save(fig, out_dir, "sobol_barplot.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Real-sim validation bar chart
# ---------------------------------------------------------------------------

def plot_realsim_validation(out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    json_path = RESULTS_DIR / "diagnostics" / "best_config_validation.json"
    if not json_path.exists():
        print(f"  [SKIP] {json_path} not found")
        return

    with open(json_path) as f:
        data = json.load(f)

    comp      = data["validation_results"]["comparison"]
    scenarios = ["pmo_mild", "pmo", "ckd_mbd", "healthy"]
    labels    = ["PMO-mild", "PMO", "CKD-MBD", "Healthy\n(FP rate)"]
    surr_vals = [comp[s]["surrogate"] for s in scenarios]
    real_vals = [comp[s]["real_sim"]  for s in scenarios]

    x = np.arange(len(scenarios))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x - w/2, surr_vals, width=w, label="Surrogate prediction",   color="#3a7dc9", alpha=0.8)
    ax.bar(x + w/2, real_vals, width=w, label="Real simulator (n = 20)", color="#e07b39", alpha=0.8)

    cis     = [data["validation_results"]["by_scenario"][s] for s in scenarios]
    errs_lo = [max(0, rv - ci["dr_ci95_low"])  for rv, ci in zip(real_vals, cis)]
    errs_hi = [max(0, ci["dr_ci95_high"] - rv) for rv, ci in zip(real_vals, cis)]
    ax.errorbar(x + w/2, real_vals, yerr=[errs_lo, errs_hi],
                fmt="none", color="black", capsize=4, linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Detection rate", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    bias = comp["pmo_mild"]["bias"]
    ax.annotate(
        f"Bias = {bias:+.2f}",
        xy=(0 + w/2, real_vals[0]),
        xytext=(0.5, 0.45), fontsize=8, color="black",
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
    )
    fig.tight_layout()

    _save(fig, out_dir, "realsim_validation.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: Convergence summary (final scores across seeds)
# ---------------------------------------------------------------------------

def plot_convergence_summary(out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    json_path = RESULTS_DIR / "convergence" / "convergence_report.json"
    if not json_path.exists():
        print(f"  [SKIP] {json_path} not found")
        return

    with open(json_path) as f:
        data = json.load(f)

    scores = [r["best_score"] for r in data["runs"]]
    seeds  = [r["seed"]       for r in data["runs"]]

    mean_s = np.mean(scores)
    std_s  = np.std(scores)

    fig, ax = plt.subplots(figsize=(7, 3.5))

    colors = ["#e07b39" if s == max(scores) else "#3a7dc9" for s in scores]
    ax.bar(np.arange(len(seeds)), scores, color=colors, alpha=0.8, width=0.6)

    ax.axhline(mean_s, color="black", linestyle="--", linewidth=1.2,
               label=f"Mean = {mean_s:.3f} +/- {std_s:.3f}")
    ax.fill_between([-0.5, len(seeds) - 0.5], mean_s - std_s, mean_s + std_s,
                    alpha=0.12, color="black")

    ax.set_xticks(np.arange(len(seeds)))
    ax.set_xticklabels([str(s) for s in seeds], fontsize=8, rotation=45)
    ax.set_xlabel("Random seed", fontsize=10)
    ax.set_ylabel("Best composite clinical score", fontsize=11)
    ax.set_ylim(0.68, 0.74)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    best_idx = int(np.argmax(scores))
    ax.annotate(
        f"Best: {max(scores):.3f}\n(seed {seeds[best_idx]})",
        xy=(best_idx, max(scores)),
        xytext=(best_idx + 0.8, max(scores) - 0.005),
        fontsize=8, color="#e07b39",
        arrowprops=dict(arrowstyle="->", color="#e07b39", lw=0.8),
    )
    fig.tight_layout()

    _save(fig, out_dir, "convergence_summary.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6: Langmuir binding isotherms (analytical)
# ---------------------------------------------------------------------------

def plot_langmuir_isotherms(out_dir: Path) -> None:
    """
    Computes Langmuir occupancy theta(c) = c / (Kd + c) analytically for each
    aptamer channel at the BO-optimised Kd values, overlaying the nominal
    biomarker concentrations for each clinical scenario. Illustrates why kd_ctx
    matters more than kd_p1np for disease discrimination.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    # Optimal Kd values (seed 888 BO result, nM)
    kd_sost    = 0.290
    kd_ctx_bo  = 10.0    # BO convergence value (boundary exploitation — majority of seeds hit upper bound)
    kd_ctx_opt = 0.278   # Scan optimum (Langmuir steep region)
    kd_p1np    = 4.951

    # Nominal concentrations (nM) from ODE steady-state calibration
    concs = {
        "SOST":  {"Healthy": 0.375, "PMO-mild": 0.563, "PMO": 0.875, "CKD-MBD": 1.125},
        "CTX":   {"Healthy": 0.200, "PMO-mild": 0.300, "PMO": 0.500, "CKD-MBD": 0.500},
        "P1NP":  {"Healthy": 0.350, "PMO-mild": 0.385, "PMO": 0.525, "CKD-MBD": 0.625},
    }

    scenario_colors = {
        "Healthy":  "#2ecc71",
        "PMO-mild": "#e07b39",
        "PMO":      "#e74c3c",
        "CKD-MBD":  "#9b59b6",
    }
    scenario_markers = {"Healthy": "o", "PMO-mild": "s", "PMO": "^", "CKD-MBD": "D"}

    c_range = np.logspace(-2, 1.3, 300)   # 0.01 to ~20 nM
    theta   = lambda c, kd: c / (kd + c)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    # --- Panel A: SOST ---
    ax = axes[0]
    ax.plot(c_range, theta(c_range, kd_sost), "#3a7dc9", linewidth=2,
            label=f"$K_d$ = {kd_sost} nM")
    for scen, conc in concs["SOST"].items():
        occ = theta(conc, kd_sost)
        ax.scatter([conc], [occ], color=scenario_colors[scen],
                   marker=scenario_markers[scen], s=70, zorder=5, label=scen)
        ax.axvline(conc, color=scenario_colors[scen], linestyle=":", alpha=0.35, linewidth=0.9)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.set_xlabel("SOST concentration (nM)", fontsize=10)
    ax.set_ylabel("Fractional occupancy", fontsize=10)
    ax.set_title("(A) Sclerostin (SOST) channel", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)

    # --- Panel B: CTX — compare BO value vs scan optimum ---
    ax = axes[1]
    ax.plot(c_range, theta(c_range, kd_ctx_bo), "#3a7dc9", linewidth=2,
            label=f"$K_d$ = {kd_ctx_bo} nM  (BO result)")
    ax.plot(c_range, theta(c_range, kd_ctx_opt), "#e74c3c", linewidth=2,
            linestyle="--", label=f"$K_d$ = {kd_ctx_opt} nM  (scan optimum)")
    for scen, conc in concs["CTX"].items():
        ax.axvline(conc, color=scenario_colors[scen], linestyle=":", alpha=0.35, linewidth=0.9)
        ax.scatter([conc], [theta(conc, kd_ctx_opt)], color=scenario_colors[scen],
                   marker=scenario_markers[scen], s=70, zorder=5)
    # Annotate the gain difference at healthy vs disease for both Kd values
    delta_bo  = theta(0.500, kd_ctx_bo)  - theta(0.200, kd_ctx_bo)
    delta_opt = theta(0.500, kd_ctx_opt) - theta(0.200, kd_ctx_opt)
    ax.text(0.15, 0.82,
            f"Delta(occ) at Kd={kd_ctx_bo}: {delta_bo:.2f}\n"
            f"Delta(occ) at Kd={kd_ctx_opt}: {delta_opt:.2f}",
            transform=ax.transAxes, fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.set_xlabel("CTX concentration (nM)", fontsize=10)
    ax.set_title("(B) CTX channel — Kd impact", fontsize=10)
    ax.legend(fontsize=7.5)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)

    # --- Panel C: P1NP ---
    ax = axes[2]
    ax.plot(c_range, theta(c_range, kd_p1np), "#3a7dc9", linewidth=2,
            label=f"$K_d$ = {kd_p1np} nM")
    for scen, conc in concs["P1NP"].items():
        occ = theta(conc, kd_p1np)
        ax.scatter([conc], [occ], color=scenario_colors[scen],
                   marker=scenario_markers[scen], s=70, zorder=5, label=scen)
        ax.axvline(conc, color=scenario_colors[scen], linestyle=":", alpha=0.35, linewidth=0.9)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.set_xlabel("P1NP concentration (nM)", fontsize=10)
    ax.set_title("(C) P1NP channel", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Langmuir Binding Isotherms — Aptamer Occupancy vs Biomarker Concentration",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    _save(fig, out_dir, "langmuir_isotherms.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 7: Robustness analysis (missing channels + gain drift)
# ---------------------------------------------------------------------------

def plot_robustness_analysis(out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    json_path = RESULTS_DIR / "diagnostics" / "distribution_shift_results.json"
    if not json_path.exists():
        print(f"  [SKIP] {json_path} not found")
        return

    with open(json_path) as f:
        data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    nominal = data["nominal_score"]

    # --- Panel A: Missing biomarker scenarios ---
    mb   = data["missing_biomarker"]
    items = [
        ("Nominal\n(all channels)", nominal,                  "#3a7dc9"),
        ("CTX channel\nmissing",    mb["no_ctx"]["score"],    "#e74c3c"),
        ("P1NP channel\nmissing",   mb["no_p1np"]["score"],   "#e07b39"),
        ("SOST only",               mb["sost_only"]["score"], "#9b59b6"),
    ]
    x_lbl = [i[0] for i in items]
    y_val = [i[1] for i in items]
    bar_c = [i[2] for i in items]

    bars = ax1.bar(range(len(items)), y_val, color=bar_c, alpha=0.82, width=0.55)
    ax1.axhline(nominal, color="black", linestyle="--", linewidth=1.0,
                label=f"Nominal = {nominal:.3f}")
    for j, (b, v) in enumerate(zip(bars, y_val)):
        delta = v - nominal
        sign  = "+" if delta >= 0 else ""
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.005,
                 f"{sign}{delta:.3f}", ha="center", va="bottom", fontsize=8)

    ax1.set_xticks(range(len(items)))
    ax1.set_xticklabels(x_lbl, fontsize=9)
    ax1.set_ylabel("Composite clinical score", fontsize=10)
    ax1.set_title("(A) Channel failure impact", fontsize=10)
    ax1.set_ylim(0.40, 0.73)
    ax1.legend(fontsize=8)
    ax1.grid(True, axis="y", alpha=0.3)

    # --- Panel B: Sensor gain drift ---
    gd     = data["sensor_gain_drift"]
    pcts   = sorted(gd.keys(), key=lambda k: float(k.replace("pct", "")), reverse=True)
    x_pct  = [float(k.replace("pct", "")) for k in pcts]
    y_sc   = [gd[k]["score"] for k in pcts]
    colors_b = ["#3a7dc9" if gd[k]["label"] == "OK" else
                "#e07b39"  if gd[k]["label"] == "WARN" else
                "#e74c3c"  for k in pcts]

    ax2.bar(range(len(pcts)), y_sc, color=colors_b, alpha=0.82, width=0.55)
    ax2.axhline(nominal, color="black", linestyle="--", linewidth=1.0,
                label=f"Nominal = {nominal:.3f}")
    ax2.axhline(0.60,    color="gray",  linestyle=":",  linewidth=0.9,
                label="Acceptable floor (0.60)")
    ax2.set_xticks(range(len(pcts)))
    ax2.set_xticklabels([f"{p:.0f}%" for p in x_pct], fontsize=9)
    ax2.set_xlabel("Sensor amplifier gain (% of nominal)", fontsize=10)
    ax2.set_ylabel("Composite clinical score", fontsize=10)
    ax2.set_title("(B) Amplifier gain drift", fontsize=10)
    ax2.set_ylim(0.48, 0.73)
    ax2.legend(fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor="#3a7dc9", label="OK (< 5% drop)"),
        Patch(facecolor="#e07b39", label="WARN (5-15% drop)"),
        Patch(facecolor="#e74c3c", label="FAIL (> 15% drop)"),
    ]
    ax2.legend(handles=legend_patches, fontsize=8, loc="lower right")

    fig.tight_layout()
    _save(fig, out_dir, "robustness_analysis.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 8: Closed-loop active learning curves
# ---------------------------------------------------------------------------

def plot_closed_loop_learning(out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    json_path = RESULTS_DIR / "closed_loop" / "results" / "closed_loop_results.json"
    if not json_path.exists():
        print(f"  [SKIP] {json_path} not found")
        return

    with open(json_path) as f:
        data = json.load(f)

    rounds  = [r["round"] for r in data["round_summaries"]]
    surr_sc = [r["best_surrogate_score"]  for r in data["round_summaries"]]
    real_dr = [r["mean_real_dr_top_k"]   for r in data["round_summaries"]]
    biases  = [r["surrogate_bias"]        for r in data["round_summaries"]]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    # Top: surrogate score vs real DR
    ax1.plot(rounds, surr_sc, "o-", color="#3a7dc9", linewidth=2,
             label="Best surrogate score (top-1)")
    ax1.plot(rounds, real_dr, "s--", color="#e07b39", linewidth=2,
             label="Mean real DR (top-10 configs)")
    ax1.set_ylabel("Score / detection rate", fontsize=10)
    ax1.set_title("Active Learning Rounds: Surrogate vs Simulator", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.58, 0.85)

    # Bottom: surrogate bias per round
    bar_colors = ["#e74c3c" if b < 0 else "#5ab05a" for b in biases]
    ax2.bar(rounds, biases, color=bar_colors, alpha=0.80, width=0.5)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Active learning round", fontsize=10)
    ax2.set_ylabel("Surrogate bias\n(predicted - real)", fontsize=10)
    ax2.set_title("Surrogate Bias Across Rounds", fontsize=10)
    ax2.set_xticks(rounds)
    ax2.grid(True, axis="y", alpha=0.3)

    mean_bias = np.mean(biases)
    ax2.text(0.98, 0.10, f"Mean bias = {mean_bias:+.3f}",
             transform=ax2.transAxes, ha="right", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))

    fig.tight_layout()
    _save(fig, out_dir, "closed_loop_learning.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 9: Per-scenario surrogate rank correlation
# ---------------------------------------------------------------------------

def plot_scenario_rank_rho(out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    json_path = RESULTS_DIR / "diagnostics" / "rank_rho_results.json"
    if not json_path.exists():
        print(f"  [SKIP] {json_path} not found")
        return

    with open(json_path) as f:
        data = json.load(f)

    scenarios   = ["pmo_mild", "pmo", "ckd_mbd"]
    labels      = ["PMO-mild", "PMO", "CKD-MBD"]
    rho_vals    = [data["by_scenario"][s]["rank_rho"]   for s in scenarios]
    n_vals      = [data["by_scenario"][s]["n"]          for s in scenarios]
    true_means  = [data["by_scenario"][s]["true_dr_mean"] for s in scenarios]
    pred_means  = [data["by_scenario"][s]["pred_dr_mean"] for s in scenarios]
    overall_rho = data["overall"]["rank_rho"]

    colors = ["#e07b39", "#3a7dc9", "#5ab05a"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel A: rank-rho per scenario
    bars = ax1.bar(range(len(scenarios)), rho_vals, color=colors, alpha=0.82, width=0.5)
    ax1.axhline(overall_rho, color="black", linestyle="--", linewidth=1.2,
                label=f"Overall rho = {overall_rho:.3f}")
    ax1.axhline(data["benchmark"]["threshold_good"], color="gray",
                linestyle=":", linewidth=1.0, label="'Good' threshold (0.70)")
    for bar, rho, n in zip(bars, rho_vals, n_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, rho + 0.01,
                 f"{rho:.3f}\n(n={n})", ha="center", va="bottom", fontsize=8.5)
    ax1.set_xticks(range(len(scenarios)))
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel("Spearman rank correlation", fontsize=10)
    ax1.set_title("(A) Surrogate accuracy per scenario", fontsize=10)
    ax1.set_ylim(0, 0.95)
    ax1.legend(fontsize=8.5)
    ax1.grid(True, axis="y", alpha=0.3)

    # Panel B: mean predicted vs mean true DR per scenario
    x = np.arange(len(scenarios))
    w = 0.35
    ax2.bar(x - w/2, true_means, width=w, label="True DR mean",      color="#e07b39", alpha=0.82)
    ax2.bar(x + w/2, pred_means, width=w, label="Predicted DR mean",  color="#3a7dc9", alpha=0.82)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel("Mean detection rate", fontsize=10)
    ax2.set_title("(B) Predicted vs true mean DR", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.set_ylim(0, 0.85)

    fig.tight_layout()
    _save(fig, out_dir, "scenario_rank_rho.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 10: Patient subtype detection rates
# ---------------------------------------------------------------------------

def plot_patient_subtypes(out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    json_path = RESULTS_DIR / "subtypes" / "subtype_comparison.json"
    if not json_path.exists():
        print(f"  [SKIP] {json_path} not found")
        return

    with open(json_path) as f:
        data = json.load(f)

    subtype_labels = {
        "young_pmo":     "Young PMO\n(age < 55)",
        "elderly_pmo":   "Elderly PMO\n(age > 70)",
        "ckd_controlled": "CKD Stage 3\n(eGFR 30-60)",
        "ckd_advanced":  "CKD Stage 4-5\n(eGFR < 30)",
    }

    subtypes = [d["subtype"] for d in data]
    dr_mild  = [d["dr_pmo_mild"]    for d in data]
    dr_pmo   = [d["dr_pmo"]         for d in data]
    dr_ckd   = [d["dr_ckd"]         for d in data]
    dr_fp    = [d["dr_healthy_fp"]  for d in data]
    labels   = [subtype_labels.get(s, s) for s in subtypes]

    x = np.arange(len(subtypes))
    w = 0.18

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - 1.5*w, dr_mild, width=w, label="PMO-mild DR",  color="#e07b39", alpha=0.85)
    ax.bar(x - 0.5*w, dr_pmo,  width=w, label="PMO DR",       color="#3a7dc9", alpha=0.85)
    ax.bar(x + 0.5*w, dr_ckd,  width=w, label="CKD-MBD DR",   color="#5ab05a", alpha=0.85)
    ax.bar(x + 1.5*w, dr_fp,   width=w, label="FP rate",       color="#e74c3c", alpha=0.85)

    ax.axhline(0.10, color="gray", linestyle=":", linewidth=1.0,
               label="FP ceiling (10%)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Detection / false-positive rate", fontsize=10)
    ax.set_title("Subtype-Specific Biosensor Optimisation Results", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_ylim(0, 1.12)
    ax.grid(True, axis="y", alpha=0.3)

    for i, d in enumerate(data):
        sc = d.get("subtype_score", 0)
        ax.text(x[i], 1.06, f"score={sc:.2f}", ha="center", fontsize=7.5, color="#555555")

    fig.tight_layout()
    _save(fig, out_dir, "patient_subtypes.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 11: MOBO Pareto front (2D projections)
# ---------------------------------------------------------------------------

def plot_mobo_pareto_front(out_dir: Path) -> None:
    """
    Three 2D scatter-plot projections of the 14-point Pareto front obtained
    from the multi-objective BO run (EHVI acquisition, 131 evaluations, 37
    feasible, 14 non-dominated). Axes: DR_mean, therapeutic_mean, specificity.
    Points are coloured by sensitivity (alpha) of the associated biosensor config.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    json_path = RESULTS_DIR / "mobo" / "mobo_results.json"
    if not json_path.exists():
        print(f"  [SKIP] {json_path} not found")
        return

    with open(json_path) as f:
        data = json.load(f)

    if not data.get("pareto_configs") or not data.get("pareto_objectives"):
        print("  [SKIP] MOBO produced no Pareto solutions")
        return

    objs    = np.array(data["pareto_objectives"])   # (n, 3)
    configs = data["pareto_configs"]
    dr      = objs[:, 0]
    ther    = objs[:, 1]
    spec    = objs[:, 2]
    alpha   = np.array([c.get("sensitivity", 2.5) for c in configs])

    norm   = plt.Normalize(alpha.min(), alpha.max())
    cmap   = cm.plasma
    colors = cmap(norm(alpha))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    def _scatter(ax, x, y, xlabel, ylabel, title):
        sc = ax.scatter(x, y, c=alpha, cmap="plasma",
                        vmin=alpha.min(), vmax=alpha.max(),
                        s=80, zorder=4, edgecolors="white", linewidths=0.4)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.25)

        best_dr_idx   = int(np.argmax(dr))
        best_th_idx   = int(np.argmax(ther))
        best_sp_idx   = int(np.argmax(spec))
        warm_start_idx = 0

        for idx, label, mkr, clr in [
            (best_dr_idx,    "Best DR",    "^", "#e74c3c"),
            (best_th_idx,    "Best Ther.", "s", "#3a7dc9"),
            (best_sp_idx,    "Best Spec.", "D", "#5ab05a"),
            (warm_start_idx, "Warm start", "o", "#e07b39"),
        ]:
            ax.scatter(x[idx], y[idx], marker=mkr, color=clr, s=110,
                       zorder=5, edgecolors="black", linewidths=0.8, label=label)

        return sc

    sc = _scatter(axes[0], dr,   ther, "Disease DR (mean)", "Therapeutic gain",
                  "(A) Disease detection vs therapeutic")
    _scatter(axes[1], dr,   spec, "Disease DR (mean)", "Specificity (1 − FP)",
             "(B) Disease detection vs specificity")
    _scatter(axes[2], ther, spec, "Therapeutic gain", "Specificity (1 − FP)",
             "(C) Therapeutic vs specificity")

    axes[0].legend(fontsize=7.5, loc="lower right")

    fig.colorbar(cm.ScalarMappable(norm=norm, cmap="plasma"), ax=axes[2],
                 label="Amplifier gain (α)", fraction=0.04, pad=0.02)

    fig.suptitle(
        f"Multi-Objective Pareto Front  |  {len(configs)} solutions  |  "
        f"Final hypervolume = {data['final_hypervolume']:.4f}",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    _save(fig, out_dir, "mobo_pareto_front.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 12: MOBO hypervolume convergence
# ---------------------------------------------------------------------------

def plot_mobo_hypervolume(out_dir: Path) -> None:
    """
    Hypervolume indicator as a function of BO evaluation index. The curve rises
    monotonically (by definition of hypervolume on a growing Pareto approximation)
    and its growth rate shows where EHVI acquisition produced the largest gains.
    """
    import matplotlib.pyplot as plt

    json_path = RESULTS_DIR / "mobo" / "mobo_results.json"
    if not json_path.exists():
        print(f"  [SKIP] {json_path} not found")
        return

    with open(json_path) as f:
        data = json.load(f)

    hv_curve = data.get("hypervolume_curve", [])
    if not hv_curve:
        print("  [SKIP] No hypervolume curve in mobo_results.json")
        return

    n_init = 30   # n_init LHS evaluations before EHVI kicks in
    evals  = np.arange(1, len(hv_curve) + 1)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(evals, hv_curve, color="#3a7dc9", linewidth=2, zorder=3)
    ax.fill_between(evals, data["reference_point"][0] if data.get("reference_point") else 0,
                    hv_curve, alpha=0.12, color="#3a7dc9")

    ax.axvline(n_init, color="gray", linestyle="--", linewidth=1.2, alpha=0.8,
               label=f"EHVI acquisition begins (eval {n_init})")

    hv_init  = hv_curve[n_init - 1] if len(hv_curve) >= n_init else hv_curve[0]
    hv_final = hv_curve[-1]
    hv_gain  = hv_final - hv_init

    ax.annotate(
        f"Final HV = {hv_final:.4f}\n(+{hv_gain:.4f} from EHVI phase)",
        xy=(len(hv_curve), hv_final),
        xytext=(len(hv_curve) * 0.6, hv_init + hv_gain * 0.3),
        fontsize=9, color="#3a7dc9",
        arrowprops=dict(arrowstyle="->", color="#3a7dc9", lw=0.9),
    )

    ax.set_xlabel("Cumulative evaluations", fontsize=11)
    ax.set_ylabel("Hypervolume indicator", fontsize=11)
    ax.set_title(
        "MOBO Hypervolume Convergence\n"
        f"({data['n_feasible']} feasible / {data['n_evaluated']} evaluated  |  "
        f"{data['n_pareto']} Pareto solutions)",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "mobo_hypervolume.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 13: Concentration shift robustness
# ---------------------------------------------------------------------------

def plot_concentration_shift(out_dir: Path) -> None:
    """
    Score versus biomarker concentration shift (−30% to +30%) from the
    distribution shift experiment. Also shows the PMO+CKD mixed population
    response. Two-panel figure.
    """
    import matplotlib.pyplot as plt

    json_path = RESULTS_DIR / "diagnostics" / "distribution_shift_results.json"
    if not json_path.exists():
        print(f"  [SKIP] {json_path} not found")
        return

    with open(json_path) as f:
        data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel A: Concentration shift
    cs = {k: v for k, v in data["concentration_shift"]["scores"].items()
          if k != "nominal"}
    shifts = sorted(cs.keys(), key=lambda k: float(k.replace("%", "")))
    x_vals = [float(k.replace("%", "")) for k in shifts]
    y_vals = [cs[k] for k in shifts]
    nominal = data["nominal_score"]

    ax1.plot(x_vals, y_vals, "o-", color="#3a7dc9", linewidth=2, markersize=7)
    ax1.axhline(nominal, color="gray", linestyle="--", linewidth=1.2,
                label=f"Nominal (0%): {nominal:.3f}")
    ax1.axhline(0.60, color="#e07b39", linestyle=":", linewidth=1.0,
                label="Clinical floor (0.60)")
    ax1.fill_between(x_vals, 0.60, y_vals, where=[y >= 0.60 for y in y_vals],
                     alpha=0.10, color="#5ab05a", label="Above floor")
    ax1.fill_between(x_vals, 0.60, y_vals, where=[y < 0.60 for y in y_vals],
                     alpha=0.15, color="#e74c3c", label="Below floor")

    ax1.set_xlabel("Global biomarker concentration shift (%)", fontsize=10)
    ax1.set_ylabel("Composite clinical score", fontsize=10)
    ax1.set_title("(A) Concentration shift robustness", fontsize=10)
    ax1.legend(fontsize=8.5)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.08, 0.78)

    # Panel B: PMO+CKD mixed population
    mixed = data.get("pmo_ckd_mixed", {})
    if mixed:
        mix_keys = sorted(mixed.keys())
        mix_alpha = [mixed[k]["alpha_ckd"] for k in mix_keys]
        mix_score = [mixed[k]["score"]     for k in mix_keys]

        ax2.plot(mix_alpha, mix_score, "s-", color="#9b59b6", linewidth=2, markersize=7)
        ax2.axhline(nominal, color="gray", linestyle="--", linewidth=1.2,
                    label=f"Pure PMO baseline: {nominal:.3f}")
        ax2.set_xlabel("CKD fraction in mixed population (α_CKD)", fontsize=10)
        ax2.set_ylabel("Composite clinical score", fontsize=10)
        ax2.set_title("(B) PMO+CKD mixed population response", fontsize=10)
        ax2.legend(fontsize=8.5)
        ax2.grid(True, alpha=0.3)

        ckd_score = mix_score[-1]
        ax2.annotate(
            f"Pure CKD: {ckd_score:.3f}\n(+{ckd_score - nominal:+.3f})",
            xy=(1.0, ckd_score),
            xytext=(0.65, ckd_score - 0.015),
            fontsize=8.5, color="#9b59b6",
            arrowprops=dict(arrowstyle="->", color="#9b59b6", lw=0.8),
        )

    fig.tight_layout()
    _save(fig, out_dir, "concentration_shift.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Generate publication figures from JSON results")
    parser.add_argument(
        "--out-dir", default=str(DEFAULT_OUT),
        help=f"Output directory (default: {DEFAULT_OUT})",
    )
    args    = parser.parse_args()
    out_dir = Path(args.out_dir)

    print(f"Generating figures -> {out_dir}")

    plot_kd_ctx_scan(out_dir)
    plot_benchmark(out_dir)
    plot_sobol(out_dir)
    plot_realsim_validation(out_dir)
    plot_convergence_summary(out_dir)
    plot_langmuir_isotherms(out_dir)
    plot_robustness_analysis(out_dir)
    plot_closed_loop_learning(out_dir)
    plot_scenario_rank_rho(out_dir)
    plot_patient_subtypes(out_dir)
    plot_mobo_pareto_front(out_dir)
    plot_mobo_hypervolume(out_dir)
    plot_concentration_shift(out_dir)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
