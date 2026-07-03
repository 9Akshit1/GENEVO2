#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Score distribution analysis: evaluate N=1000 random designs with v3 and v4 objectives.

Answers:
  - How hard is the problem? (what fraction of random configs are excellent/acceptable/poor?)
  - How different are v3 and v4 landscapes?
  - Where in parameter space do excellent designs cluster?

Usage (from project root):
    python BO/diagnostics/score_distribution.py
    python BO/diagnostics/score_distribution.py --n 2000 --seed 42
"""

import sys
import argparse
import logging
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from BO.core.surrogate_loader import SurrogateLoaderV3
from search_space.biosensor_space import BiosensorSearchSpace
from evaluation.physics_forward_model import PhysicsForwardModel
from evaluation.objective_function_v3 import ObjectiveFunctionV3
from evaluation.objective_function_v4 import ObjectiveFunctionV4
from evaluation.therapeutic_objective_v5 import TherapeuticObjectiveV5
from evaluation.therapeutic_objective_v6 import TherapeuticObjectiveV6

logging.basicConfig(level=logging.WARNING)
CONSOLE = logging.getLogger("score_dist")
CONSOLE.setLevel(logging.INFO)
CONSOLE.propagate = False
if not CONSOLE.handlers:
    _ch = logging.StreamHandler(sys.stdout)
    _ch.setFormatter(logging.Formatter("%(message)s"))
    CONSOLE.addHandler(_ch)


def _categorize(scores):
    """Return count (and fraction) of designs in each tier."""
    s = np.array(scores)
    n = len(s)
    tiers = {
        "excellent (>0.80)":    int(np.sum(s > 0.80)),
        "good     (0.70-0.80)": int(np.sum((s > 0.70) & (s <= 0.80))),
        "accept   (0.60-0.70)": int(np.sum((s > 0.60) & (s <= 0.70))),
        "marginal (0.40-0.60)": int(np.sum((s > 0.40) & (s <= 0.60))),
        "poor     (<0.40)":     int(np.sum(s <= 0.40)),
    }
    return tiers, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000,
                        help="Number of random configs to sample (default: 1000)")
    parser.add_argument("--seed", type=int, default=99,
                        help="Random seed (default: 99)")
    parser.add_argument("--surrogate-dir", type=Path, default=Path("BO/bo_results"))
    parser.add_argument("--out-dir", type=Path, default=Path("BO/diagnostics/plots"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    CONSOLE.info("=" * 70)
    CONSOLE.info("GENEVO2 -- Score Distribution Analysis")
    CONSOLE.info("=" * 70)
    CONSOLE.info(f"  N designs : {args.n}")
    CONSOLE.info(f"  Surrogate : {args.surrogate_dir}")
    CONSOLE.info(f"  Seed      : {args.seed}")
    CONSOLE.info("")

    # Load surrogates and objectives
    CONSOLE.info("[1/4] Loading surrogates ...")
    surrogate = SurrogateLoaderV3(args.surrogate_dir)
    physics = PhysicsForwardModel()
    obj_v3 = ObjectiveFunctionV3(physics, surrogate)
    obj_v4 = ObjectiveFunctionV4(physics, surrogate)
    obj_v5 = TherapeuticObjectiveV5(physics, surrogate)
    obj_v6 = TherapeuticObjectiveV6(physics, surrogate)
    space = BiosensorSearchSpace()
    CONSOLE.info("      OK")

    # Sample N random configs
    CONSOLE.info(f"[2/4] Sampling {args.n} random configs ...")
    configs = []
    for _ in range(args.n):
        x = rng.uniform(0, 1, space.n_params)
        configs.append(space.vector_to_dict(x))

    # Evaluate with all objectives
    CONSOLE.info("[3/4] Evaluating (this takes ~40s) ...")
    scores_v3, scores_v4, scores_v5, scores_v6 = [], [], [], []
    for i, cfg in enumerate(configs):
        scores_v3.append(obj_v3(cfg))
        scores_v4.append(obj_v4(cfg))
        scores_v5.append(obj_v5(cfg))
        scores_v6.append(obj_v6(cfg))
        if (i + 1) % 200 == 0:
            CONSOLE.info(f"      {i+1}/{args.n} done")

    scores_v3 = np.array(scores_v3)
    scores_v4 = np.array(scores_v4)
    scores_v5 = np.array(scores_v5)
    scores_v6 = np.array(scores_v6)

    # Print summary statistics
    CONSOLE.info("")
    CONSOLE.info("[4/4] Results")
    CONSOLE.info("=" * 70)

    for label, scores in [("v3", scores_v3), ("v4", scores_v4), ("v5", scores_v5), ("v6", scores_v6)]:
        tiers, n = _categorize(scores)
        CONSOLE.info(f"\n  Objective {label}  (n={n})")
        CONSOLE.info(f"    mean={scores.mean():.4f}  std={scores.std():.4f}  "
                     f"p5={np.percentile(scores,5):.4f}  "
                     f"p25={np.percentile(scores,25):.4f}  "
                     f"p75={np.percentile(scores,75):.4f}  "
                     f"p95={np.percentile(scores,95):.4f}")
        for tier, count in tiers.items():
            bar = "#" * int(40 * count / n)
            CONSOLE.info(f"    {tier} : {count:4d} ({100*count/n:5.1f}%)  {bar}")

    # Correlations
    corr_v3v4 = np.corrcoef(scores_v3, scores_v4)[0, 1]
    corr_v3v5 = np.corrcoef(scores_v3, scores_v5)[0, 1]
    corr_v3v6 = np.corrcoef(scores_v3, scores_v6)[0, 1]
    corr_v4v5 = np.corrcoef(scores_v4, scores_v5)[0, 1]
    corr_v4v6 = np.corrcoef(scores_v4, scores_v6)[0, 1]
    corr_v5v6 = np.corrcoef(scores_v5, scores_v6)[0, 1]
    CONSOLE.info(f"\n  Pearson correlations:")
    CONSOLE.info(f"    v3 vs v4 : {corr_v3v4:.4f}")
    CONSOLE.info(f"    v3 vs v5 : {corr_v3v5:.4f}")
    CONSOLE.info(f"    v3 vs v6 : {corr_v3v6:.4f}  <-- KEY: should be < 0.994")
    CONSOLE.info(f"    v4 vs v5 : {corr_v4v5:.4f}")
    CONSOLE.info(f"    v4 vs v6 : {corr_v4v6:.4f}")
    CONSOLE.info(f"    v5 vs v6 : {corr_v5v6:.4f}")
    v4_better = int(np.sum(scores_v4 > scores_v3))
    v5_better = int(np.sum(scores_v5 > scores_v3))
    v6_better = int(np.sum(scores_v6 > scores_v3))
    CONSOLE.info(f"  v4 > v3 for {v4_better}/{args.n} ({100*v4_better/args.n:.1f}%)")
    CONSOLE.info(f"  v5 > v3 for {v5_better}/{args.n} ({100*v5_better/args.n:.1f}%)")
    CONSOLE.info(f"  v6 > v3 for {v6_better}/{args.n} ({100*v6_better/args.n:.1f}%)")
    mean_shift = scores_v4.mean() - scores_v3.mean()
    mean_shift_v5 = scores_v5.mean() - scores_v3.mean()
    mean_shift_v6 = scores_v6.mean() - scores_v3.mean()
    CONSOLE.info(f"  Mean score shift (v4 - v3): {mean_shift:+.4f}")
    CONSOLE.info(f"  Mean score shift (v5 - v3): {mean_shift_v5:+.4f}")
    CONSOLE.info(f"  Mean score shift (v6 - v3): {mean_shift_v6:+.4f}")

    # Retain for JSON
    corr = corr_v3v4

    # Top-10 configs by v6 score (decoupled therapeutic objective) -- KEY TABLE
    CONSOLE.info("\n  Top-10 configs by v6 score (decoupled kd/sensitivity therapeutic):")
    top_idx_v6 = np.argsort(scores_v6)[-10:][::-1]
    CONSOLE.info(f"  {'Rank':<5} {'v3':>7} {'v5':>7} {'v6':>7} {'sens':>6} {'kd_nm':>6} "
                 f"{'kd_ctx':>7} {'kd_p1np':>7} {'w_ctx':>6} {'w_p1np':>6}")
    for rank, idx in enumerate(top_idx_v6, 1):
        c = configs[idx]
        CONSOLE.info(f"  {rank:<5} {scores_v3[idx]:>7.4f} {scores_v5[idx]:>7.4f} {scores_v6[idx]:>7.4f} "
                     f"{c.get('sensitivity',0):>6.3f} {c.get('kd_nm',0):>6.3f} "
                     f"{c.get('kd_ctx_nm',0):>7.4f} {c.get('kd_p1np_nm',0):>7.4f} "
                     f"{c.get('w_ctx',0):>6.3f} {c.get('w_p1np',0):>6.3f}")

    # Top-10 configs by v3 score (baseline)
    CONSOLE.info("\n  Top-10 configs by v3 score (baseline, for comparison):")
    top_idx = np.argsort(scores_v3)[-10:][::-1]
    CONSOLE.info(f"  {'Rank':<5} {'v3':>7} {'v5':>7} {'v6':>7} {'sens':>6} {'kd_nm':>6} "
                 f"{'kd_ctx':>7} {'kd_p1np':>7} {'w_ctx':>6} {'w_p1np':>6}")
    for rank, idx in enumerate(top_idx, 1):
        c = configs[idx]
        CONSOLE.info(f"  {rank:<5} {scores_v3[idx]:>7.4f} {scores_v5[idx]:>7.4f} {scores_v6[idx]:>7.4f} "
                     f"{c.get('sensitivity',0):>6.3f} {c.get('kd_nm',0):>6.3f} "
                     f"{c.get('kd_ctx_nm',0):>7.4f} {c.get('kd_p1np_nm',0):>7.4f} "
                     f"{c.get('w_ctx',0):>6.3f} {c.get('w_p1np',0):>6.3f}")

    CONSOLE.info("")

    # --- Plots ---
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

    # 1: Histogram v3 vs v5 vs v6
    ax1 = fig.add_subplot(gs[0, 0])
    bins = np.linspace(-0.5, 1.0, 40)
    ax1.hist(scores_v3, bins=bins, alpha=0.5, color="#2196F3", label="v3")
    ax1.hist(scores_v5, bins=bins, alpha=0.5, color="#4CAF50", label="v5")
    ax1.hist(scores_v6, bins=bins, alpha=0.5, color="#FF9800", label="v6")
    ax1.axvline(0.80, color="green", lw=1.5, ls="--", label="excellent")
    ax1.axvline(0.60, color="orange", lw=1.5, ls="--", label="acceptable")
    ax1.axvline(0.40, color="red", lw=1.5, ls="--", label="poor")
    ax1.set_xlabel("Objective score")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Score distribution (N={args.n})")
    ax1.legend(fontsize=8)

    # 2: CDF
    ax2 = fig.add_subplot(gs[0, 1])
    for scores, label, color in [(scores_v3, "v3", "#2196F3"), (scores_v5, "v5", "#4CAF50"),
                                  (scores_v6, "v6 (key)", "#FF9800")]:
        sorted_s = np.sort(scores)
        cdf = np.arange(1, len(sorted_s) + 1) / len(sorted_s)
        ax2.plot(sorted_s, cdf, color=color, label=label, lw=1.5)
    ax2.axvline(0.80, color="green", lw=1, ls="--")
    ax2.axvline(0.60, color="orange", lw=1, ls="--")
    ax2.set_xlabel("Objective score")
    ax2.set_ylabel("Cumulative fraction")
    ax2.set_title("CDF of scores")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3: v3 vs v6 scatter (KEY -- shows landscape decoupling)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(scores_v3, scores_v6, s=5, alpha=0.3, color="#FF9800")
    ax3.plot([-0.5, 1.0], [-0.5, 1.0], "k--", lw=1, label="v3=v6")
    ax3.set_xlabel("v3 score")
    ax3.set_ylabel("v6 score")
    ax3.set_title(f"v3 vs v6 (r={corr_v3v6:.3f})  [KEY: target < 0.90]")
    ax3.legend(fontsize=8)

    # 4: Sensitivity distribution for each tier (v6 — decoupled therapeutic)
    ax4 = fig.add_subplot(gs[1, 0])
    tier_mask = {
        "excellent": scores_v6 > 0.60,
        "good": (scores_v6 > 0.45) & (scores_v6 <= 0.60),
        "acceptable": (scores_v6 > 0.30) & (scores_v6 <= 0.45),
        "poor": scores_v6 <= 0.30,
    }
    sensitivities = np.array([c.get("sensitivity", 0) for c in configs])
    for tier, mask in tier_mask.items():
        if mask.sum() > 0:
            ax4.hist(sensitivities[mask], bins=20, alpha=0.5, label=f"{tier} (n={mask.sum()})", density=True)
    ax4.set_xlabel("Sensitivity")
    ax4.set_ylabel("Density")
    ax4.set_title("Sensitivity by v6 tier (flat=decoupled)")
    ax4.legend(fontsize=7)

    # 5: kd_ctx_nm vs kd_p1np_nm colored by v6 score
    ax5 = fig.add_subplot(gs[1, 1])
    kd_ctx = np.array([c.get("kd_ctx_nm", 0) for c in configs])
    kd_p1np = np.array([c.get("kd_p1np_nm", 0) for c in configs])
    sc = ax5.scatter(np.log10(kd_ctx + 1e-6), np.log10(kd_p1np + 1e-6),
                     c=scores_v6, cmap="RdYlGn", s=8, alpha=0.5, vmin=0.0, vmax=0.7)
    plt.colorbar(sc, ax=ax5, label="v6 score")
    ax5.axvline(np.log10(0.300), color="blue", lw=1.5, ls=":", alpha=0.8, label="CTX_mild=0.300")
    ax5.axhline(np.log10(0.385), color="red", lw=1.5, ls=":", alpha=0.8, label="P1NP_mild=0.385")
    ax5.set_xlabel("log10(kd_ctx) [nM]")
    ax5.set_ylabel("log10(kd_p1np) [nM]")
    ax5.set_title("Kd space (v6 score)")
    ax5.legend(fontsize=7)

    # 6: sensitivity vs v6 score (should show non-monotone or flat relationship)
    ax6 = fig.add_subplot(gs[1, 2])
    sc2 = ax6.scatter(sensitivities, scores_v6, s=5, alpha=0.3, color="#FF9800", label="v6")
    ax6.scatter(sensitivities, scores_v3, s=5, alpha=0.2, color="#2196F3", label="v3")
    ax6.set_xlabel("Sensitivity")
    ax6.set_ylabel("Objective score")
    ax6.set_title("Sensitivity vs score (v3=blue, v6=orange)\nFlatter v6 = decoupled")

    fig.suptitle(f"GENEVO2 Score Distribution Analysis -- N={args.n} random designs",
                 fontsize=12, fontweight="bold")

    out_png = args.out_dir / "score_distribution.png"
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    CONSOLE.info(f"  Plot saved -> {out_png}")

    # Save JSON summary
    def _stats(s):
        t, _ = _categorize(s)
        return {
            "mean": float(s.mean()), "std": float(s.std()),
            "p5":  float(np.percentile(s, 5)),
            "p25": float(np.percentile(s, 25)),
            "p50": float(np.percentile(s, 50)),
            "p75": float(np.percentile(s, 75)),
            "p95": float(np.percentile(s, 95)),
            "tiers": dict(zip(
                ["excellent_gt080", "good_070_080", "accept_060_070", "marginal_040_060", "poor_lt040"],
                list(t.values())
            )),
        }

    summary = {
        "n_samples": args.n,
        "seed": args.seed,
        "v3": _stats(scores_v3),
        "v4": _stats(scores_v4),
        "v5": _stats(scores_v5),
        "v6": _stats(scores_v6),
        "correlations": {
            "v3_vs_v4": float(corr_v3v4),
            "v3_vs_v5": float(corr_v3v5),
            "v3_vs_v6": float(corr_v3v6),
            "v4_vs_v5": float(corr_v4v5),
            "v4_vs_v6": float(corr_v4v6),
            "v5_vs_v6": float(corr_v5v6),
        },
        "mean_shifts": {
            "v4_minus_v3": float(mean_shift),
            "v5_minus_v3": float(mean_shift_v5),
            "v6_minus_v3": float(mean_shift_v6),
        },
    }
    json_path = args.out_dir / "score_distribution.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    CONSOLE.info(f"  JSON saved -> {json_path}")
    CONSOLE.info("")


if __name__ == "__main__":
    main()
