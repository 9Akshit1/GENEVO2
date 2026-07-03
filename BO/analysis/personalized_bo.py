#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Targeted Biosensor Optimizer — GP-BO with warm-start for finding the
globally optimal aptamer array configuration for a specified clinical target.

Inputs : --target {pmo|ckd_mbd|both}, optimization budget, seed
Outputs: best config JSON + 4 diagnostic plots

The TherapeuticObjectiveV6 (frozen) jointly optimises detection rate,
therapeutic dosing, specificity, and toxicity across all disease scenarios.
The --target flag specifies the primary clinical deployment context for
reporting and plot emphasis; it does NOT alter the optimization objective.

The champion config (kd_ctx=0.278 nM, seed-999) is injected as a warm-start
anchor so the GP begins from the known-good CTX affinity basin.

Usage:
    # Optimise for both PMO and CKD-MBD (default, recommended)
    python BO/analysis/personalized_bo.py

    # PMO-focused deployment reporting
    python BO/analysis/personalized_bo.py --target pmo

    # CKD-MBD-focused deployment reporting
    python BO/analysis/personalized_bo.py --target ckd_mbd

    # Skip ODE validation (surrogate scores only, fast)
    python BO/analysis/personalized_bo.py --skip-simulation

Output:
    BO/bo_results/targeted/best_config.json
    BO/bo_results/targeted/plots/convergence.png
    BO/bo_results/targeted/plots/scenario_dr.png
    BO/bo_results/targeted/plots/channel_weights.png
    BO/bo_results/targeted/plots/langmuir_curves.png
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "BO"))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Champion baseline (for comparison plots)
# ---------------------------------------------------------------------------
CHAMPION_CONFIG = {
    "name": "champion",
    "kd_nm": 1.1423227672469514,
    "sensitivity": 1.6994132503696664,
    "kd_ctx_nm": 0.278,
    "kd_p1np_nm": 0.4463926941699373,
    "w_ctx": 0.23423059135038848,
    "w_p1np": 0.001,
    "biosensor_type": "array",
    "noise_preset": "realistic",
    "response_time_s": 600.0,
}
CHAMPION_SCORES = {
    "pmo_mild": 0.65,
    "pmo": 0.95,
    "ckd_mbd": 1.00,
    "healthy": 0.05,
}

# Second warm-start: kd_ctx scan optimum (0.316 nM) — same basin as champion,
# gives GP two anchor points so it doesn't drift to boundary
SCAN_OPTIMUM_CONFIG = {
    **CHAMPION_CONFIG,
    "name": "scan_opt",
    "kd_ctx_nm": 0.316,
}

VALID_TARGETS = ("pmo", "ckd_mbd", "both")

SCENARIO_LABELS = {
    "pmo_mild": "PMO-mild",
    "pmo": "PMO",
    "ckd_mbd": "CKD-MBD",
    "healthy": "Healthy (FP)",
}


# ---------------------------------------------------------------------------
# BO with warm-start injection
# ---------------------------------------------------------------------------

class WarmStartGPBO:
    """
    Thin wrapper around GaussianProcessBO that injects known-good warm-start
    configs into the observation set after the Sobol initialisation phase.

    This ensures the optimizer has the champion (kd_ctx=0.278 nM) as a
    reference point from the start, preventing the GP from wasting iterations
    re-discovering an already-known good basin.
    """

    def __init__(self, gp_bo, objective_fn, warm_start_configs: Optional[List[dict]] = None):
        self.gp_bo = gp_bo
        self.objective_fn = objective_fn
        self.warm_starts = warm_start_configs or []

    @staticmethod
    def _config_to_vector(config: dict, space) -> np.ndarray:
        """
        Convert config dict to normalised vector, safely handling single-value
        categoricals (biosensor_type=['array'], noise_preset=['realistic']).

        BiosensorSearchSpace.dict_to_vector() divides by (n_values - 1) which
        is zero for singleton categoricals — this version maps those to 0.5.
        """
        x = np.zeros(space.n_params, dtype=np.float32)
        for i, (name, param) in enumerate(space.parameters.items()):
            if param.param_type == "categorical":
                if len(param.values) <= 1:
                    x[i] = 0.5
                else:
                    val = config.get(name, param.values[0])
                    try:
                        code = param.values.index(val)
                    except ValueError:
                        code = 0
                    x[i] = code / (len(param.values) - 1)
            else:
                try:
                    x[i] = space.normalize_continuous(name, float(config[name]))
                except (KeyError, ValueError):
                    x[i] = 0.5
        return x

    def optimize(self) -> dict:
        import numpy as _np
        bo = self.gp_bo

        # Phase 1: standard Sobol initialisation
        bo.initialize_with_random_samples()

        # Phase 2: inject warm-start points into the observation set
        for ws_cfg in self.warm_starts:
            try:
                y_ws = self.objective_fn(ws_cfg)
                x_ws = self._config_to_vector(ws_cfg, bo.search_space)
                bo.X_observed = _np.vstack([bo.X_observed, x_ws.reshape(1, -1)])
                bo.y_observed = _np.append(bo.y_observed, y_ws)
                logger.info("Warm-start injected: kd_ctx=%.4f nM  y=%.4f",
                            ws_cfg.get("kd_ctx_nm", 0.0), y_ws)
            except Exception as exc:
                logger.warning("Warm-start injection failed: %s", exc)

        # Phase 3: BO iteration loop (replicates GaussianProcessBO.optimize
        # starting from the post-injection observation set)
        iteration_history = []
        for iteration in range(bo.n_iter):
            bo.fit_gp()
            x_next = bo.maximize_acquisition()
            config_next = bo.search_space.vector_to_dict(x_next)
            y_next = self.objective_fn(config_next)

            bo.X_observed = _np.vstack([bo.X_observed, x_next.reshape(1, -1)])
            bo.y_observed = _np.append(bo.y_observed, y_next)

            mu_next, sigma_next = bo.gp.predict(
                x_next[bo._active_dims].reshape(1, -1), return_std=True
            )
            y_best = bo.y_observed.max()

            iter_info = {
                "iteration": iteration + 1,
                "y": float(y_next),
                "y_best": float(y_best),
                "gp_mean": float(mu_next[0]),
                "gp_std": float(sigma_next[0]),
                "config": config_next,
            }
            iteration_history.append(iter_info)
            bo.iteration_history = iteration_history

            if (iteration + 1) % 25 == 0 or iteration == 0:
                logger.info("Iter %3d | y=%.4f | y_best=%.4f | std=%.4f",
                            iteration + 1, y_next, y_best, sigma_next[0])

        best_idx = bo.y_observed.argmax()
        x_best = bo.X_observed[best_idx]
        y_best = float(bo.y_observed[best_idx])
        config_best = bo.search_space.vector_to_dict(x_best)

        mu_best, sigma_best = bo.gp.predict(
            x_best[bo._active_dims].reshape(1, -1), return_std=True
        )

        return {
            "x_best": x_best,
            "y_best": y_best,
            "config_best": config_best,
            "gp_mean_best": float(mu_best[0]),
            "gp_std_best": float(sigma_best[0]),
            "ci_lower": float(max(0.0, mu_best[0] - 1.96 * sigma_best[0])),
            "ci_upper": float(min(1.0, mu_best[0] + 1.96 * sigma_best[0])),
            "X_observed": bo.X_observed,
            "y_observed": bo.y_observed,
            "iteration_history": iteration_history,
            "n_init": bo.n_init,
            "n_warmstart": len(self.warm_starts),
        }


# ---------------------------------------------------------------------------
# Main BO runner
# ---------------------------------------------------------------------------

def run_targeted_bo(
    target: str = "both",
    surrogate_dir: Optional[Path] = None,
    n_init: int = 50,
    n_iter: int = 150,
    seed: int = 42,
    verbose: bool = False,
) -> dict:
    """
    Run warm-started GP-BO using the frozen TherapeuticObjectiveV6.

    V6 jointly optimises all disease scenarios (PMO-mild, PMO, CKD-MBD,
    healthy specificity). The ``target`` parameter affects reporting only.

    Returns the full BO result dict including iteration history and best config.
    """
    from BO.core.surrogate_loader import SurrogateLoaderV3
    from evaluation.physics_forward_model import PhysicsForwardModel
    from evaluation.therapeutic_objective_v6 import TherapeuticObjectiveV6
    from search_space.biosensor_space import BiosensorSearchSpace
    from optimizer.gaussian_process_bo import GaussianProcessBO
    from acquisition.acquisition_functions import ExpectedImprovement

    if surrogate_dir is None:
        surrogate_dir = ROOT / "BO" / "bo_results"

    print(f"\n  Loading surrogates from {surrogate_dir / 'saved_ml'} ...")
    loader = SurrogateLoaderV3(results_dir=surrogate_dir)
    phys = PhysicsForwardModel()
    objective = TherapeuticObjectiveV6(phys, loader, apply_constraints=True)

    space = BiosensorSearchSpace()
    space.parameters["kd_ctx_nm"].lower = 0.05
    # Langmuir physics ceiling: CTX serum peaks at ~0.7 nM in PMO.
    # Above 2 nM kd, the CTX channel is <15% occupied and contributes negligible signal.
    # Without this cap BO exploits optimistic GP extrapolation at the boundary.
    space.parameters["kd_ctx_nm"].upper = 2.0

    acq = ExpectedImprovement(xi=0.01)
    gp_bo = GaussianProcessBO(
        objective_fn=objective,
        search_space=space,
        acquisition_fn=acq,
        n_init=n_init,
        n_iter=n_iter,
        random_state=seed,
    )

    # Two warm-starts anchor the GP in the 0.27–0.32 nM CTX affinity basin
    # (champion and kd_ctx scan optimum — methodologically independent, same region)
    ws_runner = WarmStartGPBO(
        gp_bo=gp_bo,
        objective_fn=objective,
        warm_start_configs=[{**CHAMPION_CONFIG}, {**SCAN_OPTIMUM_CONFIG}],
    )

    print(f"\n  Target: {target.upper()}")
    print(f"  Objective: TherapeuticObjectiveV6 (frozen - all scenarios)")
    print(f"  kd_CTX search space: [0.05, 2.0] nM  (Langmuir physics bound)")
    print(f"  Running BO: n_init={n_init}  n_iter={n_iter}  seed={seed}")
    print(f"  Total evaluations: {n_init + 2 + n_iter} (incl. 2 warm-starts)\n")

    result = ws_runner.optimize()

    best = result["config_best"]
    w_scl = max(0.0, 1.0 - best.get("w_ctx", 0) - best.get("w_p1np", 0))
    print(f"\n  Best config found:")
    print(f"    kd_SOST   = {best.get('kd_nm', 0):.4f} nM")
    print(f"    kd_CTX    = {best.get('kd_ctx_nm', 0):.4f} nM")
    print(f"    kd_P1NP   = {best.get('kd_p1np_nm', 0):.4f} nM")
    print(f"    alpha     = {best.get('sensitivity', 0):.4f}")
    print(f"    w_SOST    = {w_scl:.3f}")
    print(f"    w_CTX     = {best.get('w_ctx', 0):.3f}")
    print(f"    w_P1NP    = {best.get('w_p1np', 0):.3f}")
    print(f"    V6 score  = {result['y_best']:.4f}")

    return result


# ---------------------------------------------------------------------------
# Surrogate-only quick scoring (for comparison bars)
# ---------------------------------------------------------------------------

def surrogate_scenario_scores(config: dict, surrogate_dir: Path) -> Dict[str, float]:
    """Query surrogates for per-scenario DR (fast, no ODE sim)."""
    from BO.core.surrogate_loader import SurrogateLoaderV3
    loader = SurrogateLoaderV3(results_dir=surrogate_dir)

    scores = {}
    for scenario in ["pmo_mild", "pmo", "ckd_mbd", "healthy"]:
        dr, _fnr, _ttd = loader.predict(
            kd_nm=config.get("kd_nm", 1.0),
            sensitivity=config.get("sensitivity", 1.0),
            response_time=600.0,
            biosensor_type=config.get("biosensor_type", "array"),
            noise_preset=config.get("noise_preset", "realistic"),
            scenario=scenario,
            kd_ctx=config.get("kd_ctx_nm", 0.0),
            kd_p1np=config.get("kd_p1np_nm", 0.0),
            w_ctx=config.get("w_ctx", 0.0),
            w_p1np=config.get("w_p1np", 0.0),
        )
        scores[scenario] = float(dr)
    return scores


# ---------------------------------------------------------------------------
# Real-simulator validation
# ---------------------------------------------------------------------------

def validate_with_simulator(config: dict, n_trials: int = 30) -> Dict[str, dict]:
    """Run n_trials ODE simulations per scenario; return per-scenario stats."""
    from simulation.dataset.generator import DatasetGenerator

    biosensor_cfg = _build_sim_biosensor_config(config)
    results: Dict[str, dict] = {}
    scenarios = ["pmo_mild", "pmo", "ckd_mbd", "healthy"]

    for scenario in scenarios:
        drs, ttds = [], []
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = DatasetGenerator(
                antimony_model_path="simulation/models/bone_environment.ant",
                output_dir=tmpdir,
                seed=None,
            )
            for _ in range(n_trials):
                try:
                    r = gen.generate_single_simulation_instrumented(
                        scenario_name=scenario,
                        biosensor_config=biosensor_cfg,
                        noise_preset="realistic",
                        duration=3600.0,
                        num_points=361,
                        apply_variability=True,
                        instrument=False,
                    )
                    if r is not None:
                        drs.append(float(r["measurement"]["detection_rate"]))
                        ttds.append(float(r["measurement"]["time_to_detection"]))
                except Exception as exc:
                    logger.debug("Trial error [%s]: %s", scenario, exc)

        label = SCENARIO_LABELS.get(scenario, scenario)
        if drs:
            n = len(drs)
            dr_m = float(np.mean(drs))
            dr_s = float(np.std(drs))
            ci = 1.96 * dr_s / max(np.sqrt(n), 1)
            results[scenario] = {
                "dr_mean": dr_m,
                "dr_std": dr_s,
                "dr_ci95": [max(0.0, dr_m - ci), min(1.0, dr_m + ci)],
                "ttd_mean": float(np.mean(ttds)) if ttds else None,
                "n_trials": n,
            }
            print(f"    {label:<24} DR = {dr_m:.3f} +/- {dr_s:.3f}")
        else:
            results[scenario] = {"dr_mean": None, "dr_std": None,
                                  "dr_ci95": [None, None], "n_trials": 0}
            print(f"    {label:<24} FAILED")

    return results


def _build_sim_biosensor_config(cfg: dict) -> dict:
    """Convert BO config dict to biosensor_config for DatasetGenerator."""
    kd      = float(cfg.get("kd_nm", 1.0))
    kd_ctx  = float(cfg.get("kd_ctx_nm", kd))
    kd_p1np = float(cfg.get("kd_p1np_nm", kd))
    w_ctx   = float(cfg.get("w_ctx", 0.0))
    w_p1np  = float(cfg.get("w_p1np", 0.0))
    w_scl   = max(0.0, 1.0 - w_ctx - w_p1np)
    sens    = float(cfg.get("sensitivity", 1.0))

    H = {"scl": 0.375, "ctx": 0.200, "p1np": 0.350}
    P = {"scl": 0.875, "ctx": 0.500, "p1np": 0.525}

    def _occ(c, kd_v): return c / (kd_v + c + 1e-12)

    ref_scl  = _occ(H["scl"],  kd)
    ref_ctx  = _occ(H["ctx"],  kd_ctx)
    ref_p1np = _occ(H["p1np"], kd_p1np)

    def _composite(cd):
        n_s = _occ(cd["scl"],  kd)      / (ref_scl  + 1e-12)
        n_c = _occ(cd["ctx"],  kd_ctx)  / (ref_ctx  + 1e-12)
        n_p = _occ(cd["p1np"], kd_p1np) / (ref_p1np + 1e-12)
        return sens * (w_scl * n_s + w_ctx * n_c + w_p1np * n_p)

    sig_h  = _composite(H)
    sig_p  = _composite(P)
    hp_gap = (sig_p / sens) - 1.0
    threshold = float(sig_h + 1.25 * hp_gap)

    return {
        "circuit_type": "array",
        "kd_scl": kd, "kd_ctx": kd_ctx, "kd_p1np": kd_p1np,
        "w_scl": w_scl, "w_ctx": w_ctx, "w_p1np": w_p1np,
        "sensitivity": sens,
        "threshold": threshold,
        "dynamic_range": (0.0, sens * 3.0),
        "kd": kd,
    }


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

def generate_targeted_plots(
    target: str,
    bo_result: dict,
    surrogate_scores: Dict[str, float],
    sim_result: Optional[Dict[str, dict]],
    out_dir: Path,
) -> None:
    """Generate 4 diagnostic plots for the targeted BO result."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available - skipping plots")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

    config_best = bo_result["config_best"]
    kd_ctx_best = config_best.get("kd_ctx_nm", 0.0)
    kd_nm_best  = config_best.get("kd_nm", 1.0)
    y_obs  = bo_result["y_observed"]
    n_init = bo_result["n_init"]
    n_ws   = bo_result.get("n_warmstart", 0)

    target_label = {"pmo": "PMO", "ckd_mbd": "CKD-MBD", "both": "PMO + CKD-MBD"}.get(target, target.upper())

    # ── Figure 1: Convergence curve ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4.5))
    n_total = len(y_obs)
    x_all = np.arange(1, n_total + 1)
    best_so_far = np.maximum.accumulate(y_obs)

    ax.plot(x_all, y_obs, "o", ms=3, alpha=0.35, color="#4C72B0", label="Eval score")
    ax.plot(x_all, best_so_far, lw=2.2, color="#DD4444", label="Best so far")
    ax.axvline(n_init + n_ws, color="gray", ls="--", lw=1.2,
               label=f"BO start (n={n_init + n_ws})")
    ax.axhline(bo_result["y_best"], color="#DD4444", ls=":", lw=1, alpha=0.5)
    ax.set_xlabel("Evaluation number")
    ax.set_ylabel("V6 composite score")
    ax.set_title(f"BO Convergence — Targeted Biosensor ({target_label})\n"
                 f"Best V6 = {bo_result['y_best']:.4f}  "
                 f"(kd_CTX = {kd_ctx_best:.3f} nM)")
    ax.legend(fontsize=9)
    ax.set_xlim(1, n_total)
    fig.tight_layout()
    fig.savefig(out_dir / "convergence.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / 'convergence.png'}")

    # ── Figure 2: Per-scenario DR bar chart ─────────────────────────────────
    scenario_keys   = ["pmo_mild", "pmo", "ckd_mbd", "healthy"]
    scenario_labels = [SCENARIO_LABELS[k] for k in scenario_keys]

    targeted_dr  = [surrogate_scores.get(k, 0) for k in scenario_keys]
    champion_dr  = [CHAMPION_SCORES.get(k, 0) for k in scenario_keys]

    if sim_result:
        sim_dr = [
            sim_result.get(k, {}).get("dr_mean") or surrogate_scores.get(k, 0)
            for k in scenario_keys
        ]
        sim_ci_lo = [(sim_result.get(k, {}).get("dr_ci95") or [0, 0])[0] for k in scenario_keys]
        sim_ci_hi = [(sim_result.get(k, {}).get("dr_ci95") or [0, 0])[1] for k in scenario_keys]
        yerr_lo = [max(0, dr - lo) for dr, lo in zip(sim_dr, sim_ci_lo)]
        yerr_hi = [max(0, hi - dr) for dr, hi in zip(sim_dr, sim_ci_hi)]
    else:
        sim_dr = targeted_dr
        yerr_lo = yerr_hi = [0] * 4

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(scenario_labels))
    width = 0.28

    ax.bar(x - width, champion_dr, width, label="Champion (seed-999)",
           color="#4C72B0", alpha=0.85)
    ax.bar(x, targeted_dr, width, label="Targeted BO (surrogate)",
           color="#55A868", alpha=0.85)
    if sim_result:
        ax.bar(x + width, sim_dr, width, label="Targeted BO (ODE validated)",
               color="#C44E52", alpha=0.85,
               yerr=[yerr_lo, yerr_hi], capsize=4, error_kw={"elinewidth": 1.2})

    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels)
    ax.set_ylabel("Detection rate")
    ax.set_ylim(0, 1.15)
    ax.axhline(1.0, color="k", lw=0.7, ls=":")
    ax.set_title(f"Per-Scenario Detection Rate — Champion vs Targeted BO ({target_label})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "scenario_dr.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / 'scenario_dr.png'}")

    # ── Figure 3: Channel weights comparison ────────────────────────────────
    w_ctx_opt  = config_best.get("w_ctx", 0)
    w_p1np_opt = config_best.get("w_p1np", 0)
    w_scl_opt  = max(0.0, 1.0 - w_ctx_opt - w_p1np_opt)

    w_ctx_ch   = CHAMPION_CONFIG.get("w_ctx", 0)
    w_p1np_ch  = CHAMPION_CONFIG.get("w_p1np", 0)
    w_scl_ch   = max(0.0, 1.0 - w_ctx_ch - w_p1np_ch)

    channels    = ["SOST", "CTX", "P1NP"]
    opt_weights = [w_scl_opt, w_ctx_opt, w_p1np_opt]
    ch_weights  = [w_scl_ch,  w_ctx_ch,  w_p1np_ch]
    colors = ["#4C72B0", "#DD8844", "#55A868"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    ax1.bar(channels, opt_weights, color=colors, alpha=0.85)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Channel weight")
    ax1.set_title(f"Targeted BO\nkd_CTX = {kd_ctx_best:.3f} nM  |  kd_SOST = {kd_nm_best:.3f} nM")
    for i, (ch, w) in enumerate(zip(channels, opt_weights)):
        ax1.text(i, w + 0.02, f"{w:.3f}", ha="center", va="bottom", fontsize=10)

    ax2.bar(channels, ch_weights, color=colors, alpha=0.85)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Channel weight")
    ax2.set_title(f"Champion (seed-999)\nkd_CTX = {CHAMPION_CONFIG['kd_ctx_nm']:.3f} nM  |  kd_SOST = {CHAMPION_CONFIG['kd_nm']:.3f} nM")
    for i, (ch, w) in enumerate(zip(channels, ch_weights)):
        ax2.text(i, w + 0.02, f"{w:.3f}", ha="center", va="bottom", fontsize=10)

    fig.suptitle("Aptamer Channel Weights: Targeted BO vs Champion", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "channel_weights.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / 'channel_weights.png'}")

    # ── Figure 4: Langmuir occupancy curves ─────────────────────────────────
    conc_range = np.linspace(0, 2.0, 400)
    sost_concs = {"Healthy": 0.375, "PMO-mild": 0.5625, "PMO": 0.875, "CKD-MBD": 1.125}
    ctx_concs  = {"Healthy": 0.200, "PMO-mild": 0.300,  "PMO": 0.500, "CKD-MBD": 0.500}

    def occ(c, kd): return c / (kd + c)

    fig, (ax_sost, ax_ctx) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax_sost.plot(conc_range, occ(conc_range, kd_nm_best),
                 lw=2.2, color="#4C72B0",
                 label=f"Targeted BO (kd={kd_nm_best:.3f} nM)")
    ax_sost.plot(conc_range, occ(conc_range, CHAMPION_CONFIG["kd_nm"]),
                 lw=2.2, ls="--", color="#4C72B0", alpha=0.5,
                 label=f"Champion (kd={CHAMPION_CONFIG['kd_nm']:.3f} nM)")
    for sc, c in sost_concs.items():
        ax_sost.axvline(c, color="gray", ls=":", lw=0.8)
        ax_sost.text(c + 0.02, 0.02, sc, fontsize=7.5, rotation=90, va="bottom", color="gray")
    ax_sost.set_xlabel("SOST concentration (nM)")
    ax_sost.set_ylabel("Langmuir occupancy θ")
    ax_sost.set_title("SOST Channel — Langmuir Occupancy")
    ax_sost.legend(fontsize=8)
    ax_sost.set_xlim(0, 2.0)
    ax_sost.set_ylim(0, 1)

    ax_ctx.plot(conc_range, occ(conc_range, kd_ctx_best),
                lw=2.2, color="#DD8844",
                label=f"Targeted BO (kd={kd_ctx_best:.3f} nM)")
    ax_ctx.plot(conc_range, occ(conc_range, CHAMPION_CONFIG["kd_ctx_nm"]),
                lw=2.2, ls="--", color="#DD8844", alpha=0.5,
                label=f"Champion (kd={CHAMPION_CONFIG['kd_ctx_nm']:.3f} nM)")
    for sc, c in ctx_concs.items():
        ax_ctx.axvline(c, color="gray", ls=":", lw=0.8)
        ax_ctx.text(c + 0.005, 0.02, sc, fontsize=7.5, rotation=90, va="bottom", color="gray")
    ax_ctx.set_xlabel("CTX concentration (nM)")
    ax_ctx.set_ylabel("Langmuir occupancy θ")
    ax_ctx.set_title("CTX Channel — Langmuir Occupancy")
    ax_ctx.legend(fontsize=8)
    ax_ctx.set_xlim(0, 0.8)
    ax_ctx.set_ylim(0, 1)

    fig.suptitle("Langmuir Occupancy Curves: Targeted BO vs Champion", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "langmuir_curves.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / 'langmuir_curves.png'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Targeted biosensor optimizer — GP-BO with warm-start"
    )
    parser.add_argument("--target",         default="both",
                        choices=VALID_TARGETS,
                        help="Primary clinical deployment target (affects reporting only)")
    parser.add_argument("--n-init",         type=int,  default=50,
                        help="BO Sobol init points")
    parser.add_argument("--n-iter",         type=int,  default=150,
                        help="BO GP-EI iterations")
    parser.add_argument("--seed",           type=int,  default=42,
                        help="Random seed")
    parser.add_argument("--n-trials",       type=int,  default=30,
                        help="ODE simulation trials per scenario")
    parser.add_argument("--skip-simulation", action="store_true",
                        help="Skip ODE validation (surrogate scores only)")
    parser.add_argument("--out-dir",        default=None,
                        help="Output directory (default: BO/bo_results/targeted/)")
    parser.add_argument("--verbose",        action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)]
    )

    target_label = {"pmo": "PMO", "ckd_mbd": "CKD-MBD", "both": "PMO + CKD-MBD"}[args.target]

    print("\n" + "=" * 65)
    print("  GENEVO Targeted Biosensor Optimizer")
    print("=" * 65)
    print(f"\n  Clinical target : {target_label}")
    print(f"  Objective       : TherapeuticObjectiveV6 (frozen, all scenarios)")
    print(f"  Warm-start      : Champion (0.278 nM) + kd_ctx scan optimum (0.316 nM)")

    results_dir = ROOT / "BO" / "bo_results"
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "targeted"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"

    # --- Run BO ---
    bo_result = run_targeted_bo(
        target=args.target,
        surrogate_dir=results_dir,
        n_init=args.n_init,
        n_iter=args.n_iter,
        seed=args.seed,
        verbose=args.verbose,
    )
    config_best = bo_result["config_best"]

    # --- Surrogate per-scenario scores ---
    print("\n  Surrogate per-scenario DR:")
    surr_scores = surrogate_scenario_scores(config_best, results_dir)
    for sc, dr in surr_scores.items():
        print(f"    {sc:<12} {dr:.3f}")

    # --- ODE validation ---
    sim_result = None
    if not args.skip_simulation:
        print(f"\n  Validating with real ODE simulator (n={args.n_trials} per scenario)...")
        sim_result = validate_with_simulator(config_best, n_trials=args.n_trials)

    # --- Compute ODE-based clinical utility (NOT the surrogate V6) ---
    # DR_mean − FP: simple, interpretable, and directly comparable across configs.
    # This is a DIFFERENT metric from the surrogate V6 that BO optimised.
    # Do NOT compare them numerically — they are different functions.
    clinical_utility_ode = None
    if sim_result:
        dr_vals = [sim_result[s]["dr_mean"] for s in ["pmo_mild", "pmo", "ckd_mbd"]
                   if sim_result[s].get("dr_mean") is not None]
        dr_disease = float(np.mean(dr_vals)) if dr_vals else 0.0
        fp_rate    = sim_result.get("healthy", {}).get("dr_mean") or 0.0
        clinical_utility_ode = dr_disease - fp_rate  # champion reference ≈ 0.90 − 0.17 = 0.73

    # --- Plots ---
    print(f"\n  Generating diagnostic plots -> {plots_dir}")
    generate_targeted_plots(
        target=args.target,
        bo_result=bo_result,
        surrogate_scores=surr_scores,
        sim_result=sim_result,
        out_dir=plots_dir,
    )

    # --- Save JSON ---
    w_scl = max(0.0, 1.0 - config_best.get("w_ctx", 0) - config_best.get("w_p1np", 0))
    output = {
        "target": args.target,
        "objective": "TherapeuticObjectiveV6",
        "bo_score": bo_result["y_best"],
        "gp_ci_95": [bo_result["ci_lower"], bo_result["ci_upper"]],
        "config": {
            "kd_nm": config_best.get("kd_nm"),
            "kd_ctx_nm": config_best.get("kd_ctx_nm"),
            "kd_p1np_nm": config_best.get("kd_p1np_nm"),
            "sensitivity": config_best.get("sensitivity"),
            "w_scl": w_scl,
            "w_ctx": config_best.get("w_ctx"),
            "w_p1np": config_best.get("w_p1np"),
            "biosensor_type": "array",
            "noise_preset": "realistic",
            "response_time_s": 600.0,
        },
        "surrogate_dr": surr_scores,
        "sim_dr": {
            k: {
                "dr_mean": v.get("dr_mean"),
                "dr_ci95": v.get("dr_ci95"),
                "n_trials": v.get("n_trials"),
            }
            for k, v in (sim_result or {}).items()
        },
        "clinical_utility_ode": clinical_utility_ode,
        "clinical_utility_note": "DR_mean(disease) - FP_rate; different metric from surrogate V6",
        "champion_clinical_utility": 0.73,
        "champion_v6_surrogate": 0.685,
        "n_init": args.n_init,
        "n_iter": args.n_iter,
        "seed": args.seed,
    }
    out_file = out_dir / "best_config.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    # --- Summary ---
    print("\n" + "=" * 65)
    print("  TARGETED BIOSENSOR SUMMARY")
    print("=" * 65)
    print(f"\n  Clinical target : {target_label}")
    print(f"\n  Design parameters:")
    print(f"    kd_SOST   = {config_best.get('kd_nm', 0):.4f} nM")
    print(f"    kd_CTX    = {config_best.get('kd_ctx_nm', 0):.4f} nM")
    print(f"    kd_P1NP   = {config_best.get('kd_p1np_nm', 0):.4f} nM")
    print(f"    alpha     = {config_best.get('sensitivity', 0):.4f}")
    print(f"    w_SOST    = {w_scl:.3f}")
    print(f"    w_CTX     = {config_best.get('w_ctx', 0):.3f}")
    print(f"    w_P1NP    = {config_best.get('w_p1np', 0):.3f}")
    print(f"\n  V6 score (surrogate, what BO optimised): {bo_result['y_best']:.4f}")
    print(f"  Champion V6 (surrogate):                 0.685")
    if clinical_utility_ode is not None:
        print(f"\n  Clinical utility ODE (DR_mean - FP):     {clinical_utility_ode:.4f}")
        print(f"  Champion reference (DR_mean=0.90, FP=0.17): ~0.73")
        print(f"  [WARNING] V6 and clinical utility are DIFFERENT metrics - do not compare them")
    print(f"\n  Per-scenario DR (surrogate):")
    for sc, dr in surr_scores.items():
        print(f"    {sc:<14} {dr:.3f}")
    if sim_result:
        print(f"\n  Per-scenario DR (ODE sim):")
        for sc in ["pmo_mild", "pmo", "ckd_mbd", "healthy"]:
            dr_m = (sim_result.get(sc) or {}).get("dr_mean")
            label = SCENARIO_LABELS.get(sc, sc)
            if dr_m is not None:
                print(f"    {label:<22} {dr_m:.3f}")
    print(f"\n  Output: {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
