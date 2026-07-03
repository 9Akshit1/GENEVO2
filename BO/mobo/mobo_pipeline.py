#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-Objective Bayesian Optimization Pipeline.

Orchestrates the MOBO loop:
  1. Latin Hypercube Sampling initialization
  2. Evaluate objectives on initial samples
  3. Fit GP surrogates per objective
  4. Maximize EHVI to find next candidate
  5. Evaluate new candidate
  6. Update archive and GP
  7. Repeat for n_iterations

Outputs:
  - Pareto front configurations
  - Full objective archive (all evaluated points)
  - Hypervolume convergence curve

Usage:
    python BO/run_mobo.py --n-init 32 --n-iter 100 --seed 42
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .mobo_objectives import MOBOObjectives, REFERENCE_POINT, OBJECTIVE_NAMES, N_OBJECTIVES
from .ehvi_acquisition import MCExpectedHypervolumeImprovement
from .pareto import pareto_front, non_dominated_sort, hypervolume

logger = logging.getLogger(__name__)


@dataclass
class MOBOResult:
    """Container for MOBO optimization results."""
    pareto_configs: List[Dict] = field(default_factory=list)
    pareto_objectives: np.ndarray = field(default_factory=lambda: np.zeros((0, N_OBJECTIVES)))
    all_configs: List[Dict] = field(default_factory=list)
    all_objectives: np.ndarray = field(default_factory=lambda: np.zeros((0, N_OBJECTIVES)))
    hypervolume_curve: List[float] = field(default_factory=list)
    n_feasible: int = 0
    best_per_objective: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "n_pareto": len(self.pareto_configs),
            "n_evaluated": len(self.all_configs),
            "n_feasible": self.n_feasible,
            "final_hypervolume": float(self.hypervolume_curve[-1]) if self.hypervolume_curve else 0.0,
            "hypervolume_curve": self.hypervolume_curve,
            "objective_names": OBJECTIVE_NAMES,
            "reference_point": REFERENCE_POINT.tolist(),
            "pareto_configs": self.pareto_configs,
            "pareto_objectives": self.pareto_objectives.tolist() if len(self.pareto_objectives) > 0 else [],
            "best_per_objective": self.best_per_objective,
        }


def _lhs_sample(n: int, d: int, bounds: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Latin Hypercube Sampling in normalized [0,1]^d space."""
    X = np.zeros((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        X[:, j] = (perm + rng.uniform(0, 1, n)) / n
    # Scale to bounds
    lo, hi = bounds[:, 0], bounds[:, 1]
    return lo + X * (hi - lo)


class MOBOPipeline:
    """
    Full MOBO pipeline with EHVI acquisition.

    The search space is continuous (d=7 effective dimensions after fixing
    biosensor_type, noise_preset, response_time to their optimal values).
    Categorical parameters (biosensor_type, noise_preset, target_scenario)
    are handled separately via fixed values.
    """

    # Fixed categorical values (frozen from BO results)
    FIXED_CATS = {
        "biosensor_type": "array",
        "noise_preset":   "realistic",
        "target_scenario": "pmo",   # Most challenging — optimize for pmo target
        "response_time_s": 600.0,   # Dead dimension — fix at geometric mean
    }

    # Continuous parameter bounds (normalized to [0,1] in search space)
    # Actual bounds in physical units:
    _CONT_PARAMS = {
        "kd_nm":      (0.1,   10.0,  "log"),
        "sensitivity":(0.5,   5.0,   "log"),
        "kd_ctx_nm":  (0.1,   10.0,  "log"),
        "kd_p1np_nm": (0.1,   10.0,  "log"),
        "w_ctx":      (0.01,  0.49,  "linear"),
        "w_p1np":     (0.01,  0.49,  "linear"),
    }

    def __init__(
        self,
        surrogate_loader,
        n_init: int = 32,
        n_iterations: int = 100,
        n_mc_samples: int = 64,
        n_restarts: int = 10,
        reference_point: Optional[np.ndarray] = None,
        seed: int = 42,
        output_dir: str = "BO/bo_results_mobo",
        warm_start_configs: Optional[List[Dict]] = None,
    ):
        self.n_init = n_init
        self.n_iterations = n_iterations
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.warm_start_configs = warm_start_configs or []

        self.objectives = MOBOObjectives(surrogate_loader)
        self.acquisition = MCExpectedHypervolumeImprovement(
            n_objectives=N_OBJECTIVES,
            reference_point=reference_point if reference_point is not None else REFERENCE_POINT,
            n_mc_samples=n_mc_samples,
            n_restarts=n_restarts,
        )

        # Build normalized bounds for continuous parameters
        param_names = list(self._CONT_PARAMS.keys())
        self._param_names = param_names
        d = len(param_names)
        self._bounds_norm = np.zeros((d, 2))
        self._bounds_norm[:, 1] = 1.0   # all params normalized to [0, 1]

    def _norm_to_config(self, x: np.ndarray) -> Dict:
        """Convert normalized [0,1]^d vector to biosensor config dict."""
        config = dict(self.FIXED_CATS)
        for j, (name, (lo, hi, scale)) in enumerate(self._CONT_PARAMS.items()):
            v = float(x[j])
            v = np.clip(v, 0.0, 1.0)
            if scale == "log":
                log_lo, log_hi = np.log10(lo), np.log10(hi)
                config[name] = float(10.0 ** (log_lo + v * (log_hi - log_lo)))
            else:
                config[name] = float(lo + v * (hi - lo))
        return config

    def _config_to_norm(self, config: Dict) -> np.ndarray:
        """Convert config dict to normalized vector."""
        x = np.zeros(len(self._param_names))
        for j, (name, (lo, hi, scale)) in enumerate(self._CONT_PARAMS.items()):
            v = config.get(name, lo)
            if scale == "log":
                log_lo, log_hi = np.log10(lo), np.log10(hi)
                x[j] = (np.log10(max(v, lo * 0.1)) - log_lo) / (log_hi - log_lo)
            else:
                x[j] = (v - lo) / (hi - lo)
        return np.clip(x, 0.0, 1.0)

    def _feasible_pareto_front(
        self, Y_arr: np.ndarray, feasible_mask: List[bool]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Pareto front restricted to feasible designs only."""
        mask = np.array(feasible_mask, dtype=bool)
        if not mask.any():
            return np.zeros((0, N_OBJECTIVES)), np.array([], dtype=int)
        feas_idx = np.where(mask)[0]
        pf_Y, pf_local = pareto_front(Y_arr[feas_idx])
        return pf_Y, feas_idx[pf_local]

    def run(self) -> MOBOResult:
        """
        Execute the full MOBO campaign.

        Returns:
            MOBOResult with Pareto front, archive, and convergence curve.
        """
        rng = np.random.RandomState(self.seed)
        result = MOBOResult()

        # --- Warm start: inject known-good configs before LHS ---
        X_all = []
        Y_all = []
        feasible_mask: List[bool] = []

        if self.warm_start_configs:
            logger.info(f"MOBO warm start: injecting {len(self.warm_start_configs)} config(s)")
            for ws_cfg in self.warm_start_configs:
                x_ws = self._config_to_norm(ws_cfg)
                y_ws = self.objectives.evaluate(ws_cfg)
                is_feas_ws = self.objectives.is_feasible(ws_cfg, y_ws)
                X_all.append(x_ws)
                Y_all.append(y_ws)
                feasible_mask.append(is_feas_ws)
                logger.info(f"  Warm start: y={y_ws.round(3)} feasible={is_feas_ws}")
                print(f"  [MOBO warm start] y={y_ws.round(3)} feasible={is_feas_ws}", flush=True)

        # --- Initialization ---
        logger.info(f"MOBO init: {self.n_init} LHS samples")
        X_init = _lhs_sample(self.n_init, len(self._param_names), self._bounds_norm, rng)

        for i, x in enumerate(X_init):
            config = self._norm_to_config(x)
            y = self.objectives.evaluate(config)
            is_feas = self.objectives.is_feasible(config, y)
            X_all.append(x)
            Y_all.append(y)
            feasible_mask.append(is_feas)
            if (i + 1) % 8 == 0:
                logger.info(f"  Init {i+1}/{self.n_init}: y={y.round(3)} feasible={is_feas}")
                print(f"  [MOBO init] {i+1}/{self.n_init} | feasible={is_feas} | y={y.round(3)}", flush=True)

        X_arr = np.array(X_all)
        Y_arr = np.array(Y_all)

        result.all_configs = [self._norm_to_config(x) for x in X_all]
        result.all_objectives = Y_arr.copy()

        # Track hypervolume after init (feasible Pareto front only)
        pf_Y, pf_idx = self._feasible_pareto_front(Y_arr, feasible_mask)
        hv = hypervolume(pf_Y, self.acquisition.ref) if len(pf_Y) > 0 else 0.0
        result.hypervolume_curve.append(hv)
        logger.info(
            f"Init HV: {hv:.4f} | Feasible Pareto size: {len(pf_Y)} | "
            f"Feasible: {sum(feasible_mask)}/{self.n_init}"
        )
        print(
            f"[MOBO] Init done. HV={hv:.4f} | Pareto={len(pf_Y)} | "
            f"Feasible={sum(feasible_mask)}/{self.n_init} | Starting {self.n_iterations} BO iters...",
            flush=True,
        )

        # --- MOBO loop ---
        for iteration in range(1, self.n_iterations + 1):
            # Fit GPs on all data (feasible and infeasible — more data = better GP)
            self.acquisition.fit(X_arr, Y_arr)

            # Maximize EHVI
            x_next, ehvi_val = self.acquisition.acquire(
                bounds=self._bounds_norm, seed=self.seed + iteration
            )
            config_next = self._norm_to_config(x_next)
            y_next = self.objectives.evaluate(config_next)
            is_feas_next = self.objectives.is_feasible(config_next, y_next)

            # Update archive
            X_arr = np.vstack([X_arr, x_next.reshape(1, -1)])
            Y_arr = np.vstack([Y_arr, y_next.reshape(1, -1)])
            feasible_mask.append(is_feas_next)
            result.all_configs.append(config_next)
            result.all_objectives = Y_arr.copy()

            # Track feasible Pareto + hypervolume
            pf_Y, pf_idx = self._feasible_pareto_front(Y_arr, feasible_mask)
            hv = hypervolume(pf_Y, self.acquisition.ref) if len(pf_Y) > 0 else 0.0
            result.hypervolume_curve.append(hv)

            if iteration % 10 == 0 or iteration == 1:
                logger.info(
                    f"Iter {iteration:3d}/{self.n_iterations} | "
                    f"EHVI={ehvi_val:.4f} | HV={hv:.4f} | "
                    f"y_next={y_next.round(3)} | Pareto size={len(pf_Y)}"
                )
                print(
                    f"[MOBO] Iter {iteration:3d}/{self.n_iterations} | "
                    f"HV={hv:.4f} | Pareto={len(pf_Y)} | feas={is_feas_next}",
                    flush=True,
                )

        # --- Finalize ---
        pf_Y, pf_idx = self._feasible_pareto_front(Y_arr, feasible_mask)
        result.pareto_objectives = pf_Y
        result.pareto_configs = [result.all_configs[i] for i in pf_idx]
        result.n_feasible = sum(feasible_mask)

        # Best per objective
        for obj_idx, name in enumerate(OBJECTIVE_NAMES):
            best_row = int(np.argmax(Y_arr[:, obj_idx]))
            result.best_per_objective[name] = {
                "value": float(Y_arr[best_row, obj_idx]),
                "config": result.all_configs[best_row],
            }

        self._save(result)
        logger.info(
            f"MOBO complete: {len(pf_Y)} Pareto solutions | "
            f"HV={result.hypervolume_curve[-1]:.4f} | "
            f"feasible={result.n_feasible}/{len(Y_arr)}"
        )
        return result

    def _save(self, result: MOBOResult):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out = self.output_dir / "mobo_results.json"
        with open(out, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Results saved to {out}")

    def print_pareto_summary(self, result: MOBOResult):
        """ASCII table of Pareto front for console output."""
        print("\nMOBO Pareto Front Summary")
        print("=" * 70)
        print(f"{'#':>3}  {'DR_mean':>8}  {'Therapeutic':>12}  {'Specificity':>12}  {'kd_nm':>7}  {'sens':>5}  {'w_ctx':>6}")
        print("-" * 70)
        for i, (y, cfg) in enumerate(zip(result.pareto_objectives, result.pareto_configs)):
            print(
                f"{i+1:>3}  {y[0]:>8.3f}  {y[1]:>12.3f}  {y[2]:>12.3f}  "
                f"{cfg.get('kd_nm', 0):>7.3f}  {cfg.get('sensitivity', 0):>5.2f}  "
                f"{cfg.get('w_ctx', 0):>6.3f}"
            )
        print("=" * 70)
        print(f"Final hypervolume: {result.hypervolume_curve[-1]:.4f}")
        print(f"Feasible designs: {result.n_feasible}/{len(result.all_configs)}")
