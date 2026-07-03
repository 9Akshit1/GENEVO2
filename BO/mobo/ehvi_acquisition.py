#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Monte Carlo Expected Hypervolume Improvement (MC-EHVI) acquisition function.

Algorithm:
  1. Fit a GP surrogate per objective (sklearn GaussianProcessRegressor).
  2. At a candidate x, sample N predictive draws from each GP.
  3. For each draw, compute the hypervolume improvement (HVI) of the
     sampled y against the current Pareto archive.
  4. EHVI(x) = mean(HVI over all samples).
  5. Maximize EHVI over the search space using random restarts.

Reference:
  Emmerich et al. 2006: "Single- and multiobjective evolutionary
  optimization assisted by Gaussian random field metamodels."

Design notes:
  - Uses the exact HVI formula (not an approximation) for m<=3 objectives.
  - GP kernel: Matern 5/2 with noise; automatic hyperparameter tuning.
  - Random restart optimization: n_restarts uniform starts + L-BFGS-B.
  - Batch mode: evaluate_batch efficiently scores multiple candidates.
"""

import warnings
import numpy as np
import logging
from typing import List, Optional, Tuple

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.exceptions import ConvergenceWarning
from scipy.optimize import minimize

from .pareto import pareto_front, hypervolume_improvement, hypervolume

logger = logging.getLogger(__name__)


class MCExpectedHypervolumeImprovement:
    """
    MC-EHVI acquisition function for multi-objective BO.

    Maintains a separate GP per objective. Call `fit()` with the
    current observed designs and `acquire()` to find the next candidate.
    """

    def __init__(
        self,
        n_objectives: int = 3,
        reference_point: Optional[np.ndarray] = None,
        n_mc_samples: int = 128,
        n_restarts: int = 10,
        noise_level: float = 1e-4,
    ):
        self.n_objectives = n_objectives
        self.ref = reference_point if reference_point is not None else np.zeros(n_objectives)
        self.n_mc_samples = n_mc_samples
        self.n_restarts = n_restarts
        self.noise_level = noise_level
        self._gps: List[Optional[GaussianProcessRegressor]] = [None] * n_objectives
        self._X_train: Optional[np.ndarray] = None
        self._Y_train: Optional[np.ndarray] = None
        self._pareto_Y: Optional[np.ndarray] = None

    def _make_kernel(self):
        return ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=np.ones(self._n_dims),
            length_scale_bounds=[(1e-3, 1e4)] * self._n_dims,
            nu=2.5,
        ) + WhiteKernel(noise_level=self.noise_level, noise_level_bounds=(1e-10, 1e-1))

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        Fit GP per objective on observed data.

        Args:
            X: (n, d) normalized parameter vectors.
            Y: (n, m) objective values (maximize all).
        """
        self._n_dims = X.shape[1]
        self._X_train = X.copy()
        self._Y_train = Y.copy()

        pf_Y, _ = pareto_front(Y)
        self._pareto_Y = pf_Y

        for obj_idx in range(self.n_objectives):
            gp = GaussianProcessRegressor(
                kernel=self._make_kernel(),
                n_restarts_optimizer=3,
                normalize_y=True,
                alpha=1e-6,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                gp.fit(X, Y[:, obj_idx])
            self._gps[obj_idx] = gp
            logger.debug(f"GP objective {obj_idx}: log_marginal_likelihood={gp.log_marginal_likelihood_value_:.3f}")

    def predict_mean_std(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        GP posterior mean and std at X for all objectives.

        Returns:
            means: (n, m), stds: (n, m)
        """
        means = np.zeros((len(X), self.n_objectives))
        stds  = np.zeros((len(X), self.n_objectives))
        for obj_idx, gp in enumerate(self._gps):
            if gp is None:
                continue
            mu, sigma = gp.predict(X, return_std=True)
            means[:, obj_idx] = mu
            stds[:, obj_idx]  = sigma
        return means, stds

    def ehvi(self, x: np.ndarray) -> float:
        """
        Monte Carlo EHVI at a single point x (shape: (d,)).

        Samples n_mc_samples from GP posteriors and averages HVI.
        """
        if any(gp is None for gp in self._gps):
            return 0.0

        x2d = x.reshape(1, -1)
        means, stds = self.predict_mean_std(x2d)
        means = means[0]
        stds  = stds[0]

        # Sample from GP posteriors
        samples = np.random.randn(self.n_mc_samples, self.n_objectives) * stds + means

        pareto_Y = self._pareto_Y if self._pareto_Y is not None else np.zeros((0, self.n_objectives))

        hvi_vals = []
        for s in samples:
            hvi = hypervolume_improvement(pareto_Y, s, self.ref)
            hvi_vals.append(hvi)

        return float(np.mean(hvi_vals))

    def ehvi_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate EHVI for a batch of candidates. Returns (n,) array."""
        return np.array([self.ehvi(X[i]) for i in range(len(X))], dtype=float)

    def acquire(
        self,
        bounds: np.ndarray,
        n_restarts: Optional[int] = None,
        seed: int = 0,
    ) -> Tuple[np.ndarray, float]:
        """
        Find the next candidate by maximizing EHVI.

        Uses random multi-start L-BFGS-B (gradient-free fallback via
        numerical differentiation).

        Args:
            bounds: (d, 2) array of [lower, upper] for each dimension.
            n_restarts: number of random restarts (default: self.n_restarts).
            seed: random seed.

        Returns:
            (x_best, ehvi_best) — x_best is in original (normalized) scale.
        """
        n_restarts = n_restarts or self.n_restarts
        rng = np.random.RandomState(seed)
        d = bounds.shape[0]

        best_x = None
        best_val = -np.inf

        # Random initial points
        x0_set = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, d))

        for x0 in x0_set:
            result = minimize(
                fun=lambda x: -self.ehvi(x),    # negate to maximize
                x0=x0,
                bounds=list(zip(bounds[:, 0], bounds[:, 1])),
                method="L-BFGS-B",
                options={"maxiter": 100, "ftol": 1e-6},
            )
            val = -result.fun
            if val > best_val:
                best_val = val
                best_x = result.x.copy()

        if best_x is None:
            best_x = rng.uniform(bounds[:, 0], bounds[:, 1])
            best_val = self.ehvi(best_x)

        return best_x, float(best_val)

    def ucb_batch(self, X: np.ndarray, beta: float = 2.0) -> np.ndarray:
        """
        Multi-objective UCB (scalarized) as cheaper alternative to EHVI.

        Score = sum_m (mean_m + beta * std_m)

        Use this for cheap candidate screening before expensive EHVI.
        """
        means, stds = self.predict_mean_std(X)
        return float(np.sum(means + beta * stds, axis=1))

    @property
    def pareto_archive(self) -> np.ndarray:
        """Current Pareto front objective values."""
        return self._pareto_Y if self._pareto_Y is not None else np.zeros((0, self.n_objectives))

    @property
    def current_hypervolume(self) -> float:
        """Hypervolume of the current Pareto archive."""
        if self._pareto_Y is None or len(self._pareto_Y) == 0:
            return 0.0
        return hypervolume(self._pareto_Y, self.ref)
