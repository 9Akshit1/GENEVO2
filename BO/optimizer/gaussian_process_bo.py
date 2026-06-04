#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gaussian Process-based Bayesian Optimization.

Main BO algorithm using sklearn's GaussianProcessRegressor with Matern kernel
and Expected Improvement (EI) acquisition function.
"""

import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
import logging

logger = logging.getLogger(__name__)


class GaussianProcessBO:
    """
    Gaussian Process-based Bayesian Optimization.

    Uses sklearn's GP with Matern kernel and EI acquisition for optimization.
    """

    def __init__(
        self,
        objective_fn,
        search_space,
        acquisition_fn,
        n_init: int = 20,
        n_iter: int = 80,
        random_state: int = 42,
    ):
        """
        Initialize BO optimizer.

        Args:
            objective_fn: Callable objective function to maximize
            search_space: BiosensorSearchSpace instance
            acquisition_fn: AcquisitionFunction instance (e.g., ExpectedImprovement)
            n_init: Number of initial random samples
            n_iter: Number of BO iterations
            random_state: Random seed for reproducibility
        """
        self.objective_fn = objective_fn
        self.search_space = search_space
        self.acquisition_fn = acquisition_fn
        self.n_init = n_init
        self.n_iter = n_iter
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        # Initialize GP
        kernel = Matern(nu=2.5, length_scale_bounds=(0.01, 10.0)) + WhiteKernel(
            noise_level_bounds=(1e-5, 1.0)
        )
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=random_state,
        )

        # Storage for optimization history
        self.X_observed = []
        self.y_observed = []
        self.iteration_history = []

    def initialize_with_random_samples(self) -> None:
        """Generate initial random samples using Sobol sequence."""
        logger.info(f"Generating {self.n_init} initial random samples...")

        # Use Sobol quasi-random sampling for better coverage
        try:
            # Try modern scipy API first (random_state parameter)
            try:
                sampler = qmc.Sobol(d=self.search_space.n_params, random_state=self.rng)
            except TypeError:
                # Fall back to older scipy API (seed parameter)
                sampler = qmc.Sobol(d=self.search_space.n_params, seed=self.random_state)
            X_init_normalized = sampler.random(n=self.n_init)
        except Exception as e:
            logger.warning(f"Sobol sampling failed: {e}. Using uniform random sampling.")
            X_init_normalized = self.rng.uniform(0, 1, (self.n_init, self.search_space.n_params))

        # Evaluate at initial points
        for i, x_norm in enumerate(X_init_normalized):
            config = self.search_space.vector_to_dict(x_norm)
            y = self.objective_fn(config)

            self.X_observed.append(x_norm)
            self.y_observed.append(y)

            if (i + 1) % 5 == 0:
                logger.info(f"  Evaluated {i + 1}/{self.n_init} initial samples")

        self.X_observed = np.array(self.X_observed, dtype=np.float32)
        self.y_observed = np.array(self.y_observed, dtype=np.float32)

        logger.info(
            f"Initial sampling complete. "
            f"Best initial y: {self.y_observed.max():.4f}"
        )

    def fit_gp(self) -> None:
        """Fit GP to observed data."""
        self.gp.fit(self.X_observed, self.y_observed)
        logger.debug(f"GP fitted on {len(self.y_observed)} samples")

    def maximize_acquisition(self) -> np.ndarray:
        """
        Maximize acquisition function to find next candidate point.

        Uses L-BFGS-B with multiple random restarts for robustness.

        Returns:
            Optimal normalized parameters, shape (n_params,)
        """
        y_best = self.y_observed.max()

        def neg_acq(x):
            x = x.reshape(1, -1)
            acq = self.acquisition_fn(x, self.gp, y_best)
            return -acq[0]

        def grad_acq(x):
            eps = 1e-5
            grad = np.zeros_like(x)
            for i in range(len(x)):
                x_plus = x.copy()
                x_plus[i] += eps
                x_minus = x.copy()
                x_minus[i] -= eps
                grad[i] = (neg_acq(x_plus) - neg_acq(x_minus)) / (2 * eps)
            return grad

        # Multiple restarts
        best_x = None
        best_acq = -np.inf

        n_restarts = min(10, self.search_space.n_params * 2)
        for _ in range(n_restarts):
            x0 = self.rng.uniform(0, 1, self.search_space.n_params)

            result = minimize(
                neg_acq,
                x0,
                method="L-BFGS-B",
                bounds=[(0, 1)] * self.search_space.n_params,
                options={"maxiter": 100},
            )

            acq_val = -result.fun
            if acq_val > best_acq:
                best_acq = acq_val
                best_x = result.x

        # Clip to valid range
        best_x = np.clip(best_x, 0, 1)

        return best_x

    def optimize(self) -> dict:
        """
        Run full BO optimization loop.

        Returns:
            Dictionary with optimization results
        """
        logger.info("=" * 80)
        logger.info("Starting Bayesian Optimization")
        logger.info("=" * 80)
        logger.info(f"Initial samples: {self.n_init}")
        logger.info(f"BO iterations: {self.n_iter}")
        logger.info(f"Total evaluations: {self.n_init + self.n_iter}")

        # Step 1: Initial random sampling
        self.initialize_with_random_samples()

        # Step 2: BO iterations
        logger.info("\n" + "=" * 80)
        logger.info("BO Iterations")
        logger.info("=" * 80)

        for iteration in range(self.n_iter):
            # Fit GP to current data
            self.fit_gp()

            # Maximize acquisition
            x_next = self.maximize_acquisition()

            # Evaluate at new point
            config_next = self.search_space.vector_to_dict(x_next)
            y_next = self.objective_fn(config_next)

            # Update observed data
            self.X_observed = np.vstack([self.X_observed, x_next.reshape(1, -1)])
            self.y_observed = np.append(self.y_observed, y_next)

            # Compute GP predictions for logging
            mu_next, sigma_next = self.gp.predict(x_next.reshape(1, -1), return_std=True)
            y_best = self.y_observed.max()

            # Log iteration
            iter_info = {
                "iteration": iteration + 1,
                "y": float(y_next),
                "y_best": float(y_best),
                "gp_mean": float(mu_next[0]),
                "gp_std": float(sigma_next[0]),
                "config": config_next,
            }
            self.iteration_history.append(iter_info)

            if (iteration + 1) % 10 == 0 or iteration == 0:
                logger.info(
                    f"Iteration {iteration + 1:3d} | "
                    f"y={y_next:.4f} | "
                    f"y_best={y_best:.4f} | "
                    f"μ={mu_next[0]:.4f} | "
                    f"σ={sigma_next[0]:.4f}"
                )

        logger.info("\n" + "=" * 80)
        logger.info("BO Optimization Complete")
        logger.info("=" * 80)

        # Find best configuration
        best_idx = self.y_observed.argmax()
        x_best = self.X_observed[best_idx]
        y_best = self.y_observed[best_idx]
        config_best = self.search_space.vector_to_dict(x_best)

        # Compute uncertainty bounds
        mu_best, sigma_best = self.gp.predict(x_best.reshape(1, -1), return_std=True)
        ci_lower = max(0.0, mu_best[0] - 1.96 * sigma_best[0])
        ci_upper = min(1.0, mu_best[0] + 1.96 * sigma_best[0])

        logger.info(f"Best score: {y_best:.4f}")
        logger.info(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        logger.info(f"Best config:\n  {config_best}")

        return {
            "x_best": x_best,
            "y_best": float(y_best),
            "config_best": config_best,
            "gp_mean_best": float(mu_best[0]),
            "gp_std_best": float(sigma_best[0]),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "X_observed": self.X_observed,
            "y_observed": self.y_observed,
            "iteration_history": self.iteration_history,
            "gp": self.gp,
        }
