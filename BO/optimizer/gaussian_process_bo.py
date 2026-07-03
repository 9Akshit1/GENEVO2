#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gaussian Process-based Bayesian Optimization.

Main BO algorithm using sklearn's GaussianProcessRegressor with Matern kernel
and Expected Improvement (EI) acquisition function.
"""

import warnings
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.exceptions import ConvergenceWarning
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

        # Identify which search-space dimensions actually affect the objective.
        # Single-value categoricals (biosensor_type, noise_preset) are always
        # decoded to the same value regardless of the Sobol sample — pure noise.
        # response_time_s was removed from the search space (PI=0, fixed at 600s).
        # target_scenario is explicitly documented as "not used" by ObjectiveFunctionV3.
        # Giving the GP noisy dims corrupts the kernel — fit the GP on real dims only.
        self._active_dims = self._get_active_dims()
        logger.info(
            f"GP active dims: {len(self._active_dims)}/{search_space.n_params} "
            f"({[list(search_space.parameters.keys())[i] for i in self._active_dims]})"
        )

        # Initialize GP — wider bounds prevent hyperparameter-at-bound ConvergenceWarnings
        kernel = Matern(nu=2.5, length_scale_bounds=(0.01, 100.0)) + WhiteKernel(
            noise_level_bounds=(1e-8, 1.0)
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

    def _get_active_dims(self) -> np.ndarray:
        """Return indices of dimensions that genuinely affect the objective.

        Excluded:
          - Single-value categoricals (biosensor_type=['array'],
            noise_preset=['realistic']): Sobol samples a random float but
            vector_to_dict always decodes to the same string → y is uncorrelated
            with this dimension.
          - target_scenario: ObjectiveFunctionV3 explicitly ignores this field and
            always evaluates all four scenarios.
          (response_time_s was removed from BiosensorSearchSpace; fixed at 600s.)
        """
        active = []
        skip_names = {"target_scenario"}
        for i, (name, param) in enumerate(self.search_space.parameters.items()):
            if param.param_type == "categorical" and len(param.values) <= 1:
                continue
            if name in skip_names:
                continue
            active.append(i)
        return np.array(active, dtype=int)

    def initialize_with_random_samples(self) -> None:
        """Generate initial random samples using Sobol sequence."""
        logger.info(f"Generating {self.n_init} initial random samples...")

        # Use Sobol quasi-random sampling for better coverage.
        # Sobol sequences require n to be a power of 2 for correct balance
        # properties.  We round n_init up to the next power of 2, generate
        # that many points, then truncate to exactly n_init so the number of
        # evaluations remains predictable.
        n_sobol = 1 << (self.n_init - 1).bit_length()  # next power of 2 >= n_init
        try:
            try:
                sampler = qmc.Sobol(d=self.search_space.n_params, random_state=self.rng)
            except TypeError:
                sampler = qmc.Sobol(d=self.search_space.n_params, seed=self.random_state)
            X_init_normalized = sampler.random(n=n_sobol)[: self.n_init]
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
        """Fit GP on active (non-noise) dimensions only."""
        X_gp = self.X_observed[:, self._active_dims]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.gp.fit(X_gp, self.y_observed)
        logger.debug(f"GP fitted on {len(self.y_observed)} samples, {len(self._active_dims)} active dims")

    def maximize_acquisition(self) -> np.ndarray:
        """
        Maximize acquisition function in the active-dimension subspace.

        The GP is trained on active dims only, so we optimize in that reduced
        space (fewer dims = faster, cleaner gradient signal). Inactive dims
        are filled with 0.5 in the returned full-length vector — they decode
        to their single valid value and do not affect the objective.

        Returns:
            Optimal normalized parameters, shape (n_params,)
        """
        y_best = self.y_observed.max()
        active = self._active_dims
        n_active = len(active)

        def neg_acq(x_active):
            x_active = x_active.reshape(1, -1)
            acq = self.acquisition_fn(x_active, self.gp, y_best)
            return -acq[0]

        best_x_active = None
        best_acq_val = -np.inf

        n_restarts = min(10, n_active * 2 + 2)
        for _ in range(n_restarts):
            x0 = self.rng.uniform(0, 1, n_active)
            result = minimize(
                neg_acq,
                x0,
                method="L-BFGS-B",
                bounds=[(0, 1)] * n_active,
                options={"maxiter": 200},
            )
            if -result.fun > best_acq_val:
                best_acq_val = -result.fun
                best_x_active = result.x

        best_x_active = np.clip(best_x_active, 0, 1)

        # Reconstruct full parameter vector; inactive dims set to midpoint
        # (they decode to fixed constants and do not affect objective)
        best_x = np.full(self.search_space.n_params, 0.5)
        best_x[active] = best_x_active
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

            # Compute GP predictions for logging (use active dims only)
            mu_next, sigma_next = self.gp.predict(
                x_next[self._active_dims].reshape(1, -1), return_std=True
            )
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
                    f"mu={mu_next[0]:.4f} | "
                    f"std={sigma_next[0]:.4f}"
                )

        logger.info("\n" + "=" * 80)
        logger.info("BO Optimization Complete")
        logger.info("=" * 80)

        # Find best configuration
        best_idx = self.y_observed.argmax()
        x_best = self.X_observed[best_idx]
        y_best = self.y_observed[best_idx]
        config_best = self.search_space.vector_to_dict(x_best)

        # Compute uncertainty bounds (GP uses active dims only)
        mu_best, sigma_best = self.gp.predict(
            x_best[self._active_dims].reshape(1, -1), return_std=True
        )
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
