#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Acquisition functions for Bayesian Optimization.

Implements Expected Improvement (EI), Upper Confidence Bound (UCB),
and Probability of Improvement (PI).
"""

import numpy as np
from scipy.stats import norm
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class AcquisitionFunction(ABC):
    """Base class for acquisition functions."""

    @abstractmethod
    def __call__(self, X: np.ndarray, gp, y_best: float) -> np.ndarray:
        """
        Evaluate acquisition function.

        Args:
            X: Points at which to evaluate, shape (n_points, n_features)
            gp: Fitted Gaussian Process
            y_best: Best observed function value so far

        Returns:
            Acquisition values, shape (n_points,)
        """
        pass


class ExpectedImprovement(AcquisitionFunction):
    """
    Expected Improvement acquisition function.

    EI(x) = E[max(f(x) - y_best - xi, 0)]
          = (mu - y_best - xi) * Phi(Z) + sigma * phi(Z)
    where Z = (mu - y_best - xi) / sigma

    Args:
        xi: Exploration parameter (jitter). Higher xi = more exploration.
    """

    def __init__(self, xi: float = 0.01):
        """
        Initialize EI acquisition function.

        Args:
            xi: Jitter for exploration (default 0.01)
        """
        self.xi = xi

    def __call__(self, X: np.ndarray, gp, y_best: float) -> np.ndarray:
        """
        Evaluate EI at given points.

        Args:
            X: Points to evaluate, shape (n_points, n_features)
            gp: Fitted GP
            y_best: Best observed value

        Returns:
            EI values, shape (n_points,)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        mu, sigma = gp.predict(X, return_std=True)

        # Avoid division by zero
        sigma = np.maximum(sigma, 1e-9)

        Z = (mu - y_best - self.xi) / sigma
        ei = (mu - y_best - self.xi) * norm.cdf(Z) + sigma * norm.pdf(Z)

        # Set EI to zero where sigma is very small (highly certain predictions)
        ei[sigma < 1e-9] = 0.0

        return ei


class UpperConfidenceBound(AcquisitionFunction):
    """
    Upper Confidence Bound acquisition function.

    UCB(x) = mu(x) + beta * sigma(x)

    Args:
        beta: Exploration-exploitation trade-off parameter.
              Higher beta = more exploration.
    """

    def __init__(self, beta: float = 2.576):
        """
        Initialize UCB acquisition function.

        Args:
            beta: Confidence parameter (default 2.576 for ~2 sigma confidence)
        """
        self.beta = beta

    def __call__(self, X: np.ndarray, gp, y_best: float) -> np.ndarray:
        """
        Evaluate UCB at given points.

        Args:
            X: Points to evaluate, shape (n_points, n_features)
            gp: Fitted GP (y_best is ignored)
            y_best: Unused for UCB

        Returns:
            UCB values, shape (n_points,)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        mu, sigma = gp.predict(X, return_std=True)
        ucb = mu + self.beta * sigma

        return ucb


class ProbabilityOfImprovement(AcquisitionFunction):
    """
    Probability of Improvement acquisition function.

    PI(x) = P(f(x) >= y_best + xi)
          = Phi((mu - y_best - xi) / sigma)

    Args:
        xi: Minimum improvement threshold (default 0.0)
    """

    def __init__(self, xi: float = 0.0):
        """
        Initialize PI acquisition function.

        Args:
            xi: Minimum improvement (default 0.0)
        """
        self.xi = xi

    def __call__(self, X: np.ndarray, gp, y_best: float) -> np.ndarray:
        """
        Evaluate PI at given points.

        Args:
            X: Points to evaluate, shape (n_points, n_features)
            gp: Fitted GP
            y_best: Best observed value

        Returns:
            PI values, shape (n_points,)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        mu, sigma = gp.predict(X, return_std=True)

        # Avoid division by zero
        sigma = np.maximum(sigma, 1e-9)

        Z = (mu - y_best - self.xi) / sigma
        pi = norm.cdf(Z)

        return pi
