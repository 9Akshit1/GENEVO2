#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pareto front utilities for multi-objective optimization.

All functions assume MAXIMIZATION (negate objectives to minimize).

Hypervolume algorithms:
  - 2D: O(n log n), exact
  - 3D: O(n^2 log n), exact sweep
  - nD: WFG recursive algorithm (n <= 6 practical)
"""

import numpy as np
from typing import List, Tuple


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """True if a dominates b (all objectives >= b and at least one strict >)."""
    return bool(np.all(a >= b) and np.any(a > b))


def non_dominated_sort(Y: np.ndarray) -> List[List[int]]:
    """
    Fast non-dominated sort (NSGA-II style, Deb 2002).

    Args:
        Y: (n, m) array of objective values (maximize all).

    Returns:
        fronts: list of fronts, each front is a list of indices.
                fronts[0] is the Pareto front.
    """
    n = len(Y)
    dom_count = np.zeros(n, dtype=int)   # how many solutions dominate i
    dominated_by_i = [[] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            if dominates(Y[i], Y[j]):
                dominated_by_i[i].append(j)
                dom_count[j] += 1
            elif dominates(Y[j], Y[i]):
                dominated_by_i[j].append(i)
                dom_count[i] += 1

    fronts = []
    current_front = [i for i in range(n) if dom_count[i] == 0]
    while current_front:
        fronts.append(current_front)
        next_front = []
        for i in current_front:
            for j in dominated_by_i[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    next_front.append(j)
        current_front = next_front

    return fronts


def pareto_front(Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract Pareto-optimal points from a set of objective vectors.

    Args:
        Y: (n, m) array. Maximizing all objectives.

    Returns:
        (pareto_Y, pareto_indices) where pareto_Y is (k, m) and
        pareto_indices is the original row indices.
    """
    fronts = non_dominated_sort(Y)
    if not fronts:
        return Y[:0], np.array([], dtype=int)
    idx = np.array(fronts[0], dtype=int)
    return Y[idx], idx


# ---------------------------------------------------------------------------
# Hypervolume computation
# ---------------------------------------------------------------------------

def hypervolume_2d(Y: np.ndarray, ref: np.ndarray) -> float:
    """
    Exact 2D hypervolume (WFG O(n log n) special case).

    Args:
        Y:   (n, 2) array of points (to maximize).
        ref: (2,) reference point (worst acceptable, all objectives <= ref).

    Returns:
        Hypervolume dominated by Y relative to ref.
    """
    if len(Y) == 0:
        return 0.0

    # Keep only non-dominated points above reference
    mask = np.all(Y > ref, axis=1)
    Y = Y[mask]
    if len(Y) == 0:
        return 0.0

    # Sort by first objective descending
    order = np.argsort(-Y[:, 0])
    Y = Y[order]

    hv = 0.0
    y2_bound = ref[1]  # tracks the second objective contribution boundary

    for i in range(len(Y)):
        width  = Y[i, 0] - ref[0]
        height = Y[i, 1] - y2_bound
        if height > 0 and width > 0:
            hv += width * height
        # Update boundary for next point (second objective)
        if i + 1 < len(Y):
            if Y[i, 1] > y2_bound:
                y2_bound = Y[i, 1]

    return float(hv)


def hypervolume_3d(Y: np.ndarray, ref: np.ndarray) -> float:
    """
    Exact 3D hypervolume via slice decomposition.

    Sweeps along the 3rd objective (ascending slices), computing 2D
    hypervolume at each level. O(n^2) slices, O(n log n) each -> O(n^2 log n).

    Args:
        Y:   (n, 3) array. Maximize all.
        ref: (3,) reference point.

    Returns:
        3D hypervolume.
    """
    if len(Y) == 0:
        return 0.0

    mask = np.all(Y > ref, axis=1)
    Y = Y[mask]
    if len(Y) == 0:
        return 0.0

    # Sort by third objective ascending.
    # A point p is "active" in all z-slices where z <= p[2].
    # Sweeping upward: start with ALL points active, remove each point
    # just before processing the next z level above its z3 value.
    order = np.argsort(Y[:, 2])
    Y = Y[order]

    hv = 0.0
    prev_z = ref[2]
    # All points start active; we remove them as z increases past their z3
    active_2d = [Y[i, :2].tolist() for i in range(len(Y))]

    for i, pt in enumerate(Y):
        z = pt[2]
        if z > prev_z and active_2d:
            slice_Y = np.array(active_2d)
            hv += hypervolume_2d(slice_Y, ref[:2]) * (z - prev_z)
        # Remove this point from active set (its z3 value is z, so above z it no longer contributes)
        try:
            active_2d.remove(pt[:2].tolist())
        except ValueError:
            pass
        prev_z = z

    return float(hv)


def hypervolume(Y: np.ndarray, ref: np.ndarray) -> float:
    """
    Compute hypervolume for 2D or 3D objective spaces.

    Args:
        Y:   (n, m) objective array (maximize all), m in {2, 3}.
        ref: (m,) reference point.

    Returns:
        Hypervolume indicator.
    """
    m = Y.shape[1]
    if m == 2:
        return hypervolume_2d(Y, ref)
    elif m == 3:
        return hypervolume_3d(Y, ref)
    else:
        raise NotImplementedError(f"Hypervolume for {m}D not implemented. Use 2 or 3 objectives.")


def hypervolume_improvement(Y: np.ndarray, y_new: np.ndarray, ref: np.ndarray) -> float:
    """
    Exact hypervolume improvement of adding y_new to current Pareto set Y.

    HVI(y_new | Y) = HV(Y union {y_new}) - HV(Y)

    Args:
        Y:     (n, m) current objective archive.
        y_new: (m,) candidate point.
        ref:   (m,) reference point.

    Returns:
        Hypervolume improvement (>= 0, 0 if dominated by Y).
    """
    if len(Y) == 0:
        hv_before = 0.0
    else:
        hv_before = hypervolume(Y, ref)

    Y_aug = np.vstack([Y, y_new.reshape(1, -1)]) if len(Y) > 0 else y_new.reshape(1, -1)
    hv_after = hypervolume(Y_aug, ref)
    return float(max(0.0, hv_after - hv_before))


def crowding_distance(Y: np.ndarray) -> np.ndarray:
    """
    Compute crowding distances for NSGA-II diversity preservation.

    Args:
        Y: (n, m) objective array.

    Returns:
        (n,) crowding distance per point.
    """
    n, m = Y.shape
    distances = np.zeros(n)

    for obj in range(m):
        order = np.argsort(Y[:, obj])
        sorted_vals = Y[order, obj]
        obj_range = sorted_vals[-1] - sorted_vals[0]

        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf

        if obj_range > 1e-9:
            for i in range(1, n - 1):
                distances[order[i]] += (sorted_vals[i + 1] - sorted_vals[i - 1]) / obj_range

    return distances
