#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-Optimizer Benchmark: BO vs DE vs CMA-ES vs Random on v6 objective.

Tests whether BO is actually the best architecture for this optimization problem.

Optimizers:
  BO          Bayesian Optimization with GP (current default)
  DE          Differential Evolution (scipy) — strong continuous baseline
  CMA-ES      Covariance Matrix Adaptation ES (custom impl, no cma package needed)
  Random      Pure random search — always the minimum baseline

Budget: 100 evaluations per run, 20 runs per optimizer.

Usage (from project root):
    python BO/benchmarks/multi_optimizer_benchmark.py
    python BO/benchmarks/multi_optimizer_benchmark.py --budget 200 --runs 30
    python BO/benchmarks/multi_optimizer_benchmark.py --objective v3
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from BO.core.surrogate_loader import SurrogateLoaderV3
from search_space.biosensor_space import BiosensorSearchSpace
from evaluation.physics_forward_model import PhysicsForwardModel
from evaluation.therapeutic_objective_v6 import TherapeuticObjectiveV6
from acquisition.acquisition_functions import ExpectedImprovement
from optimizer.gaussian_process_bo import GaussianProcessBO

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
CONSOLE = logging.getLogger("multi_opt")
CONSOLE.setLevel(logging.INFO)
CONSOLE.propagate = False
if not CONSOLE.handlers:
    _ch = logging.StreamHandler(sys.stdout)
    _ch.setFormatter(logging.Formatter("%(message)s"))
    CONSOLE.addHandler(_ch)


# ──────────────────────────────────────────────────────────────────────────────
# Optimizer implementations
# ──────────────────────────────────────────────────────────────────────────────

def run_bo(objective_fn, search_space, n_init, n_iter, seed):
    """BO with GP-EI (existing GaussianProcessBO implementation)."""
    acq = ExpectedImprovement(xi=0.01)
    bo = GaussianProcessBO(
        objective_fn, search_space, acq,
        n_init=n_init, n_iter=n_iter, random_state=seed,
    )
    results = bo.optimize()
    return float(results["y_best"])


def run_random(objective_fn, search_space, n_budget, seed):
    """Pure random search."""
    rng = np.random.RandomState(seed)
    best = -np.inf
    for _ in range(n_budget):
        x = rng.uniform(0, 1, search_space.n_params)
        score = objective_fn(search_space.vector_to_dict(x))
        best = max(best, score)
    return best


def run_de(objective_fn, search_space, n_budget, seed):
    """
    Differential Evolution (scipy).

    scipy's DE handles bounds internally. We wrap objective to negate
    (DE minimizes; we maximize). Budget is approximately popsize * maxiter.
    """
    n = search_space.n_params
    bounds = [(0.0, 1.0)] * n

    call_count = [0]
    best_seen = [-np.inf]

    def neg_obj(x):
        if call_count[0] >= n_budget:
            return 0.0
        cfg = search_space.vector_to_dict(np.clip(x, 0, 1))
        score = objective_fn(cfg)
        call_count[0] += 1
        best_seen[0] = max(best_seen[0], score)
        return -score

    # popsize=8 → population of 8*n each generation
    # maxiter * popsize * n ≈ n_budget
    popsize = 6
    maxiter = max(1, n_budget // (popsize * n))

    differential_evolution(
        neg_obj, bounds,
        seed=seed, maxiter=maxiter, popsize=popsize,
        mutation=(0.5, 1.0), recombination=0.7,
        tol=1e-4, init="latinhypercube",
    )
    return best_seen[0]


def run_cma_es(objective_fn, search_space, n_budget, seed):
    """
    Simplified CMA-ES (Covariance Matrix Adaptation Evolution Strategy).

    Custom implementation — does not require the 'cma' package.
    Operates in normalized [0,1]^n space with periodic restarts on stagnation.

    Reference: Hansen 2016 "The CMA Evolution Strategy: A Tutorial"
    """
    n = search_space.n_params
    rng = np.random.RandomState(seed)

    # Strategy parameters (from Hansen's tutorial defaults)
    lam = 4 + int(3 * np.log(n))        # offspring per generation
    mu = lam // 2                         # parents used for update
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()
    mu_eff = 1.0 / np.sum(weights ** 2)  # effective sample size

    cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
    cs = (mu_eff + 2) / (n + mu_eff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mu_eff)
    cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
    ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + cs
    chi_n = n ** 0.5 * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

    # State variables
    m = rng.uniform(0.2, 0.8, n)
    sigma = 0.3
    pc = np.zeros(n)
    ps = np.zeros(n)
    C = np.eye(n)
    sqrtC = np.eye(n)      # initialized to I; updated lazily by eigendecomposition
    invsqrtC = np.eye(n)   # initialized to I; updated lazily by eigendecomposition
    eigeneval = 0

    best_score = -np.inf
    total_evals = 0

    while total_evals < n_budget:
        # Eigendecomposition (cached for efficiency)
        if total_evals - eigeneval > lam / (c1 + cmu) / n / 10:
            eigeneval = total_evals
            C = np.triu(C) + np.triu(C, 1).T  # enforce symmetry
            C = np.clip(C, -5, 5)
            try:
                D, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D, 1e-20))
                invsqrtC = B @ np.diag(1.0 / D) @ B.T
                sqrtC = B @ np.diag(D) @ B.T
            except np.linalg.LinAlgError:
                C = np.eye(n)
                D, B = np.ones(n), np.eye(n)
                invsqrtC = np.eye(n)
                sqrtC = np.eye(n)

        # Sample offspring
        z_list = [rng.randn(n) for _ in range(lam)]
        y_list = [sqrtC @ z for z in z_list]
        x_list = [np.clip(m + sigma * y, 0, 1) for y in y_list]

        # Evaluate
        fitness = []
        for x in x_list:
            if total_evals >= n_budget:
                break
            score = objective_fn(search_space.vector_to_dict(x))
            fitness.append(score)
            best_score = max(best_score, score)
            total_evals += 1

        if len(fitness) == 0:
            break

        # Rank (maximize = negate for CMA indexing)
        ranked_idx = np.argsort(fitness)[::-1][:mu]

        # Update mean
        x_top = np.array([x_list[i] for i in ranked_idx[:len(fitness)]])
        y_top = np.array([y_list[i] for i in ranked_idx[:len(fitness)]])
        actual_mu = min(mu, len(ranked_idx))
        if actual_mu == 0:
            continue

        w = weights[:actual_mu]
        w /= w.sum()

        m_old = m.copy()
        m = np.clip(np.dot(w, x_top[:actual_mu]), 0, 1)

        # Cumulation: update evolution paths
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * invsqrtC @ (m - m_old) / sigma
        hs = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * total_evals / lam)) / chi_n
              < 1.4 + 2 / (n + 1))
        pc = (1 - cc) * pc + hs * np.sqrt(cc * (2 - cc) * mu_eff) * (m - m_old) / sigma

        # Update covariance
        artmp = (1 / sigma) * (x_top[:actual_mu] - m_old)
        C = ((1 - c1 - cmu) * C
             + c1 * (np.outer(pc, pc) + (1 - hs) * cc * (2 - cc) * C)
             + cmu * artmp.T @ np.diag(w) @ artmp)

        # Update step size
        sigma = sigma * np.exp((cs / ds) * (np.linalg.norm(ps) / chi_n - 1))
        sigma = np.clip(sigma, 1e-4, 1.0)

        # Restart if stagnated
        if sigma < 1e-3 or sigma > 2.0:
            m = rng.uniform(0.1, 0.9, n)
            sigma = 0.3
            C = np.eye(n)
            ps = np.zeros(n)
            pc = np.zeros(n)

    return best_score


def run_nsga2_single(objective_fn, search_space, n_budget, seed):
    """
    NSGA-II reduced to single objective (= elitist (μ+λ)-EA).

    When called with a single-objective function, NSGA-II's non-dominated sorting
    reduces to simple fitness ranking. We use crossover + mutation from the NSGA-II
    spec with polynomial mutation and simulated binary crossover (SBX).
    """
    n = search_space.n_params
    rng = np.random.RandomState(seed)

    pop_size = min(20, n_budget // 5)
    eta_c = 15.0   # SBX distribution index
    eta_m = 20.0   # polynomial mutation distribution index
    p_m = 1.0 / n  # mutation probability per gene

    def sbx_crossover(p1, p2):
        child1, child2 = p1.copy(), p2.copy()
        for i in range(n):
            if rng.rand() < 0.5:
                if abs(p1[i] - p2[i]) > 1e-10:
                    y1, y2 = min(p1[i], p2[i]), max(p1[i], p2[i])
                    beta = 1.0 + 2.0 * min(y1, 1 - y2) / (y2 - y1 + 1e-10)
                    alpha = 2.0 - beta ** (-(eta_c + 1))
                    u = rng.rand()
                    if u <= 1.0 / alpha:
                        betaq = (u * alpha) ** (1.0 / (eta_c + 1))
                    else:
                        betaq = (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta_c + 1))
                    child1[i] = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                    child2[i] = 0.5 * ((y1 + y2) + betaq * (y2 - y1))
        return np.clip(child1, 0, 1), np.clip(child2, 0, 1)

    def polynomial_mutation(x):
        x = x.copy()
        for i in range(n):
            if rng.rand() < p_m:
                u = rng.rand()
                if u < 0.5:
                    delta = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                x[i] = np.clip(x[i] + delta, 0, 1)
        return x

    # Initialize population
    pop = [rng.uniform(0, 1, n) for _ in range(pop_size)]
    fitness = [objective_fn(search_space.vector_to_dict(x)) for x in pop]
    n_evals = pop_size
    best = max(fitness)

    while n_evals < n_budget:
        # Create offspring
        offspring = []
        n_needed = min(pop_size, n_budget - n_evals)
        while len(offspring) < n_needed:
            idx1, idx2 = rng.choice(pop_size, 2, replace=False)
            c1, c2 = sbx_crossover(pop[idx1], pop[idx2])
            offspring.append(polynomial_mutation(c1))
            if len(offspring) < n_needed:
                offspring.append(polynomial_mutation(c2))

        off_fitness = [objective_fn(search_space.vector_to_dict(x)) for x in offspring]
        n_evals += len(off_fitness)
        best = max(best, max(off_fitness) if off_fitness else best)

        # Elitist (μ+λ) selection: keep top pop_size from combined pool
        combined_x = pop + offspring
        combined_f = fitness + off_fitness
        ranked = sorted(zip(combined_f, range(len(combined_x))), reverse=True)
        pop = [combined_x[i] for _, i in ranked[:pop_size]]
        fitness = [combined_f[i] for _, i in ranked[:pop_size]]

    return best


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-Optimizer Benchmark")
    parser.add_argument("--surrogate-dir", type=Path, default=Path("BO/bo_results"))
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--n-init", type=int, default=20,
                        help="BO initial random samples per run")
    parser.add_argument("--objective", choices=["v6"], default="v6")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = Path(f"BO/bo_results/results/multi_optimizer_benchmark_{args.objective}.json")

    CONSOLE.info("=" * 70)
    CONSOLE.info("GENEVO2 — Multi-Optimizer Benchmark")
    CONSOLE.info("=" * 70)
    CONSOLE.info(f"  Objective  : {args.objective}")
    CONSOLE.info(f"  Budget/run : {args.budget}")
    CONSOLE.info(f"  Runs/opt   : {args.runs}")
    CONSOLE.info(f"  Surrogate  : {args.surrogate_dir}")
    CONSOLE.info("")

    surrogate = SurrogateLoaderV3(args.surrogate_dir)
    physics = PhysicsForwardModel()
    search_space = BiosensorSearchSpace()

    obj = TherapeuticObjectiveV6(physics, surrogate)

    n_bo_init = args.n_init
    n_bo_iter = args.budget - n_bo_init

    results = {}
    optimizers = [
        ("BO",       lambda seed: run_bo(obj, search_space, n_bo_init, n_bo_iter, seed)),
        ("DE",       lambda seed: run_de(obj, search_space, args.budget, seed)),
        ("CMA-ES",   lambda seed: run_cma_es(obj, search_space, args.budget, seed)),
        ("NSGA2-SE", lambda seed: run_nsga2_single(obj, search_space, args.budget, seed)),
        ("Random",   lambda seed: run_random(obj, search_space, args.budget, seed + 1000)),
    ]

    scores_by_opt = {}
    for name, opt_fn in optimizers:
        CONSOLE.info(f"[{name}] running {args.runs} trials ...")
        scores = []
        for i in range(args.runs):
            s = opt_fn(i)
            scores.append(s)
            CONSOLE.info(f"  {name:10s}  run {i+1:2d}/{args.runs}  best={s:.4f}")
        scores_by_opt[name] = scores
        m, sd = np.mean(scores), np.std(scores)
        CONSOLE.info(f"  {name:10s}  SUMMARY  mean={m:.4f} ± {sd:.4f}  "
                     f"min={np.min(scores):.4f}  max={np.max(scores):.4f}")
        CONSOLE.info("")

    # Statistics table
    CONSOLE.info("=" * 70)
    CONSOLE.info("RESULTS SUMMARY")
    CONSOLE.info("=" * 70)
    CONSOLE.info(f"  {'Optimizer':<12}  {'Mean':>8}  {'Std':>7}  {'Min':>7}  {'Max':>7}")
    CONSOLE.info(f"  {'-'*12}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}")
    for name in scores_by_opt:
        s = np.array(scores_by_opt[name])
        CONSOLE.info(f"  {name:<12}  {s.mean():>8.4f}  {s.std():>7.4f}  {s.min():>7.4f}  {s.max():>7.4f}")

    # Pairwise significance tests vs Random
    rand_scores = np.array(scores_by_opt["Random"])
    CONSOLE.info("")
    CONSOLE.info("Wilcoxon tests vs Random:")
    for name in scores_by_opt:
        if name == "Random":
            continue
        s = np.array(scores_by_opt[name])
        try:
            stat, p = wilcoxon(s, rand_scores)
            wins = int(np.sum(s > rand_scores))
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
            CONSOLE.info(f"  {name:<12}  lift={s.mean()-rand_scores.mean():+.4f}  "
                         f"wins={wins}/{args.runs}  p={p:.4f}  {sig}")
        except Exception as e:
            CONSOLE.info(f"  {name:<12}  test failed: {e}")

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "timestamp": datetime.now().isoformat(),
        "objective": args.objective,
        "budget": args.budget,
        "runs": args.runs,
        "scores": {k: list(v) for k, v in scores_by_opt.items()},
        "summary": {
            k: {
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
                "min": float(np.min(v)),
                "max": float(np.max(v)),
            }
            for k, v in scores_by_opt.items()
        },
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    CONSOLE.info(f"\nResults saved -> {args.output}")


if __name__ == "__main__":
    main()
