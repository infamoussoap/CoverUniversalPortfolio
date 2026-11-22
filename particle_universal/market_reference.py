"""Simplex sampling and the best constant-rebalanced portfolio."""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc


def _normalize(returns):
    returns = np.asarray(returns, dtype=float)
    if returns.ndim != 2 or len(returns) == 0 or returns.shape[1] < 2:
        raise ValueError("Expected a nonempty market with at least two assets")

    scale = returns.max(axis=1)
    if (not np.all(np.isfinite(returns)) or np.any(returns < 0)
            or np.any(scale <= 0)):
        raise ValueError("Gross returns need a finite positive row maximum")

    return returns / scale[:, None], np.log(scale)


def best_fixed(returns):
    """ Maximize realized log wealth over constant simplex allocations.

        Parameters
        ----------
        returns : (T, d) np.ndarray
            Gross asset returns for T rounds and d assets.

        Returns
        -------
        weights : (d,) np.ndarray
            Best constant-rebalanced allocation in hindsight.
        log_wealth : float
            Cumulative natural-log return on the supplied market.
    """
    market, log_scale = _normalize(returns)
    T, d = market.shape
    floor = np.finfo(float).tiny

    def objective(w):
        payoff = market @ w
        if np.any(payoff <= 0):
            return np.inf

        return -np.log(payoff).mean()

    def gradient(w):
        payoff = np.maximum(market @ w, floor)
        return -market.T @ ((1 / T) / payoff)

    constraint = {"type": "eq", "fun": lambda w: w.sum() - 1,
                  "jac": lambda w: np.ones_like(w)}
    solution = minimize(objective, np.full(d, 1 / d), jac=gradient,
                        method="SLSQP", bounds=[(0, 1)] * d,
                        constraints=constraint,
                        options={"ftol": 1e-14, "maxiter": 500})

    if (not np.isfinite(solution.fun) or np.min(solution.x) < -1e-10
            or abs(solution.x.sum() - 1) > 1e-10):
        raise RuntimeError(f"Invalid hindsight optimum: {solution.message}")

    weights = np.maximum(solution.x, 0)
    weights /= weights.sum()
    grad = gradient(weights)

    # Convexity makes the linearized objective gap a global error bound.
    gap = grad @ weights - grad.min()
    if not np.isfinite(gap) or gap > 1e-7:
        raise RuntimeError(f"Hindsight optimizer gap {gap:g}")

    return weights, float(log_scale.sum() - T * objective(weights))


def _prior_particles(d, power, rng):
    uniforms = qmc.Sobol(d - 1, scramble=True, rng=rng).random_base2(power)
    log_remaining = np.log1p(-uniforms) / np.arange(d - 1, 0, -1)

    # The remaining stick fractions have Beta(d-i-1, 1) distributions.
    remaining = np.ones((len(uniforms), d))
    remaining[:, 1:] = np.exp(np.cumsum(log_remaining, axis=1))
    particles = np.empty_like(remaining)
    particles[:, :-1] = remaining[:, :-1] * (-np.expm1(log_remaining))
    particles[:, -1] = remaining[:, -1]
    return particles
